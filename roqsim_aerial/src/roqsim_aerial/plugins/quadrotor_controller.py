"""Controller plugin: quadrotor position + attitude control over collective thrust and body moments.

The aerial counterpart to :mod:`roqsim_mobile.plugins.diff_drive`. It closes the loop a quadrotor
MJCF cannot: Menagerie's Crazyflie exposes ``body_thrust`` plus three body-moment actuators and no
stabiliser at all, so an uncommanded drone is not a robot standing still -- it is a falling brick.
Every aerial experiment needs this layer before it can ask any other question.

Cascaded, in the usual quadrotor form:

1. **Position -> desired acceleration.** A PD on position and velocity error, plus gravity
   feed-forward. Saturated as an acceleration, which is the physically meaningful place to limit
   aggressiveness (a tilt limit alone still lets the controller ask for infinite thrust).
2. **Desired acceleration -> thrust + desired attitude.** Thrust is the desired force projected on
   the *current* body z, so the drone does not command thrust it cannot yet direct; the desired
   attitude is the rotation whose z axis is the desired acceleration direction, carrying the
   commanded yaw.
3. **Attitude -> body moments.** The standard SO(3) error ``e_R = 0.5 * (Rd^T R - R^T Rd)^vee`` with
   a rate damping term. This is used rather than Euler angles because it does not degenerate as the
   drone tilts, and a quadrotor recovering from a large disturbance does tilt.

Config -- a component of the entry that spawns the drone, since ownership is where the entry
sits rather than a config key::

    quadrotor_controller:
      namespace: ""                 # transport scope (default: inherited from spawn_robot)
      body: cf2                     # the drone's root body (default: the entity's root)
      thrust_actuator: body_thrust  # collective thrust, in newtons
      moment_actuators: [x_moment, y_moment, z_moment]
      target: [0.0, 0.0, 1.0]       # position setpoint (x, y, z), world frame
      yaw: 0.0                      # heading setpoint (rad)
      max_tilt: 0.5                 # rad, cap on commanded tilt from vertical
      max_accel: 4.0                # m/s^2, cap on the commanded horizontal acceleration
      max_vel: 1.5                  # m/s, cap on the velocity a position error may ask for
      kp_pos: [3.0, 3.0, 12.0]      # position gains (x, y, z)
      kd_pos: [2.4, 2.4, 6.0]       # velocity gains
      kp_att: [0.0096, 0.0096, 0.0038]    # attitude gains (roll, pitch, yaw), N*m per unit error
      kd_att: [0.00086, 0.00086, 0.00051] # body-rate damping, N*m per rad/s
      cmd_vel_timeout: 0.0          # s; > 0 brings a velocity-commanded drone to a hover (see below)

**Odometry and a stale command**, in three dimensions. ``odom`` is the 6-DOF pose in the spawn
frame (:class:`roqsim.odometry.SpawnFrame`): x and y from the spawn point, turned to the spawn
heading, ``z`` the altitude as in the world, tilt kept; its twist is in the body frame, as
``nav_msgs/Odometry`` states it. ``read_state`` is the world-frame truth. ``cmd_vel_timeout`` applies
to the velocity command (``drive``): once it is stale the drone brakes to a stop at its altitude
setpoint and then holds the position where it stopped, a hover. A position setpoint (``target``,
``cmd_pos``) does not expire, since holding one already is a hover. Reset clears both commands.

**A position setpoint is read in the frame it names.** ``cmd_pos`` carries a frame: ``odom`` is the
spawn frame above (``z`` the altitude), ``world`` or ``map`` is the world, and so is an empty
frame; the configured ``target`` is a world position. Any other frame is refused by name rather
than flown to as if it were one of these. A yaw is read in the same frame as the position.

**The moment actuators carry a negative gear**, so a positive ``ctrl`` produces a *negative* body
moment. The sign is read from the model at configure time rather than hardcoded -- it is upstream's
convention, and a future airframe need not share it.

**The attitude gains are in newton-metres, not normalised units**, because the controller emits a
moment and the model's actuator gear converts it to ``ctrl``. They are sized from the airframe: for a
body inertia I and a target attitude bandwidth wn with damping zeta, ``kp_att ~ I*wn^2`` and
``kd_att ~ 2*zeta*I*wn``. The defaults are I = 2.4e-5 kg*m^2 at wn = 20 rad/s, zeta = 0.9. This is
also why the Crazyflie's moment gear is tuned rather than upstream's: at the arbitrary 1e-5 N*m
upstream ships, full deflection buys 0.42 rad/s^2 and no attitude loop can track a position
controller's tilt command -- the drone hovers perfectly and flies away the moment it is asked to
translate. See the port log.

**Air matters.** ``density``/``viscosity`` default to 0 in MuJoCo, so a world that does not set
the world's ``density``/``viscosity`` flies the drone through a vacuum -- no drag, and a lateral step never settles.
The plugin logs a warning rather than silently flying in vacuum.
"""

from __future__ import annotations

import logging

import mujoco
import numpy as np

from roqsim import endpoint
from roqsim.context import RobotHandle, SimContext
from roqsim.kinematics import body_twist
from roqsim.odometry import CommandWatchdog, SpawnFrame
from roqsim.plugin import Plugin
from roqsim.pose import yaw_of
from roqsim.types import Odometry, Point3, Pose, Quaternion

logger = logging.getLogger(__name__)

_DEFAULTS = {
    "thrust_actuator": "body_thrust",
    "moment_actuators": ["x_moment", "y_moment", "z_moment"],
    "target": [0.0, 0.0, 1.0],
    "yaw": 0.0,
    "max_tilt": 0.5,
    "max_accel": 4.0,
    "max_vel": 1.5,
    "kp_pos": [3.0, 3.0, 12.0],
    "kd_pos": [2.4, 2.4, 6.0],
    "kp_att": [0.0096, 0.0096, 0.0038],
    "kd_att": [0.00086, 0.00086, 0.00051],
}

#: Frame names a position setpoint may carry for the world; empty is an unstamped setpoint.
_WORLD_FRAMES = ("", "world", "map")

#: m/s: below this horizontal speed a drone braking on a stale command holds where it is.
_HOVER_SPEED = 0.05


def _hat_vee(matrix: np.ndarray) -> np.ndarray:
    """The vee map: the axial vector of a 3x3 skew-symmetric matrix."""
    return np.array([matrix[2, 1], matrix[0, 2], matrix[1, 0]])


class QuadrotorControllerPlugin(Plugin):
    #: Drives an entity's actuators, so it belongs inside that entity's ``components:`` block.
    requires_owner = True

    def __init__(self, config=None, *, name=None, entity=None, label=None):
        super().__init__(config, name=name, entity=entity, label=label)
        self.robot = self.entity
        self._aid_thrust = -1
        self._aid_moments: list[int] = []
        self._bid = -1
        self._target = np.array(self.cfg("target"), dtype=float)
        self._yaw = float(self.cfg("yaw"))
        self._vel_cmd: np.ndarray | None = None
        self._yaw_rate = 0.0
        self.watchdog = CommandWatchdog.from_config(self.config)
        self._odom_frame = SpawnFrame()
        self._ctx: SimContext | None = None

    def cfg(self, key):
        return self.config.get(key, _DEFAULTS[key])

    def validate_config(self, config: dict) -> list[str]:
        errors = self.validate_topics(config)
        for key in ("target", "kp_pos", "kd_pos", "kp_att", "kd_att"):
            if key in config and len(config[key]) != 3:
                errors.append(f"'{key}' must be 3 numbers")
        if "moment_actuators" in config and len(config["moment_actuators"]) != 3:
            errors.append("'moment_actuators' must name exactly 3 actuators (x, y, z)")
        for key in ("max_tilt", "max_accel", "max_vel"):
            if key in config and float(config[key]) <= 0:
                errors.append(f"'{key}' must be > 0")
        if "max_tilt" in config and float(config["max_tilt"]) >= np.pi / 2:
            errors.append("'max_tilt' must be < pi/2: at 90 degrees a quadrotor has no lift left")
        errors += CommandWatchdog.validate(config)
        return errors

    def configure(self, ctx: SimContext) -> None:
        entity = ctx.entities.get(self.robot)
        prefix = entity.meta.get("prefix", "") if entity else ""
        model = ctx.model
        self._ctx = ctx

        def actuator(n):
            return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, prefix + n)

        self._aid_thrust = actuator(self.cfg("thrust_actuator"))
        self._aid_moments = [actuator(n) for n in self.cfg("moment_actuators")]
        # `entity.body` is the root body spawn_robot resolved against the compiled model, so it is
        # already prefixed; a config `body:` is in the model's own namespace and takes the prefix.
        configured = self.config.get("body")
        body = (prefix + str(configured)) if configured else (entity.body if entity else None)
        if body:
            # A named body that does not resolve is an error, not a cue to fly another one.
            self._bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
        elif self._aid_thrust >= 0:
            # No body named anywhere: a drone declared without spawn_robot (a bare MJCF world, a
            # test harness) has an entity with no body. The thrust actuator's site is where the
            # force is applied and therefore on the body being flown.
            site = model.actuator_trnid[self._aid_thrust, 0]
            self._bid = int(model.site_bodyid[site]) if site >= 0 else -1
        else:
            self._bid = -1

        missing = [
            name
            for name, aid in [
                (self.cfg("thrust_actuator"), self._aid_thrust),
                *zip(self.cfg("moment_actuators"), self._aid_moments, strict=True),
            ]
            if aid < 0
        ]
        if missing or self._bid < 0:
            raise RuntimeError(
                f"quadrotor_controller: could not resolve {missing or f'the drone body {body!r}'} "
                f"for robot {self.robot!r}"
            )

        # A vacuum is a silent, plausible-looking failure mode: the drone still hovers, but nothing
        # damps it, so a lateral step rings forever and the run looks like bad gains.
        if float(model.opt.density) == 0.0 and float(model.opt.viscosity) == 0.0:
            logger.warning(
                "quadrotor_controller (%s): the world has no medium (density and viscosity are 0), "
                "so this drone is flying in a vacuum and has no aerodynamic damping. Set "
                "sim: {density: 1.225, viscosity: 1.8e-5} for air.",
                self.robot,
            )

        self._mass = float(model.body_subtreemass[self._bid])
        self._gravity = float(-model.opt.gravity[2])
        self._thrust_range = tuple(float(v) for v in model.actuator_ctrlrange[self._aid_thrust])
        # gear[3:6] is the moment axis scaling; upstream's is negative, so remember the sign rather
        # than hardcoding it -- a future airframe may not share the convention.
        self._moment_gear = np.array(
            [float(model.actuator_gear[a][3 + i]) for i, a in enumerate(self._aid_moments)]
        )

        ctx.blackboard.set(
            f"robot:{self.robot}",
            RobotHandle(name=self.robot, drive=self.drive, read_odom=self.read_odom),
        )

    # -- commands ----------------------------------------------------------------------------

    def set_target(self, x, y, z, yaw=None, frame: str = "world") -> None:
        """Position setpoint in ``frame`` (see the module docstring); ``yaw`` keeps the current
        heading if omitted."""
        if frame == "odom":
            x, y, z = self._odom_frame.world_position((x, y, z))
            if yaw is not None:
                yaw = self._odom_frame.world_yaw(yaw)
        elif frame not in _WORLD_FRAMES:
            raise ValueError(
                f"quadrotor_controller ({self.robot}): a position setpoint in frame {frame!r} is "
                f"refused; it takes 'odom' (the spawn frame) or 'world'/'map'"
            )
        self._target = np.array([float(x), float(y), float(z)])
        self._vel_cmd = None
        if yaw is not None:
            self._yaw = float(yaw)

    # An airframe holds pitch and roll to fly, so only the heading of the commanded orientation is a
    # setpoint for it. The projection is here, with the consumer that wants it, rather than in the
    # transport: a Cartesian controller taking the same type needs the full orientation.
    @endpoint.stream(Pose)
    def cmd_pos(
        self, position: Point3, orientation: Quaternion | None = None, frame_id: str = ""
    ) -> None:
        """Position setpoint and heading, in the frame it names; applied once per step.

        Args:
            position: setpoint (x, y, z)
            orientation: quaternion (w, x, y, z) whose heading is flown; none keeps the heading
            frame_id: 'odom' (the spawn frame), 'world', 'map' or empty (the world); any other
                is refused and the setpoint kept
        """
        yaw = yaw_of(orientation) if orientation is not None else None
        self.set_target(*position, yaw, frame=frame_id)

    def drive(self, vx: float, vy: float, w: float) -> None:
        """:class:`RobotHandle` contract: body-frame planar velocity, altitude held.

        A quadrotor is not a ground robot, so this is a projection rather than its native command --
        it exists so teleop and the generic in-process consumers work unchanged. Altitude comes from
        the standing target; ``set_target`` is the full-authority command.
        """
        self._vel_cmd = np.array([float(vx), float(vy)])
        self._yaw_rate = float(w)  # integrated in pre_step, where dt is known
        self.watchdog.stamp(self._ctx)

    def read_state(self):
        """World-frame truth: ``(x, y, z, vx, vy, vz, yaw, yaw_rate)``."""
        return self._state

    def read_odom(self):
        o = self.read_odom6()
        return (o["x"], o["y"], self._odom_frame.yaw(self._quat), *self._planar_vel, o["wz"])

    @endpoint.out
    def odom(self) -> Odometry:
        """The 6-DOF pose from the spawn pose, tilt kept, and the body-frame twist."""
        o = self.read_odom6()
        return Odometry(
            np.array([o["x"], o["y"], o["z"]]),
            np.array([o["qw"], o["qx"], o["qy"], o["qz"]]),
            np.array([o["vx"], o["vy"], o["vz"]]),
            np.array([o["wx"], o["wy"], o["wz"]]),
        )

    def read_odom6(self):
        """Full 6-DOF odometry, keyed x, y, z, qx, qy, qz, qw, vx, vy, vz, wx, wy, wz.

        The planar tuple ``read_odom`` returns satisfies :class:`RobotHandle`, whose consumers are
        2D by construction; it must NOT be what reaches ``odom``. Flattened to yaw, a
        quadrotor publishes zero tilt and no vertical speed -- which reads not as a coarse
        measurement but as a level, hovering aircraft whatever it is actually doing.
        """
        x, y, z = self._odom_frame.position(self._state[:3])
        qw, qx, qy, qz = self._odom_frame.orientation(self._quat)
        vx, vy, vz = self._body_vel
        wx, wy, wz = self._omega
        return {
            "x": x,
            "y": y,
            "z": z,
            "qx": qx,
            "qy": qy,
            "qz": qz,
            "qw": qw,
            "vx": vx,
            "vy": vy,
            "vz": vz,
            "wx": wx,
            "wy": wy,
            "wz": wz,
        }

    # -- lifecycle ---------------------------------------------------------------------------

    def on_reset(self, ctx: SimContext) -> None:
        # A commanded setpoint belongs to the episode that commanded it: the next trial takes off
        # toward the configured target, not toward wherever the previous one was sent.
        self._target = np.array(self.cfg("target"), dtype=float)
        self._yaw = float(self.cfg("yaw"))
        self._vel_cmd = None
        self._yaw_rate = 0.0
        self.watchdog.clear()
        self._sense(ctx.model, ctx.data)
        self._odom_frame.capture(self._state[:3], self._quat)

    def _sense(self, model, data):
        """Read the drone's state; returns ``(pos, rot, vel, omega)`` for the control law."""
        pos = np.array(data.xpos[self._bid])
        rot = np.array(data.xmat[self._bid]).reshape(3, 3)
        twist = body_twist(model, data, self._bid)
        vel = np.array(twist.linear)
        omega = rot.T @ np.array(twist.angular)

        yaw = float(np.arctan2(rot[1, 0], rot[0, 0]))
        self._state = (*(float(v) for v in pos), *(float(v) for v in vel), yaw, float(omega[2]))
        # Keep the FULL rotation as well. Yaw alone is what a ground robot may report; an airframe
        # holds attitude to fly, so tilt is the signal a flight-envelope experiment measures and a
        # yaw-only projection reports it as identically zero. mju_mat2Quat rather than a hand-rolled
        # conversion so the sign convention is MuJoCo's own.
        quat = np.empty(4)
        mujoco.mju_mat2Quat(quat, np.asarray(rot, dtype=float).reshape(9))
        self._quat = tuple(float(v) for v in quat)  # (w, x, y, z)
        self._omega = tuple(float(v) for v in omega)
        self._body_vel = tuple(float(v) for v in rot.T @ vel)
        c, s = np.cos(yaw), np.sin(yaw)
        self._planar_vel = (float(c * vel[0] + s * vel[1]), float(-s * vel[0] + c * vel[1]))
        return pos, rot, vel, omega

    def pre_step(self, ctx: SimContext) -> None:
        model, data = ctx.model, ctx.data
        pos, rot, vel, omega = self._sense(model, data)

        if self._vel_cmd is not None and self.watchdog.expired(ctx):
            # Stale: brake at the altitude setpoint, then hold where the drone stopped.
            self._vel_cmd[:] = 0.0
            self._yaw_rate = 0.0
            if np.hypot(vel[0], vel[1]) < _HOVER_SPEED:
                self._target = np.array([pos[0], pos[1], self._target[2]])
                self._vel_cmd = None

        if self._yaw_rate:
            self._yaw += self._yaw_rate * ctx.dt

        kp_pos = np.array(self.cfg("kp_pos"))
        kd_pos = np.array(self.cfg("kd_pos"))
        max_vel = float(self.cfg("max_vel"))

        if self._vel_cmd is not None:
            # Teleop mode: the horizontal command IS a velocity; hold the standing target's altitude.
            world_v = rot[:2, :2] @ self._vel_cmd
            vel_des = np.array([world_v[0], world_v[1], 0.0])
            pos_err = np.array([0.0, 0.0, self._target[2] - pos[2]])
        else:
            pos_err = self._target - pos
            vel_des = np.clip(kp_pos * pos_err / np.maximum(kd_pos, 1e-6), -max_vel, max_vel)
            pos_err = np.zeros(3)

        accel = kp_pos * pos_err + kd_pos * (vel_des - vel)
        max_accel = float(self.cfg("max_accel"))
        horiz = np.linalg.norm(accel[:2])
        if horiz > max_accel:
            accel[:2] *= max_accel / horiz
        accel[2] += self._gravity

        # Cap the tilt the acceleration implies, before it becomes an attitude command: a request
        # steeper than max_tilt is scaled back, not clipped per-axis, so the direction survives.
        max_tilt = float(self.cfg("max_tilt"))
        max_horiz = abs(accel[2]) * np.tan(max_tilt)
        horiz = np.linalg.norm(accel[:2])
        if horiz > max_horiz > 0:
            accel[:2] *= max_horiz / horiz

        force = self._mass * accel
        # Thrust along the CURRENT body z: commanding force the drone cannot yet point at is what
        # makes a tilted quadrotor climb when it was asked to translate.
        thrust = float(force @ rot[:, 2])
        thrust = float(np.clip(thrust, *self._thrust_range))

        # Desired attitude: body z along the desired force, x carrying the commanded yaw.
        z_des = force / max(np.linalg.norm(force), 1e-9)
        x_head = np.array([np.cos(self._yaw), np.sin(self._yaw), 0.0])
        y_des = np.cross(z_des, x_head)
        norm = np.linalg.norm(y_des)
        if norm < 1e-6:  # heading parallel to thrust: keep the current y axis
            y_des, norm = rot[:, 1], 1.0
        y_des = y_des / norm
        rot_des = np.column_stack((np.cross(y_des, z_des), y_des, z_des))

        err_att = 0.5 * _hat_vee(rot_des.T @ rot - rot.T @ rot_des)
        moment = -np.array(self.cfg("kp_att")) * err_att - np.array(self.cfg("kd_att")) * omega

        data.ctrl[self._aid_thrust] = thrust
        for i, aid in enumerate(self._aid_moments):
            # ctrl = moment / gear, so the model's (negative) gear sign is undone here.
            gear = self._moment_gear[i] if abs(self._moment_gear[i]) > 1e-12 else 1.0
            lo, hi = model.actuator_ctrlrange[aid]
            data.ctrl[aid] = float(np.clip(moment[i] / gear, lo, hi))
