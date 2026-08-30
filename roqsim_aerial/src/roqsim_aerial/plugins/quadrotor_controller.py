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

Config::

    quadrotor_controller:
      robot: drone                  # entity name registered by spawn_robot
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

**The moment actuators carry a negative gear**, so a positive ``ctrl`` produces a *negative* body
moment. The sign is read from the model at configure time rather than hardcoded -- it is upstream's
convention, and a future airframe need not share it.

**The attitude gains are in newton-metres, not normalised units**, because the controller emits a
moment and the model's actuator gear converts it to ``ctrl``. They are sized from the airframe: for a
body inertia I and a target attitude bandwidth wn with damping zeta, ``kp_att ~ I*wn^2`` and
``kd_att ~ 2*zeta*I*wn``. The defaults are I = 2.4e-5 kg*m^2 at wn = 20 rad/s, zeta = 0.9. This is
also why the Crazyflie's moment gear had to be tuned during the port: at the arbitrary 1e-5 N*m
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

from roqsim.context import Endpoint, RobotHandle, SimContext
from roqsim.plugin import Plugin

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
        return errors

    def configure(self, ctx: SimContext) -> None:
        entity = ctx.entities.get(self.robot)
        prefix = entity.meta.get("prefix", "") if entity else ""
        ns = self.config.get("namespace") or (entity.meta.get("namespace", "") if entity else "")
        model = ctx.model

        def actuator(n):
            return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, prefix + n)

        self._aid_thrust = actuator(self.cfg("thrust_actuator"))
        self._aid_moments = [actuator(n) for n in self.cfg("moment_actuators")]
        body = self.config.get("body") or (entity.meta.get("root_body") if entity else None)
        self._bid = (
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, prefix + body) if body else -1
        )
        if self._bid < 0:
            # Fall back to the body owning the thrust actuator's site, which is where the force is
            # applied and therefore the body being flown by construction.
            site = model.actuator_trnid[self._aid_thrust, 0]
            self._bid = int(model.site_bodyid[site]) if site >= 0 else -1

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
                f"quadrotor_controller: could not resolve {missing or 'the drone body'} "
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
        ctx.interface.add(
            Endpoint(
                name="cmd_pos",
                direction="in",
                owner=self.robot,
                namespace=ns,
                write=lambda p: self.set_target(p[0], p[1], p[2], p[3] if len(p) > 3 else None),
                backend={"ros2": {"type": "geometry_msgs.msg.PoseStamped", "topic": "cmd_pos"}},
            )
        )
        ctx.interface.add(
            Endpoint(
                name="odom",
                direction="out",
                owner=self.robot,
                namespace=ns,
                read=self.read_odom6,
                backend={"ros2": {"type": "nav_msgs.msg.Odometry", "topic": "odom"}},
            )
        )

    # -- commands ----------------------------------------------------------------------------

    def set_target(self, x, y, z, yaw=None) -> None:
        """Position setpoint in the world frame; ``yaw`` keeps the current heading if omitted."""
        self._target = np.array([float(x), float(y), float(z)])
        self._vel_cmd = None
        if yaw is not None:
            self._yaw = float(yaw)

    def drive(self, vx: float, vy: float, w: float) -> None:
        """:class:`RobotHandle` contract: body-frame planar velocity, altitude held.

        A quadrotor is not a ground robot, so this is a projection rather than its native command --
        it exists so teleop and the generic in-process consumers work unchanged. Altitude comes from
        the standing target; ``set_target`` is the full-authority command.
        """
        self._vel_cmd = np.array([float(vx), float(vy)])
        self._yaw += float(w) * 0.0  # yaw-rate integration happens in pre_step, where dt is known
        self._yaw_rate = float(w)

    def read_state(self):
        return self._state

    def read_odom(self):
        x, y, z, vx, vy, vz, yaw, w = self._state
        return (x, y, yaw, vx, vy, w)

    def read_odom6(self):
        """Full 6-DOF odometry payload (the bridge's ``ODOM6_KEYS`` mapping).

        The planar tuple ``read_odom`` returns satisfies :class:`RobotHandle`, whose consumers are
        2D by construction; it must NOT be what reaches the odometry topic. Flattened to yaw, a
        quadrotor publishes zero tilt and no vertical speed -- which reads not as a coarse
        measurement but as a level, hovering aircraft whatever it is actually doing.
        """
        x, y, z, vx, vy, vz, _yaw, _w = self._state
        qw, qx, qy, qz = self._quat
        wx, wy, wz = self._omega
        return {
            "x": x, "y": y, "z": z,
            "qx": qx, "qy": qy, "qz": qz, "qw": qw,
            "vx": vx, "vy": vy, "vz": vz,
            "wx": wx, "wy": wy, "wz": wz,
        }

    # -- lifecycle ---------------------------------------------------------------------------

    def on_reset(self, ctx: SimContext) -> None:
        self._vel_cmd = None
        self._yaw_rate = 0.0
        self._state = (0.0,) * 8
        self._quat = (1.0, 0.0, 0.0, 0.0)
        self._omega = (0.0, 0.0, 0.0)

    def pre_step(self, ctx: SimContext) -> None:
        model, data = ctx.model, ctx.data
        pos = np.array(data.xpos[self._bid])
        rot = np.array(data.xmat[self._bid]).reshape(3, 3)
        vel = np.array(data.cvel[self._bid][3:6])
        omega = rot.T @ np.array(data.cvel[self._bid][0:3])

        yaw = float(np.arctan2(rot[1, 0], rot[0, 0]))
        self._state = (*pos, *vel, yaw, float(omega[2]))
        # Keep the FULL rotation as well. Yaw alone is what a ground robot may report; an airframe
        # holds attitude to fly, so tilt is the signal a flight-envelope experiment measures and a
        # yaw-only projection reports it as identically zero. mju_mat2Quat rather than a hand-rolled
        # conversion so the sign convention is MuJoCo's own.
        quat = np.empty(4)
        mujoco.mju_mat2Quat(quat, np.asarray(rot, dtype=float).reshape(9))
        self._quat = tuple(float(v) for v in quat)  # (w, x, y, z)
        self._omega = tuple(float(v) for v in omega)

        if getattr(self, "_yaw_rate", 0.0):
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
