# Copyright (C) 2026 Frederik Pasch
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""Controller plugin: tricycle kinematics -- one steered wheel and a fixed axle -- plus odometry.

The fourth base geometry, beside ``diff_drive``, ``omni_drive`` and ``ackermann_drive``: one steered
wheel on the centre line at a signed distance ``steer_offset`` from ``base_link``, and a fixed axle
through ``base_link``. Three-wheel counterbalance forklifts, tuggers and pallet trucks are built this
way. Unlike a car it has one steered wheel rather than a linked pair, and the wheel may sit behind
the axle (``steer_offset < 0``) or in front of it (``> 0``).

``base_link`` must be the centre of the fixed axle: the instantaneous centre of rotation always lies
on the axle's line, so that is the one point whose velocity is always along the heading, and the
point a twist is expressed about.

**Geometry.** With ``a = steer_offset``, a body twist ``(v, w)`` moves the steered wheel's point at
``(v, w * a)``, so the wheel points at ``delta = atan(w * a / v)`` and rolls at ``v / cos(delta)``.
For a rear wheel ``a < 0``, so a left turn going forward steers the wheel right; reversing with the
same yaw rate steers it left. A yaw rate beyond the lock is clamped to the lock at the commanded
speed, so the vehicle follows its tightest circle.

**Which wheels are driven** is the ``drive`` key:

* ``axle`` -- the two fixed wheels are driven and the steered one is passive. They are split across
  ``track`` as a differential splits them, ``v -/+ w * track / 2`` with ``w = v * tan(delta) / a``,
  so neither scrubs. Near full lock the inner wheel runs backwards; ``max_wheel_speed`` caps the
  outer one, scaling both down together.
* ``steer_wheel`` -- the steered wheel is driven, at ``v / cos(delta)``, and the axle is passive.

**The split uses the measured steering angle**, not the commanded one. While the wheel slews the
vehicle turns about wherever the wheel points; a split computed from the command would drive the
axle for a curve the vehicle is not yet on, and scrub all three wheels.

**A zero-speed turn command moves nothing.** ``cmd_vel`` with ``v = 0`` and a yaw rate asks for a
rotation about ``base_link``, which needs the steered wheel at 90 degrees. ``max_steer_angle`` must
be below that, since the formulation divides by ``cos(delta)``, and the vehicle it describes cannot
pivot either: at its lock it turns about a point ``|a| / tan(max_steer_angle)`` to the side.
Counter-rotating the axle wheels would rotate the base anyway and hide the failure a car-like stack
must not have. The steered wheel holds its angle, as the rack of ``ackermann_drive`` does, and the
drive ramps to a stop. The plugin declares ``kinematics="ackermann"`` on its
:class:`~roqsim.context.RobotHandle`: a twist states a curvature.

Config::

    tricycle_drive:
      drive: axle                   # axle | steer_wheel -- which wheels the motors turn
      steer_offset: -1.39           # m; x of the steering axis from base_link, < 0 = rear
      wheel_radius: 0.23            # driven axle wheels (drive: axle)
      track: 0.93                   # driven axle width (drive: axle)
      steer_wheel_radius: 0.18      # the steered wheel (drive: steer_wheel)
      max_linear_vel: 2.0           # m/s at base_link
      max_wheel_speed: 2.0          # m/s at any driven tread; default max_linear_vel
      max_steer_angle: 1.2          # rad, < pi/2; the mechanical lock
      steer_rate: 1.5               # rad/s slew on the steering angle (0 = instant)
      accel_limit: 1.0              # m/s^2 on the commanded speed (0 = instant)
      steer_actuator: steer_motor   # POSITION servo
      steer_joint: steer_joint
      drive_actuators: [left_motor, right_motor]   # VELOCITY servos; left then right for `axle`,
      drive_joints: [left_wheel, right_wheel]      # the one steered wheel for `steer_wheel`
      passive_joints: [steer_wheel_joint]          # published in joint_states, never commanded
      base_body: base_link
      odom_child_frame: base_link
      odom_rate_hz: 50.0
      cmd_vel_timeout: 0.0          # s; > 0 stops the vehicle when no command arrives for this long
      publish_joint_states: true
      stamped_cmd_vel: false        # true when the stack publishes TwistStamped
      test_cmd: [1.0, 0.3]          # optional [v, w] applied every tick (standalone demo)

``steer_offset`` and ``track`` are checked against the model at ``configure``: the steering joint's
anchor and the two axle wheels' anchors, expressed in ``base_body``, must agree with them to within a
centimetre, and the steering axis must be vertical. A kinematic model that disagrees with its vehicle
turns about the wrong point, so the disagreement is refused by name.

``passive_joints`` are joints nothing actuates but a URDF names, such as the steered wheel's roll on
an axle-driven truck. ``robot_state_publisher`` publishes a link's transform only once it has a state
for the joint above it, so a joint left out here leaves that wheel's frame missing from TF.

Endpoints are the siblings', declared on typed methods: ``cmd_vel`` in (a ``Twist``; the ROS bridge
carries it as ``geometry_msgs/Twist``, or ``TwistStamped`` with ``stamped_cmd_vel``), ``odom`` out
(an ``Odometry``) with its TF to ``odom_child_frame``, and ``joint_states`` out (a ``JointState``,
off with ``publish_joint_states: false``) at ``odom_rate_hz``. ``joint_states`` carries the steering
joint first, then the driven joints, then the passive ones.

**Odometry is what the encoders say.** ``drive: axle`` takes the speed from the mean of the two
driven wheels and the yaw rate from the measured steering angle, ``w = v * tan(delta) / a``;
``drive: steer_wheel`` splits the wheel's own speed ``s`` into ``v = s * cos(delta)`` and
``w = s * sin(delta) / a``. It is dead reckoning and drifts where the tyres slip, uncorrected for the
reason ``ackermann_drive`` gives: a tyre's slip angle varies with speed and load, so a constant would
make the estimate look better than the sensor it stands for.
"""

from __future__ import annotations

import mujoco
import numpy as np

from roqsim import endpoint
from roqsim.context import RobotHandle, SimContext
from roqsim.odometry import CommandWatchdog
from roqsim.plugin import Plugin
from roqsim.types import AngularSpeed, JointState, Odometry, Speed, Twist

#: Below this speed a curvature command has no meaning (see the module docstring).
_MIN_SPEED = 1e-3

#: How far a configured length may sit from the model's own before configure refuses it (m).
_GEOMETRY_TOLERANCE = 0.01

_DRIVES = ("axle", "steer_wheel")


def _is_name_list(value) -> bool:
    """A list (or tuple) of strings: what a key naming actuators or joints takes."""
    return isinstance(value, (list, tuple)) and all(isinstance(n, str) for n in value)


class TricycleDrivePlugin(Plugin):
    """See the module docstring."""

    #: Drives an entity's actuators, so it belongs inside that entity's ``components:`` block.
    requires_owner = True

    def __init__(self, config=None, *, name=None, entity=None, label=None):
        super().__init__(config, name=name, entity=entity, label=label)
        self.robot = self.entity
        self.drive_mode = str(self.config.get("drive", "axle"))
        self.a = float(self.config.get("steer_offset", -1.0))
        self.r = float(self.config.get("wheel_radius", 0.1))
        self.track = float(self.config.get("track", 0.5))
        self.r_steer = float(self.config.get("steer_wheel_radius", self.r))
        self.max_v = float(self.config.get("max_linear_vel", 1.0))
        self.max_wheel = float(self.config.get("max_wheel_speed", self.max_v))
        self.max_steer = float(self.config.get("max_steer_angle", 1.0))
        self.steer_rate = float(self.config.get("steer_rate", 1.5))
        self.accel_limit = float(self.config.get("accel_limit", 1.0))
        self.steer_actuator_name = self.config.get("steer_actuator")
        self.steer_joint_name = self.config.get("steer_joint")
        self.drive_actuator_names = list(self.config.get("drive_actuators") or [])
        self.drive_joint_names = list(self.config.get("drive_joints") or [])
        self.passive_joint_names = list(self.config.get("passive_joints") or [])
        self.base_body = self.config.get("base_body", "base_link")
        self.odom_child_frame = self.config.get("odom_child_frame", "base_link")
        self.odom_rate_hz = float(self.config.get("odom_rate_hz", 50.0))
        self.publish_joint_states = bool(self.config.get("publish_joint_states", True))
        #: Message type of the velocity command: the stack decides it, not the kinematics.
        self.stamped_cmd_vel = bool(self.config.get("stamped_cmd_vel", False))
        #: ``cmd_vel_timeout``: a command older than this stops the vehicle; 0 holds it forever.
        self.watchdog = CommandWatchdog.from_config(self.config)

        self._ctx: SimContext | None = None
        self._target_v = 0.0
        self._target_w = 0.0
        self._cmd_v = 0.0  # ramped base_link speed
        self._steer = 0.0  # slewed steering-angle command
        self._odom = [0.0, 0.0, 0.0, 0.0, 0.0]  # x, y, yaw, v, w
        self._steer_aid = -1
        self._steer_jid = -1
        self._steer_sign = 1.0
        self._drive_aid: list[int] = []
        self._drive_jid: list[int] = []
        self._roll_sign: list[float] = []
        self._passive_jid: list[int] = []
        self._jnames: list[str] = []
        self._jpos = np.zeros(0)
        self._jvel = np.zeros(0)

    # -- validation ---------------------------------------------------------------------------

    def validate_config(self, config: dict) -> list[str]:
        errors = self.validate_topics(config)
        drive = config.get("drive", "axle")
        if drive not in _DRIVES:
            errors.append(f"'drive' must be one of {list(_DRIVES)}, got {drive!r}")
        if "steer_offset" not in config:
            errors.append(
                "'steer_offset' is required: the signed x of the steering axis from base_link "
                "(negative for a rear-steered vehicle)"
            )
        elif abs(float(config["steer_offset"])) < 1e-3:
            errors.append(
                "'steer_offset' must be non-zero: a steered wheel on the axle cannot turn the vehicle"
            )
        for key in (
            "wheel_radius",
            "track",
            "steer_wheel_radius",
            "max_linear_vel",
            "max_wheel_speed",
            "max_steer_angle",
            "odom_rate_hz",
        ):
            if key in config and float(config[key]) <= 0:
                errors.append(f"'{key}' must be > 0")
        for key in ("steer_rate", "accel_limit"):
            if key in config and float(config[key]) < 0:
                errors.append(f"'{key}' must be >= 0 (0 means no limit)")
        if float(config.get("max_steer_angle", 1.0)) >= np.pi / 2:
            # At 90 degrees the vehicle pivots about base_link and a twist's v no longer fixes the
            # wheel speed (it is zero); the formulation here divides by cos(delta). See the docstring.
            errors.append(
                "'max_steer_angle' must be < pi/2: a wheel that reaches 90 degrees pivots the "
                "vehicle about its axle centre, which this plugin does not model"
            )
        for key in ("steer_actuator", "steer_joint"):
            value = config.get(key)
            if not value:
                errors.append(f"'{key}' is required: name the model's one steering {key[6:]}")
            elif not isinstance(value, str):
                errors.append(f"'{key}' names ONE {key[6:]} (a string); a tricycle has one")
        want = 2 if drive == "axle" else 1
        which = "left then right" if drive == "axle" else "the steered wheel"
        for key in ("drive_actuators", "drive_joints", "passive_joints"):
            names = config.get(key)
            if names is not None and not _is_name_list(names):
                # Checked before the length: a bare string has a length too, its character count.
                errors.append(f"'{key}' must be a list of names, got {type(names).__name__}")
        for key in ("drive_actuators", "drive_joints"):
            names = config.get(key)
            if not names:
                errors.append(f"'{key}' is required: name {want} ({which})")
            elif _is_name_list(names) and len(names) != want:
                errors.append(f"'{key}' must name exactly {want} for drive: {drive} ({which})")
        if drive == "axle":
            for key in ("wheel_radius", "track"):
                if key not in config:
                    errors.append(f"'{key}' is required for drive: axle")
        elif drive == "steer_wheel" and "steer_wheel_radius" not in config:
            errors.append("'steer_wheel_radius' is required for drive: steer_wheel")
        if "test_cmd" in config:
            cmd = config["test_cmd"]
            if (
                not isinstance(cmd, (list, tuple))
                or len(cmd) != 2
                or not all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in cmd)
            ):
                errors.append("'test_cmd' must be [v, w]")
        errors += CommandWatchdog.validate(config)
        return errors

    # -- lifecycle ----------------------------------------------------------------------------

    def configure(self, ctx: SimContext) -> None:
        self._ctx = ctx
        entity = ctx.entities.get(self.robot)
        prefix = entity.meta.get("prefix", "") if entity else ""
        m = ctx.model

        def resolve(kind, names):
            ids = [mujoco.mj_name2id(m, kind, prefix + n) for n in names]
            missing = [n for n, i in zip(names, ids, strict=True) if i < 0]
            if missing:
                raise RuntimeError(
                    f"tricycle_drive: could not resolve {missing} for robot {self.robot!r}"
                )
            return ids

        (self._steer_aid,) = resolve(mujoco.mjtObj.mjOBJ_ACTUATOR, [self.steer_actuator_name])
        (self._steer_jid,) = resolve(mujoco.mjtObj.mjOBJ_JOINT, [self.steer_joint_name])
        self._drive_aid = resolve(mujoco.mjtObj.mjOBJ_ACTUATOR, self.drive_actuator_names)
        self._drive_jid = resolve(mujoco.mjtObj.mjOBJ_JOINT, self.drive_joint_names)
        self._passive_jid = resolve(mujoco.mjtObj.mjOBJ_JOINT, self.passive_joint_names)

        base_b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, prefix + self.base_body)
        if base_b < 0:
            raise RuntimeError(
                f"tricycle_drive: base body {prefix + self.base_body!r} not found; the geometry and "
                f"the wheel signs are read in it"
            )
        # The reference pose, where every joint is at zero: the steered wheel points straight ahead.
        d0 = mujoco.MjData(m)
        mujoco.mj_forward(m, d0)
        rb = d0.xmat[base_b].reshape(3, 3)
        pb = d0.xpos[base_b]

        def axis_in_base(jid: int) -> np.ndarray:
            return rb.T @ (d0.xmat[m.jnt_bodyid[jid]].reshape(3, 3) @ m.jnt_axis[jid])

        def anchor_in_base(jid: int) -> np.ndarray:
            return rb.T @ (d0.xanchor[jid] - pb)

        # The steering axis: vertical, and where the config says it is.
        steer_axis = axis_in_base(self._steer_jid)
        if abs(float(steer_axis[2])) < 0.99:
            raise RuntimeError(
                f"tricycle_drive: steering joint {self.steer_joint_name!r} is not about the base's "
                f"z axis (axis in {self.base_body!r}: {np.round(steer_axis, 3).tolist()})"
            )
        # A positive steering angle here is counter-clockwise seen from above, whichever way the
        # model states its joint axis.
        self._steer_sign = 1.0 if steer_axis[2] > 0 else -1.0
        steer_at = anchor_in_base(self._steer_jid)
        if abs(float(steer_at[0]) - self.a) > _GEOMETRY_TOLERANCE:
            raise RuntimeError(
                f"tricycle_drive: steer_offset {self.a} disagrees with the model, whose steering "
                f"axis {self.steer_joint_name!r} is at x = {float(steer_at[0]):.4f} in "
                f"{self.base_body!r}; base_body must be the centre of the fixed axle"
            )
        if abs(float(steer_at[1])) > _GEOMETRY_TOLERANCE:
            raise RuntimeError(
                f"tricycle_drive: the steering axis {self.steer_joint_name!r} is "
                f"{float(steer_at[1]):.4f} m off the centre line of {self.base_body!r}; this plugin "
                f"models a steered wheel on the centre line"
            )
        if self.drive_mode == "axle":
            left, right = (anchor_in_base(j) for j in self._drive_jid)
            measured_track = float(left[1] - right[1])
            if abs(measured_track - self.track) > _GEOMETRY_TOLERANCE:
                raise RuntimeError(
                    f"tricycle_drive: track {self.track} disagrees with the model, whose driven "
                    f"wheels (left then right) are {measured_track:.4f} m apart in y"
                )
            if max(abs(float(left[0])), abs(float(right[0]))) > _GEOMETRY_TOLERANCE:
                raise RuntimeError(
                    f"tricycle_drive: the driven wheels are not on the axle through "
                    f"{self.base_body!r} (x = {float(left[0]):.4f}, {float(right[0]):.4f}); "
                    f"base_body must be the centre of the fixed axle"
                )

        # Per-wheel roll sign, read off the model as diff_drive does: a wheel carries the vehicle
        # forward when it spins about +y of the frame it steers in, and a source URDF may express
        # the same wheel about either y direction with both being correct.
        self._roll_sign = [1.0 if float(axis_in_base(j)[1]) > 0 else -1.0 for j in self._drive_jid]

        self._jnames = (
            [self.steer_joint_name] + list(self.drive_joint_names) + list(self.passive_joint_names)
        )
        self._jpos = np.zeros(len(self._jnames))
        self._jvel = np.zeros(len(self._jnames))

        ctx.blackboard.set(
            f"robot:{self.robot}",
            RobotHandle(
                name=self.robot,
                drive=self.drive,
                read_odom=self.read_odom,
                # A twist states a curvature, and w with v == 0 moves nothing (module docstring).
                kinematics="ackermann",
            ),
        )

    # -- commands -----------------------------------------------------------------------------

    @endpoint.stream(Twist, ros2=lambda self: {"stamped": self.stamped_cmd_vel})
    def cmd_vel(self, vx: Speed, vy: Speed = 0.0, wz: AngularSpeed = 0.0) -> None:
        """Body-frame velocity command at base_link, applied once per step.

        Args:
            vx: forward speed
            vy: sideways speed; a tricycle drops it
            wz: yaw rate, steered through the tricycle relation
        """
        self.drive(vx, vy, wz)

    def drive(self, vx: float, vy: float, w: float) -> None:
        """Body-frame twist target at base_link (``vy`` dropped: a tricycle cannot strafe)."""
        self._target_v = float(np.clip(vx, -self.max_v, self.max_v))
        self._target_w = float(w)
        self.watchdog.stamp(self._ctx)

    def steer_angle_for(self, v: float, w: float) -> float | None:
        """The steering angle a twist asks for, clamped to the lock -- or None when it asks for none.

        ``atan(w * a / v)``: the direction the steered wheel's own point moves in. Below
        ``_MIN_SPEED`` a twist states no curvature and the wheel holds (module docstring).
        """
        if abs(v) < _MIN_SPEED:
            return None
        return float(np.clip(np.arctan(w * self.a / v), -self.max_steer, self.max_steer))

    def wheel_speeds(self, v: float, delta: float) -> list[float]:
        """Tread speeds (m/s) of the driven wheels for a base_link speed and a steering angle.

        ``axle``: ``[left, right]`` split across the track. ``steer_wheel``: ``[v / cos(delta)]``.
        Scaled down together when one would exceed ``max_wheel_speed``, so the vehicle slows on
        the curve rather than leaving it.
        """
        if self.drive_mode == "axle":
            w = v * np.tan(delta) / self.a
            speeds = [v - w * self.track / 2.0, v + w * self.track / 2.0]
        else:
            speeds = [v / np.cos(delta)]
        peak = max(abs(s) for s in speeds)
        if peak > self.max_wheel:
            speeds = [s * self.max_wheel / peak for s in speeds]
        return [float(s) for s in speeds]

    def _measured_steer(self, ctx: SimContext) -> float:
        m, d = ctx.model, ctx.data
        return self._steer_sign * float(d.qpos[m.jnt_qposadr[self._steer_jid]])

    def pre_step(self, ctx: SimContext) -> None:
        if ctx.manual_control:
            return  # the viewer's sliders own the actuators this run
        if "test_cmd" in self.config:
            v, w = self.config["test_cmd"]
            self.drive(float(v), 0.0, float(w))
        if self.watchdog.expired(ctx):
            # The watchdog: the last command has expired, so the target is a stop, through the ramp.
            self._target_v = self._target_w = 0.0

        if self.accel_limit > 0:
            dv = self.accel_limit * ctx.dt
            self._cmd_v += float(np.clip(self._target_v - self._cmd_v, -dv, dv))
        else:
            self._cmd_v = self._target_v

        target = self.steer_angle_for(self._target_v, self._target_w)
        if target is not None:
            if self.steer_rate > 0:
                step = self.steer_rate * ctx.dt
                self._steer += float(np.clip(target - self._steer, -step, step))
            else:
                self._steer = target
        ctx.data.ctrl[self._steer_aid] = self._steer_sign * self._steer

        # The drive follows the wheel where it IS, not where it was sent (module docstring).
        delta = float(np.clip(self._measured_steer(ctx), -self.max_steer, self.max_steer))
        speeds = self.wheel_speeds(self._cmd_v, delta)
        radius = self.r if self.drive_mode == "axle" else self.r_steer
        for aid, sign, s in zip(self._drive_aid, self._roll_sign, speeds, strict=True):
            ctx.data.ctrl[aid] = sign * s / radius

    # -- odometry -----------------------------------------------------------------------------

    def post_step(self, ctx: SimContext) -> None:
        m, d = ctx.model, ctx.data
        delta = self._measured_steer(ctx)
        if self.drive_mode == "axle":
            v = float(
                np.mean(
                    [
                        sign * d.qvel[m.jnt_dofadr[j]] * self.r
                        for j, sign in zip(self._drive_jid, self._roll_sign, strict=True)
                    ]
                )
            )
            w = v * np.tan(delta) / self.a
        else:
            s = self._roll_sign[0] * float(d.qvel[m.jnt_dofadr[self._drive_jid[0]]]) * self.r_steer
            v = s * np.cos(delta)
            w = s * np.sin(delta) / self.a

        o = self._odom
        o[0] += v * np.cos(o[2]) * ctx.dt
        o[1] += v * np.sin(o[2]) * ctx.dt
        o[2] = (o[2] + w * ctx.dt + np.pi) % (2 * np.pi) - np.pi
        o[3], o[4] = v, float(w)

        self._read_joints(m, d)

    @endpoint.out(
        rate="odom_rate_hz",
        ros2=lambda self: {"child_frame_id": self.odom_child_frame, "emit_tf": True},
    )
    def odom(self) -> Odometry:
        """Dead reckoning from the driven wheels and the measured steering angle."""
        x, y, yaw, v, w = self._odom
        return Odometry.planar(x, y, yaw, v, 0.0, w)

    def read_odom(self) -> tuple[float, float, float, float, float, float]:
        """The latest ``(x, y, yaw, vx, vy, w)``, what the :class:`RobotHandle` reads."""
        x, y, yaw, v, w = self._odom
        return (x, y, yaw, v, 0.0, w)

    @endpoint.out(rate="odom_rate_hz", when="publish_joint_states")
    def joint_states(self) -> JointState:
        """The steering joint, then the driven wheels, then the passive joints."""
        return JointState(self._jnames, self._jpos, self._jvel)

    def on_reset(self, ctx: SimContext) -> None:
        self._target_v = self._target_w = 0.0
        self._cmd_v = 0.0
        self._steer = 0.0
        self._odom = [0.0, 0.0, 0.0, 0.0, 0.0]
        self.watchdog.clear()
        # The reset pose, not the previous episode's last one, until the first step.
        self._read_joints(ctx.model, ctx.data)

    def _read_joints(self, m, d) -> None:
        """The joint_states payload, written in place so ``joint_states`` is zero-copy."""
        for k, jid in enumerate([self._steer_jid, *self._drive_jid, *self._passive_jid]):
            self._jpos[k] = d.qpos[m.jnt_qposadr[jid]]
            self._jvel[k] = d.qvel[m.jnt_dofadr[jid]]
