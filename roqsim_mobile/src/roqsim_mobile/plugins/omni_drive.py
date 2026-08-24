"""Controller plugin: holonomic (omnidirectional) base kinematics + odometry.

The holonomic counterpart of :mod:`roqsim_mobile.plugins.diff_drive`. Consumes a body-frame twist
(``vx``, ``vy``, ``wz``) -- a diff-drive drops ``vy``, this one does not -- and drives a planar base
that can translate in any direction while holding, or independently changing, its heading. Written
for PAL Robotics' OMNI base (TIAGo Pro), whose ROS 2 stack uses
``omni_drive_controller/OmniDriveController``.

Config::

    omni_drive:
      robot: robot                  # entity name registered by spawn_robot
      namespace: ""                 # transport scope (default: inherited from spawn_robot)
      base_joint: base_free         # the base's free joint
      vx_actuator: base_vx          # planar drive actuators (see "Planar drive" below)
      vy_actuator: base_vy
      wz_actuator: base_wz
      max_linear_vel: 1.0           # m/s, applies to vx
      max_lateral_vel: 1.0          # m/s, applies to vy (defaults to max_linear_vel)
      max_combined_linear_vel: 0.7  # m/s, cap on hypot(vx, vy); 0 disables
      max_angular_vel: 2.09         # rad/s
      accel_limit: 1.0              # m/s^2, ramps vx and vy
      angular_accel_limit: 2.09     # rad/s^2, ramps wz
      # Mecanum wheels: OBSERVATIONAL only (see "Wheels" below). Omit to skip wheel handling.
      wheel_radius: 0.0762
      wheel_separation: 0.44715     # lateral, left <-> right
      axis_separation: 0.488        # longitudinal, front <-> rear
      wheels: [front_left, front_right, rear_left, rear_right]   # joint name stems
      wheel_actuators: [...]        # same order; defaults to <stem>_motor
      test_cmd: [0.2, 0.1, 0.0]     # optional [vx, vy, wz] applied every tick (standalone demo)

**Planar drive.** A real omnidirectional base translates sideways because each mecanum wheel's passive rollers let the
contact patch slide along one diagonal. Those rollers are **not modelled** (~9 per wheel would mean
~36 extra bodies and as many contact pairs, and MuJoCo's cylinder-on-plane contact frame is not
roller-aligned, so anisotropic friction is not a reliable substitute). Instead the commanded twist is
applied to the base's free joint through three velocity actuators, and the wheels are near-frictionless
load carriers -- which is what an omni wheel *is*, in the directions that matter. This is the approach
PAL themselves sketched: ``base_x`` / ``base_y`` / ``base_tau`` velocity actuators on the floating
base, commented out in ``omni_base_description/mujoco/mj_tags.xacro``.

The drive stays inside MuJoCo's force path rather than writing ``qvel`` directly, so contacts still
win: a wall stops the base, and it cannot be shoved through thin geometry.

**Frames.** A free joint's translational DOFs are expressed in the **world** frame and its rotational
DOFs in the **body** frame (both verified against MuJoCo 3.11), and ``gear`` inherits that. So the
body-frame ``vx``/``vy`` command is rotated by the base yaw before it is written to ``ctrl``, while
``wz`` needs no rotation. Getting this wrong yields a base that drives correctly only while its
heading is zero -- which a straight-line test would not catch.

**Wheels.** The wheel velocity servos are driven from the mecanum inverse kinematics purely so that the visual
and ``joint_states`` are right (the wheels turn, and turn *differently* when strafing). They are not
the motive force: at the model's wheel friction they transmit almost nothing. Each wheel's roll sign
is derived from its joint axis in the base frame at configure time rather than hardcoded, because
which way "positive" spins depends on how the source URDF mirrored that wheel.

**Odometry.** Integrated from the base's **achieved** twist, so a base held against a wall reports no progress.
Note the consequence: with no wheel slip in the model, wheel-encoder odometry and ground truth
coincide by construction. This port therefore cannot be used to study odometry drift -- a
skid-steer's characteristic error source is absent here by design, not by accident.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np

from roqsim.context import Endpoint, RobotHandle, SimContext
from roqsim.plugin import Plugin

WHEEL_ORDER = ("front_left", "front_right", "rear_left", "rear_right")


class OmniDrivePlugin(Plugin):
    #: Drives an entity's actuators, so it cannot function without one: it belongs inside that
    #: entity's ``components:`` block. (A *sensor* may be world-mounted and does not set this.)
    requires_owner = True

    def __init__(self, config=None, *, name=None, entity=None, label=None):
        super().__init__(config, name=name, entity=entity, label=label)
        self.robot = self.entity
        self.base_joint = self.config.get("base_joint", "base_free")
        # Body the wheel axes are expressed in when deriving their roll signs (see configure()).
        self.base_body = self.config.get("base_body", "base_link")
        self._act_names = (
            self.config.get("vx_actuator", "base_vx"),
            self.config.get("vy_actuator", "base_vy"),
            self.config.get("wz_actuator", "base_wz"),
        )
        self.max_v = float(self.config.get("max_linear_vel", 1.0))
        self.max_vy = float(self.config.get("max_lateral_vel", self.max_v))
        # A holonomic controller caps the RESULTANT planar speed as well as each axis: PAL's
        # mobile_base_controller.yaml carries a `space: xy: max_velocity` alongside its per-axis
        # limits, which is lower than either (0.7 vs 1.0 for the OMNI base). Without it the model
        # would allow a 1.41 m/s diagonal the real stack refuses. 0 disables.
        self.max_combined = float(self.config.get("max_combined_linear_vel", 0.0))
        self.max_w = float(self.config.get("max_angular_vel", 2.09))
        self.accel = float(self.config.get("accel_limit", 1.0))
        self.ang_accel = float(self.config.get("angular_accel_limit", 2.09))

        self.r = float(self.config.get("wheel_radius", 0.0))
        self.lx = float(self.config.get("axis_separation", 0.0)) / 2.0
        self.ly = float(self.config.get("wheel_separation", 0.0)) / 2.0
        self._wheels = list(self.config.get("wheels", WHEEL_ORDER)) if self.r > 0 else []
        self._wa_names = list(
            self.config.get("wheel_actuators", [f"{w}_motor" for w in self._wheels])
        )
        self._wj_names = [f"{w}_joint" for w in self._wheels]

        self._target = np.zeros(3)  # commanded body-frame [vx, vy, wz], clipped
        self._cmd = np.zeros(3)  # ramped body-frame command actually written
        self._odom = np.zeros(6)  # x, y, yaw, vx, vy, wz  (pose world, twist body)
        self._jpos = np.zeros(len(self._wj_names))
        self._jvel = np.zeros(len(self._wj_names))
        # resolved in configure()
        self._aid = [-1, -1, -1]
        self._waid: list[int] = []
        self._wjid: list[int] = []
        self._wsign = np.ones(len(self._wj_names))
        self._qadr = -1
        self._vadr = -1

    def validate_config(self, config: dict) -> list[str]:
        errors = self.validate_topics(config)
        for key in ("max_linear_vel", "max_lateral_vel", "max_angular_vel"):
            if key in config and float(config[key]) <= 0:
                errors.append(f"'{key}' must be > 0")
        wheeled = float(config.get("wheel_radius", 0)) > 0
        if wheeled:
            # The mecanum kinematics need both track widths; a missing one would silently degrade
            # the wheel spin to a pure-translation rolling model and hide a config mistake.
            for key in ("wheel_separation", "axis_separation"):
                if float(config.get(key, 0)) <= 0:
                    errors.append(f"'{key}' must be > 0 when 'wheel_radius' is set")
            n = len(config.get("wheels", WHEEL_ORDER))
            if n != 4:
                errors.append("'wheels' must list exactly 4 wheels (front/rear x left/right)")
            if "wheel_actuators" in config and len(config["wheel_actuators"]) != n:
                errors.append("'wheel_actuators' must have the same length as 'wheels'")
        if "test_cmd" in config and len(config["test_cmd"]) != 3:
            errors.append("'test_cmd' must be [vx, vy, wz]")
        return errors

    def configure(self, ctx: SimContext) -> None:
        entity = ctx.entities.get(self.robot)
        prefix = entity.meta.get("prefix", "") if entity else ""
        ns = self.config.get("namespace") or (entity.meta.get("namespace", "") if entity else "")
        m = ctx.model

        def act(n):
            return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, prefix + n)

        def jnt(n):
            return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, prefix + n)

        self._aid = [act(n) for n in self._act_names]
        self._waid = [act(n) for n in self._wa_names]
        self._wjid = [jnt(n) for n in self._wj_names]
        bjid = jnt(self.base_joint)
        missing = [
            n
            for n, v in [
                *zip(self._act_names, self._aid, strict=True),
                *zip(self._wa_names, self._waid, strict=True),
                *zip(self._wj_names, self._wjid, strict=True),
                (self.base_joint, bjid),
            ]
            if v < 0
        ]
        if missing:
            raise RuntimeError(f"omni_drive: could not resolve {missing} for robot {self.robot!r}")
        if m.jnt_type[bjid] != mujoco.mjtJoint.mjJNT_FREE:
            raise RuntimeError(
                f"omni_drive: {prefix + self.base_joint!r} is not a free joint; the planar drive "
                f"applies its command to the base's 6-DOF joint"
            )
        self._qadr = int(m.jnt_qposadr[bjid])
        self._vadr = int(m.jnt_dofadr[bjid])

        # Per-wheel roll sign, read off the model at the reference pose: a wheel rolls the base
        # forward when it spins about the base's -y, so sign = -axis_y. Derived rather than
        # hardcoded because the source URDF mirrors left/right wheels (here the two reflections
        # happen to cancel and all four axes come out +y, which is exactly the kind of thing that
        # should not be assumed).
        if self._wjid:
            d0 = mujoco.MjData(m)
            mujoco.mj_forward(m, d0)
            base_b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, prefix + self.base_body)
            if base_b < 0:
                raise RuntimeError(
                    f"omni_drive: base body {prefix + self.base_body!r} not found; the wheel roll "
                    f"signs are read off the wheel axes expressed in it"
                )
            rb = d0.xmat[base_b].reshape(3, 3)
            for k, jid in enumerate(self._wjid):
                axis_w = d0.xmat[m.jnt_bodyid[jid]].reshape(3, 3) @ m.jnt_axis[jid]
                axis_y = float((rb.T @ axis_w)[1])
                self._wsign[k] = -1.0 if axis_y > 0 else 1.0

        ctx.blackboard.set(
            f"robot:{self.robot}",
            RobotHandle(name=self.robot, drive=self.drive, read_odom=self.read_odom),
        )
        ctx.interface.add(
            Endpoint(
                name="cmd_vel",
                direction="in",
                owner=self.robot,
                namespace=ns,
                write=lambda twist: self.drive(twist[0], twist[1], twist[2]),
                backend={
                    "ros2": {
                        "type": "geometry_msgs.msg.Twist",
                        "topic": self.topic_override("cmd_vel") or "cmd_vel",
                    }
                },
            )
        )
        ctx.interface.add(
            Endpoint(
                name="odom",
                direction="out",
                owner=self.robot,
                namespace=ns,
                read=self.read_odom,
                rate_hz=50.0,
                backend={
                    "ros2": {
                        "type": "nav_msgs.msg.Odometry",
                        "topic": self.topic_override("odom") or "odom",
                        "frame_id": "odom",
                        # PAL's mobile_base_controller reports odom in base_footprint, which is also
                        # the body the free joint drives.
                        "child_frame_id": "base_footprint",
                        "emit_tf": True,
                    }
                },
            )
        )
        if self._wj_names:
            ctx.interface.add(
                Endpoint(
                    name="joint_states",
                    direction="out",
                    owner=self.robot,
                    namespace=ns,
                    read=self.read_joint_states,
                    rate_hz=50.0,
                    backend={
                        "ros2": {
                            "type": "sensor_msgs.msg.JointState",
                            "topic": self.topic_override("joint_states") or "joint_states",
                        }
                    },
                )
            )

    def drive(self, vx: float, vy: float, w: float) -> None:
        """Body-frame twist target. Unlike a differential drive, ``vy`` is honoured."""
        vx = float(np.clip(vx, -self.max_v, self.max_v))
        vy = float(np.clip(vy, -self.max_vy, self.max_vy))
        if self.max_combined > 0:
            # Scale the pair down together, so the commanded DIRECTION of travel survives the cap.
            speed = math.hypot(vx, vy)
            if speed > self.max_combined:
                vx *= self.max_combined / speed
                vy *= self.max_combined / speed
        self._target[:] = (vx, vy, float(np.clip(w, -self.max_w, self.max_w)))

    def read_odom(self):
        x, y, yaw, vx, vy, w = self._odom
        return (x, y, yaw, vx, vy, w)

    def read_joint_states(self):
        return (self._wj_names, self._jpos, self._jvel)

    def on_reset(self, ctx: SimContext) -> None:
        self._target[:] = 0.0
        self._cmd[:] = 0.0
        self._odom[:] = 0.0

    def _yaw(self, d) -> float:
        qw, qx, qy, qz = d.qpos[self._qadr + 3 : self._qadr + 7]
        return float(np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz)))

    def pre_step(self, ctx: SimContext) -> None:
        if ctx.manual_control:
            return  # the viewer's sliders own the actuators this run
        if "test_cmd" in self.config:
            self.drive(*(float(v) for v in self.config["test_cmd"]))

        # Ramp the body-frame command toward the target under the acceleration limits.
        lim = np.array([self.accel, self.accel, self.ang_accel]) * ctx.dt
        self._cmd += np.clip(self._target - self._cmd, -lim, lim)

        # Body -> world for the translational pair; wz is already a body-frame DOF (see docstring).
        c, s = np.cos(self._yaw(ctx.data)), np.sin(self._yaw(ctx.data))
        ctx.data.ctrl[self._aid[0]] = c * self._cmd[0] - s * self._cmd[1]
        ctx.data.ctrl[self._aid[1]] = s * self._cmd[0] + c * self._cmd[1]
        ctx.data.ctrl[self._aid[2]] = self._cmd[2]

        # Observational mecanum wheel spin, from the body-frame command.
        if self._waid:
            for k, aid in enumerate(self._waid):
                ctx.data.ctrl[aid] = self._wsign[k] * self._wheel_rate(k, *self._cmd)

    # Mecanum sign table, indexed by position in ``wheels`` (which validate_config pins to exactly 4
    # in WHEEL_ORDER): (lateral sign, yaw sign). Standard X-roller layout -- the diagonal pairs share
    # a lateral sign, and each side shares a yaw sign. Keyed by INDEX, not by name: the caller's
    # `wheels` entries are joint-name stems for the model at hand (the TIAGo Pro's are
    # `wheel_front_left`, not `front_left`), so matching on the name silently degrades every wheel to
    # the same sign -- which looks like a working forward drive and a broken strafe.
    _MECANUM = ((-1.0, -1.0), (1.0, 1.0), (1.0, -1.0), (-1.0, 1.0))

    def _wheel_rate(self, k: int, vx: float, vy: float, wz: float) -> float:
        """Mecanum inverse kinematics for wheel *k* in ``WHEEL_ORDER``, in rad/s.

        The roller handedness is not modelled, so this is the nominal convention (see "Wheels" in
        the module docstring).
        """
        lat, yaw = self._MECANUM[k]
        return (vx + lat * vy + yaw * (self.lx + self.ly) * wz) / self.r

    def post_step(self, ctx: SimContext) -> None:
        m, d = ctx.model, ctx.data
        # Achieved twist: linear DOFs are world-frame, angular are body-frame (see docstring).
        vx_w, vy_w = d.qvel[self._vadr], d.qvel[self._vadr + 1]
        wz = float(d.qvel[self._vadr + 5])
        yaw = self._yaw(d)
        c, s = np.cos(yaw), np.sin(yaw)
        vx_b = c * vx_w + s * vy_w
        vy_b = -s * vx_w + c * vy_w

        o = self._odom
        o[0] += vx_w * ctx.dt
        o[1] += vy_w * ctx.dt
        o[2] = (o[2] + wz * ctx.dt + np.pi) % (2 * np.pi) - np.pi
        o[3], o[4], o[5] = vx_b, vy_b, wz

        for k, jid in enumerate(self._wjid):
            self._jpos[k] = d.qpos[m.jnt_qposadr[jid]]
            self._jvel[k] = d.qvel[m.jnt_dofadr[jid]]
