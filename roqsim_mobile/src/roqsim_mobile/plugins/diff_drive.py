"""Controller plugin: differential-drive kinematics + wheel-encoder odometry.

Ported from our earlier in-house nav prototype's ``TurtleBot4``. Consumes a body-frame twist target (set via the published
:class:`RobotHandle` — e.g. by the ROS bridge — or a scripted ``test_cmd`` for standalone demos),
writes wheel velocity-servo targets in ``pre_step``, and integrates encoder odometry in ``post_step``.

Config::

    diff_drive:
      robot: robot                 # entity name registered by spawn_robot
      namespace: ""                # transport scope (default: inherited from spawn_robot's namespace)
      wheel_radius: 0.03575
      wheel_separation: 0.233
      max_linear_vel: 0.31
      max_angular_vel: 1.90
      wheel_accel_limit: 0.9       # m/s^2 per wheel; ramps commands (Create 3 default/max 900 mm/s^2)
      left_actuator: left_wheel_motor
      right_actuator: right_wheel_motor
      left_joint: left_wheel_joint
      right_joint: right_wheel_joint
      test_cmd: [0.15, 0.4]        # optional [v, w] applied every tick (standalone demo)

Skid-steer (>1 wheel per side, e.g. Husky A200): give the per-side actuator/joint *lists* instead of
the singular keys; every left wheel gets the same command, every right wheel the same, and odometry
averages each side's wheel velocities. ``slip_factor`` compensates the lateral scrub of a skid-steer
(see below)::

    diff_drive:
      wheel_radius: 0.17775
      wheel_separation: 0.5708      # track width
      slip_factor: 4.0              # ICR slip compensation (1.0 = ideal diff-drive)
      left_actuators:  [front_left_wheel_motor,  rear_left_wheel_motor]
      right_actuators: [front_right_wheel_motor, rear_right_wheel_motor]
      left_joints:  [front_left_wheel_joint,  rear_left_wheel_joint]
      right_joints: [front_right_wheel_joint, rear_right_wheel_joint]

``slip_factor`` exists because a 4-wheel skid-steer turns by scrubbing its wheels sideways: with
MuJoCo point contacts the base yaws at only ~15% of the ideal differential-drive prediction, which
would leave a planner unable to rotate. The factor inflates the yaw term of the wheel command and is
divided back out of odometry, so commanded yaw is achieved and odometry stays consistent with the
base's real motion. It is a per-robot *calibration* against the model's contact/friction setup --
re-measure it (achieved vs commanded yaw rate) if wheel friction, mass, or the timestep change.
"""

from __future__ import annotations

import mujoco
import numpy as np

from roqsim.context import Endpoint, RobotHandle, SimContext
from roqsim.plugin import Plugin


class DiffDrivePlugin(Plugin):
    def __init__(self, config=None, *, name=None):
        super().__init__(config, name=name)
        self.robot = self.config.get("robot", "robot")
        self.r = float(self.config.get("wheel_radius", 0.03575))
        self.L = float(self.config.get("wheel_separation", 0.233))
        self.max_v = float(self.config.get("max_linear_vel", 0.31))
        self.max_w = float(self.config.get("max_angular_vel", 1.90))
        # Skid-steer ICR slip compensation. A 4-wheel skid-steer scrubs laterally when turning, so
        # the base yaws at only a fraction of the ideal differential-drive prediction; `slip_factor`
        # (>1) inflates the yaw term of the wheel command to restore the commanded yaw, and divides
        # it back out of odometry so the two stay consistent. 1.0 = ideal diff-drive (e.g. TB4).
        self.slip = float(self.config.get("slip_factor", 1.0))
        # Wheel linear-acceleration limit: the real Create 3 ramps wheel velocity commands with
        # wheel_accel_limit (defaults to its max settable 900 mm/s^2). Without this the wheels jump
        # to full speed in one step and the robot lurches on the first cmd_vel.
        # https://iroboteducation.github.io/create3_docs/api/safety/
        self.wheel_accel = float(self.config.get("wheel_accel_limit", 0.9))
        self._target_v = 0.0
        self._target_w = 0.0
        self._cmd_wl = 0.0  # ramped wheel angular-velocity commands (rad/s)
        self._cmd_wr = 0.0
        self._odom = [0.0, 0.0, 0.0, 0.0, 0.0]  # x, y, yaw, v, w
        # Per-side actuator/joint names: accept singular keys (2-wheel diff-drive) or plural lists
        # (skid-steer, >1 wheel per side). The lists drive the shared kinematics unchanged.
        self._la_names = self._as_list(
            self.config.get("left_actuators"), self.config.get("left_actuator", "left_wheel_motor")
        )
        self._ra_names = self._as_list(
            self.config.get("right_actuators"),
            self.config.get("right_actuator", "right_wheel_motor"),
        )
        self._lj_names = self._as_list(
            self.config.get("left_joints"), self.config.get("left_joint", "left_wheel_joint")
        )
        self._rj_names = self._as_list(
            self.config.get("right_joints"), self.config.get("right_joint", "right_wheel_joint")
        )
        # Wheel joint state, updated in place each post_step (read by the joint_states endpoint).
        self._jnames = self._lj_names + self._rj_names
        self._jpos = np.zeros(len(self._jnames))
        self._jvel = np.zeros(len(self._jnames))
        # resolved in configure()
        self._aid_l: list[int] = []
        self._aid_r: list[int] = []
        self._jid_l: list[int] = []
        self._jid_r: list[int] = []

    @staticmethod
    def _as_list(plural, singular):
        if plural:
            return list(plural)
        return [singular]

    def validate_config(self, config: dict) -> list[str]:
        errors = self.validate_topics(config)
        for key in ("wheel_radius", "wheel_separation", "slip_factor"):
            if key in config and float(config[key]) <= 0:
                errors.append(f"'{key}' must be > 0")
        for side in ("left", "right"):
            acts, joints = config.get(f"{side}_actuators"), config.get(f"{side}_joints")
            if acts is not None and joints is not None and len(acts) != len(joints):
                errors.append(f"'{side}_actuators' and '{side}_joints' must have the same length")
        if "test_cmd" in config and len(config["test_cmd"]) != 2:
            errors.append("'test_cmd' must be [v, w]")
        return errors

    def configure(self, ctx: SimContext) -> None:
        entity = ctx.entities.get(self.robot)
        prefix = entity.meta.get("prefix", "") if entity else ""
        # Transport scope for this robot's endpoints: own config wins, else inherited from the spawn.
        ns = self.config.get("namespace") or (entity.meta.get("namespace", "") if entity else "")
        m = ctx.model

        def act(n):
            return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, prefix + n)

        def jnt(n):
            return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, prefix + n)

        self._aid_l = [act(n) for n in self._la_names]
        self._aid_r = [act(n) for n in self._ra_names]
        self._jid_l = [jnt(n) for n in self._lj_names]
        self._jid_r = [jnt(n) for n in self._rj_names]
        missing = [
            name
            for names, ids in (
                (self._la_names, self._aid_l),
                (self._ra_names, self._aid_r),
                (self._lj_names, self._jid_l),
                (self._rj_names, self._jid_r),
            )
            for name, v in zip(names, ids, strict=True)
            if v < 0
        ]
        if missing:
            raise RuntimeError(f"diff_drive: could not resolve {missing} for robot {self.robot!r}")

        # RobotHandle: kept for in-process consumers (teleop, standalone driver).
        ctx.blackboard.set(
            f"robot:{self.robot}",
            RobotHandle(name=self.robot, drive=self.drive, read_odom=self.read_odom),
        )

        # Declare this robot's I/O as backend-neutral endpoints. A bridge (ROS 2, zenoh, ...) reads
        # ctx.interface and wires them up; nothing ROS-specific is imported here -- the message type
        # is named as a string under a backend hint block, resolved by the bridge. ``namespace``
        # scopes topics/frames per robot so one bridge can serve a many-robot world.
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
                        "child_frame_id": "base_link",
                        "emit_tf": True,
                    }
                },
            )
        )
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
        """Body-frame twist target (vy dropped: differential drive cannot strafe)."""
        self._target_v = float(np.clip(vx, -self.max_v, self.max_v))
        self._target_w = float(np.clip(w, -self.max_w, self.max_w))

    def read_odom(self):
        x, y, yaw, v, w = self._odom
        return (x, y, yaw, v, 0.0, w)

    def read_joint_states(self):
        return (self._jnames, self._jpos, self._jvel)

    def on_reset(self, ctx: SimContext) -> None:
        self._target_v = self._target_w = 0.0
        self._cmd_wl = self._cmd_wr = 0.0
        self._odom = [0.0, 0.0, 0.0, 0.0, 0.0]

    def pre_step(self, ctx: SimContext) -> None:
        if ctx.manual_control:
            return  # the viewer's sliders own the wheel actuators this run
        if "test_cmd" in self.config:
            v, w = self.config["test_cmd"]
            self.drive(float(v), 0.0, float(w))
        # differential-drive inverse kinematics -> target wheel angular velocities
        # (slip == 1.0 is the ideal model; > 1.0 compensates skid-steer scrub, see __init__)
        yaw_term = self._target_w * self.slip * self.L / 2.0
        tgt_wl = (self._target_v - yaw_term) / self.r
        tgt_wr = (self._target_v + yaw_term) / self.r
        # Ramp each wheel toward its target, capped at the wheel accel limit (as angular rate).
        dw_max = (self.wheel_accel / self.r) * ctx.dt if self.wheel_accel > 0 else np.inf
        self._cmd_wl += float(np.clip(tgt_wl - self._cmd_wl, -dw_max, dw_max))
        self._cmd_wr += float(np.clip(tgt_wr - self._cmd_wr, -dw_max, dw_max))
        for aid in self._aid_l:
            ctx.data.ctrl[aid] = self._cmd_wl
        for aid in self._aid_r:
            ctx.data.ctrl[aid] = self._cmd_wr

    def post_step(self, ctx: SimContext) -> None:
        m, d = ctx.model, ctx.data
        # Per-side wheel angular velocity: average across that side's wheels (>1 for skid-steer).
        wl = float(np.mean([d.qvel[m.jnt_dofadr[j]] for j in self._jid_l]))
        wr = float(np.mean([d.qvel[m.jnt_dofadr[j]] for j in self._jid_r]))
        v = self.r * (wr + wl) / 2.0
        # Invert the same slip model used to command the wheels, so odometry tracks the base's
        # actual yaw rather than the (inflated) ideal-differential prediction.
        w = self.r * (wr - wl) / (self.L * self.slip)
        o = self._odom
        o[0] += v * np.cos(o[2]) * ctx.dt
        o[1] += v * np.sin(o[2]) * ctx.dt
        o[2] = (o[2] + w * ctx.dt + np.pi) % (2 * np.pi) - np.pi
        o[3], o[4] = v, w
        # Wheel joint state for the joint_states endpoint: written in place so read() is zero-copy.
        jids = self._jid_l + self._jid_r
        for k, jid in enumerate(jids):
            self._jpos[k] = d.qpos[m.jnt_qposadr[jid]]
            self._jvel[k] = d.qvel[m.jnt_dofadr[jid]]
