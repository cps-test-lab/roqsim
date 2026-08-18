"""Controller plugin: ONNX whole-body walk policy + PD torque loop for the LimX Oli (HU_D04_01).

The 31-DoF whole-body analogue of :mod:`g1_locomotion`. Consumes a body-frame twist target (set via
the published :class:`RobotHandle` -- e.g. by the ROS bridge -- or a scripted ``test_cmd``), and every
physics step applies a PD torque loop to all 31 actuators. Every ``decimation`` steps (-> 100 Hz) it
builds the 102-dim observation, pushes it onto a 5-deep history buffer, and evaluates the pretrained
ONNX policy (input 510 = 102x5, output 31) to refresh the joint position targets. Reports the floating
base as ``odom`` and the joints as ``joint_states``.

The observation layout, PD gains, default angles, scales, torque limits and timing are lifted verbatim
from humanoid-rl-deploy-python's ``walk_controller.py`` + ``walk_param.yaml`` (bundled under
``policy/oli/``), so the policy runs on exactly the conventions it was trained on. Unlike the vendor
deploy code the IMU is read noise-free from MuJoCo ground truth (base quat/omega from qpos/qvel), as
``g1_locomotion`` does; the gait-phase observation is left out because the vendor policy was exported
with it commented out (obs is 102, not 107).

**The world must set ``sim.timestep: 0.001``** -- the PD loop and ``decimation`` are tuned for a
1000 Hz step / 100 Hz policy (see the demo world).

Config::

    oli_locomotion:
      robot: robot                 # entity name registered by spawn_robot
      namespace: ""                # transport scope (default: inherited from spawn_robot)
      policy_path: <bundled>       # override the ONNX policy (default: policy/oli/policy.onnx)
      config_path: <bundled>       # override the deploy config (default: policy/oli/walk_param.yaml)
      max_linear_vel: 0.5          # |vx| clamp (m/s)   -- vendor max_vx
      max_lateral_vel: 0.3         # |vy| clamp (m/s)   -- vendor max_vy
      max_angular_vel: 0.5         # |yaw_rate| clamp (rad/s) -- vendor max_vz
      test_cmd: [0.3, 0.0, 0.0]    # optional [vx, vy, w] applied every tick (standalone demo)
"""

from __future__ import annotations

import mujoco
import numpy as np
import onnxruntime as ort
import yaml

from roqsim.context import Endpoint, RobotHandle, SimContext
from roqsim.plugin import Plugin

from ..policy import OLI_CONFIG, OLI_POLICY

# The 31 joints in the policy's PR order -- the exact index order of every walk_param.yaml array and
# of the policy's obs/action vectors. Head is pitch-then-yaw (opposite the MJCF body order); the
# resolve-by-name below makes the MJCF ordering irrelevant. See parallel_joint_mapping_en.md.
JOINTS = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "head_pitch_joint",
    "head_yaw_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_yaw_joint",
    "left_wrist_pitch_joint",
    "left_wrist_roll_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_yaw_joint",
    "right_wrist_pitch_joint",
    "right_wrist_roll_joint",
)
NUM = 31


class OliLocomotionPlugin(Plugin):
    def __init__(self, config=None, *, name=None):
        super().__init__(config, name=name)
        self.robot = self.config.get("robot", "robot")
        self.policy_path = str(self.config.get("policy_path", OLI_POLICY))
        self.config_path = str(self.config.get("config_path", OLI_CONFIG))
        self.max_v = float(self.config.get("max_linear_vel", 0.5))
        self.max_vy = float(self.config.get("max_lateral_vel", 0.3))
        self.max_w = float(self.config.get("max_angular_vel", 0.5))

        # Command (body-frame [vx, vy, yaw_rate]); written by drive(), read into the policy obs.
        self._cmd = np.zeros(3, dtype=np.float32)

        # Resolved in configure().
        self._session = self._in_name = None
        self._kp = self._kd = self._default_angle = self._action_scale = self._tlim = None
        self._ang_vel_scale = self._dof_pos_scale = self._dof_vel_scale = 0.0
        self._clip_obs = self._clip_act = 100.0
        self._num_obs = 102
        self._hist_len = 5
        self._decimation = 10

        # Per-run policy state.
        self._action = np.zeros(NUM, dtype=np.float32)
        self._target_q = None
        self._hist = None
        self._counter = 0

        # MuJoCo id/address caches (resolved in configure()).
        self._base_bid = self._base_qadr = self._base_dadr = -1
        self._qadr = self._dadr = self._aid = None
        self._ctx = None

    def validate_config(self, config: dict) -> list[str]:
        errors = self.validate_topics(config)
        for key in ("max_linear_vel", "max_lateral_vel", "max_angular_vel"):
            if key in config and float(config[key]) <= 0:
                errors.append(f"'{key}' must be > 0")
        if "test_cmd" in config and len(config["test_cmd"]) != 3:
            errors.append("'test_cmd' must be [vx, vy, w]")
        return errors

    def configure(self, ctx: SimContext) -> None:
        entity = ctx.entities.get(self.robot)
        prefix = entity.meta.get("prefix", "") if entity else ""
        ns = self.config.get("namespace") or (entity.meta.get("namespace", "") if entity else "")
        m = ctx.model
        self._ctx = ctx

        # -- deploy config (nested HumanoidRobotCfg from walk_param.yaml) -----------------------
        with open(self.config_path) as fh:
            cfg = yaml.safe_load(fh)["HumanoidRobotCfg"]
        control, norm, size = cfg["control"], cfg["normalization"], cfg["size"]
        self._kp = np.array(control["kp"], dtype=np.float32)
        self._kd = np.array(control["kd"], dtype=np.float32)
        self._default_angle = np.array(control["default_angle"], dtype=np.float32)
        self._action_scale = np.array(control["action_scale"], dtype=np.float32)
        self._tlim = np.array(control["user_torque_limit"], dtype=np.float32)
        self._decimation = int(control["decimation"])
        self._ang_vel_scale = float(norm["obs_scales"]["ang_vel"])
        self._dof_pos_scale = float(norm["obs_scales"]["dof_pos"])
        self._dof_vel_scale = float(norm["obs_scales"]["dof_vel"])
        self._clip_obs = float(norm["clip_scales"]["clip_observations"])
        self._clip_act = float(norm["clip_scales"]["clip_actions"])
        self._num_obs = int(size["observations_size"])
        self._hist_len = int(size["obs_history_length"])
        self._target_q = self._default_angle.copy()

        # -- ONNX policy (single-threaded: a small MLP at 100 Hz, cf. g1's torch.set_num_threads) -
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        so.inter_op_num_threads = 1
        self._session = ort.InferenceSession(
            self.policy_path, sess_options=so, providers=["CPUExecutionProvider"]
        )
        self._in_name = self._session.get_inputs()[0].name

        # -- resolve MuJoCo ids/addresses (prefixed) --------------------------------------------
        def jnt(n):
            return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, prefix + n)

        def act(n):
            return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, prefix + n)

        self._base_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, prefix + "base_link")
        base_jid = jnt("base_free")
        if self._base_bid < 0 or base_jid < 0:
            raise RuntimeError(
                f"oli_locomotion: base body/joint not found for robot {self.robot!r} "
                f"(prefix {prefix!r}); expected {prefix}base_link / {prefix}base_free"
            )
        self._base_qadr = m.jnt_qposadr[base_jid]
        self._base_dadr = m.jnt_dofadr[base_jid]

        jids = [jnt(n) for n in JOINTS]
        self._aid = np.array([act(n) for n in JOINTS], dtype=np.int32)
        missing = [n for n, j in zip(JOINTS, jids, strict=True) if j < 0]
        missing += [n for n, a in zip(JOINTS, self._aid, strict=True) if a < 0]
        if missing:
            raise RuntimeError(f"oli_locomotion: could not resolve joints/actuators {missing}")
        self._qadr = np.array([m.jnt_qposadr[j] for j in jids], dtype=np.int32)
        self._dadr = np.array([m.jnt_dofadr[j] for j in jids], dtype=np.int32)

        # -- RobotHandle + backend-neutral endpoints (identical contract to g1/diff_drive) -------
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

    # -- command / readback -------------------------------------------------------------------
    def drive(self, vx: float, vy: float, w: float) -> None:
        """Body-frame twist target (vx forward, vy left, w yaw-rate), clamped to the trained range."""
        self._cmd[0] = float(np.clip(vx, -self.max_v, self.max_v))
        self._cmd[1] = float(np.clip(vy, -self.max_vy, self.max_vy))
        self._cmd[2] = float(np.clip(w, -self.max_w, self.max_w))

    def read_odom(self):
        # Computed on demand at the endpoint rate (cf. g1). Trailing z is the true base height (the
        # Oli pelvis stands ~0.9 m up); the bridge tf/odom carry it, nav2 stays 2D.
        d = self._ctx.data
        x, y, z = d.xpos[self._base_bid]
        qw, qx, qy, qz = d.xquat[self._base_bid]
        yaw = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
        vgx, vgy = d.qvel[self._base_dadr : self._base_dadr + 2]
        c, s = np.cos(yaw), np.sin(yaw)
        return (
            float(x),
            float(y),
            float(yaw),
            float(c * vgx + s * vgy),
            float(-s * vgx + c * vgy),
            float(d.qvel[self._base_dadr + 5]),
            float(z),
        )

    def read_joint_states(self):
        d = self._ctx.data
        return (JOINTS, d.qpos[self._qadr], d.qvel[self._dadr])

    def on_reset(self, ctx: SimContext) -> None:
        # spawn_robot.on_reset (runs first) placed the base; set the default stance so the robot
        # starts from the pose the policy expects, and clear the policy state.
        d = ctx.data
        if self._qadr is not None:
            d.qpos[self._qadr] = self._default_angle
            d.qvel[self._dadr] = 0.0
            mujoco.mj_forward(ctx.model, d)
        self._cmd[:] = 0.0
        self._action[:] = 0.0
        self._target_q = self._default_angle.copy()
        self._hist = None
        self._counter = 0

    # -- control loop -------------------------------------------------------------------------
    def pre_step(self, ctx: SimContext) -> None:
        if ctx.manual_control:
            # Viewer sliders own the motors this run; skip the policy so its history can't drift
            # against what the user drags. These are TORQUE actuators: with the policy off the Oli
            # simply folds under gravity -- expected, not a bug.
            return
        if "test_cmd" in self.config:
            vx, vy, w = self.config["test_cmd"]
            self.drive(float(vx), float(vy), float(w))

        d = ctx.data
        qj = d.qpos[self._qadr]
        dqj = d.qvel[self._dadr]

        # 100 Hz policy update: build the 102-dim obs, push onto the 5-deep history, run the policy.
        if self._counter % self._decimation == 0:
            omega = d.qvel[self._base_dadr + 3 : self._base_dadr + 6]  # body-frame angular velocity
            rot = d.xmat[self._base_bid].reshape(3, 3)  # world-from-body
            proj_grav = rot.T @ np.array([0.0, 0.0, -1.0])  # gravity in body frame

            obs = np.concatenate(
                [
                    omega * self._ang_vel_scale,
                    proj_grav,
                    self._cmd,
                    (qj - self._default_angle) * self._dof_pos_scale,
                    dqj * self._dof_vel_scale,
                    self._action,
                ]
            ).astype(np.float32)
            np.clip(obs, -self._clip_obs, self._clip_obs, out=obs)

            if self._hist is None:
                self._hist = np.tile(obs, self._hist_len)  # prime with the first obs (cf. vendor)
            else:
                self._hist[self._num_obs :] = self._hist[: -self._num_obs]  # shift older back
                self._hist[: self._num_obs] = obs  # newest at front

            out = self._session.run(None, {self._in_name: self._hist[None]})[0].flatten()
            self._action = np.clip(out, -self._clip_act, self._clip_act).astype(np.float32)
            self._target_q = self._action * self._action_scale + self._default_angle

        # 1000 Hz PD torque loop on all 31 motors, clamped to the vendor per-joint torque limits.
        tau = (self._target_q - qj) * self._kp - dqj * self._kd
        d.ctrl[self._aid] = np.clip(tau, -self._tlim, self._tlim)
        self._counter += 1
