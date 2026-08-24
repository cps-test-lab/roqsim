"""Controller plugin: RL walking policy + PD torque loop for the Unitree G1 (12-DoF legs).

The legged analogue of :mod:`roqsim_mobile.plugins.diff_drive`. Consumes a body-frame twist target
(set via the published :class:`RobotHandle` -- e.g. by the ROS bridge -- or a scripted ``test_cmd``
for standalone demos), and every physics step applies a PD torque loop to the 12 leg actuators. Every
``control_decimation`` steps (-> 50 Hz) it builds the 47-dim observation and evaluates the pretrained
TorchScript policy to update the leg position targets. Reports the floating base as ``odom`` and the
leg joints as ``joint_states``.

The observation/action conventions, PD gains, default angles, scales and timing are lifted verbatim
from unitree_rl_gym's ``deploy/deploy_mujoco/deploy_mujoco.py`` + ``configs/g1.yaml`` (bundled in this
package under ``policy/``), so the policy runs on exactly what it was trained on.

Config::

    g1_locomotion:
      robot: robot                 # entity name registered by spawn_robot
      namespace: ""                # transport scope (default: inherited from spawn_robot's namespace)
      policy: g1_stand             # use a spec-driven policy instead of the bundled walk one: names a
                                   #   directory under policy/ holding <name>.spec.yaml + its checkpoint
                                   #   (roqsim.policy.PolicySpec). The spec carries the observation layout,
                                   #   gains and trained envelope, so adding a policy edits no Python.
                                   #   Omitted -> the bundled walk policy, unchanged.
      policy_path: <bundled>       # override the TorchScript policy (default: policy/motion.pt)
      config_path: <bundled>       # override the deploy config (default: policy/g1.yaml)
      gait_period: 0.8             # gait-phase period (s) feeding the sin/cos phase obs
      max_linear_vel: 1.0          # |vx| clamp (m/s)
      max_lateral_vel: 0.5         # |vy| clamp (m/s)
      max_angular_vel: 1.0         # |yaw_rate| clamp (rad/s)
      test_cmd: [0.4, 0.0, 0.0]    # optional [vx, vy, w] applied every tick (standalone demo)
      station_keeping: true        # hold position when commanded to stop (see below)
      station_gain: [2.5, 2.5, 2.0]  # P gains on [x, y, yaw] error -> body-frame twist
      station_deadband: [0.02, 0.05]  # [m, rad] inside which no correction is applied

``station_keeping`` closes a position loop so a zero command means *stay here* rather than *walk at
zero velocity*. It is off by default (the policy's raw behaviour is what a locomotion study wants to
measure), but anything standing still needs it: the policy takes only a velocity command and has no
notion of where it is, so a stationary G1 drifts at roughly 0.06--0.09 m/s -- **0.6--0.9 m over ten
seconds**, measured on both this model and ``unitree_g1``. That is enough to walk a robot away from the
table it was reaching for, and it also means a navigating robot creeps away after arriving at its goal.

The hold target arms itself whenever the external command returns to (near) zero, capturing the pose
the robot is standing at, and releases the moment a non-zero command arrives -- so nav2 drives normally
and only the standstill is corrected. The correction is a P term on the world-frame error rotated into
the body frame, clamped by the same ``max_*`` limits as any other command, with a deadband so the robot
is not permanently taking small steps to chase millimetres.

Measured on ``unitree_g1_dex1``, 10 s at rest -- settled offset from the armed pose, and how far the
base still wanders in a subsequent 2 s (the number a manipulation task actually cares about):

===================  ===============  ====================
station_gain         settled offset   further motion / 2 s
===================  ===============  ====================
off                  0.905 m, rising  --
[1.0, 1.0, 1.5]      0.116 m          0.0045 m
[2.5, 2.5, 2.0]      0.046 m          0.0049 m
[5.0, 5.0, 3.0]      0.026 m          0.0031 m
===================  ===============  ====================

Residual wander is ~5 mm regardless, so the gain only trades settled offset against how hard the policy
is pushed; the default is the middle row. A P term leaves a steady-state offset because the policy
needs a finite velocity command to step at all -- plan in the robot's own base frame (as MoveIt should
here anyway) and a constant offset costs nothing.

This is what the hardware does too: the real G1's sport mode has a stand state that holds pose rather
than integrating a zero velocity.
"""

from __future__ import annotations

import mujoco
import numpy as np
import torch
import yaml

from roqsim.context import Endpoint, RobotHandle, SimContext
from roqsim.plugin import Plugin
from roqsim.policy import ObservationState, PolicySpec

from ..policy import DEFAULT_CONFIG, DEFAULT_POLICY, find_spec

# The 12 leg joints in policy order -- matches the g1.yaml kps/kds/default_angles ordering and the
# <motor> actuator order in unitree_g1.xml. Do not reorder: the policy's observation and action
# vectors are indexed by this exact sequence.
LEG_JOINTS = (
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
)


def _gravity_orientation(quat: np.ndarray) -> np.ndarray:
    """Projected-gravity direction in the base frame from a (w, x, y, z) quaternion.

    Verbatim from unitree_rl_gym deploy_mujoco.get_gravity_orientation -- the policy's torso-tilt obs.
    """
    qw, qx, qy, qz = quat
    return np.array(
        [
            2 * (-qz * qx + qw * qy),
            -2 * (qz * qy + qw * qx),
            1 - 2 * (qw * qw + qz * qz),
        ],
        dtype=np.float32,
    )


class G1LocomotionPlugin(Plugin):
    #: Drives an entity's actuators, so it cannot function without one: it belongs inside that
    #: entity's ``components:`` block. (A *sensor* may be world-mounted and does not set this.)
    requires_owner = True

    def __init__(self, config=None, *, name=None, entity=None, label=None):
        super().__init__(config, name=name, entity=entity, label=label)
        self.robot = self.entity
        self.policy_path = str(self.config.get("policy_path", DEFAULT_POLICY))
        self.config_path = str(self.config.get("config_path", DEFAULT_CONFIG))
        self.gait_period = float(self.config.get("gait_period", 0.8))
        # Opt-in: a spec-driven policy replaces the bundled walk policy's hardcoded observation and
        # gains. Absent, every existing world behaves exactly as before.
        policy_ref = self.config.get("policy")
        self._spec: PolicySpec | None = (
            PolicySpec.from_yaml(find_spec(str(policy_ref))) if policy_ref else None
        )
        self._obs_qadr = np.zeros(0, dtype=np.int32)
        self._obs_dadr = np.zeros(0, dtype=np.int32)
        self.max_v = float(self.config.get("max_linear_vel", 1.0))
        self.max_vy = float(self.config.get("max_lateral_vel", 0.5))
        self.max_w = float(self.config.get("max_angular_vel", 1.0))

        # Command (body-frame [vx, vy, yaw_rate]); written by drive(), read by the policy obs.
        self._cmd = np.zeros(3, dtype=np.float32)

        # Station keeping: hold the pose the robot was standing at when the command went to zero.
        self.station_keeping = bool(self.config.get("station_keeping", False))
        gain = self.config.get("station_gain", [2.5, 2.5, 2.0])
        self.station_gain = np.array([float(v) for v in gain], dtype=np.float32)
        band = self.config.get("station_deadband", [0.02, 0.05])
        self.station_pos_band, self.station_yaw_band = float(band[0]), float(band[1])
        self._hold: np.ndarray | None = None  # (x, y, yaw) target, or None while driving

        # Loaded/resolved in configure().
        self._policy = None
        self._kps = self._kds = self._default_angles = None
        self._ang_vel_scale = self._dof_pos_scale = self._dof_vel_scale = self._action_scale = 0.0
        self._cmd_scale = None
        self._num_obs = 47
        self._num_actions = 12
        self._decimation = 10

        # Per-run policy state.
        self._action = np.zeros(12, dtype=np.float32)
        self._target_q = None  # leg position targets (default_angles + action*scale)
        self._obs = None
        self._counter = 0

        # MuJoCo id/address caches (resolved in configure()).
        self._base_bid = -1
        self._base_qadr = -1
        self._base_dadr = -1
        self._leg_qadr = None  # (12,) qpos addresses, policy order
        self._leg_dadr = None  # (12,) qvel/dof addresses, policy order
        self._leg_aid = None  # (12,) actuator ids, policy order
        self._ctx = None  # set in configure; the out-endpoint readbacks compute from its data

    def validate_config(self, config: dict) -> list[str]:
        errors = self.validate_topics(config)
        for key in ("max_linear_vel", "max_lateral_vel", "max_angular_vel", "gait_period"):
            if key in config and float(config[key]) <= 0:
                errors.append(f"'{key}' must be > 0")
        if config.get("policy"):
            spec_path = find_spec(str(config["policy"]))
            if not spec_path.exists():
                errors.append(
                    f"'policy': no spec at {spec_path}. A policy is a directory under "
                    f"roqsim_humanoid/policy/ holding <name>.spec.yaml and its checkpoint."
                )
            if "gait_period" in config:
                # The gait phase is a term the spec either declares or does not; a period set beside a
                # spec that has no phase term reads as configuration that silently does nothing.
                errors.append("'gait_period' does not apply with 'policy'; it is part of the spec")
        if "test_cmd" in config and len(config["test_cmd"]) != 3:
            errors.append("'test_cmd' must be [vx, vy, w]")
        if "station_gain" in config and len(config["station_gain"]) != 3:
            errors.append("'station_gain' must be [x, y, yaw] gains")
        if "station_deadband" in config and len(config["station_deadband"]) != 2:
            errors.append("'station_deadband' must be [position_m, yaw_rad]")
        return errors

    def configure(self, ctx: SimContext) -> None:
        entity = ctx.entities.get(self.robot)
        prefix = entity.meta.get("prefix", "") if entity else ""
        ns = self.config.get("namespace") or (entity.meta.get("namespace", "") if entity else "")
        m = ctx.model
        self._ctx = (
            ctx  # odom/joint readbacks compute from ctx.data on demand (bridge reads at rate)
        )

        # -- load the deploy config (gains, default angles, scales, timing) --------------------
        # Skipped entirely under `policy:`; the spec carries all of it (see the block further down),
        # and loading the bundled walk checkpoint only to discard it would be misleading as well as
        # wasteful.
        if self._spec is None:
            with open(self.config_path) as fh:
                cfg = yaml.safe_load(fh)
            self._kps = np.array(cfg["kps"], dtype=np.float32)
            self._kds = np.array(cfg["kds"], dtype=np.float32)
            self._default_angles = np.array(cfg["default_angles"], dtype=np.float32)
            self._ang_vel_scale = float(cfg["ang_vel_scale"])
            self._dof_pos_scale = float(cfg["dof_pos_scale"])
            self._dof_vel_scale = float(cfg["dof_vel_scale"])
            self._action_scale = float(cfg["action_scale"])
            self._cmd_scale = np.array(cfg["cmd_scale"], dtype=np.float32)
            self._num_obs = int(cfg["num_obs"])
            self._num_actions = int(cfg["num_actions"])
            self._decimation = int(cfg["control_decimation"])
            self._obs = np.zeros(self._num_obs, dtype=np.float32)
            self._target_q = self._default_angles.copy()

        # -- load the TorchScript policy -------------------------------------------------------
        # Pin torch to one thread: the policy is a small MLP run at 50 Hz, so single-threaded
        # inference is fastest and avoids fanning out across every core (which oversubscribes the
        # box against MuJoCo's physics thread and any co-running policies/nav stacks).
        if self._spec is None:
            torch.set_num_threads(1)
            self._policy = torch.jit.load(self.policy_path)
            self._policy.eval()

        # -- resolve MuJoCo ids/addresses (prefixed) -------------------------------------------
        def jnt(n):
            return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, prefix + n)

        def act(n):
            return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, prefix + n)

        self._base_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, prefix + "base_link")
        base_jid = jnt("base_free")
        if self._base_bid < 0 or base_jid < 0:
            raise RuntimeError(
                f"g1_locomotion: base body/joint not found for robot {self.robot!r} "
                f"(prefix {prefix!r}); expected {prefix}base_link / {prefix}base_free"
            )
        self._base_qadr = m.jnt_qposadr[base_jid]
        self._base_dadr = m.jnt_dofadr[base_jid]

        leg_jids = [jnt(n) for n in LEG_JOINTS]
        self._leg_aid = np.array([act(n) for n in LEG_JOINTS], dtype=np.int32)
        missing = [n for n, j in zip(LEG_JOINTS, leg_jids, strict=True) if j < 0]
        missing += [n for n, a in zip(LEG_JOINTS, self._leg_aid, strict=True) if a < 0]
        if missing:
            raise RuntimeError(f"g1_locomotion: could not resolve joints/actuators {missing}")
        self._leg_qadr = np.array([m.jnt_qposadr[j] for j in leg_jids], dtype=np.int32)
        self._leg_dadr = np.array([m.jnt_dofadr[j] for j in leg_jids], dtype=np.int32)

        # -- spec-driven policy (opt-in via `policy:`) ------------------------------------------
        # Resolves the joints the spec *observes but does not command* -- the arms, for a stand policy
        # that has to see the CoM shift coming. The actuated set stays LEG_JOINTS: this plugin owns the
        # twelve leg motors and nothing else, which is what keeps the arms MoveIt's.
        if self._spec is not None:
            if tuple(self._spec.actuated) != LEG_JOINTS:
                raise RuntimeError(
                    f"g1_locomotion: spec {self._spec.name!r} actuates {self._spec.actuated}, but this "
                    f"plugin owns exactly the 12 leg joints. A spec that commands the arms would fight "
                    f"the arm_controllers for the same actuators."
                )
            obs_jids = [jnt(n) for n in self._spec.observed]
            if missing_obs := [
                n for n, j in zip(self._spec.observed, obs_jids, strict=True) if j < 0
            ]:
                raise RuntimeError(f"g1_locomotion: spec observes unknown joints {missing_obs}")
            self._obs_qadr = np.array([m.jnt_qposadr[j] for j in obs_jids], dtype=np.int32)
            self._obs_dadr = np.array([m.jnt_dofadr[j] for j in obs_jids], dtype=np.int32)
            if not self._spec.checkpoint.exists():
                raise RuntimeError(
                    f"g1_locomotion: policy {self._spec.name!r} has no checkpoint at "
                    f"{self._spec.checkpoint}. Train it with `make train-{self._spec.name}` (see "
                    f"external/train/README.md), or point `policy:` at one that exists."
                )
            torch.set_num_threads(1)
            self._policy = torch.jit.load(str(self._spec.checkpoint))
            self._policy.eval()
            self._obs = np.zeros(self._spec.num_obs, dtype=np.float32)
            self._action = np.zeros(len(LEG_JOINTS), dtype=np.float32)
            self._kps = self._spec.kp
            self._kds = self._spec.kd
            self._default_angles = self._spec.default_angles
            self._action_scale = self._spec.action_scale
            self._decimation = self._spec.decimation
            self._target_q = self._default_angles.copy()

        # -- RobotHandle for in-process consumers (teleop, standalone driver) ------------------
        ctx.blackboard.set(
            f"robot:{self.robot}",
            RobotHandle(name=self.robot, drive=self.drive, read_odom=self.read_odom),
        )

        # -- backend-neutral endpoints (identical contract to diff_drive) ----------------------
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

    def _station_keeping_cmd(self, d) -> np.ndarray:
        """The command to actually give the policy: the external one, or a hold correction at rest.

        Arms the hold target when the external command falls to ~zero, so it captures wherever the
        robot came to rest, and drops it the moment a real command arrives -- nav2 then drives with
        nothing in its way and only the standstill is corrected.
        """
        if float(np.abs(self._cmd).max()) > 1e-3:
            self._hold = None
            return self._cmd

        x, y, _ = d.xpos[self._base_bid]
        qw, qx, qy, qz = d.xquat[self._base_bid]
        yaw = float(np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz)))
        if self._hold is None:
            self._hold = np.array([x, y, yaw], dtype=np.float32)
            return self._cmd

        ex, ey = float(self._hold[0] - x), float(self._hold[1] - y)
        eyaw = float(np.arctan2(np.sin(self._hold[2] - yaw), np.cos(self._hold[2] - yaw)))
        # Deadband: without it the robot takes endless small corrective steps, which is both unlike a
        # standing robot and a worse disturbance to a carried payload than the drift it corrects.
        if np.hypot(ex, ey) < self.station_pos_band and abs(eyaw) < self.station_yaw_band:
            return np.zeros(3, dtype=np.float32)

        # World-frame error into the body frame; the policy's command is body-frame.
        c, s = np.cos(yaw), np.sin(yaw)
        gx, gy, gw = self.station_gain
        return np.array(
            [
                np.clip(gx * (c * ex + s * ey), -self.max_v, self.max_v),
                np.clip(gy * (-s * ex + c * ey), -self.max_vy, self.max_vy),
                np.clip(gw * eyaw, -self.max_w, self.max_w),
            ],
            dtype=np.float32,
        )

    def read_odom(self):
        # Computed on demand: the bridge calls this only at the endpoint rate, not every physics step,
        # so there is no per-step cost. Runs on the physics thread inside the bridge's post_step,
        # reading the same post-mj_step data. Returns (x, y, yaw, vx, vy, w, z); trailing z is the true
        # base height (the G1 pelvis stands ~0.7 m up; the bridge tf/odom carry it, nav2 stays 2D).
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
        # Computed on demand (see read_odom). Fancy-indexing qpos/qvel returns fresh arrays.
        d = self._ctx.data
        return (LEG_JOINTS, d.qpos[self._leg_qadr], d.qvel[self._leg_dadr])

    def on_reset(self, ctx: SimContext) -> None:
        # spawn_robot.on_reset (runs first) placed the base; put the legs in the default standing
        # stance so the robot starts stable, and clear the policy state.
        d = ctx.data
        if self._leg_qadr is not None:
            d.qpos[self._leg_qadr] = self._default_angles
            d.qvel[self._leg_dadr] = 0.0
            mujoco.mj_forward(ctx.model, d)
        self._cmd[:] = 0.0
        self._action[:] = 0.0
        self._target_q = self._default_angles.copy()
        self._counter = 0
        # Re-arm on the next tick at the freshly placed base pose, not the previous episode's.
        self._hold = None

    # -- control loop -------------------------------------------------------------------------
    def pre_step(self, ctx: SimContext) -> None:
        if ctx.manual_control:
            # The viewer's sliders own the leg motors this run; skip the policy entirely rather than
            # run it and drop its output, so its recurrent state can't drift against what the user
            # is dragging. Note these are TORQUE actuators: with the policy off, the sliders command
            # torques and a G1 left at zero simply folds -- expected, not a bug.
            return
        if "test_cmd" in self.config:
            vx, vy, w = self.config["test_cmd"]
            self.drive(float(vx), float(vy), float(w))

        d = ctx.data
        cmd = self._station_keeping_cmd(d) if self.station_keeping else self._cmd
        qj = d.qpos[self._leg_qadr]
        dqj = d.qvel[self._leg_dadr]

        # 50 Hz policy update: rebuild the observation and refresh the leg position targets.
        if self._counter % self._decimation == 0:
            quat = d.qpos[self._base_qadr + 3 : self._base_qadr + 7]  # (w, x, y, z)
            omega = d.qvel[self._base_dadr + 3 : self._base_dadr + 6]  # body-frame angular velocity

            phase = (self._counter * ctx.dt) % self.gait_period / self.gait_period
            sin_phase, cos_phase = np.sin(2 * np.pi * phase), np.cos(2 * np.pi * phase)

            obs = self._obs
            if self._spec is not None:
                # The spec's term list IS the layout; nothing here knows what is in it. `phase` is
                # supplied regardless and simply goes unused when the spec declares no gait_phase term
                # -- which a stand policy must not, since the cycling phase is what makes the walk
                # policy step at zero command.
                self._spec.build_observation(
                    ObservationState(
                        base_ang_vel=omega,
                        base_quat=quat,
                        command=cmd,
                        actuated_pos=qj,
                        actuated_vel=dqj,
                        prev_action=self._action,
                        observed_pos=d.qpos[self._obs_qadr],
                        observed_vel=d.qvel[self._obs_dadr],
                        phase=phase,
                    ),
                    out=obs,
                )
            else:
                n = self._num_actions
                obs[:3] = omega * self._ang_vel_scale
                obs[3:6] = _gravity_orientation(quat)
                obs[6:9] = cmd * self._cmd_scale
                obs[9 : 9 + n] = (qj - self._default_angles) * self._dof_pos_scale
                obs[9 + n : 9 + 2 * n] = dqj * self._dof_vel_scale
                obs[9 + 2 * n : 9 + 3 * n] = self._action
                obs[9 + 3 * n : 9 + 3 * n + 2] = (sin_phase, cos_phase)

            with torch.no_grad():
                self._action = self._policy(torch.from_numpy(obs).unsqueeze(0)).numpy().squeeze()
            self._target_q = self._action * self._action_scale + self._default_angles

        # 500 Hz PD torque loop on the 12 leg motors.
        tau = (self._target_q - qj) * self._kps - dqj * self._kds
        d.ctrl[self._leg_aid] = tau
        self._counter += 1

    # odom / joint_states are computed on demand in read_odom / read_joint_states (the bridge reads
    # them at endpoint rate), so there is no post_step readout here -- nothing runs every step.
