"""Controller plugin: RL flat-terrain walking policy for the Boston Dynamics Spot (12-DoF quadruped).

The quadruped analogue of :mod:`roqsim_humanoid.plugins.g1_locomotion` (and, at the interface,
:mod:`roqsim_mobile.plugins.diff_drive`). Consumes a body-frame twist target (set via the
published :class:`RobotHandle` -- e.g. by the ROS bridge -- or a scripted ``test_cmd`` for standalone
demos). Every ``control_decimation`` steps (-> 50 Hz) it builds the 48-dim observation and evaluates
the pretrained TorchScript policy to update the leg **position targets**, which it writes straight to
Spot's MuJoCo position actuators (kp=60 / kv=1.5 in ``spot.xml`` close the PD loop -- there is no
per-step torque loop, unlike the G1). Reports the floating base as ``odom`` and the leg joints as
``joint_states``.

The observation/action convention, default pose, action scale and joint order are transcribed from
NVIDIA's Isaac ``Isaac-Velocity-Flat-Spot-v0`` env (bundled as ``policy/spot.yaml``), so the policy
runs on exactly what it was trained on.

Policy weights are NVIDIA-licensed and *not committed*: fetch them with
``python -m roqsim_quadruped.policy.fetch_policy`` or point ``policy_path`` / ``$SPOT_POLICY_PATH``
at a local ``spot_policy.pt``.

Config -- a component of the entry that spawns the robot, since ownership is where the entry
sits rather than a config key::

    spot_locomotion:
      namespace: ""                # transport scope (default: inherited from spawn_robot's namespace)
      policy_path: <fetched>       # TorchScript policy (default: $SPOT_POLICY_PATH or policy/spot_policy.pt)
      config_path: <bundled>       # deploy config (default: policy/spot.yaml)
      max_linear_vel: 1.5          # |vx| clamp (m/s)
      max_lateral_vel: 0.8         # |vy| clamp (m/s)
      max_angular_vel: 1.5         # |yaw_rate| clamp (rad/s)
      test_cmd: [0.5, 0.0, 0.0]    # optional [vx, vy, w] applied every tick (standalone demo)
"""

from __future__ import annotations

import os

import mujoco
import numpy as np
import torch
import yaml

from roqsim.context import Endpoint, RobotHandle, SimContext
from roqsim.plugin import Plugin

from ..policy import DEFAULT_CONFIG, DEFAULT_POLICY


def _quat_rotate_inverse(quat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """Rotate ``vec`` from the world frame into the base frame (R(q)^T v).

    Verbatim from Isaac Lab ``omni.isaac.lab.utils.math.quat_rotate_inverse`` for a (w, x, y, z)
    quaternion -- used for both the base linear velocity and projected-gravity observations, exactly
    as the Spot flat env computes them.
    """
    w = quat[0]
    qvec = quat[1:4]
    a = vec * (2.0 * w * w - 1.0)
    b = np.cross(qvec, vec) * (2.0 * w)
    c = qvec * (np.dot(qvec, vec) * 2.0)
    return (a - b + c).astype(np.float32)


# Gravity direction in the world frame (Isaac projected_gravity uses [0, 0, -1]).
_GRAVITY_W = np.array([0.0, 0.0, -1.0], dtype=np.float32)


class SpotLocomotionPlugin(Plugin):
    #: Drives an entity's actuators, so it cannot function without one: it belongs inside that
    #: entity's ``components:`` block. (A *sensor* may be world-mounted and does not set this.)
    requires_owner = True

    def __init__(self, config=None, *, name=None, entity=None, label=None):
        super().__init__(config, name=name, entity=entity, label=label)
        self.robot = self.entity
        # Policy path resolution order: explicit config > $SPOT_POLICY_PATH > bundled (fetched) path.
        self.policy_path = str(
            self.config.get("policy_path") or os.environ.get("SPOT_POLICY_PATH") or DEFAULT_POLICY
        )
        self.config_path = str(self.config.get("config_path", DEFAULT_CONFIG))
        self.max_v = float(self.config.get("max_linear_vel", 1.5))
        self.max_vy = float(self.config.get("max_lateral_vel", 0.8))
        self.max_w = float(self.config.get("max_angular_vel", 1.5))

        # Command (body-frame [vx, vy, yaw_rate]); written by drive(), read by the policy obs.
        self._cmd = np.zeros(3, dtype=np.float32)

        # Loaded/resolved in configure().
        self._policy = None
        self._joint_order: tuple[str, ...] = ()
        self._default_angles = None
        self._lin_vel_scale = self._ang_vel_scale = 1.0
        self._dof_pos_scale = self._dof_vel_scale = 1.0
        self._cmd_scale = None
        self._action_scale = 0.2
        self._num_obs = 48
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
        self._ctx = (
            ctx  # odom/joint readbacks compute from ctx.data on demand (bridge reads at rate)
        )

        # -- load the deploy config (joint order, default pose, scales, timing) -----------------
        with open(self.config_path) as fh:
            cfg = yaml.safe_load(fh)
        self._joint_order = tuple(cfg["joint_order"])
        self._default_angles = np.array(cfg["default_angles"], dtype=np.float32)
        self._action_scale = float(cfg["action_scale"])
        self._lin_vel_scale = float(cfg.get("lin_vel_scale", 1.0))
        self._ang_vel_scale = float(cfg.get("ang_vel_scale", 1.0))
        self._dof_pos_scale = float(cfg.get("dof_pos_scale", 1.0))
        self._dof_vel_scale = float(cfg.get("dof_vel_scale", 1.0))
        self._cmd_scale = np.array(cfg.get("cmd_scale", [1.0, 1.0, 1.0]), dtype=np.float32)
        self._num_obs = int(cfg["num_obs"])
        self._num_actions = int(cfg["num_actions"])
        self._decimation = int(cfg["control_decimation"])
        self._obs = np.zeros(self._num_obs, dtype=np.float32)
        self._target_q = self._default_angles.copy()

        # -- load the TorchScript policy -------------------------------------------------------
        if not os.path.exists(self.policy_path):
            raise RuntimeError(
                f"spot_locomotion: policy not found at {self.policy_path!r}. The Spot policy is "
                "NVIDIA-licensed and not committed -- fetch it with "
                "`python -m roqsim_quadruped.policy.fetch_policy`, or set the plugin's "
                "`policy_path` / the SPOT_POLICY_PATH env var to a local spot_policy.pt."
            )
        # Pin torch to one thread: the policy is a small MLP run at 50 Hz, so single-threaded
        # inference is fastest and avoids fanning out across every core (which oversubscribes the
        # box against MuJoCo's physics thread and any co-running policies/nav stacks).
        torch.set_num_threads(1)
        self._policy = torch.jit.load(self.policy_path)
        self._policy.eval()

        # -- resolve MuJoCo ids/addresses (prefixed), mapping each name to the policy's slot ---
        def jnt(n):
            return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, prefix + n)

        def act(n):
            return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, prefix + n)

        self._base_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, prefix + "base_link")
        base_jid = jnt("base_free")
        if self._base_bid < 0 or base_jid < 0:
            raise RuntimeError(
                f"spot_locomotion: base body/joint not found for robot {self.robot!r} "
                f"(prefix {prefix!r}); expected {prefix}base_link / {prefix}base_free"
            )
        self._base_qadr = m.jnt_qposadr[base_jid]
        self._base_dadr = m.jnt_dofadr[base_jid]

        leg_jids = [jnt(n) for n in self._joint_order]
        self._leg_aid = np.array([act(n) for n in self._joint_order], dtype=np.int32)
        missing = [n for n, j in zip(self._joint_order, leg_jids, strict=True) if j < 0]
        missing += [n for n, a in zip(self._joint_order, self._leg_aid, strict=True) if a < 0]
        if missing:
            raise RuntimeError(f"spot_locomotion: could not resolve joints/actuators {missing}")
        self._leg_qadr = np.array([m.jnt_qposadr[j] for j in leg_jids], dtype=np.int32)
        self._leg_dadr = np.array([m.jnt_dofadr[j] for j in leg_jids], dtype=np.int32)

        # -- RobotHandle for in-process consumers (teleop, standalone driver) ------------------
        ctx.blackboard.set(
            f"robot:{self.robot}",
            RobotHandle(name=self.robot, drive=self.drive, read_odom=self.read_odom),
        )

        # -- backend-neutral endpoints (identical contract to diff_drive / g1_locomotion) ------
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
        # Computed on demand: the bridge calls this only at the endpoint rate (50 Hz), not every
        # physics step, so there is no per-step cost. Runs on the physics thread inside the bridge's
        # post_step, reading the same post-mj_step data. Returns (x, y, yaw, vx, vy, w, z); trailing z
        # is the true base height (Spot stands ~0.5 m up; the bridge tf/odom carry it, nav2 stays 2D).
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
        return (list(self._joint_order), d.qpos[self._leg_qadr], d.qvel[self._leg_dadr])

    def on_reset(self, ctx: SimContext) -> None:
        # spawn_robot.on_reset (runs first) placed the base; put the legs in the default standing
        # stance so the robot starts stable, hold the position targets there, and clear policy state.
        d = ctx.data
        if self._leg_qadr is not None:
            d.qpos[self._leg_qadr] = self._default_angles
            d.qvel[self._leg_dadr] = 0.0
            d.ctrl[self._leg_aid] = self._default_angles
            mujoco.mj_forward(ctx.model, d)
        self._cmd[:] = 0.0
        self._action[:] = 0.0
        self._target_q = self._default_angles.copy()
        self._counter = 0

    # -- control loop -------------------------------------------------------------------------
    def pre_step(self, ctx: SimContext) -> None:
        if ctx.manual_control:
            # The viewer's sliders own the leg actuators this run; skip the policy entirely rather
            # than run it and drop its output, so its recurrent state can't drift against the pose
            # the user is dragging. on_reset left ctrl at the default stance for the sliders.
            return
        if "test_cmd" in self.config:
            vx, vy, w = self.config["test_cmd"]
            self.drive(float(vx), float(vy), float(w))

        d = ctx.data
        qj = d.qpos[self._leg_qadr]
        dqj = d.qvel[self._leg_dadr]

        # 50 Hz policy update: rebuild the observation and refresh the leg position targets.
        if self._counter % self._decimation == 0:
            quat = np.asarray(d.qpos[self._base_qadr + 3 : self._base_qadr + 7], dtype=np.float32)
            lin_w = np.asarray(d.qvel[self._base_dadr : self._base_dadr + 3], dtype=np.float32)
            omega = np.asarray(d.qvel[self._base_dadr + 3 : self._base_dadr + 6], dtype=np.float32)

            n = self._num_actions
            obs = self._obs
            obs[0:3] = _quat_rotate_inverse(quat, lin_w) * self._lin_vel_scale
            obs[3:6] = omega * self._ang_vel_scale
            obs[6:9] = _quat_rotate_inverse(quat, _GRAVITY_W)
            obs[9:12] = self._cmd * self._cmd_scale
            obs[12 : 12 + n] = (qj - self._default_angles) * self._dof_pos_scale
            obs[12 + n : 12 + 2 * n] = dqj * self._dof_vel_scale
            obs[12 + 2 * n : 12 + 3 * n] = self._action

            with torch.no_grad():
                self._action = self._policy(torch.from_numpy(obs).unsqueeze(0)).numpy().squeeze()
            self._target_q = self._action * self._action_scale + self._default_angles

        # Write position targets to Spot's position actuators (MuJoCo closes the PD loop each step;
        # ctrl is clamped to the joint range via the actuators' inherited ctrlrange).
        d.ctrl[self._leg_aid] = self._target_q
        self._counter += 1

    # odom / joint_states are computed on demand in read_odom / read_joint_states (the bridge reads
    # them at endpoint rate), so there is no post_step readout here -- nothing runs every step.
