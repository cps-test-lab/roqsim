"""Controller plugin: joint-position hold for a manipulator + joint-state publishing.

The mobile analogue is :mod:`roqsim_mobile.plugins.diff_drive`. This plugin resolves the arm's
(prefixed) position actuators, holds a target joint vector every ``pre_step``, and declares the
arm's I/O as backend-neutral :class:`~roqsim.context.Endpoint`s that any bridge serves: a
``joint_states`` output and a ``follow_joint_trajectory`` action (MoveIt2 execution).

Optionally (``stream_commands: true``) it also declares a high-rate ``<controller>/joint_trajectory``
*topic* input -- the same command interface a ros2_control JointTrajectoryController exposes. This is
the reusable path for streaming controllers (e.g. ``moveit_servo``): each inbound single-point
``JointTrajectory`` just sets the held target, so a fast stream of positions servos the arm. It is
arm-agnostic (any arm using this plugin gets it) and needs no downstream changes -- the sim now
matches what a real driver offers, so one servo/controller config drives sim and hardware alike.

It also registers an :class:`ArmHandle` on the blackboard under ``arm:<name>`` exposing
``joint_names``, ``set_targets(names, positions)`` and ``read_state()`` for in-process consumers;
a scripted ``test_target`` drives the arm standalone.

Config::

    arm_controller:
      arm: ur10e                 # entity name registered by spawn_arm
      joints: [shoulder_pan_joint, ...]  # optional: the joints this controller owns. Omitted, the
                                 #   plugin claims every joint actuator sharing the entity's prefix,
                                 #   which is right for a standalone arm and wrong for an arm that
                                 #   shares its entity with other actuated parts -- a humanoid's legs,
                                 #   a mobile manipulator's wheels. There the scan claims those too
                                 #   and this plugin then fights their owner, writing position targets
                                 #   into what may be torque actuators. Naming the joints also scopes
                                 #   `joint_states` to this arm, so several controllers can share one
                                 #   topic without each restating the others' joints.
      gripper_actuator: left_gripper  # required WITH `joints:` for a gripper, and the ONLY way to
                                 #   declare one that is a plain joint actuator (the X-Series arms
                                 #   drive their jaws from a `left_finger` slide, which the scan
                                 #   below would otherwise claim as a seventh arm joint). Resolved by
                                 #   ACTUATOR NAME, so it need not be a tendon -- and a tendon one is
                                 #   not inferable anyway once an entity carries two (left/right).
      namespace: ur10e           # transport scope (default: inherited from spawn_arm's namespace)
      topics: {joint_states: /joint_states}  # optional: hardwire the joint_states topic to an
                                 #   absolute name, overriding namespace (see Plugin.topic_override)
      controller_name: arm_controller   # action at <controller_name>/follow_joint_trajectory
      goal_tolerance: 0.5        # rad the joints may end from the trajectory's last waypoint before
                                 #   the action reports GOAL_TOLERANCE_VIOLATED instead of success.
                                 #   A scalar applies to every joint; {joint: rad} sets them apart;
                                 #   0 disables the check. Loose on purpose -- it exists to catch an
                                 #   arm that never arrived (blocked, saturated, planned through the
                                 #   furniture), not to grade a servo's steady-state error.
      goal_time_tolerance: 1.0   # s the joints get, after the last waypoint, to reach that
      stream_commands: false     # also expose <controller_name>/joint_trajectory as a high-rate topic
                                 #   input (mirrors ros2_control's JointTrajectoryController): the path
                                 #   moveit_servo streams position targets to. Off by default.
      velocity_commands: false   # also accept JOINT VELOCITIES at <controller_name>/joint_velocity
                                 #   (see "Velocity commands" below). Off by default.
      velocity_timeout_s: 0.5    # watchdog: a velocity command decays to zero if not refreshed within
                                 #   this window, so a dropped stream cannot leave the arm drifting.
      gripper_ctrl: 255.0        # ctrl held on any non-joint (tendon) actuator, e.g. the gripper
      rest: {joint1: 0.0, ...}   # {joint: angle} spawn+hold stance. Seeds BOTH the reset qpos (so the
                                 #   arm spawns in the pose) and the held target (so it stays there).
                                 #   Needed whenever the arm is carried by `spawn_robot`, which sets
                                 #   only the base pose and no joint stance -- see below.
      test_target: [...]         # optional joint vector held every tick (standalone demo)

To pose the arm by hand with the viewer's control sliders instead, run ``roqsim --manual-control``
(a run-level switch; see :attr:`roqsim.context.SimContext.manual_control`).

**Velocity commands** (``velocity_commands: true``). Reactive whole-body controllers -- resolved-rate
or QP redundancy resolution, e.g. Haviland et al.'s holistic mobile manipulation -- emit joint
**velocities**, not positions. This plugin's actuators are
position servos, so a velocity command is integrated into the held target at the physics rate:
``target += qd * dt``, clamped to each joint's range. That is what a real velocity-mode driver does on
top of a position-controlled joint, and it keeps the servo's gravity-compensated hold -- a MuJoCo
``<velocity>`` actuator would sag under gravity whenever the command is zero.

Two consequences worth knowing before using it for a metric:

- **The achieved profile is shaped by the servo, not only by the command.** Integrating and then
  tracking with a stiff PD adds the actuator's own dynamics, so end-effector acceleration is not purely
  the controller's. Where acceleration *is* the measured quantity, verify tracking error and report the
  servo gains as part of the setup.
- **A stream that stops must stop the arm.** ``velocity_timeout_s`` zeroes a stale command; without a
  watchdog an interrupted stream integrates the last velocity forever.

``ArmHandle.set_velocities(names, velocities)`` is the in-process entry point; the transport endpoint is
``<controller_name>/joint_velocity``.

**The ``rest`` stance.** ``spawn_arm`` supplies a per-model ``home`` that this plugin seeds its targets from. ``spawn_robot``
does **not**: a robot spawn sets the base pose only, so an arm carried by a mobile base falls back to
``qpos0`` -- all joints zero. For the Panda that is not a neutral default but an actively bad pose (its
``link5`` and ``hand`` collision geoms overlap by 0.030 m there), so a mobile manipulator must declare
``rest`` in its manifest. It seeds the reset ``qpos`` *and* the held target, by joint name, which is
attach-safe where a model ``<keyframe>`` is not (``spawn_robot`` strips keyframes -- they cannot merge
into a composed world). Mirrors ``agibot_g2_controller``'s ``rest``.

If the arm has a non-joint (tendon) actuator -- a parallel gripper -- it also becomes commandable: the
plugin declares a ``control_msgs/GripperCommand`` action endpoint at
``<gripper_controller_name>/gripper_cmd`` and publishes a ``() -> (position, velocity)`` reader on the
blackboard under ``gripper:<arm>`` (the bridge's GripperCommand handler watches it to report
reached/stalled). The commanded position (the gripper joint angle, e.g. 0=open .. 0.8=closed for a
Robotiq 2F-85) is mapped linearly onto the tendon actuator's ctrlrange. Gripper config::

      gripper_controller_name: gripper_controller   # action at <name>/gripper_cmd
      gripper_joint: right_driver_joint  # joint whose angle is the reported gripper position
      gripper_open: 0.0          # position value that maps to the open end of the actuator ctrlrange
      gripper_close: 0.8         # position value that maps to the closed end
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import mujoco

from roqsim.context import Endpoint, SimContext
from roqsim.plugin import Plugin

from ._arm import (
    named_actuators,
    named_joints,
    prefixed_actuators,
    prefixed_joints,
    strip_prefix,
)


@dataclass
class ArmHandle:
    """Published by :class:`ArmControllerPlugin`; consumed by in-process drivers (scripts, tests).

    ``set_targets`` accepts (joint names, positions) — unknown names are ignored, so a MoveIt
    trajectory naming a subset of joints works. ``read_state`` returns the latest
    ``(names, positions, velocities, efforts)`` for all arm joints — effort included because a real
    driver reports it. Both run on the physics thread.
    """

    name: str
    joint_names: list[str]  # controllable joints, in actuator order (unprefixed)
    set_targets: Callable[[list[str], list[float]], None]
    read_state: Callable[[], tuple[list[str], list[float], list[float]]]
    # Joint-velocity command (rad/s), integrated into the held target every tick. Present only when
    # `velocity_commands: true`; None otherwise, so a consumer can detect the capability rather than
    # discovering it by silent no-op.
    set_velocities: Callable[[list[str], list[float]], None] | None = None


class ArmControllerPlugin(Plugin):
    #: Drives an entity's actuators, so it cannot function without one: it belongs inside that
    #: entity's ``components:`` block. (A *sensor* may be world-mounted and does not set this.)
    requires_owner = True

    def validate_config(self, config: dict) -> list[str]:
        # Only the ``joint_states`` endpoint topic is hardwireable here (the trajectory action is not
        # a topic). ``topics: {joint_states: /joint_states}`` matches external/hardware names.
        errors = self.validate_topics(config)
        if "joints" in config and not isinstance(config["joints"], list):
            errors.append("arm_controller: `joints` must be a list of joint names")
        if config.get("gripper_actuator") and not config.get("joints"):
            # Without `joints:` the prefix scan already finds tendon actuators, so naming one here
            # would be silently ignored -- and on a two-gripper entity that reads as a working config.
            errors.append(
                "arm_controller: `gripper_actuator` only applies together with `joints`; "
                "without it the prefix scan picks up tendon actuators itself"
            )
        errors += self._validate_tolerances(config)
        return errors

    @staticmethod
    def _validate_tolerances(config: dict) -> list[str]:
        """Refuse a goal tolerance that would not do what it says.

        A negative one is meaningless, and a per-joint one naming a joint this controller does not
        own is worse than meaningless: it is silently dropped, so the author reads the config as
        setting a tolerance that was never in force. Only checkable against an explicit ``joints:``
        -- without one the plugin claims joints by prefix scan, which needs the compiled model.
        """
        errors: list[str] = []
        for key in ("goal_tolerance", "goal_time_tolerance"):
            value = config.get(key)
            if value is None or isinstance(value, dict):
                continue
            if not isinstance(value, int | float) or value < 0.0:
                errors.append(
                    f"arm_controller: `{key}` must be a non-negative number, got {value!r}"
                )
        tol = config.get("goal_tolerance")
        if isinstance(tol, dict):
            for joint, value in tol.items():
                if not isinstance(value, int | float) or value < 0.0:
                    errors.append(
                        f"arm_controller: `goal_tolerance[{joint}]` must be a non-negative number, "
                        f"got {value!r}"
                    )
            owned = config.get("joints")
            if isinstance(owned, list):
                unknown = [j for j in tol if j not in owned]
                if unknown:
                    errors.append(
                        f"arm_controller: `goal_tolerance` names {unknown}, which `joints` does not "
                        "list -- it would be dropped rather than applied"
                    )
        return errors

    def __init__(self, config=None, *, name=None, entity=None, label=None):
        super().__init__(config, name=name, entity=entity, label=label)
        # Entity name. `spawn_arm` wires its manifest plugins with `arm: <name>`, `spawn_robot` with
        # `robot: <name>` (roqsim.manifest.expand_manifest sets the spawn's own target key). Accepting
        # either lets one controller serve a standalone arm and an arm carried by a robot, without the
        # manifest having to hardcode the entity name the world happens to choose.
        self.arm = self.entity
        self.gripper_ctrl = float(self.config.get("gripper_ctrl", 255.0))
        self.stream_commands = bool(self.config.get("stream_commands", False))
        self.velocity_commands = bool(self.config.get("velocity_commands", False))
        self.velocity_timeout_s = float(self.config.get("velocity_timeout_s", 0.5))
        self._vel_cmd: dict[str, float] = {}
        self._vel_stamp = -1.0  # sim time of the last velocity command; -1 = never
        self._jnt_range: dict[
            str, tuple[float, float]
        ] = {}  # clamp integration to the joint limits
        # Optional explicit ownership (see the module docstring). Absent -> prefix scan, unchanged.
        self.joints = list(self.config.get("joints", []))
        self.gripper_actuator = self.config.get("gripper_actuator")
        self._joint_acts: list[tuple[int, int]] = []  # (actuator_id, joint_id)
        self._aux_acts: list[int] = []
        self._report_jids: list[int] = []
        self._ctrl_names: list[str] = []  # unprefixed, actuator order
        self._report_names: list[str] = []  # unprefixed, all arm joints
        self._target: dict[str, float] = {}
        self._ctx = None  # set in configure; read_state/read_gripper_state compute from its data
        # Gripper (present iff the arm has a non-joint/tendon actuator). ctrl held on the aux
        # actuator(s); defaults to gripper_ctrl until a GripperCommand goal moves it.
        self._gripper_ctrl_target = self.gripper_ctrl
        self._grip_jid: int | None = None  # joint whose angle is the reported gripper position
        self._grip_qposadr = 0
        self._grip_dofadr = 0
        self._grip_open = float(self.config.get("gripper_open", 0.0))
        self._grip_close = float(self.config.get("gripper_close", 0.8))
        self._grip_ctrl_lo = 0.0
        self._grip_ctrl_hi = 255.0
        self._gripper_key = ""  # set in configure; the reader key the bridge is pointed at

    def configure(self, ctx: SimContext) -> None:
        self._ctx = (
            ctx  # joint/gripper readbacks compute from ctx.data on demand (bridge reads at rate)
        )
        entity = ctx.entities.get(self.arm)
        prefix = entity.meta.get("prefix", "") if entity else ""
        home = list(entity.meta.get("home", [])) if entity else []
        # Transport scope for this arm's endpoints: own config wins, else inherited from the spawn
        # (spawn_arm stores its `namespace` on the entity), else unscoped.
        ns = self.config.get("namespace") or (entity.meta.get("namespace", "") if entity else "")
        m = ctx.model

        if self.joints:
            # Explicit ownership: resolve only the named joints, and only the named aux actuator.
            # Two arms on one entity each need their own gripper actuator, so it cannot be inferred.
            try:
                self._joint_acts, self._aux_acts = named_actuators(
                    m, prefix, self.joints, self.gripper_actuator
                )
            except RuntimeError as exc:
                raise RuntimeError(f"arm_controller[{self.arm}]: {exc}") from exc
        else:
            self._joint_acts, self._aux_acts = prefixed_actuators(m, prefix)
        if not self._joint_acts:
            raise RuntimeError(f"arm_controller: no joint actuators found for arm {self.arm!r}")
        self._ctrl_names = [
            strip_prefix(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, jid), prefix)
            for _, jid in self._joint_acts
        ]
        if self.joints:
            # Report only this arm's joints (plus its gripper joint, which MoveIt needs): with several
            # controllers publishing to one /joint_states topic, a prefix-wide readout would have each
            # of them restating every other subsystem's joints.
            report = list(self.joints)
            if (gjoint := self.config.get("gripper_joint")) and gjoint not in report:
                report.append(gjoint)
            self._report_jids = named_joints(m, prefix, report)
        else:
            self._report_jids = prefixed_joints(m, prefix)
        self._report_names = [
            strip_prefix(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, jid), prefix)
            for jid in self._report_jids
        ]

        # Joint ranges for velocity integration. A limitless joint (range 0 0 with autolimits off) must
        # not be clamped to zero, so treat an empty range as unbounded.
        for name, (_, jid) in zip(self._ctrl_names, self._joint_acts, strict=True):
            lo, hi = (float(v) for v in m.jnt_range[jid])
            self._jnt_range[name] = (
                (lo, hi) if (hi > lo and bool(m.jnt_limited[jid])) else (-1e9, 1e9)
            )

        # Seed targets from the home pose (mapped by joint name), falling back to current qpos.
        home_map = dict(zip(self._report_names, home, strict=False))  # home may be partial/empty
        for name, (_, jid) in zip(self._ctrl_names, self._joint_acts, strict=True):
            self._target[name] = float(home_map.get(name, ctx.data.qpos[m.jnt_qposadr[jid]]))
        self._apply_rest(ctx)
        if "test_target" in self.config:
            for name, val in zip(self._ctrl_names, self.config["test_target"], strict=False):
                self._target[name] = float(val)

        # Blackboard keys. One entity can carry several controllers (a humanoid's two arms), and they
        # would then all publish under `arm:<entity>` / `gripper:<entity>`, each silently overwriting
        # the last -- which is worse than it looks for the gripper: the bridge's GripperCommand handler
        # watches that reader to decide reached/stalled, so both arms' gripper actions would end up
        # reporting the SAME gripper's motion. In explicit-ownership mode the controller name
        # disambiguates; the single-arm default key is untouched.
        controller = self.config.get("controller_name", "arm_controller")
        gripper_controller = self.config.get("gripper_controller_name", "gripper_controller")
        arm_key = f"arm:{self.arm}:{controller}" if self.joints else f"arm:{self.arm}"
        self._gripper_key = (
            f"gripper:{self.arm}:{gripper_controller}" if self.joints else f"gripper:{self.arm}"
        )
        for key in (arm_key, self._gripper_key if self._aux_acts else None):
            if key and ctx.blackboard.get(key) is not None:
                raise RuntimeError(
                    f"arm_controller[{self.arm}]: blackboard key {key!r} is already registered. Two "
                    f"controllers on one entity need distinct `controller_name` / "
                    f"`gripper_controller_name`, else they overwrite each other's handles."
                )

        # ArmHandle: for in-process consumers (scripted drivers, tests) that bypass any transport.
        ctx.blackboard.set(
            arm_key,
            ArmHandle(
                name=self.arm,
                joint_names=list(self._ctrl_names),
                set_targets=self.set_targets,
                read_state=self.read_state,
                set_velocities=self.set_velocities if self.velocity_commands else None,
            ),
        )

        # Declare joint_states as a backend-neutral output endpoint (no ROS import here). The bridge
        # resolves the type string and publishes it; ``namespace`` keeps several arms' topics apart
        # under one shared bridge.
        ctx.interface.add(
            Endpoint(
                name="joint_states",
                direction="out",
                owner=self.arm,
                namespace=ns,
                read=self.read_state,
                rate_hz=50.0,
                backend={
                    "ros2": {
                        "type": "sensor_msgs.msg.JointState",
                        "topic": self.topic_override("joint_states") or "joint_states",
                    }
                },
            )
        )

        # Trajectory execution as a backend-neutral *action* endpoint: the hint names the action
        # type as a string; a bridge with a handler for it (see roqsim_ros_bridge.actions) runs the
        # goal and feeds each waypoint through ``write`` as a neutral (names, positions) payload.
        ctx.interface.add(
            Endpoint(
                name="follow_joint_trajectory",
                direction="in",
                owner=self.arm,
                namespace=ns,
                write=lambda wp: self.set_targets(*wp),
                backend={
                    "ros2": {
                        "action": "control_msgs.action.FollowJointTrajectory",
                        "name": f"{controller}/follow_joint_trajectory",
                        # Which ArmHandle the handler reads back, so its feedback reports measured
                        # `actual` against commanded `desired` -- as a real JTC does. Its default
                        # (arm:<owner>) cannot tell two arms on one entity apart.
                        "arm_state_key": arm_key,
                        # What "reached the goal" means for THIS arm. The handler grades the result
                        # on the joints rather than on the trajectory's clock; a goal may tighten
                        # these per joint, and a heavier or softer arm loosens them here.
                        "goal_tolerance": self.config.get("goal_tolerance", 0.5),
                        "goal_time_tolerance": float(self.config.get("goal_time_tolerance", 1.0)),
                    }
                },
            )
        )

        # Controller state, the third interface a ros2_control JointTrajectoryController exposes
        # alongside the action and the command topic (rqt_joint_trajectory_controller and most
        # diagnostics read it). Reported against the held target, which IS this controller's setpoint.
        ctx.interface.add(
            Endpoint(
                name="controller_state",
                direction="out",
                owner=self.arm,
                namespace=ns,
                read=self.read_controller_state,
                rate_hz=50.0,
                backend={
                    "ros2": {
                        "type": "control_msgs.msg.JointTrajectoryControllerState",
                        "topic": f"{controller}/controller_state",
                    }
                },
            )
        )

        # Streaming joint-position command as a *topic* input, mirroring ros2_control's
        # JointTrajectoryController ``<controller>/joint_trajectory`` topic (the real UR driver exposes
        # both that topic and the action). This is the high-rate path moveit_servo drives: each inbound
        # single-point JointTrajectory just sets the held target, so a stream of positions servos the arm
        # (``pre_step`` holds ``data.ctrl`` at the target every tick). Same neutral payload as the action.
        if self.stream_commands:
            ctx.interface.add(
                Endpoint(
                    name="joint_command",
                    direction="in",
                    owner=self.arm,
                    namespace=ns,
                    write=lambda wp: self.set_targets(*wp),
                    backend={
                        "ros2": {
                            "type": "trajectory_msgs.msg.JointTrajectory",
                            "topic": f"{controller}/joint_trajectory",
                        }
                    },
                )
            )

        # Joint-VELOCITY command input, for reactive controllers that resolve to joint rates rather
        # than poses (see "Velocity commands" in the module docstring). Integrated in pre_step.
        if self.velocity_commands:
            ctx.interface.add(
                Endpoint(
                    name="joint_velocity",
                    direction="in",
                    owner=self.arm,
                    namespace=ns,
                    write=lambda wp: self.set_velocities(*wp),
                    backend={
                        "ros2": {
                            "type": "trajectory_msgs.msg.JointTrajectory",
                            "topic": f"{controller}/joint_velocity",
                        }
                    },
                )
            )

        # Gripper: a non-joint (tendon) actuator makes this arm's hand commandable. Map the tendon's
        # ctrlrange to a commanded gripper position and expose a GripperCommand action + a state reader.
        if self._aux_acts:
            lo, hi = m.actuator_ctrlrange[self._aux_acts[0]]
            self._grip_ctrl_lo, self._grip_ctrl_hi = float(lo), float(hi)
            gjoint = self.config.get("gripper_joint")
            if gjoint:
                jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{prefix}{gjoint}")
                if jid < 0:
                    raise RuntimeError(
                        f"arm_controller: gripper_joint {gjoint!r} not found for arm {self.arm!r}"
                    )
                self._grip_jid = jid
                self._grip_qposadr = int(m.jnt_qposadr[jid])
                self._grip_dofadr = int(m.jnt_dofadr[jid])
            ctx.blackboard.set(self._gripper_key, self.read_gripper_state)
            ctx.interface.add(
                Endpoint(
                    name="gripper_cmd",
                    direction="in",
                    owner=self.arm,
                    namespace=ns,
                    write=self.set_gripper,
                    backend={
                        "ros2": {
                            "action": "control_msgs.action.GripperCommand",
                            "name": f"{gripper_controller}/gripper_cmd",
                            # Tell the bridge's handler which reader to watch; its default is
                            # gripper:<owner>, which cannot distinguish two grippers on one entity.
                            "state_key": self._gripper_key,
                        }
                    },
                )
            )

    def _apply_rest(self, ctx: SimContext) -> None:
        """Overlay the `rest` stance onto data.qpos and the held target, by joint name.

        Runs under configure/on_reset, both on the physics thread, so writing ``data`` obeys the
        single-writer rule. Joints the stance does not name keep whatever they already had.
        """
        rest = self.config.get("rest") or {}
        if not rest:
            return
        m = ctx.model
        by_name = dict(zip(self._ctrl_names, self._joint_acts, strict=True))
        for jn, ang in rest.items():
            if jn not in by_name:
                raise RuntimeError(
                    f"arm_controller[{self.arm}]: rest stance names joint {jn!r}, which is not one of "
                    f"this arm's controllable joints {sorted(by_name)}"
                )
            _, jid = by_name[jn]
            ctx.data.qpos[m.jnt_qposadr[jid]] = float(ang)
            self._target[jn] = float(ang)

    def set_targets(self, names, positions) -> None:
        for n, p in zip(names, positions, strict=False):  # tolerate external/partial input
            if n in self._target:
                self._target[n] = float(p)
        # A position command supersedes any in-flight velocity command; otherwise the integrator would
        # keep walking away from the pose the caller just asked for.
        self._vel_cmd.clear()

    def set_velocities(self, names, velocities) -> None:
        """Command joint velocities (rad/s), integrated into the held target each tick.

        Only meaningful with ``velocity_commands: true``; unknown joint names are ignored, so a
        controller naming a subset (or a superset, e.g. a whole-body QP that also solves base DOFs)
        works without the caller filtering first.
        """
        if not self.velocity_commands:
            return
        for n, v in zip(names, velocities, strict=False):
            if n in self._target:
                self._vel_cmd[n] = float(v)
        self._vel_stamp = float(self._ctx.data.time) if self._ctx is not None else 0.0

    def _integrate_velocity(self, d) -> None:
        """target += qd*dt, clamped to the joint range; stale commands decay to a hold."""
        if not self._vel_cmd:
            return
        if self._vel_stamp >= 0.0 and (d.time - self._vel_stamp) > self.velocity_timeout_s:
            # Watchdog: hold position rather than integrate a command nobody refreshed.
            self._vel_cmd.clear()
            return
        dt = float(self._ctx.model.opt.timestep) if self._ctx is not None else 0.0
        for name, qd in self._vel_cmd.items():
            lo, hi = self._jnt_range[name]
            self._target[name] = min(max(self._target[name] + qd * dt, lo), hi)

    def read_state(self):
        # Computed on demand: the bridge calls this only at the joint_states rate, not every physics
        # step, so there is no per-step cost. Runs on the physics thread inside the bridge's post_step.
        #
        # Effort is included because a real driver reports it: ros2_control fills the effort interface
        # and the G1's own /lowstate carries per-motor tau_est. ``qfrc_actuator`` is the actuator force
        # already projected onto the joint's DOF, so it is the per-joint effort even for a gripper
        # finger driven through a tendon.
        m, d = self._ctx.model, self._ctx.data
        pos = [float(d.qpos[m.jnt_qposadr[jid]]) for jid in self._report_jids]
        vel = [float(d.qvel[m.jnt_dofadr[jid]]) for jid in self._report_jids]
        eff = [float(d.qfrc_actuator[m.jnt_dofadr[jid]]) for jid in self._report_jids]
        return (self._report_names, pos, vel, eff)

    def read_controller_state(self):
        """``(names, desired, actual, velocities)`` for the controlled joints, in actuator order.

        Only the joints this controller commands, not everything it reports in ``joint_states``: a
        JointTrajectoryControllerState describes the control loop, and a joint with no actuator (a
        mimicked finger) has no setpoint to state.
        """
        m, d = self._ctx.model, self._ctx.data
        desired = [self._target[n] for n in self._ctrl_names]
        actual = [float(d.qpos[m.jnt_qposadr[jid]]) for _, jid in self._joint_acts]
        vel = [float(d.qvel[m.jnt_dofadr[jid]]) for _, jid in self._joint_acts]
        return (list(self._ctrl_names), desired, actual, vel)

    def set_gripper(self, position) -> None:
        """Map a commanded gripper position (gripper_joint angle) onto the tendon actuator ctrl."""
        span = self._grip_close - self._grip_open
        frac = 0.0 if span == 0 else (float(position) - self._grip_open) / span
        frac = max(0.0, min(1.0, frac))
        self._gripper_ctrl_target = self._grip_ctrl_lo + frac * (
            self._grip_ctrl_hi - self._grip_ctrl_lo
        )

    def read_gripper_state(self):
        # Computed on demand (see read_state); (0, 0) when the arm has no gripper joint.
        if self._grip_jid is None:
            return (0.0, 0.0)
        d = self._ctx.data
        return (float(d.qpos[self._grip_qposadr]), float(d.qvel[self._grip_dofadr]))

    def _write_ctrl(self, d) -> None:
        for name, (aid, _) in zip(self._ctrl_names, self._joint_acts, strict=True):
            d.ctrl[aid] = self._target[name]
        for aid in self._aux_acts:
            d.ctrl[aid] = self._gripper_ctrl_target

    def on_reset(self, ctx: SimContext) -> None:
        # spawn_arm re-applies the home qpos on reset; resync targets to hold there.
        m = ctx.model
        for name, (_, jid) in zip(self._ctrl_names, self._joint_acts, strict=True):
            self._target[name] = float(ctx.data.qpos[m.jnt_qposadr[jid]])
        # Then re-seat the `rest` stance, so repeated trials start from an identical arm pose. Order
        # matters: the resync above reads whatever qpos the spawn left, which for spawn_robot (no joint
        # stance) is qpos0 -- `rest` is what makes a mobile manipulator's arm reproducible per trial.
        self._apply_rest(ctx)
        self._vel_cmd.clear()  # a stale velocity command must not survive a reset
        self._gripper_ctrl_target = self.gripper_ctrl
        if "test_target" in self.config:
            for name, val in zip(self._ctrl_names, self.config["test_target"], strict=False):
                self._target[name] = float(val)
        # Manual mode: seed ctrl to the home target once so the arm starts there (and the sliders
        # open at that pose); pre_step then leaves ctrl alone for the user to drag.
        if ctx.manual_control:
            self._write_ctrl(ctx.data)

    def pre_step(self, ctx: SimContext) -> None:
        if not ctx.manual_control:
            self._integrate_velocity(ctx.data)
            self._write_ctrl(ctx.data)

    # joint / gripper state are computed on demand in read_state / read_gripper_state (the bridge reads
    # them at endpoint rate), so there is no post_step readout here -- nothing runs every step.
