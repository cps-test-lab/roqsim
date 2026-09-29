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

Config -- a component of the entry that spawns the arm, since ownership is where the entry
sits rather than a config key::

    arm_controller:
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
      joint_prefix: ""           # prepended to every joint name this controller REPORTS in
                                 #   `joint_states` and accepts in a trajectory. Empty (the default)
                                 #   reports the model's own names. Set it -- conventionally to the
                                 #   arm's MJCF prefix -- when two arms must appear in ONE robot
                                 #   description: a URDF is a flat namespace, so two `shoulder_pan_joint`
                                 #   cannot coexist there, and MoveIt matches states and trajectory
                                 #   points to the description by name.
      namespace: ur10e           # transport scope (default: inherited from spawn_arm's namespace)
      topics: {joint_states: /joint_states}  # optional: rename an endpoint, here to an absolute
                                 #   name that overrides the namespace (see Plugin.topic_override)
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
reached/stalled). The commanded position is the ``gripper_joint``'s position (e.g. 0=open .. 0.8=closed
for a Robotiq 2F-85, 0.057=open .. 0.021=closed for a ViperX 300 finger slide), mapped linearly onto the
actuator's ctrlrange. Which end of the ctrlrange is open is read from the model -- the actuator's gain
and its signed moment on that joint -- so ``gripper_open``/``gripper_close`` are the joint's positions
with the fingers open and closed, the two positions the ctrlrange's ends hold it at, whichever way the
actuator runs. A gripper needs its ``gripper_joint``, driven with a constant moment; configure fails
without one.

**Grip force.** Beside the position the plugin publishes a :class:`GripperEffort` under
``gripper_effort:<arm>``, the key the endpoint's ``effort_key`` hint names. It takes GripperCommand's
``max_effort`` as ros2_control's gripper action controller does: a clamp on the gripper joint's
effort, in that joint's own unit -- newtons for a slide jaw, newton-metres for a knuckle, the unit
``/joint_states`` reports -- which is what that controller's effort adapter applies
(``gripper_controllers/hardware_interface_adapter.hpp``). A goal is never refused for its effort: a
request at or above the model's own limit saturates there, as a drive does, and ``max_effort <= 0``
and every reset restore the model's own force range. The clamp reaches the actuator through the
transmission's constant moment. A gripper actuator without a force limit keeps its range and executes
the position alone, as a position-interface controller does. Gripper config::

      gripper_controller_name: gripper_controller   # action at <name>/gripper_cmd
      gripper_joint: right_driver_joint  # joint whose angle is the reported gripper position
      gripper_open: 0.0          # gripper_joint position with the fingers open
      gripper_close: 0.8         # gripper_joint position with the fingers closed
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import mujoco
import numpy as np
from numpy.typing import NDArray

from roqsim import endpoint
from roqsim.context import SimContext
from roqsim.controllers import ACTIVE, INACTIVE, Controller, registry_for
from roqsim.plugin import Plugin
from roqsim.types import JointPositions, JointState

from ._arm import (
    named_actuators,
    named_joints,
    prefixed_actuators,
    prefixed_joints,
    strip_prefix,
)


def joint_moment(model, actuator_id: int, joint_id: int) -> float:
    """The actuator's length change per unit of ``joint_id``'s position, signed; 0.0 where not constant.

    Read from the model alone, so it holds in every configuration or not at all: an actuator on the
    joint carries its gear, and one on a fixed tendon carries gear times the tendon's coefficient on
    that joint. A spatial tendon, a site transmission, or a joint the transmission does not reach has
    no constant moment, and gets 0.0.
    """
    gear = float(model.actuator_gear[actuator_id][0])
    target = int(model.actuator_trnid[actuator_id][0])
    trn = model.actuator_trntype[actuator_id]
    if trn == mujoco.mjtTrn.mjTRN_JOINT:
        return gear if target == joint_id else 0.0
    if trn != mujoco.mjtTrn.mjTRN_TENDON:
        return 0.0
    first = int(model.tendon_adr[target])
    coef = 0.0
    for wrap in range(first, first + int(model.tendon_num[target])):
        if model.wrap_type[wrap] != mujoco.mjtWrap.mjWRAP_JOINT:
            return 0.0
        if int(model.wrap_objid[wrap]) == joint_id:
            coef = float(model.wrap_prm[wrap])
    return gear * coef


def joint_effort_per_actuator_force(model, actuator_id: int, joint_id: int) -> float:
    """The effort on ``joint_id`` per unit of the actuator's force, or 0.0 where it is not constant.

    The magnitude of :func:`joint_moment`. The unit is the joint's own -- newtons for a slide joint,
    newton-metres for a hinge -- the unit ``/joint_states`` reports its effort in.
    """
    return abs(joint_moment(model, actuator_id, joint_id))


def ctrl_direction_on_joint(model, actuator_id: int, joint_id: int) -> int:
    """+1 if raising the actuator's ctrl moves ``joint_id`` toward larger positions, -1 if smaller.

    0 where the model does not say: no constant moment on the joint, or no gain. A servo settles where
    ``gain * ctrl + bias0 + bias1 * length = 0``, so its length rises with ctrl as ``gain / -bias1``;
    a motor without a position bias pushes its length the way ``gain`` points. The joint then follows
    the length through the transmission's signed moment.
    """
    gain = float(model.actuator_gainprm[actuator_id][0])
    bias = float(model.actuator_biasprm[actuator_id][1])
    along_length = gain / -bias if bias != 0.0 else gain
    slope = along_length * joint_moment(model, actuator_id, joint_id)
    return (slope > 0.0) - (slope < 0.0)


@dataclass
class JointVelocities:
    """Target velocities for named joints.

    ROS carries it as ``trajectory_msgs/JointTrajectory`` with the velocities in the last point's
    ``positions`` (:mod:`roqsim_manipulation.ros2_types`).

    Attributes:
        names: joint names, in the order of ``velocities``
        velocities: rad/s for a revolute joint, m/s for a prismatic one
    """

    names: list[str]
    velocities: NDArray[np.float64]


@dataclass
class TrajectoryPoint:
    """Joint values at one point of a trajectory controller's loop.

    Attributes:
        positions: rad for a revolute joint, m for a prismatic one
        velocities: rad/s or m/s; empty where the point states none
    """

    positions: list[float]
    velocities: list[float] = field(default_factory=list)


@dataclass
class ControllerState:
    """A trajectory controller's loop: the setpoint it holds against what the joints do.

    Shaped as ``control_msgs/JointTrajectoryControllerState``, so the ROS bridge maps it by field
    name.

    Attributes:
        joint_names: the controlled joints, in actuator order
        reference: the held target
        feedback: the measured positions and velocities
        error: reference - feedback, the sign the message states and ros2_control publishes
    """

    joint_names: list[str]
    reference: TrajectoryPoint
    feedback: TrajectoryPoint
    error: TrajectoryPoint


@dataclass
class GripperEffort:
    """GripperCommand's ``max_effort`` for one gripper: a clamp on its joint's effort.

    Published by :class:`ArmControllerPlugin` beside the gripper's position reader, under the key the
    ``gripper_cmd`` endpoint names in its ``effort_key`` hint. ``limit`` is the most effort the
    gripper joint can carry, in its own unit (N or N*m), and 0.0 where no clamp can be applied.
    ``set_max_effort`` runs on the physics thread; ``read_effort`` returns the joint's effort now, the
    value ``/joint_states`` reports for it.
    """

    limit: float
    set_max_effort: Callable[[float], None]
    read_effort: Callable[[], float]


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
    #: Register the one plugin that computes this arm's targets each step (see
    #: :meth:`ArmControllerPlugin.set_command_source`).
    set_command_source: Callable[[Callable[[object], None], str], None] | None = None
    #: Run that source for this step if it has not run yet. Whoever reaches it first triggers it,
    #: so the result does not depend on where either plugin sits in the world file.
    ensure_updated: Callable[[object], None] | None = None
    #: Whether this controller currently holds the arm. An inactive one keeps its last target and
    #: takes no new ones, the way a deactivated ros2_control controller does.
    is_active: Callable[[], bool] | None = None
    set_active: Callable[[bool], None] | None = None


class ArmControllerPlugin(Plugin):
    #: Drives an entity's actuators, so it cannot function without one: it belongs inside that
    #: entity's ``components:`` block. (A *sensor* may be world-mounted and does not set this.)
    requires_owner = True

    def validate_config(self, config: dict) -> list[str]:
        # ``topics:`` renames an endpoint, e.g. ``{joint_states: /joint_states}`` to match
        # external/hardware names; an action's rename is its action name.
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
        # The one plugin that computes this arm's targets, and the step its work last ran
        # for. Pulled from `pre_step`, so declaration order cannot decide whether a command
        # lands this step or the next.
        self._command_source = None
        self._command_source_owner = ""
        self._commanded_step = -1
        self._registered = None
        # Whether this controller holds the arm. Active unless the world says otherwise, so a world
        # that never switches behaves exactly as it always has; `inactive` is what ros2_control's
        # `spawner --inactive` leaves behind.
        self._active = str(self.config.get("initial_state", "active")) != "inactive"
        #: What a reset returns `_active` to: `initial_state`, or inactive once a command source has
        #: claimed the arm at configure.
        self._configured_active = self._active
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
        # The gripper_joint positions at the ctrlrange's low and high end: gripper_open and
        # gripper_close in the order the actuator runs, set in configure.
        self._grip_q_at_ctrl_lo = self._grip_open
        self._grip_q_at_ctrl_hi = self._grip_close
        self._gripper_key = ""  # set in configure; the reader key the bridge is pointed at
        # Names the endpoints' hints carry, resolved in configure.
        self._controller = ""
        self._gripper_controller = ""
        self._arm_key = ""
        self._effort_key = ""
        #: A non-joint (tendon) actuator makes the hand commandable, with a ``gripper_cmd``.
        self.has_gripper = False
        # Grip force: the model's own force range on the gripper actuator (what a reset restores), the
        # gripper joint's effort per unit of actuator force, and the joint-effort limit that range
        # allows. `_grip_cap_warned` keeps a gripper that cannot take a clamp to one warning.
        self._grip_forcerange: tuple[float, float] | None = None
        self._grip_gain = 0.0
        self._grip_limit = 0.0
        self._grip_cap_warned = False

    def configure(self, ctx: SimContext) -> None:
        self._ctx = (
            ctx  # joint/gripper readbacks compute from ctx.data on demand (bridge reads at rate)
        )
        entity = ctx.entities.get(self.arm)
        prefix = entity.meta.get("prefix", "") if entity else ""
        home = list(entity.meta.get("home", [])) if entity else []
        # Transport scope for this arm's endpoints and controllers: own config wins, else inherited
        # from the spawn (spawn_arm stores its `namespace` on the entity), else unscoped.
        ns = self.endpoint_namespace(ctx)
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
        # What this controller CALLS its joints, everywhere it reports or accepts them. The MJCF
        # prefix is stripped so the names are the model's own; `joint_prefix` puts one back, which is
        # what a cell needs when two arms of the same model must appear in ONE robot description --
        # `shoulder_pan_joint` cannot name two joints there, and MoveIt matches joint states and
        # trajectory points to that description BY NAME.
        jp = str(self.config.get("joint_prefix", "") or "")
        self._ctrl_names = [
            jp + strip_prefix(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, jid), prefix)
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
            jp + strip_prefix(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, jid), prefix)
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
        self._controller, self._gripper_controller, self._arm_key = (
            controller,
            gripper_controller,
            arm_key,
        )
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

        # This arm's controllers, as ros2_control would list them. The trajectory controller
        # claims the joints it drives; the broadcaster claims nothing and reads them, which is how
        # both are active at once on every real robot.
        registry = registry_for(ctx)
        self._registered = registry.register(
            Controller(
                name=controller,
                type="joint_trajectory_controller/JointTrajectoryController",
                claims=tuple(f"{j}/position" for j in self._ctrl_names),
                state=ACTIVE if self._active else INACTIVE,
                namespace=ns,
                owner=self.arm,
                apply=self.set_active,
            )
        )
        # A robot has ONE joint_state_broadcaster reading every joint, however many arms it has, so
        # a second arm on the same entity extends the broadcaster the first one registered.
        broadcaster = self.config.get("joint_state_broadcaster_name", "joint_state_broadcaster")
        broadcaster_type = "joint_state_broadcaster/JointStateBroadcaster"
        reads = tuple(f"{j}/position" for j in self._ctrl_names)
        shared = next(
            (
                c
                for c in registry.all(ns)
                if c.owner == self.arm and c.name == broadcaster and c.type == broadcaster_type
            ),
            None,
        )
        if shared is not None:
            shared.reads += tuple(r for r in reads if r not in shared.reads)
        else:
            registry.register(
                Controller(
                    name=broadcaster,
                    type=broadcaster_type,
                    reads=reads,
                    state=ACTIVE,
                    namespace=ns,
                    owner=self.arm,
                )
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
                set_command_source=self.set_command_source,
                is_active=self.is_active,
                set_active=self.set_active,
                ensure_updated=self.ensure_updated,
            ),
        )

        # Gripper: a non-joint (tendon) actuator makes this arm's hand commandable. Map the tendon's
        # ctrlrange to a commanded gripper position and expose a GripperCommand action + a state reader.
        if self._aux_acts:
            grip = self._aux_acts[0]
            lo, hi = m.actuator_ctrlrange[grip]
            self._grip_ctrl_lo, self._grip_ctrl_hi = float(lo), float(hi)
            gjoint = self.config.get("gripper_joint")
            if not gjoint:
                raise RuntimeError(
                    f"arm_controller[{self.arm}]: the arm has a gripper actuator but no "
                    "`gripper_joint`; name the joint whose position a GripperCommand sets and reports"
                )
            jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{prefix}{gjoint}")
            if jid < 0:
                raise RuntimeError(
                    f"arm_controller: gripper_joint {gjoint!r} not found for arm {self.arm!r}"
                )
            self._grip_jid = jid
            self._grip_qposadr = int(m.jnt_qposadr[jid])
            self._grip_dofadr = int(m.jnt_dofadr[jid])
            # Which end of the ctrlrange opens depends on the actuator, not on a convention: a
            # Robotiq tendon closes as its ctrl rises, an Interbotix finger slide opens.
            direction = ctrl_direction_on_joint(m, grip, jid)
            if direction == 0:
                raise RuntimeError(
                    f"arm_controller[{self.arm}]: the gripper actuator has no constant moment on "
                    f"gripper_joint {gjoint!r}, so the direction a command moves it is unknown; "
                    "drive the gripper through that joint or a fixed tendon over it"
                )
            ends = sorted((self._grip_open, self._grip_close))
            if direction < 0:
                ends.reverse()
            self._grip_q_at_ctrl_lo, self._grip_q_at_ctrl_hi = ends
            ctx.blackboard.set(self._gripper_key, self.read_gripper_state)
            self._grip_forcerange = tuple(float(v) for v in m.actuator_forcerange[grip])
            if m.actuator_forcelimited[grip]:
                self._grip_gain = joint_effort_per_actuator_force(m, grip, jid)
            self._grip_limit = min(abs(v) for v in self._grip_forcerange) * self._grip_gain
            self._effort_key = self._gripper_key.replace("gripper:", "gripper_effort:", 1)
            ctx.blackboard.set(
                self._effort_key,
                GripperEffort(
                    limit=self._grip_limit,
                    set_max_effort=self.set_gripper_max_effort,
                    read_effort=self.read_gripper_effort,
                ),
            )
            self.has_gripper = True

    # -- endpoints ----------------------------------------------------------------------------
    # Each takes a waypoint's names and positions, the fields of a JointPositions. Joints this
    # controller does not own are ignored, so a message may name a subset. The inputs are commands,
    # not streams: every message is applied, in order, because each merges into the held target and
    # a later one in the same step must not replace an earlier one that named other joints.

    @endpoint.out(rate=50.0)
    def joint_states(self) -> JointState:
        """Positions, velocities and efforts of the joints this controller reports."""
        names, pos, vel, eff = self.read_state()
        return JointState(names, np.array(pos), np.array(vel), np.array(eff))

    # Trajectory execution as an action: a bridge with a handler for the type (see
    # roqsim_ros_bridge.actions) runs the goal and feeds each waypoint through this command.
    @endpoint.command(
        JointPositions,
        ros2=lambda self: {
            "action": "control_msgs.action.FollowJointTrajectory",
            "name": f"{self._controller}/follow_joint_trajectory",
            # Which ArmHandle the handler reads back, so its feedback reports measured `actual`
            # against commanded `desired` -- as a real JTC does. Its default (arm:<owner>) cannot
            # tell two arms on one entity apart.
            "arm_state_key": self._arm_key,
            # What "reached the goal" means for THIS arm. The handler grades the result on the
            # joints rather than on the trajectory's clock; a goal may tighten these per joint, and
            # a heavier or softer arm loosens them here.
            "goal_tolerance": self.config.get("goal_tolerance", 0.5),
            "goal_time_tolerance": float(self.config.get("goal_time_tolerance", 1.0)),
        },
    )
    def follow_joint_trajectory(self, names: list[str], positions: NDArray[np.float64]) -> None:
        """One waypoint of a trajectory goal, held from now.

        Args:
            names: the joints the waypoint names
            positions: one per name
        """
        self.set_targets(names, positions)

    # Controller state, the third interface a ros2_control JointTrajectoryController exposes
    # alongside the action and the command topic (rqt_joint_trajectory_controller and most
    # diagnostics read it).
    @endpoint.out(
        rate=50.0,
        ros2=lambda self: {
            "type": "control_msgs.msg.JointTrajectoryControllerState",
            "topic": f"{self._controller}/controller_state",
        },
    )
    def controller_state(self) -> ControllerState:
        """The held target against the measured joints, for the joints this controller commands.

        Only those, not everything ``joint_states`` reports: the state describes the control loop,
        and a joint with no actuator (a mimicked finger) has no setpoint to state.
        """
        m, d = self._ctx.model, self._ctx.data
        desired = [self._target[n] for n in self._ctrl_names]
        actual = [float(d.qpos[m.jnt_qposadr[jid]]) for _, jid in self._joint_acts]
        vel = [float(d.qvel[m.jnt_dofadr[jid]]) for _, jid in self._joint_acts]
        return ControllerState(
            list(self._ctrl_names),
            TrajectoryPoint(desired),
            TrajectoryPoint(actual, vel),
            TrajectoryPoint([d - a for d, a in zip(desired, actual, strict=True)]),
        )

    # Streaming joint-position command as a *topic* input, mirroring ros2_control's
    # JointTrajectoryController ``<controller>/joint_trajectory`` topic (the real UR driver exposes
    # both that topic and the action). This is the high-rate path moveit_servo drives: each
    # single-point trajectory sets the held target, so a stream of positions servos the arm.
    @endpoint.command(
        JointPositions,
        when="stream_commands",
        ros2=lambda self: {"topic": f"{self._controller}/joint_trajectory"},
    )
    def joint_command(self, names: list[str], positions: NDArray[np.float64]) -> None:
        """A joint-position target, held from now.

        Args:
            names: the joints it names
            positions: one per name
        """
        self.set_targets(names, positions)

    # Joint-VELOCITY command input, for reactive controllers that resolve to joint rates rather
    # than poses (see "Velocity commands" in the module docstring). Integrated in pre_step.
    @endpoint.command(
        JointVelocities,
        when="velocity_commands",
        ros2=lambda self: {"topic": f"{self._controller}/joint_velocity"},
    )
    def joint_velocity(self, names: list[str], velocities: NDArray[np.float64]) -> None:
        """Joint velocities, integrated into the held target each step.

        Args:
            names: the joints it names
            velocities: one per name, rad/s or m/s
        """
        self.set_velocities(names, velocities)

    @endpoint.command(
        when="has_gripper",
        ros2=lambda self: {
            "action": "control_msgs.action.GripperCommand",
            "name": f"{self._gripper_controller}/gripper_cmd",
            # Tell the bridge's handler which reader to watch; its default is gripper:<owner>,
            # which cannot distinguish two grippers on one entity.
            "state_key": self._gripper_key,
            # Where the handler finds the gripper's effort clamp (GripperCommand's max_effort), a
            # second command beside the position.
            "effort_key": self._effort_key,
        },
    )
    def gripper_cmd(self, position: float) -> None:
        """The gripper's commanded position.

        Args:
            position: the gripper joint's target (rad, or m for a slide), between gripper_open and
                gripper_close
        """
        self.set_gripper(position)

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
        #
        # ``qfrc_gravcomp`` is added because a compensated arm splits one physical quantity across
        # two fields. A real drive that holds its own weight delivers that torque and its sensor
        # reads it; MuJoCo's ``body_gravcomp`` supplies the same torque outside the actuator, so
        # ``qfrc_actuator`` alone reports a motor doing nothing while the arm hangs off it. The sum
        # is what the joint carries, and it matches an uncompensated arm's reading in the same pose.
        m, d = self._ctx.model, self._ctx.data
        pos = [float(d.qpos[m.jnt_qposadr[jid]]) for jid in self._report_jids]
        vel = [float(d.qvel[m.jnt_dofadr[jid]]) for jid in self._report_jids]
        eff = [
            float(d.qfrc_actuator[m.jnt_dofadr[jid]] + d.qfrc_gravcomp[m.jnt_dofadr[jid]])
            for jid in self._report_jids
        ]
        return (self._report_names, pos, vel, eff)

    def set_gripper(self, position) -> None:
        """Map a commanded gripper position (gripper_joint position) onto the gripper actuator ctrl."""
        span = self._grip_q_at_ctrl_hi - self._grip_q_at_ctrl_lo
        frac = 0.0 if span == 0 else (float(position) - self._grip_q_at_ctrl_lo) / span
        frac = max(0.0, min(1.0, frac))
        self._gripper_ctrl_target = self._grip_ctrl_lo + frac * (
            self._grip_ctrl_hi - self._grip_ctrl_lo
        )

    def set_gripper_max_effort(self, effort: float) -> None:
        """Clamp the gripper joint's effort at GripperCommand's ``max_effort``. Physics thread.

        The unit is the joint's own. ``effort <= 0``, and ``effort`` at or above the model's limit,
        leave the model's own force range: a drive saturates at what it can give. A gripper that cannot
        take a clamp (``limit == 0``) keeps its range and executes the position alone, as a
        position-interface controller does, and says so once.
        """
        grip = self._aux_acts[0]
        if self._grip_limit <= 0.0 and effort > 0.0 and not self._grip_cap_warned:
            self._grip_cap_warned = True
            self._ctx.logger.warning(
                "arm_controller[%s]: the gripper actuator has no force limit, so max_effort cannot "
                "be applied; the position runs with the model's own force range",
                self.arm,
            )
        if effort <= 0.0 or effort >= self._grip_limit:
            self._ctx.model.actuator_forcerange[grip] = self._grip_forcerange
            return
        limit = effort / self._grip_gain
        self._ctx.model.actuator_forcerange[grip] = (-limit, limit)

    def read_gripper_effort(self) -> float:
        """The gripper joint's effort now, as ``/joint_states`` reports it."""
        d = self._ctx.data
        return float(d.qfrc_actuator[self._grip_dofadr] + d.qfrc_gravcomp[self._grip_dofadr])

    def read_gripper_state(self):
        # Computed on demand (see read_state).
        d = self._ctx.data
        return (float(d.qpos[self._grip_qposadr]), float(d.qvel[self._grip_dofadr]))

    def _write_ctrl(self, d) -> None:
        for name, (aid, _) in zip(self._ctrl_names, self._joint_acts, strict=True):
            d.ctrl[aid] = self._target[name]
        for aid in self._aux_acts:
            d.ctrl[aid] = self._gripper_ctrl_target

    def on_reset(self, ctx: SimContext) -> None:
        # A trial's controller switches end with it.
        self._active = self._configured_active
        if self._registered is not None:
            state = ACTIVE if self._active else INACTIVE
            registry_for(ctx).restore(self._registered, state, ctx.sim_time)
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
        if self._grip_forcerange is not None:
            # A trial's grip force is that trial's: the next one starts from the model's own.
            ctx.model.actuator_forcerange[self._aux_acts[0]] = self._grip_forcerange
        if "test_target" in self.config:
            for name, val in zip(self._ctrl_names, self.config["test_target"], strict=False):
                self._target[name] = float(val)
        # The commands that hold this pose, written now rather than at the first step: a reset zeroes
        # every actuator command, so until they are written the physics state is the arm pulled
        # toward zero, and anything read before the first step -- a wrench, a tare -- reads that
        # transient. In manual mode this is also what opens the sliders at the pose; pre_step then
        # leaves ctrl alone for the user to drag.
        self._write_ctrl(ctx.data)

    def set_command_source(self, update, owner: str = "") -> None:
        """Name the plugin that computes this arm's targets, so its work can be PULLED.

        One source per arm: two plugins computing targets for the same joints would each overwrite
        the other's within a step, and which one won would be decided by their order in the world
        file. Refused by naming both, rather than silently letting the later one win.
        """
        if self._command_source is not None and self._command_source_owner != owner:
            raise RuntimeError(
                f"arm_controller[{self.name}]: {owner!r} wants to command arm "
                f"{self.arm!r}, but {self._command_source_owner!r} already does. An arm takes its "
                f"targets from ONE controller; deactivate one, or give them separate `joints:`."
            )
        self._command_source = update
        self._command_source_owner = owner
        # A world that declares a Cartesian controller has declared which controller drives the
        # arm. Leaving the trajectory role active as well would come up in a state real
        # ros2_control refuses outright -- two active controllers claiming the same command
        # interfaces -- so it releases them here and a scenario switches back when it wants them.
        self._active = self._configured_active = False
        if self._registered is not None:
            self._registered.state = INACTIVE

    def ensure_updated(self, ctx: SimContext) -> None:
        """Run the command source for this step, once, whoever asks first.

        Pulled rather than ordered. A Cartesian controller can only be DECLARED after the arm --
        its `configure` needs the handle this plugin publishes -- so in `pre_step` order the arm
        wrote `ctrl` first and the controller computed the next targets immediately after, landing
        them a step late, every tick. Stamping the work with the step it ran for makes the order
        irrelevant instead of merely correct, which is the same reason the avoidance solve is
        stamped rather than placed.
        """
        step = round(ctx.sim_time / ctx.dt) if ctx.dt else 0
        if self._command_source is None or step == self._commanded_step:
            return
        self._commanded_step = step
        self._command_source(ctx)

    def is_active(self) -> bool:
        return self._active

    def set_active(self, active: bool) -> None:
        """Take or release the arm's TRAJECTORY role -- not its role as the joint command writer.

        Two things live in this plugin and only one of them is a ros2_control controller. Writing
        `ctrl` from the held target is the hardware: it never stops, or the joints would fall.
        Executing trajectories is the controller, and that is what activates and deactivates -- an
        inactive one rejects goals and integrates no velocity, and the joints stay where they are
        while another controller commands them through `set_targets`.

        """
        self._active = bool(active)

    def pre_step(self, ctx: SimContext) -> None:
        if not ctx.manual_control:
            # The command source is a controller in its own right and gates itself, so the pull is
            # not conditioned on the trajectory role. Velocity integration IS that role's, and
            # stops with it.
            self.ensure_updated(ctx)
            if self._active:
                self._integrate_velocity(ctx.data)
            self._write_ctrl(ctx.data)

    # joint / gripper state are computed on demand in read_state / read_gripper_state (the bridge reads
    # them at endpoint rate), so there is no post_step readout here -- nothing runs every step.
