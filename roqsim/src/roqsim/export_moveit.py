"""Generate a complete MoveIt 2 configuration for an arm, from the world the simulator loads.

Why all six files and not two
=============================
``roqsim export urdf`` and ``roqsim export srdf`` produce the two files a human would call the robot
description. ``move_group`` needs four more -- ``kinematics``, ``joint_limits``,
``moveit_controllers`` and ``ompl_planning`` -- and those were left to whoever was standing up a
cell. Three separate copies of that code existed, byte-for-byte identical in the parts that matter,
and most of what they wrote was not theirs to choose:

* ``moveit_controllers.yaml`` maps MoveIt's controller names onto **the action names this substrate's
  bridge actually serves** and onto the joint list ``arm_controller`` publishes. Every copy carried a
  comment saying so and then hardcoded the values anyway, which is how they drifted.
* ``joint_limits.yaml`` states the kinematic limits that turn a geometric path into a timed one. The
  positions and efforts are already in the URDF; what is added here is an *execution* property of the
  bridge and the position servo behind it.
* ``kinematics.yaml`` is a solver over the chain the SRDF already names.
* ``ompl_planning.yaml`` is the one that genuinely belongs to the experiment -- a planner comparison's
  whole factor can live in it -- so it is emitted as a starting point and meant to be overridden.

So this reads the answers off the compiled world instead of asking for them. The joint names, their
order, the home posture, the controller and action names, the gripper's units and which subtree is a
closed loop are all facts about the model and the plugins configured on it. A flag that restates a
value the model already carries is exactly the drift those three copies suffered, so each flag here
only *overrides* a derived value.

What "derived" means, file by file
==================================
* joints and their order -- the ``ArmHandle`` ``arm_controller`` published, i.e. the same list and the
  same order that reach ``/joint_states``.
* the home posture -- ``data.qpos`` after ``setup()``. Not the world's ``home:`` key: the controller
  applies that (and any ``rest`` stance) itself, so the simulator's actual starting posture is the
  one thing ``CheckStartStateBounds`` will be handed, and it is what must be in the SRDF.
* controller and action names -- the ``Endpoint`` objects ``arm_controller`` declared on
  ``ctx.interface``. That registry *is* what the bridge wires, so a name taken from it cannot
  disagree with the action the trajectory is executed against.
* the collapse root -- the lowest common ancestor of every body an ``equality`` constraint touches.
  A closed linkage is what URDF cannot express, and MuJoCo says exactly where one is.
* ``start_state_max_bounds_error`` -- present only if the arm has a CONTINUOUS joint. MoveIt maps such
  a joint onto [-pi, pi] and ``CheckStartStateBounds`` then refuses to plan from a start state that
  has drifted a hair outside; the symptom is the next phase failing instantly with
  START_STATE_INVALID (-26) after a phase that succeeded, at a different phase each run. A
  range-limited arm has no such problem and gets no such setting.

Several arms in one configuration
=================================
``--arm left,right`` describes both arms as ONE robot: one URDF with both chains under a common root,
one group per arm, and a group spanning all of them. That last group is the point of it -- a plan for
it is a single trajectory through both arms' joint space, so each arm's motion is checked against
where the other IS at that instant rather than against where it was before it started, and the two
can move at once. Planning the arms one after the other cannot express that.

Three things follow from a shared description, and each is refused rather than repaired:

* **Names stay apart.** One URDF is one flat namespace, so each arm's links and joints keep its MJCF
  prefix. Joint names are the controller's, not ours to rename -- they are what reaches
  ``/joint_states`` and what a trajectory point carries -- so an arm whose controller reports
  unprefixed names is refused, naming ``arm_controller``'s ``joint_prefix``.
* **The combined group gets no IK solver.** A chain solver has no chain to solve; see
  ``kinematics_yaml``.
* **Each arm keeps its own controller entry**, with the action its own endpoints declared, so a goal
  for one arm cannot be executed against the other.

Usage::

    roqsim export moveit --world w.yaml --prefix ur5e_ --out cfg/ --tip-site pinch --check
    roqsim export moveit --world cell.yaml --arm left,right --out cfg/ --tip-site pinch
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import yaml

from . import logging_setup
from .export_srdf import ARM_GROUP, ArmGroup, build_srdf, links_from_urdf
from .export_urdf import UrdfExporter, _first_body, combine_urdfs, round_trip_error

logger = logging.getLogger(__name__)

#: MoveIt's own group names. Not configurable: ``kinematics.yaml``, the SRDF's ``<group>`` and
#: ``ompl_planning.yaml`` must all use the same string, and a knob whose only valid setting is "the
#: same everywhere" is a way to get it wrong. ``ARM_GROUP`` (imported from ``export_srdf``, which
#: emits the group) is the name a ONE-arm description uses; a description holding several arms names
#: each group after the entity it was spawned as, because one string cannot serve two groups.
GRIPPER_GROUP = "gripper"

#: Name of the group spanning EVERY arm in a multi-arm description. Nothing in the model implies it,
#: so it is the one group name that is a choice rather than a reading.
COMBINED_GROUP = "both_arms"

#: Per-joint kinematic limits, in SI. Deliberately far below any vendor maximum, and this is an
#: execution property of the substrate rather than a fact about the robot: a trajectory is followed by
#: a position servo whose own dynamics shape the achieved profile, so a plan timed for the datasheet's
#: velocities is one the arm lags behind -- and a grasped object leaves the jaws when it does. Override
#: per cell if its arm is lightly loaded.
MAX_VELOCITY = 1.0
MAX_ACCELERATION = 2.0
SCALING = 0.15

#: Gripper effort reported to a ``GripperCommand`` goal. Not read by the substrate -- the commanded
#: position is mapped onto the actuator's ctrlrange -- but MoveIt requires the field.
GRIPPER_EFFORT = 50.0

_GENERATED = (
    "# GENERATED by `roqsim export moveit` from the world the simulator loads. Do not edit.\n#\n"
)


@dataclass
class ArmFacts:
    """Everything the six files need, as read off a compiled world."""

    arm: str
    prefix: str
    namespace: str
    joints: list[str]
    home: dict[str, float]
    controller: str
    trajectory_action: str
    collapse: tuple[str, ...]
    continuous_joints: list[str] = field(default_factory=list)
    gripper_controller: str = ""
    gripper_action: str = ""
    gripper_joint: str = ""
    gripper_open: float = 0.0
    gripper_close: float = 0.0
    #: The SRDF group this arm's chain is in. ``ARM_GROUP`` for a description holding one arm; the
    #: entity's own name where several share a description.
    group: str = ARM_GROUP
    #: Whether the emitted link and joint names keep the arm's MJCF prefix. True only in a
    #: description holding several robots, where one flat namespace has to hold both.
    keep_prefix: bool = False

    @property
    def has_gripper(self) -> bool:
        return bool(self.gripper_joint)

    @property
    def gripper_group(self) -> str:
        return GRIPPER_GROUP if self.group == ARM_GROUP else f"{self.group}_gripper"

    @property
    def urdf_gripper_joint(self) -> str:
        """The gripper joint as the DESCRIPTION spells it.

        Unlike the arm joints -- which come from the controller and are therefore already in the
        spelling that reaches ``/joint_states`` -- this one is read from the plugin's config, which
        names it model-locally. Where prefixes are kept it needs the arm's own.
        """
        if not self.gripper_joint or not self.keep_prefix:
            return self.gripper_joint
        return self.prefix + self.gripper_joint

    @property
    def controller_key(self) -> str:
        """What ``moveit_controllers.yaml`` calls this arm's controller.

        MoveIt joins the entry's name and ``action_ns`` into the action it sends goals to, so an arm
        whose endpoints are scoped by a namespace only reaches its own action when that namespace is
        part of the name. In a one-arm description the name is left as the controller's own, which is
        what a bringup that remaps the whole node expects.
        """
        if self.group != ARM_GROUP and self.namespace:
            return f"{self.namespace}/{self.controller}"
        return self.controller

    @property
    def gripper_controller_key(self) -> str:
        if self.group != ARM_GROUP and self.namespace:
            return f"{self.namespace}/{self.gripper_controller}"
        return self.gripper_controller


@dataclass
class _Exported:
    """What ``_run`` reports about a URDF it wrote, whether from one exporter or several."""

    mesh_files: dict
    dropped_dofs: list
    mesh_dir: Path
    strip_prefix: str


def infer_collapse(model: mujoco.MjModel, bodies: set[int]) -> tuple[str, ...]:
    """Bodies whose subtree must be lumped into one fixed link, read off the model's equalities.

    URDF is a tree, so a closed kinematic loop -- a Robotiq 2F-85's four-bar, a PAL PRO jaw -- has no
    URDF spelling at all. MuJoCo closes such a loop with an ``equality`` constraint, which makes the
    question answerable rather than a matter of the caller knowing their gripper: the loop lives under
    the lowest common ancestor of every body the equalities touch, and collapsing there removes it.

    Getting this wrong is quiet. The constraint is simply not exported, so the URDF keeps the loop's
    branches as free revolute DOFs that no controller publishes, and move_group then never assembles a
    complete robot state -- "The complete state of the robot is not yet known. Missing
    <joint>", then "Found empty JointState message", then a planner that cannot sample a valid state.

    Returns model-local (prefix-stripped is the CALLER's job) body names, or ``()`` for a model with no
    closed loop -- a suction cup, a bare flange, an arm with no tool at all.
    """
    involved: set[int] = set()
    for i in range(model.neq):
        eq_type = int(model.eq_type[i])
        for obj in (int(model.eq_obj1id[i]), int(model.eq_obj2id[i])):
            if eq_type == mujoco.mjtEq.mjEQ_JOINT:
                involved.add(int(model.jnt_bodyid[obj]))
            elif eq_type in (mujoco.mjtEq.mjEQ_CONNECT, mujoco.mjtEq.mjEQ_WELD):
                involved.add(obj)
    # Only loops INSIDE the robot being exported. A world may weld a prop to a shelf or couple a
    # conveyor's rollers; collapsing the arm at their common ancestor would lump the whole robot into
    # one link, which is worse than not collapsing at all.
    involved &= bodies
    if not involved:
        return ()

    common = _common_ancestor(model, involved)
    if common <= 0:
        return ()
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, common) or ""
    return (name,) if name else ()


def _common_ancestor(model: mujoco.MjModel, bodies: set[int]) -> int:
    """Lowest body that is an ancestor of (or is) every one of ``bodies``; -1 if there is none."""
    if not bodies:
        return -1

    def ancestry(body: int) -> list[int]:
        out = []
        while body > 0:
            out.append(body)
            body = int(model.body_parentid[body])
        return list(reversed(out))

    common = -1
    for step in zip(*(ancestry(b) for b in bodies), strict=False):
        if len(set(step)) != 1:
            break
        common = step[0]
    return common


def _subtree(model: mujoco.MjModel, root: int) -> set[int]:
    """``root`` and every body below it.

    Asked of the arm's own root body rather than of a name prefix, because a prefix is optional: a
    single-arm cell usually has none, and "every body" would then take a prop welded to a shelf or a
    conveyor's coupled rollers for part of the robot.
    """
    out = {root}
    for body in range(1, model.nbody):
        cur = body
        while cur > 0:
            if cur == root:
                out.add(body)
                break
            cur = int(model.body_parentid[cur])
    return out


def _endpoint(endpoints: list, arm: str, name: str):
    return next((e for e in endpoints if e.owner == arm and e.name == name), None)


def arm_facts(engine, arm: str | None = None) -> ArmFacts:
    """Read one arm's configuration off a set-up ``Engine``.

    ``arm`` names the entity (``spawn_arm``'s ``name:``); with one arm in the world it is optional.
    """
    ctx = engine.ctx
    endpoints = ctx.interface.all()
    arms = sorted({e.owner for e in endpoints if e.name == "follow_joint_trajectory"})
    if not arms:
        raise ValueError(
            "no arm in this world declares a follow_joint_trajectory endpoint. MoveIt executes "
            "trajectories against that action, so there is nothing to generate a configuration for -- "
            "check that the world spawns an arm and that its arm_controller is not disabled."
        )
    if arm is None:
        if len(arms) > 1:
            raise ValueError(
                f"this world has {len(arms)} arms ({', '.join(arms)}); name one with --arm, or "
                "several (--arm a,b) to describe them as one robot with a group spanning both. "
                "Which it is changes every file, so it is not guessed."
            )
        arm = arms[0]
    elif arm not in arms:
        raise ValueError(f"--arm {arm!r} is not one of this world's arms: {', '.join(arms)}")

    traj = _endpoint(endpoints, arm, "follow_joint_trajectory")
    ros = traj.backend.get("ros2", {})
    trajectory_action = str(ros.get("name", ""))
    controller = (
        trajectory_action.rsplit("/", 1)[0] if "/" in trajectory_action else "arm_controller"
    )

    # The joint list comes from the handle the controller published, not from the world's `joints:`
    # key: those are the names, in the order, that reach /joint_states, and robot_state_publisher
    # matches joint states to URDF joints BY NAME. Anything else here can be right about the robot and
    # still leave MoveIt planning from a pose the arm is not in.
    handle = ctx.blackboard.get(str(ros.get("arm_state_key") or f"arm:{arm}"))
    if handle is None or not getattr(handle, "joint_names", None):
        raise ValueError(
            f"arm {arm!r} published no ArmHandle with joint names; cannot tell which joints MoveIt "
            "should command."
        )
    joints = list(handle.joint_names)

    prefix = ""
    entity = ctx.entities.get(arm)
    if entity is not None:
        prefix = entity.meta.get("prefix", "") or ""

    model, data = ctx.model, ctx.data
    home: dict[str, float] = {}
    continuous: list[str] = []
    for name in joints:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, prefix + name)
        if jid < 0:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise ValueError(f"joint {name!r} is in the arm's handle but not in the model")
        home[name] = float(data.qpos[int(model.jnt_qposadr[jid])])
        # MuJoCo's spelling of "unlimited": no range, or a range that does not increase. The URDF
        # exporter emits such a joint without limits, which is what makes MoveIt treat it as
        # continuous.
        lo, hi = (float(v) for v in model.jnt_range[jid])
        if not bool(model.jnt_limited[jid]) or hi <= lo:
            continuous.append(name)

    facts = ArmFacts(
        arm=arm,
        prefix=prefix,
        namespace=traj.namespace or "",
        joints=joints,
        home=home,
        controller=controller,
        trajectory_action=trajectory_action,
        collapse=(),
        continuous_joints=continuous,
    )

    grip = _endpoint(endpoints, arm, "gripper_cmd")
    if grip is not None:
        gros = grip.backend.get("ros2", {})
        facts.gripper_action = str(gros.get("name", ""))
        facts.gripper_controller = (
            facts.gripper_action.rsplit("/", 1)[0]
            if "/" in facts.gripper_action
            else "gripper_controller"
        )
        # The gripper's joint and its open/closed angles are the CONTROLLER's, merged from the gripper
        # model's manifest. Read from the plugin rather than from the world YAML so a gripper that
        # states them only in its manifest -- which is every gripper the substrate ships -- is handled.
        plugin = next(
            (
                p
                for p in engine.plugins
                if type(p).__name__ == "ArmControllerPlugin"
                and getattr(p, "arm", None) == arm
                and p.config.get("gripper_controller_name", "gripper_controller")
                == facts.gripper_controller
            ),
            None,
        )
        if plugin is not None:
            facts.gripper_joint = str(plugin.config.get("gripper_joint", "") or "")
            facts.gripper_open = float(getattr(plugin, "_grip_open", 0.0))
            facts.gripper_close = float(getattr(plugin, "_grip_close", 0.0))
        if not facts.gripper_joint:
            logger.warning(
                "arm %r serves a GripperCommand action but names no gripper joint, so no gripper "
                "group is emitted. MoveIt will have no way to open or close the hand.",
                arm,
            )

    # `spawn_arm` records the arm's root body on its entity, which is what bounds "this robot" for
    # the collapse search below. Without an entity (a hand-built world) fall back to the common
    # ancestor of the arm's own joints, which is the same body by another route.
    root_body = -1
    if entity is not None and getattr(entity, "body", None):
        root_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, entity.body)
    if root_body < 0:
        root_body = _common_ancestor(
            model,
            {
                int(model.jnt_bodyid[j])
                for j in (
                    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, prefix + n) for n in joints
                )
                if j >= 0
            },
        )
    facts.collapse = tuple(
        n[len(prefix) :] if prefix and n.startswith(prefix) else n
        for n in infer_collapse(model, _subtree(model, root_body) if root_body > 0 else set())
    )
    return facts


def all_arm_facts(engine, arms: list[str] | None = None) -> list[ArmFacts]:
    """Read every arm a configuration is being generated for, in the order named.

    One arm is the ordinary case and reads exactly like :func:`arm_facts`. Several arms are a
    description they SHARE, which one flat namespace has to hold, so two things are required of them
    and refused rather than repaired here:

    * a distinct, non-empty MJCF prefix each -- what keeps their link names apart;
    * joint names that already carry that prefix, i.e. ``arm_controller``'s ``joint_prefix``. Those
      names are the ones reaching ``/joint_states`` and the ones MoveIt puts in a trajectory point,
      so the description cannot rename them: two arms of one model would both command
      ``shoulder_pan_joint`` and each would answer for the other's trajectory.
    """
    if not arms:
        return [arm_facts(engine)]
    if len(arms) == 1:
        return [arm_facts(engine, arms[0])]

    facts = []
    for arm in arms:
        one = arm_facts(engine, arm)
        one.group = one.arm
        one.keep_prefix = True
        if not one.prefix:
            raise ValueError(
                f"arm {arm!r} has no MJCF prefix, so its links cannot be told from the other arms' "
                "in one description. Give each arm in a multi-arm cell a distinct `prefix:`."
            )
        outside = [j for j in one.joints if not j.startswith(one.prefix)]
        if outside:
            raise ValueError(
                f"arm {arm!r} publishes joints that do not carry its prefix {one.prefix!r} "
                f"({', '.join(outside)}). A description holding several arms is one flat namespace, "
                "and these are the names that reach /joint_states and a trajectory point, so they "
                f"cannot be renamed here. Set `joint_prefix: {one.prefix!r}` on this arm's "
                "arm_controller."
            )
        facts.append(one)

    for label, seen in (("prefix", {}), ("joint", {})):
        for one in facts:
            for value in [one.prefix] if label == "prefix" else one.joints:
                if value in seen:
                    raise ValueError(
                        f"arms {seen[value]!r} and {one.arm!r} share the {label} {value!r}. One "
                        "description cannot hold two of it, so MoveIt would command one arm and "
                        "read the other's state back."
                    )
                seen[value] = one.arm
    return facts


def _facts_list(facts) -> list[ArmFacts]:
    """Accept either one arm's facts or a whole cell's, so every generator below takes both."""
    return [facts] if isinstance(facts, ArmFacts) else list(facts)


# -- the four YAMLs ------------------------------------------------------------------------------


def kinematics_yaml(facts, combined_group: str = "") -> str:
    arms = _facts_list(facts)
    body = {
        one.group: {
            "kinematics_solver": "kdl_kinematics_plugin/KDLKinematicsPlugin",
            "kinematics_solver_search_resolution": 0.005,
            "kinematics_solver_timeout": 0.05,
            "kinematics_solver_attempts": 3,
        }
        for one in arms
    }
    combined = (
        "#\n"
        f"# {combined_group} gets NO solver, deliberately. KDL solves a single serial chain, and that\n"
        "# group is several -- there is no pose for it to solve for, only a pair of poses whose IK is\n"
        "# each arm's own. What the group is for is planning in the JOINT space of both arms at once,\n"
        "# so that one arm's motion is checked against where the other actually is at that instant\n"
        "# rather than against where it was before it started. Give each arm a pose through its own\n"
        "# group, or plan for this one in joint space.\n"
        if combined_group
        else ""
    )
    return (
        _GENERATED
        + "# KDL is MoveIt's default and solves a single serial chain, which is what an arm group is.\n"
        + "# The gripper group gets no solver: it is one joint driven by a GripperCommand controller,\n"
        + "# never by IK.\n"
        + combined
        + yaml.safe_dump(body, sort_keys=False)
    )


def joint_limits_yaml(facts, max_velocity: float, max_acceleration: float) -> str:
    limits = {
        j: {
            "has_velocity_limits": True,
            "max_velocity": max_velocity,
            "has_acceleration_limits": True,
            "max_acceleration": max_acceleration,
        }
        for one in _facts_list(facts)
        for j in one.joints
    }
    return (
        _GENERATED
        + "# The URDF already carries each joint's POSITION and EFFORT limits, exported from the MJCF.\n"
        + "# These add the KINEMATIC limits MoveIt needs to turn a geometric path into a timed one.\n"
        + "#\n"
        + "# They are far below any vendor maximum, and that is an execution property of this substrate\n"
        + "# rather than a fact about the robot: a trajectory is followed by a position servo, and the\n"
        + "# achieved profile is shaped by the servo's own dynamics as well as by the command. A plan\n"
        + "# timed for the datasheet is one the arm lags behind, and a grasped object leaves the jaws\n"
        + "# when it does. Raise them for a lightly loaded arm.\n"
        + f"default_velocity_scaling_factor: {SCALING}\n"
        + f"default_acceleration_scaling_factor: {SCALING}\n"
        + yaml.safe_dump({"joint_limits": limits}, sort_keys=False)
    )


def moveit_controllers_yaml(facts, gripper_effort: float) -> str:
    arms = _facts_list(facts)
    names: list[str] = []
    manager: dict = {"controller_names": names}
    served_lines = ""
    for one in arms:
        if one.controller_key in manager:
            raise ValueError(
                f"two arms would both be called {one.controller_key!r} in moveit_controllers.yaml. "
                "MoveIt maps a trajectory onto a controller by that name, so one arm's goals would "
                "go to the other. Give each arm's arm_controller its own `controller_name` or its "
                "own `namespace`."
            )
        names.append(one.controller_key)
        manager[one.controller_key] = {
            "type": "FollowJointTrajectory",
            "action_ns": "follow_joint_trajectory",
            "default": True,
            "joints": one.joints,
        }
        served = f"{one.namespace}/" if one.namespace else ""
        served_lines += f"#   /{served}{one.trajectory_action}\n"
        if one.has_gripper:
            names.append(one.gripper_controller_key)
            manager[one.gripper_controller_key] = {
                "type": "GripperCommand",
                "action_ns": "gripper_cmd",
                "default": True,
                "joints": [one.urdf_gripper_joint],
                "max_effort": gripper_effort,
            }
            served_lines += f"#   /{served}{one.gripper_action}\n"
    body: dict = {
        "moveit_controller_manager": (
            "moveit_simple_controller_manager/MoveItSimpleControllerManager"
        ),
        "moveit_simple_controller_manager": manager,
    }
    return (
        _GENERATED
        + "# Every name below was read from the endpoints the arm's controller declared, so it names an\n"
        + "# action this substrate's bridge actually serves:\n"
        + served_lines
        + "#\n"
        + "# moveit_simple_controller_manager is the right manager: the bridge exposes those actions\n"
        + "# DIRECTLY, with no ros2_control controller_manager behind them, so\n"
        + "# moveit_ros_control_interface would have nothing to talk to.\n"
        + yaml.safe_dump(body, sort_keys=False)
    )


def ompl_planning_yaml(facts, combined_group: str = "") -> str:
    arms = _facts_list(facts)
    body: dict = {
        "planning_plugins": ["ompl_interface/OMPLPlanner"],
        "request_adapters": [
            "default_planning_request_adapters/ResolveConstraintFrames",
            "default_planning_request_adapters/ValidateWorkspaceBounds",
            "default_planning_request_adapters/CheckStartStateBounds",
            "default_planning_request_adapters/CheckStartStateCollision",
        ],
        "response_adapters": [
            "default_planning_response_adapters/AddTimeOptimalParameterization",
            "default_planning_response_adapters/ValidateSolution",
            "default_planning_response_adapters/DisplayMotionPath",
        ],
    }
    for one in arms:
        body[one.group] = {
            "planner_configs": ["RRTConnectkConfigDefault"],
            # Must name joints that ARE in the group: naming one outside it makes move_group refuse
            # every request with "joint ... is not known to the group", which reads like a planner
            # problem rather than a configuration typo.
            "projection_evaluator": f"joints({one.joints[0]},{one.joints[1]})",
            "longest_valid_segment_fraction": 0.002,
        }
    if combined_group:
        # One joint from each of the first two arms, so the projection separates states by what the
        # ARMS are doing rather than by one arm's shoulder alone -- and both are in the group, which
        # is the property move_group checks.
        body[combined_group] = {
            "planner_configs": ["RRTConnectkConfigDefault"],
            "projection_evaluator": f"joints({arms[0].joints[0]},{arms[1].joints[0]})",
            "longest_valid_segment_fraction": 0.002,
        }
    body["planner_configs"] = {
        "RRTConnectkConfigDefault": {"type": "geometric::RRTConnect", "range": 0.0}
    }
    note = ""
    continuous = sorted({j for one in arms for j in one.continuous_joints})
    if continuous:
        # Only for an arm that has one. On a range-limited arm the setting is noise, and a reader who
        # sees it everywhere learns nothing from it being there.
        body["start_state_max_bounds_error"] = 0.1
        note = (
            "#\n"
            "# start_state_max_bounds_error is set because this arm has CONTINUOUS joints "
            f"({', '.join(continuous)}).\n"
            "# MoveIt maps such a joint onto [-pi, pi], so after any motion one can sit a hair\n"
            "# outside, and CheckStartStateBounds' default of 0.0 then REFUSES to plan from it. The\n"
            "# symptom is a phase failing instantly with START_STATE_INVALID (-26) right after a phase\n"
            "# that succeeded, at a different phase each run. 0.1 rad wraps that drift and is far less\n"
            "# than a real configuration error.\n"
        )
    return (
        _GENERATED
        + "# THE ONE FILE HERE THAT IS YOURS. Planner choice, its parameters and the validity-checking\n"
        + "# resolution are experiment decisions -- a planner comparison's whole factor can live in\n"
        + "# this file -- so this is a working starting point rather than an answer.\n"
        + "#\n"
        + "# RRTConnect is OMPL's default and suits an arm reaching across an open workspace.\n"
        + "# longest_valid_segment_fraction is tightened from the 0.005 default because a small object\n"
        + "# is small relative to the links: a coarser check can step the gripper through a table edge\n"
        + "# between two states that both validate.\n"
        + note
        + yaml.safe_dump(body, sort_keys=False)
    )


# -- the postcondition ---------------------------------------------------------------------------


def assert_agrees(
    out: Path, urdf: Path, srdf: Path, facts, arm_tip, combined_group: str = ""
) -> str:
    """Check the generated set agrees with itself, and return the planning frame it settled on.

    Every check here catches a configuration that LOADS and then cannot plan, which is the expensive
    failure: move_group logs a complaint once a second and the task hangs rather than failing, so a
    caller waits out its whole timeout to discover it. Owning both files is what makes these checkable at
    all -- ``export srdf`` alone cannot validate ``--arm-tip``, and accepts a tip link that is not in
    the URDF at all.

    ``facts`` and ``arm_tip`` are per arm, so every check runs once per arm: a second arm whose chain
    is short by a joint, or whose home disagrees with the simulator, fails exactly as loudly as the
    first. The root-link check does not multiply -- however many robots a description holds, it is
    still one tree with one root.
    """
    arms = _facts_list(facts)
    tips = [arm_tip] if isinstance(arm_tip, str) else list(arm_tip)
    if len(tips) != len(arms):
        raise ValueError(f"{len(arms)} arms but {len(tips)} tip links; they are read pairwise")
    urdf_root = ET.parse(urdf).getroot()
    srdf_root = ET.parse(srdf).getroot()

    links = {link.get("name") for link in urdf_root.findall("link")}
    joint_by_child = {
        j.find("child").get("link"): j
        for j in urdf_root.findall("joint")
        if j.find("child") is not None
    }
    roots = [name for name in links if name not in joint_by_child]
    if len(roots) != 1:
        raise ValueError(f"the URDF has {len(roots)} root links ({sorted(roots)}); a URDF has one")
    urdf_root_link = roots[0]

    virtual = srdf_root.find("virtual_joint")
    if virtual is None:
        # A welded robot: MoveIt's planning frame IS the root link. Nothing publishes a transform
        # above it, so any other frame would be rejected as unknown.
        planning_frame = urdf_root_link
    else:
        planning_frame = virtual.get("parent_frame")
        if virtual.get("child_link") != urdf_root_link:
            raise ValueError(
                f"the SRDF's virtual joint hangs off {virtual.get('child_link')!r} but the URDF's root "
                f"is {urdf_root_link!r}. srdfdom SKIPS a virtual joint whose child is any other link, "
                "which degrades exactly like declaring none."
            )

    def spanned_joints(chain: ET.Element, group: str) -> list[str]:
        """The ACTUATED joints a chain spans. Fixed joints are skipped deliberately: the chain may
        end at a fixed frame link, and counting that as a DOF would make an over-long chain look
        right while a genuinely missing joint also looked right."""
        spanned, link = [], chain.get("tip_link")
        while link != chain.get("base_link"):
            joint = joint_by_child.get(link)
            if joint is None:
                raise ValueError(
                    f"the {group!r} chain {chain.get('base_link')} -> {chain.get('tip_link')} is "
                    f"broken: nothing in the URDF is the parent of {link!r}"
                )
            if joint.get("type") != "fixed":
                spanned.append(joint.get("name"))
            link = joint.find("parent").get("link")
        return spanned

    controllers = yaml.safe_load((out / "moveit_controllers.yaml").read_text(encoding="utf-8"))
    for one, tip in zip(arms, tips, strict=True):
        group = next((g for g in srdf_root.findall("group") if g.get("name") == one.group), None)
        chain = group.find("chain") if group is not None else None
        if chain is None:
            raise ValueError(f"the SRDF has no {one.group!r} group with a chain")
        if chain.get("tip_link") != tip:
            raise ValueError(
                f"the SRDF's {one.group!r} chain ends at {chain.get('tip_link')!r}, but this export "
                f"asked for {tip!r}."
            )
        if chain.get("tip_link") not in links:
            raise ValueError(
                f"the {one.group!r} chain's tip is {chain.get('tip_link')!r}, which is not a link in "
                "the URDF. MoveIt would load and then never plan. Use --tip-site to end the chain at "
                "a real frame."
            )

        spanned = spanned_joints(chain, one.group)
        if sorted(spanned) != sorted(one.joints):
            raise ValueError(
                f"the {one.group!r} chain spans {sorted(spanned)} but the controller commands "
                f"{sorted(one.joints)}. An arm exposed as fewer DOF than it has plans badly and "
                "silently."
            )

        state = next(
            (
                s
                for s in srdf_root.findall("group_state")
                if s.get("name") == "home" and s.get("group") == one.group
            ),
            None,
        )
        if state is None:
            raise ValueError(f"the SRDF has no 'home' state for {one.group!r}")
        got = {j.get("name"): float(j.get("value")) for j in state.findall("joint")}
        for joint, want in one.home.items():
            if abs(got.get(joint, 1e9) - want) > 1e-5:
                raise ValueError(
                    f"the SRDF's home has {joint}={got.get(joint)} but the simulator starts it at "
                    f"{want}. They must agree: MoveIt's start-state bounds check runs against the "
                    "arm's real posture."
                )

        declared = controllers["moveit_simple_controller_manager"][one.controller_key]["joints"]
        if declared != one.joints:
            raise ValueError(
                f"moveit_controllers.yaml names {declared} for {one.controller_key!r} but that "
                f"controller publishes {one.joints}. move_group cannot execute a trajectory whose "
                "joints it cannot map."
            )

    if combined_group:
        group = next(
            (g for g in srdf_root.findall("group") if g.get("name") == combined_group), None
        )
        if group is None:
            raise ValueError(f"the SRDF has no {combined_group!r} group")
        joints = sorted(
            j for c in group.findall("chain") for j in spanned_joints(c, combined_group)
        )
        want = sorted(j for one in arms for j in one.joints)
        if joints != want:
            raise ValueError(
                f"the {combined_group!r} group spans {joints} but the arms together command {want}. "
                "A group that is missing an arm's joints plans that arm as if it were not there, "
                "which is exactly the collision the group exists to avoid."
            )
        solvers = yaml.safe_load((out / "kinematics.yaml").read_text(encoding="utf-8")) or {}
        if combined_group in solvers:
            raise ValueError(
                f"kinematics.yaml configures a solver for {combined_group!r}. That group is several "
                "chains, and a chain solver cannot solve it -- it would load and then fail every "
                "pose request. Plan for it in joint space."
            )
    return planning_frame


# -- CLI -----------------------------------------------------------------------------------------


def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="roqsim export moveit",
        description="Generate a complete MoveIt 2 configuration for an arm, or for a cell's arms "
        "together, from a roqsim world.",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--world", help="world YAML (compiled via the plugin pipeline)")
    source.add_argument("--mjcf", help="a bare MJCF file")
    parser.add_argument("--out", required=True, help="output DIRECTORY for the six files")
    parser.add_argument(
        "--arm",
        default=None,
        help="entity name of the arm; optional with one arm in the world. Several names, "
        "comma-separated, put those arms in ONE description with a group spanning all of them -- "
        "what a planner needs to move them together instead of one after the other",
    )
    parser.add_argument(
        "--combined-group",
        default=COMBINED_GROUP,
        help="name of the group spanning every arm (several --arm only). Nothing in the model "
        "implies it, so it is the one group name that is chosen rather than read",
    )
    parser.add_argument("--name", default=None, help="robot name (default: the arm's entity name)")
    parser.add_argument(
        "--prefix",
        default=None,
        help="MJCF name prefix selecting the robot. Default: the arm entity's own prefix",
    )
    parser.add_argument("--root-link", default="base_link", help="name for the robot's root link")
    parser.add_argument(
        "--collapse",
        default=None,
        help="comma-separated bodies whose subtree is lumped into one fixed link. Default: READ OFF "
        "THE MODEL -- the lowest common ancestor of every body an `equality` constraint touches, "
        "which is where a closed gripper linkage URDF cannot express lives",
    )
    parser.add_argument(
        "--tip-site",
        default="",
        help="MuJoCo site the arm chain should end at, e.g. `pinch` on a Robotiq 2F-85. Strongly "
        "recommended: without it the chain ends at the flange and every orientation tolerance is "
        "multiplied by the distance to the fingers",
    )
    parser.add_argument("--tip-link", default="tcp", help="name of the link --tip-site emits")
    parser.add_argument(
        "--arm-base",
        default=None,
        help="first link of the arm chain (default: the URDF's root link)",
    )
    parser.add_argument(
        "--arm-tip",
        default=None,
        help="last link of the arm chain (default: --tip-link when --tip-site is given, else the "
        "last link the arm's joints reach)",
    )
    parser.add_argument("--base-joint", default="auto", help="passed to the SRDF export")
    parser.add_argument("--parent-frame", default="odom", help="passed to the SRDF export")
    parser.add_argument("--max-velocity", type=float, default=MAX_VELOCITY)
    parser.add_argument("--max-acceleration", type=float, default=MAX_ACCELERATION)
    parser.add_argument("--gripper-effort", type=float, default=GRIPPER_EFFORT)
    parser.add_argument(
        "--samples", type=int, default=10000, help="configurations sampled for the collision matrix"
    )
    parser.add_argument(
        "--mesh-package",
        default="",
        help="emit meshes as package://<PKG>/meshes/<file> instead of file://<abs path>",
    )
    parser.add_argument(
        "--manifest",
        help="also write {'inputs': [...]} so a caller can tell when this output is stale",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify the URDF's FK against the MJCF and fail if it diverges. The self-consistency "
        "checks between the generated files always run",
    )
    parser.add_argument(
        "--tolerance", type=float, default=1e-6, help="--check: max allowed FK error in metres"
    )
    parser.add_argument("--skip-plugins", default="", help="extra plugin names/refs to drop")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging_setup.configure(verbose=args.verbose)
    log = logging.getLogger("roqsim.export_moveit")

    try:
        return _run(args, log)
    except ValueError as err:
        # Every ValueError raised here describes a configuration that would be WRONG -- a chain that
        # spans the wrong joints, a tip link nothing contains, a home the simulator disagrees with.
        # Those are answers to the operator, not bugs, so they read as a message not a traceback.
        raise SystemExit(f"roqsim export moveit: {err}") from err


def _run(args, log) -> int:
    from .config import drop_transport_plugins, load_config
    from .engine import Engine

    if args.mjcf:
        raise ValueError(
            "--mjcf is not supported: the controller and gripper configuration this reads come from "
            "the PLUGINS a world declares, and a bare MJCF has none. Point --world at the world the "
            "world the simulation runs, which is what makes the export match the simulated scene."
        )

    cfg = load_config(args.world)
    transport, unavailable = drop_transport_plugins(cfg)
    if transport:
        log.info("skipping transport plugins: %s", ", ".join(transport))
    skip = {s.strip() for s in args.skip_plugins.split(",") if s.strip()}
    if skip:
        cfg.plugins = [p for p in cfg.plugins if p.ref not in skip and (p.name or "") not in skip]
    engine = Engine(cfg)
    engine.setup()

    named = [s.strip() for s in (args.arm or "").split(",") if s.strip()]
    facts_list = all_arm_facts(engine, named)
    facts = facts_list[0]
    multi = len(facts_list) > 1
    combined_group = args.combined_group if multi else ""
    if multi:
        for flag, value in (
            ("--prefix", args.prefix),
            ("--collapse", args.collapse),
            ("--arm-base", args.arm_base),
            ("--arm-tip", args.arm_tip),
        ):
            if value is not None:
                raise ValueError(
                    f"{flag} names one robot, and this export covers "
                    f"{', '.join(f.arm for f in facts_list)}. Each arm's value is read off the model."
                )
        log.warning(
            "%d arms in one description: their link and joint names keep each arm's MJCF prefix, "
            "and %r spans all of them so a plan is one trajectory through both arms' joint space.",
            len(facts_list),
            combined_group,
        )
    prefix = args.prefix if args.prefix is not None else facts.prefix
    collapse = (
        tuple(s.strip() for s in args.collapse.split(",") if s.strip())
        if args.collapse is not None
        else facts.collapse
    )
    name = args.name or "_".join(f.arm for f in facts_list)
    log.info(
        "arm %r: %d joints (%s), controller %r%s; collapse %s",
        facts.arm,
        len(facts.joints),
        ", ".join(facts.joints),
        facts.controller,
        f", gripper {facts.gripper_controller!r} on {facts.gripper_joint!r}"
        if facts.has_gripper
        else " (no gripper)",
        ", ".join(collapse) or "(nothing: no closed loop in this robot)",
    )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    urdf = out / f"{name}.urdf"
    srdf = out / f"{name}.srdf"
    model = engine.ctx.model

    def tip_of(tree: ET.ElementTree, one: ArmFacts, base: str, tip_link: str) -> str:
        """Where this arm's chain ends: the asked-for tip, the tip site's link, or the last link."""
        if args.arm_tip:
            return args.arm_tip
        if args.tip_site:
            return tip_link
        tip = _last_link_of_chain(tree.getroot(), base, one.joints)
        log.warning(
            "no --tip-site given, so the %s chain ends at %r -- the outermost rigidly attached "
            "link, not the point that grasps. Every orientation tolerance is then multiplied by the "
            "distance from there to the "
            "fingers; a cell measured 61 mm of lateral error against 12.2 mm of jaw clearance that "
            "way. Pass --tip-site to fix the goal to the frame that does the work.",
            one.group,
            tip,
        )
        return tip

    if multi:
        parts, groups, tips, meshes, dropped = [], [], [], {}, []
        for one in facts_list:
            entity = engine.ctx.entities.get(one.arm)
            root_body = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_BODY, getattr(entity, "body", "") or ""
            )
            if root_body < 0:
                root_body = _first_body(model, one.prefix)
            arm_root_link = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, root_body)
            part = UrdfExporter(
                model,
                prefix=one.prefix,
                name=f"{name}_{one.arm}",
                # Names are kept, so this arm's root link keeps the model's own (prefixed) body
                # name and --root-link names the common root every arm hangs off instead.
                root_link=arm_root_link,
                collapse=tuple(one.prefix + c for c in one.collapse),
                mesh_dir=out / "meshes",
                gripper_joint=one.urdf_gripper_joint,
                mesh_package=args.mesh_package,
                tip_site=args.tip_site,
                tip_link=f"{one.prefix}{args.tip_link}",
                strip="",
                link_strip="",
            )
            part_tree = part.export()
            parts.append((part_tree, root_body))
            meshes |= part.mesh_files
            dropped += part.dropped_dofs
            tip = tip_of(part_tree, one, arm_root_link, f"{one.prefix}{args.tip_link}")
            tips.append(tip)
            groups.append(
                ArmGroup(
                    name=one.group,
                    base_link=arm_root_link,
                    tip_link=tip,
                    gripper_joint=one.urdf_gripper_joint,
                    gripper_open=one.gripper_open,
                    gripper_close=one.gripper_close,
                    home=one.home,
                )
            )
        tree = combine_urdfs(model, parts, name=name, root_link=args.root_link)
        exporter = _Exported(meshes, dropped, out / "meshes", "")
    else:
        exporter = UrdfExporter(
            model,
            prefix=prefix,
            name=name,
            root_link=args.root_link,
            collapse=collapse,
            mesh_dir=out / "meshes",
            gripper_joint=facts.gripper_joint,
            mesh_package=args.mesh_package,
            tip_site=args.tip_site,
            tip_link=args.tip_link,
        )
        tree = exporter.export()
    ET.indent(tree, space="  ")
    tree.write(urdf, encoding="utf-8", xml_declaration=True)
    log.info(
        "wrote %s (%d links, %d meshes)%s",
        urdf,
        len(tree.getroot().findall("link")),
        len(exporter.mesh_files),
        f"; dropped {len(exporter.dropped_dofs)} collapsed DOF(s)" if exporter.dropped_dofs else "",
    )

    if multi:
        arm_tip = tips
        strip = ""
    else:
        arm_base = args.arm_base or args.root_link
        arm_tip = tip_of(tree, facts, arm_base, args.tip_link)
        strip = prefix
        groups = None

    links = links_from_urdf(model, urdf, strip, roots=len(facts_list))
    if not links:
        raise ValueError(f"no URDF link matched a body in the model (prefix={strip!r})")
    srdf_tree = build_srdf(
        model,
        links,
        name=name,
        arm_base="" if multi else arm_base,
        arm_tip="" if multi else arm_tip,
        gripper_joint=facts.gripper_joint,
        gripper_open=facts.gripper_open,
        gripper_close=facts.gripper_close,
        home=facts.home,
        arms=groups,
        combined_group=combined_group,
        urdf_root_link=args.root_link if multi else "",
        base_joint=args.base_joint,
        parent_frame=args.parent_frame,
        samples=args.samples,
    )
    ET.indent(srdf_tree, space="  ")
    srdf_tree.write(srdf, encoding="utf-8", xml_declaration=True)
    log.info("wrote %s", srdf)

    for fname, text in (
        ("kinematics.yaml", kinematics_yaml(facts_list, combined_group)),
        (
            "joint_limits.yaml",
            joint_limits_yaml(facts_list, args.max_velocity, args.max_acceleration),
        ),
        ("moveit_controllers.yaml", moveit_controllers_yaml(facts_list, args.gripper_effort)),
        ("ompl_planning.yaml", ompl_planning_yaml(facts_list, combined_group)),
    ):
        (out / fname).write_text(text, encoding="utf-8")
    log.info("wrote kinematics, joint_limits, moveit_controllers and ompl_planning YAMLs")

    planning_frame = assert_agrees(out, urdf, srdf, facts_list, arm_tip, combined_group)
    log.info(
        "planning frame: %s; chains %s",
        planning_frame,
        "; ".join(f"{g.base_link} -> {g.tip_link}" for g in groups)
        if multi
        else f"{arm_base} -> {arm_tip}",
    )

    if args.check:
        err, where = round_trip_error(urdf, model, strip, mesh_dir=exporter.mesh_dir)
        if err > args.tolerance:
            log.error(
                "the exported URDF diverges from the MJCF by %.3e m at %r (tolerance %.1e). MoveIt "
                "would plan against a different robot than the one being simulated.",
                err,
                where,
                args.tolerance,
            )
            return 1
        log.info("FK round trip: %.3e m worst error (at %r)", err, where)

    if args.manifest:
        from .config import world_sources

        sources = [str(p) for p in world_sources(args.world)]
        Path(args.manifest).parent.mkdir(parents=True, exist_ok=True)
        with open(args.manifest, "w", encoding="utf-8") as fh:
            json.dump({"inputs": sources}, fh, indent=2)
        log.info("wrote source manifest (%d files) to %s", len(sources), args.manifest)
    return 0


def _last_link_of_chain(urdf_root: ET.Element, arm_base: str, joints: list[str]) -> str:
    """The outermost rigidly attached frame: the last actuated joint's child, then all FIXED links.

    Only used when no ``--tip-site`` was given, and the fixed descent is why: a UR arm's
    ``wrist_3_link`` carries a fixed ``tool0``, which carries the tool, which carries the gripper's own
    base -- all one rigid body. Stopping at the joint's child would put the chain several frames short
    of anything a plan cares about. It still ends short of the point that GRASPS, which is a site and
    therefore has no link unless ``--tip-site`` made one.
    """
    child_of = {j.get("name"): j.find("child").get("link") for j in urdf_root.findall("joint")}
    fixed_children: dict[str, str] = {
        j.find("parent").get("link"): j.find("child").get("link")
        for j in urdf_root.findall("joint")
        if j.get("type") == "fixed"
    }
    for jname in reversed(joints):
        if jname not in child_of:
            continue
        link = child_of[jname]
        seen = {link}
        while (nxt := fixed_children.get(link)) is not None and nxt not in seen:
            link = nxt
            seen.add(link)
        return link
    return arm_base


if __name__ == "__main__":
    sys.exit(main())
