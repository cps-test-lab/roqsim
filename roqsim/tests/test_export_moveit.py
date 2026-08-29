"""The complete MoveIt 2 configuration export, and the facts it must read rather than restate.

Three hand-written copies of this generator existed before it, byte-for-byte identical in the parts
that matter, and each one had drifted somewhere. So the tests here are mostly about PROVENANCE: not
"does it emit a controller_names list" but "does that list name the action the bridge actually
serves, and this arm's joints in the order the controller publishes them". That is the assertion the
three copies would have failed.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

import mujoco
import pytest
import yaml

from roqsim.config import load_config_from_dict
from roqsim.engine import Engine
from roqsim.export_moveit import (
    ARM_GROUP,
    arm_facts,
    infer_collapse,
    moveit_controllers_yaml,
    ompl_planning_yaml,
)


def _cell(tmp_path, model="ur5e", *, gripper="robotiq_2f85", name="ur5e", namespace=None):
    arm = {
        "model": model,
        "prefix": f"{model}_",
        "pos": [0.0, 0.0, 0.76],
    }
    if gripper:
        arm["end_effector"] = {
            "model": gripper,
            "site": "attachment_site",
            "pos": [0.0, 0.0, 0.011],
        }
    if namespace:
        arm["namespace"] = namespace
    return load_config_from_dict(
        {"sim": {}, "plugins": [{"spawn_arm": arm, "name": name}]}, base_dir=tmp_path
    )


@pytest.fixture(scope="module")
def ur5e(tmp_path_factory):
    """UR5e + Robotiq 2F-85 on a bench, compiled once: the tests below only read it."""
    engine = Engine(_cell(tmp_path_factory.mktemp("ur5e")))
    engine.setup()
    return engine


# -- provenance: the facts come off the model, not off flags -------------------------------------


def test_joints_come_from_the_handle_the_controller_published(ur5e):
    """In the controller's ORDER, not sorted and not the world's own list.

    robot_state_publisher matches joint states to URDF joints by name, and MoveIt executes a
    trajectory whose joint order must be one the controller accepts. A list that is right about the
    robot but wrong about the order leaves MoveIt planning from a pose the arm is not in.
    """
    facts = arm_facts(ur5e)
    assert facts.joints == [
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint",
    ]


def test_action_names_are_the_ones_the_bridge_serves(ur5e):
    """Read off the declared endpoints, which is what the bridge wires -- so they cannot disagree."""
    facts = arm_facts(ur5e)
    endpoints = {e.name: e for e in ur5e.ctx.interface.all() if e.owner == facts.arm}
    assert facts.trajectory_action == endpoints["follow_joint_trajectory"].backend["ros2"]["name"]
    assert facts.gripper_action == endpoints["gripper_cmd"].backend["ros2"]["name"]

    body = yaml.safe_load(moveit_controllers_yaml(facts, 50.0))["moveit_simple_controller_manager"]
    assert body["controller_names"] == [facts.controller, facts.gripper_controller]
    assert f"{facts.controller}/{body[facts.controller]['action_ns']}" == facts.trajectory_action, (
        "the controller name and action_ns must reassemble into the action the bridge serves"
    )
    assert (
        f"{facts.gripper_controller}/{body[facts.gripper_controller]['action_ns']}"
        == facts.gripper_action
    )
    assert body[facts.controller]["joints"] == facts.joints


def test_home_is_the_posture_the_simulator_actually_starts_in(ur5e):
    """Not the world's ``home:`` key: the controller applies that itself, and a ``rest`` stance can
    overlay it. MoveIt's start-state bounds check runs against the real qpos, so the real qpos is
    what the SRDF's ``home`` has to be."""
    facts = arm_facts(ur5e)
    model, data = ur5e.ctx.model, ur5e.ctx.data
    for joint, value in facts.home.items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, facts.prefix + joint)
        assert jid >= 0
        assert value == pytest.approx(float(data.qpos[int(model.jnt_qposadr[jid])]))
    assert any(abs(v) > 1e-6 for v in facts.home.values()), (
        "fixture is wrong: an all-zero home would make this test pass on a bug that reads nothing"
    )


def test_gripper_units_come_from_the_gripper_manifest(ur5e):
    """A world pairing a bare arm with a gripper states neither number; the gripper's manifest does.

    They must be the joint's own units, so that a MoveIt named target and a GripperCommand position
    mean the same thing.
    """
    facts = arm_facts(ur5e)
    assert facts.gripper_joint == "robotiq_85_left_knuckle_joint"
    assert facts.gripper_open == pytest.approx(0.0)
    assert facts.gripper_close == pytest.approx(0.8)


def test_the_arms_namespace_reaches_the_reported_actions(tmp_path):
    """Two arms under one bridge are told apart by namespace, so a config that drops it points at
    an action nothing serves."""
    engine = Engine(_cell(tmp_path, namespace="left"))
    engine.setup()
    facts = arm_facts(engine)
    assert facts.namespace == "left"
    assert "/left/arm_controller/follow_joint_trajectory" in moveit_controllers_yaml(facts, 50.0)


# -- the collapse root, read off the model's equality constraints --------------------------------


def test_collapse_is_inferred_at_the_closed_loops_common_ancestor(ur5e):
    """The 2F-85's four-bar is closed by ``equality``, so the model says where the loop is.

    Getting this wrong is quiet: the constraint is simply not exported, the URDF keeps the loop's
    branches as revolute DOFs nothing publishes, and move_group then never assembles a complete robot
    state.
    """
    facts = arm_facts(ur5e)
    assert facts.collapse == ("robotiq_85_base_link",)


def test_an_arm_with_no_closed_loop_collapses_nothing(tmp_path):
    """A bare flange has no linkage URDF cannot express, so inventing a collapse would only lump
    away geometry MoveIt should plan around."""
    engine = Engine(_cell(tmp_path, gripper=None))
    engine.setup()
    assert arm_facts(engine).collapse == ()


def test_equalities_outside_the_robot_are_ignored(ur5e):
    """A world may weld a prop to a shelf or couple a conveyor's rollers. Taking those into the
    common ancestor would lump the entire robot into one link -- worse than not collapsing at all."""
    model = ur5e.ctx.model
    assert model.neq > 0, "fixture is wrong: this gripper should carry equality constraints"
    # An empty robot subtree means nothing of the robot is involved, whatever the world constrains.
    assert infer_collapse(model, set()) == ()


# -- ompl_planning: the one derived value in the file that is otherwise the experiment's ----------


def test_a_range_limited_arm_gets_no_start_state_bounds_slack(ur5e):
    """The UR joints are range-limited, so the setting would be noise -- and a reader who sees it on
    every arm learns nothing from its presence."""
    body = yaml.safe_load(ompl_planning_yaml(arm_facts(ur5e)))
    assert "start_state_max_bounds_error" not in body
    assert arm_facts(ur5e).continuous_joints == []


def test_a_continuous_joint_arm_gets_the_slack_and_is_told_why(tmp_path):
    """The expensive failure this prevents: a phase failing instantly with START_STATE_INVALID
    (-26) right after a phase that succeeded, at a different phase each run."""
    engine = Engine(_cell(tmp_path, model="gen3", gripper=None, name="gen3"))
    engine.setup()
    facts = arm_facts(engine)
    # Read off the model, and these four are exactly the ones a hand-written config named.
    assert facts.continuous_joints == ["joint_1", "joint_3", "joint_5", "joint_7"]
    text = ompl_planning_yaml(facts)
    assert yaml.safe_load(text)["start_state_max_bounds_error"] == 0.1
    assert "joint_1" in text, "the file should say WHICH joints made it necessary"


def test_the_projection_evaluator_names_joints_in_the_group(ur5e):
    """Naming a joint outside the group makes move_group refuse every request with 'joint ... is not
    known to the group', which reads like a planner problem rather than a config typo."""
    facts = arm_facts(ur5e)
    body = yaml.safe_load(ompl_planning_yaml(facts))
    named = body[ARM_GROUP]["projection_evaluator"].removeprefix("joints(").removesuffix(")")
    assert all(j in facts.joints for j in named.split(","))


# -- the whole set, through the CLI --------------------------------------------------------------


def _run_cli(tmp_path, world: dict, *extra):
    """Drive the CLI the way a build step does: a world file in, a directory out."""
    from roqsim.export_moveit import main

    (tmp_path / "cell.yaml").write_text(yaml.safe_dump(world), encoding="utf-8")
    out = tmp_path / "gen"
    code = main(
        [
            "--world",
            str(tmp_path / "cell.yaml"),
            "--out",
            str(out),
            # Far below the Setup Assistant's 10000: this is not testing the matrix, and 10000 samples
            # of a 12-link robot dominate the runtime of the whole file.
            "--samples",
            "60",
            *extra,
        ]
    )
    return code, out


_WORLD = {
    "sim": {},
    "plugins": [
        {
            "spawn_arm": {
                "model": "ur5e",
                "prefix": "ur5e_",
                "pos": [0.0, 0.0, 0.76],
                "end_effector": {
                    "model": "robotiq_2f85",
                    "site": "attachment_site",
                    "pos": [0.0, 0.0, 0.011],
                },
            },
            "name": "ur5e",
        }
    ],
}


def test_the_cli_writes_the_six_files_and_they_parse(tmp_path):
    code, out = _run_cli(tmp_path, _WORLD, "--tip-site", "pinch", "--check")
    assert code == 0
    assert (out / "ur5e.urdf").is_file() and (out / "ur5e.srdf").is_file()
    assert list((out / "meshes").glob("*.stl")), "the URDF references meshes that must be written"
    for fname in (
        "kinematics.yaml",
        "joint_limits.yaml",
        "moveit_controllers.yaml",
        "ompl_planning.yaml",
    ):
        assert yaml.safe_load((out / fname).read_text(encoding="utf-8")), f"{fname} is empty"
    ET.parse(out / "ur5e.urdf")
    ET.parse(out / "ur5e.srdf")


def test_no_planning_yaml_is_emitted(tmp_path):
    """It is not a MoveIt file. Every hand-written copy shipped one -- the trial node's own
    parameters -- and that belongs to the experiment, not to the substrate."""
    _code, out = _run_cli(tmp_path, _WORLD, "--tip-site", "pinch")
    assert not (out / "planning.yaml").exists()


def test_the_chain_ends_at_the_tip_site_and_the_srdf_agrees(tmp_path):
    _code, out = _run_cli(tmp_path, _WORLD, "--tip-site", "pinch")
    srdf = ET.parse(out / "ur5e.srdf").getroot()
    chain = srdf.find(f"group[@name='{ARM_GROUP}']/chain")
    assert chain.get("tip_link") == "tcp"
    urdf_links = {
        link.get("name") for link in ET.parse(out / "ur5e.urdf").getroot().findall("link")
    }
    assert chain.get("tip_link") in urdf_links, (
        "export srdf alone would accept a tip that is absent"
    )


def test_a_welded_arm_gets_no_virtual_joint_and_plans_in_its_root_link(tmp_path):
    """A bolted-down arm has nothing publishing a transform above its base, so the planning frame is
    the root link itself -- and a virtual joint here would be one nothing ever publishes."""
    _code, out = _run_cli(tmp_path, _WORLD, "--tip-site", "pinch")
    srdf = ET.parse(out / "ur5e.srdf").getroot()
    assert srdf.find("virtual_joint") is None


def test_a_tip_that_is_not_a_link_is_refused(tmp_path):
    """``export srdf`` alone accepts this and move_group then loads and never plans -- the failure
    that hangs rather than erroring. Owning both files is what makes it checkable."""
    with pytest.raises(SystemExit, match="not a link in the URDF"):
        _run_cli(tmp_path, _WORLD, "--arm-tip", "no_such_link")


def test_a_world_with_no_arm_says_so(tmp_path):
    world = {
        "sim": {},
        "plugins": [
            {"spawn_model": {"model": "industrial_table", "pos": [0, 0, 0]}, "name": "bench"}
        ],
    }
    with pytest.raises(SystemExit, match="follow_joint_trajectory"):
        _run_cli(tmp_path, world)


def test_two_arms_must_be_disambiguated(tmp_path):
    world = {
        "sim": {},
        "plugins": [
            {"spawn_arm": {"model": "ur5e", "prefix": "a_", "pos": [0, 0, 0]}, "name": "a"},
            {"spawn_arm": {"model": "ur5e", "prefix": "b_", "pos": [1, 0, 0]}, "name": "b"},
        ],
    }
    with pytest.raises(SystemExit, match="--arm"):
        _run_cli(tmp_path, world)
