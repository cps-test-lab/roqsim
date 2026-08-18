"""Integration: an arm riding a linear axis is one 7-DOF robot, in the sim and in the URDF.

A 6-DOF arm on a rail is the substrate's kinematically redundant manipulator, so the properties these
tests pin are the ones a redundancy-resolving planner depends on: the rail is a commandable joint, it
comes FIRST in every joint ordering, and the description MoveIt plans against has it too.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

import mujoco
import pytest

from roqsim.config import load_config_from_dict
from roqsim.engine import Engine

RAIL = {"axis": [1, 0, 0], "range": [-1.5, 1.5], "home": 0.3}
# Ceiling mount: 2.6 m up and rolled 180 deg, so the arm hangs -- the configuration the rail option
# exists for (a floor pedestal needs no carriage).
CEILING = {"pos": [0.0, 0.0, 2.6], "rpy": [3.14159265, 0.0, 0.0]}


def _world(tmp_path, arm_extra=None):
    plugins = [
        {
            "spawn_arm": {
                "model": "ur10e",
                "name": "ur10e",
                "prefix": "ur10e_",
                **CEILING,
                **(arm_extra or {}),
            }
        },
        {"arm_controller": {"arm": "ur10e"}},
    ]
    return load_config_from_dict({"sim": {}, "plugins": plugins}, base_dir=tmp_path)


def _engine(tmp_path, **kwargs):
    engine = Engine(_world(tmp_path, **kwargs))
    engine.setup()
    engine.reset()
    return engine


def _positions(handle):
    state = handle.read_state()
    pos = state[0] if len(state) == 2 else state[1]
    return dict(zip(handle.joint_names, (float(p) for p in pos), strict=True))


def test_rail_adds_one_commandable_dof_before_the_arm(tmp_path):
    engine = _engine(tmp_path, arm_extra={"rail": RAIL})
    m = engine.ctx.model
    assert m.nu == 7, "six arm actuators plus the carriage drive"

    handle = engine.ctx.blackboard.require("arm:ur10e")
    # Order is the contract: arm_controller reports and commands in model order, and the URDF below
    # puts the prismatic joint at the root of the chain. A rail that sorted anywhere else would make
    # /joint_states and the planning model disagree about which number is which joint.
    assert handle.joint_names[0] == "rail_joint"
    assert handle.joint_names[1] == "shoulder_pan_joint"

    jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "ur10e_rail_joint")
    assert m.jnt_type[jid] == mujoco.mjtJoint.mjJNT_SLIDE


def test_rail_home_seats_the_carriage_without_shifting_the_arm(tmp_path):
    # The per-model `home` is the ARM's six joints; the carriage's start is rail.home. If the two were
    # one vector, this ur10e default would land on [rail, j1..j5] and leave wrist_3 unset.
    engine = _engine(tmp_path, arm_extra={"rail": RAIL})
    q = _positions(engine.ctx.blackboard.require("arm:ur10e"))
    assert q["rail_joint"] == pytest.approx(0.3)
    assert q["shoulder_pan_joint"] == pytest.approx(-1.5708, abs=1e-3)
    assert q["wrist_3_joint"] == pytest.approx(0.0, abs=1e-3)
    # The mount pose is where the axis sits, so the arm base spawns at rail.home along it.
    assert engine.ctx.data.body("ur10e_base").xpos[0] == pytest.approx(0.3, abs=1e-3)


def test_carriage_tracks_a_commanded_position(tmp_path):
    engine = _engine(tmp_path, arm_extra={"rail": RAIL})
    handle = engine.ctx.blackboard.require("arm:ur10e")
    handle.set_targets(["rail_joint"], [-1.2])
    for _ in range(6000):
        engine.step()
    q = _positions(handle)
    assert q["rail_joint"] == pytest.approx(-1.2, abs=1e-2)
    # The whole arm rides along -- the point of the axis.
    assert engine.ctx.data.body("ur10e_base").xpos[0] == pytest.approx(-1.2, abs=1e-2)


def test_rail_geometry_is_visual_only(tmp_path):
    # A ceiling track that collides traps the arm against its own support from the first step, and the
    # planner's collision model comes from the URDF, not from these geoms.
    engine = _engine(tmp_path, arm_extra={"rail": RAIL})
    m = engine.ctx.model
    for name in ("ur10e_rail_beam", "ur10e_rail_carriage_geom"):
        gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, name)
        assert gid >= 0, f"missing {name}"
        assert m.geom_contype[gid] == 0 and m.geom_conaffinity[gid] == 0


@pytest.mark.parametrize(
    ("rail", "expected"),
    [
        ({"range": [1.0, -1.0]}, "min < max"),
        ({"range": [0.0, 1.0], "home": 2.0}, "'home' must lie within"),
        ({"axis": [0, 0, 0]}, "non-zero"),
    ],
)
def test_rail_config_is_refused_when_unusable(rail, expected):
    from roqsim_manipulation.plugins.spawn_arm import SpawnArmPlugin

    errors = SpawnArmPlugin.validate_config(
        SpawnArmPlugin, {"model": "ur10e", "rail": {**RAIL, **rail}}
    )
    assert any(expected in e for e in errors), errors


def test_rail_and_mount_are_mutually_exclusive():
    from roqsim_manipulation.plugins.spawn_arm import SpawnArmPlugin

    errors = SpawnArmPlugin.validate_config(
        SpawnArmPlugin,
        {"model": "ur10e", "prefix": "a_", "rail": RAIL, "mount": {"robot": "base"}},
    )
    assert any("mutually exclusive" in e for e in errors), errors


def test_exported_urdf_keeps_the_rail_dof(tmp_path):
    # The exporter emits a joint per NON-root body, and the carriage is the root -- so without the
    # jointed-root case the rail would vanish from the description while still existing in
    # /joint_states, and MoveIt would reject the 7-name trajectory the controller reports.
    from roqsim.export_urdf import UrdfExporter

    engine = _engine(tmp_path, arm_extra={"rail": RAIL})
    exporter = UrdfExporter(
        engine.ctx.model, prefix="ur10e_", name="ur10e_rail", root_link="rail_carriage"
    )
    root = exporter.export().getroot()

    joints = {j.get("name"): j for j in root.findall("joint")}
    rail = joints["rail_joint"]
    assert rail.get("type") == "prismatic"
    assert rail.find("parent").get("link") == "world"
    assert rail.find("child").get("link") == "rail_carriage"
    assert rail.find("axis").get("xyz") == "1 0 0"
    limit = rail.find("limit")
    assert (float(limit.get("lower")), float(limit.get("upper"))) == (-1.5, 1.5)
    # The mount pose becomes the axis origin, so plan-space and sim-space place the track identically.
    assert rail.find("origin").get("xyz") == "0 0 2.6"

    # Exactly one root: `world` is the only link that is never a child.
    children = {j.find("child").get("link") for j in root.findall("joint")}
    roots = [ln.get("name") for ln in root.findall("link") if ln.get("name") not in children]
    assert roots == ["world"]


def test_unrailed_arm_has_no_world_link(tmp_path):
    # The synthetic parent is emitted only for a jointed root; a welded-down arm keeps root_link as
    # the URDF root, so this change cannot perturb the existing descriptions.
    from roqsim.export_urdf import UrdfExporter

    engine = _engine(tmp_path)
    root = UrdfExporter(engine.ctx.model, prefix="ur10e_", name="ur10e").export().getroot()
    assert "world" not in {ln.get("name") for ln in root.findall("link")}
    assert not any(j.get("type") == "prismatic" for j in root.findall("joint"))


def test_urdf_is_wellformed_xml(tmp_path):
    from roqsim.export_urdf import UrdfExporter

    engine = _engine(tmp_path, arm_extra={"rail": RAIL})
    out = tmp_path / "robot.urdf"
    tree = UrdfExporter(
        engine.ctx.model, prefix="ur10e_", name="r", root_link="rail_carriage", mesh_dir=tmp_path
    ).export()
    tree.write(out)
    assert ET.parse(out).getroot().tag == "robot"
