"""``roqsim export moveit`` on every shipped world with an arm, and every arm model on its own.

Each export is checked against what the simulator publishes: a MoveIt configuration whose joint
names differ from ``/joint_states`` loads and then never has a complete robot state.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

import pytest
import yaml

from roqsim import exit_status
from roqsim.config import drop_transport_plugins, load_config
from roqsim.engine import Engine
from roqsim.export_moveit import arm_facts, main

_SEVERAL_CONTROLLERS = pytest.mark.xfail(
    strict=True,
    reason="the entity carries several arm_controllers, and --arm selects an entity, not one of "
    "its controllers",
)

SHIPPED = [
    "roqsim_manipulation_assets:gen3_demo",
    "roqsim_manipulation_assets:lbr_iiwa7_demo",
    "roqsim_manipulation_assets:lbr_iiwa14_demo",
    "roqsim_manipulation_assets:m1013_demo",
    "roqsim_manipulation_assets:open_manipulator_x_demo",
    "roqsim_manipulation_assets:ur10e_demo",
    "roqsim_manipulation_assets:vx300s_demo",
    "roqsim_manipulation_assets:xarm7_demo",
    "roqsim_mobile_manipulation:frankie_demo",
    pytest.param("roqsim_mobile_manipulation:tiago_pro_demo", marks=_SEVERAL_CONTROLLERS),
    pytest.param("roqsim_humanoid:unitree_g1_dex1_demo", marks=_SEVERAL_CONTROLLERS),
]

#: Arm models no shipped world spawns on its own.
MODELS = ["wx250s", "panda", "ur5e"]


def _world_of(model: str, tmp_path) -> str:
    path = tmp_path / f"{model}.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "sim": {},
                "components": [
                    {"spawn_arm": {"model": model, "prefix": f"{model}_"}, "name": model}
                ],
            }
        ),
        encoding="utf-8",
    )
    return str(path)


def _published_joint_names(world: str) -> set[str]:
    """Every name the exported robot's ``joint_states`` endpoints report: its arm's, and any other
    part's, such as a mobile base's wheels."""
    cfg = load_config(world)
    drop_transport_plugins(cfg)
    with Engine(cfg, preview=True) as engine:
        arm = arm_facts(engine).arm
        return {
            name
            for e in engine.ctx.interface.all()
            if e.owner == arm and e.name == "joint_states"
            for name in e.read().names
        }


def _export_and_compare(world: str, tmp_path) -> None:
    out = tmp_path / "gen"
    assert main(["--world", world, "--out", str(out), "--samples", "60", "--check"]) == (
        exit_status.OK
    )
    published = _published_joint_names(world)

    (urdf,) = out.glob("*.urdf")
    movable = {
        j.get("name"): j
        for j in ET.parse(urdf).getroot().findall("joint")
        if j.get("type") != "fixed"
    }
    # A mimic joint follows its leader and is never published; every other movable joint is, or
    # MoveIt waits for it forever.
    driven = {name for name, j in movable.items() if j.find("mimic") is None}
    assert driven <= published, f"joints nothing publishes: {sorted(driven - published)}"

    (srdf,) = out.glob("*.srdf")
    grouped = {
        j.get("name") for g in ET.parse(srdf).getroot().iter("group") for j in g.iter("joint")
    }
    controllers = yaml.safe_load((out / "moveit_controllers.yaml").read_text(encoding="utf-8"))
    commanded = {
        j
        for entry in controllers["moveit_simple_controller_manager"].values()
        if isinstance(entry, dict)
        for j in entry.get("joints", [])
    }
    for what, names in (("SRDF group", grouped), ("moveit_controllers.yaml", commanded)):
        assert names <= driven, f"{what} names {sorted(names - driven)}, not driven URDF joints"
        assert names <= published, f"{what} names {sorted(names - published)}, never published"


@pytest.mark.parametrize("world", SHIPPED)
def test_every_shipped_arm_world_exports_with_the_published_joint_names(world, tmp_path):
    _export_and_compare(world, tmp_path)


@pytest.mark.parametrize("model", MODELS)
def test_every_arm_model_exports_with_the_published_joint_names(model, tmp_path):
    _export_and_compare(_world_of(model, tmp_path), tmp_path)
