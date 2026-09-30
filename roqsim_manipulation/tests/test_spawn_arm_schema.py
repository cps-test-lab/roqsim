"""spawn_arm's config is its schema: every key it reads is declared, and no other is accepted."""

from __future__ import annotations

from pathlib import Path

import pytest

from roqsim.config import instantiate_plugins, load_config, load_config_from_dict
from roqsim.introspection import get_plugin_details
from roqsim.plugin import PluginError
from roqsim_manipulation.plugins.spawn_arm import SpawnArmPlugin

REPO = Path(__file__).resolve().parents[2]


def _errors(**config) -> list[str]:
    """Every config-stage error for a ur10e spawned with *config* -- what a load is held to."""
    config = {"model": "ur10e", **config}
    return SpawnArmPlugin(config).config_errors(config)


def _load(tmp_path, **config):
    world = {"sim": {}, "components": [{"spawn_arm": {"model": "ur10e", **config}}]}
    return instantiate_plugins(load_config_from_dict(world, base_dir=tmp_path))


def test_a_complete_config_loads(tmp_path):
    (arm, *_) = _load(
        tmp_path,
        prefix="ur10e_",
        base_body="base",
        pose={"position": {"z": 0.76}, "orientation": {"yaw": 3.14159}},
        home=[-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.0],
        default_plugins=False,
        actuators={
            "control": "impedance",
            "stiffness": 2.0,
            "damping": 0.02,
            "each": {"wrist_3": {"control": "position", "p": 2000, "d": 500}},
        },
        gravity_compensation=True,
        pedestal=True,
        pedestal_half_width=0.12,
        rail={"axis": [1, 0, 0], "range": [-1.5, 1.5], "home": 0.5, "kp": 1000, "damping": 50},
        end_effector={
            "model": "robotiq_2f85",
            "prefix": "g_",
            "pose": {"position": {"z": 0.011}},
            "replaces": ["ee_plate"],
        },
    )
    assert isinstance(arm, SpawnArmPlugin)
    assert (arm.rail_home, arm.rail_kp, arm.rail_joint) == (0.5, 1000.0, "rail_joint")


def test_a_top_level_key_it_does_not_read_is_refused(tmp_path):
    with pytest.raises(PluginError, match=r"'stiffnes' is not a setting of this component"):
        _load(tmp_path, stiffnes=2.0)


@pytest.mark.parametrize(
    "config, path",
    [
        ({"rail": {"axes": [1, 0, 0]}}, "rail.axes"),
        ({"mount": {"robot": "base", "bdy": "x"}, "prefix": "a_"}, "mount.bdy"),
        ({"end_effector": {"model": "robotiq_2f85", "sit": "x"}}, "end_effector.sit"),
        ({"actuators": {"each": {"wrist_3": {"stifness": 2.0}}}}, "actuators.each.wrist_3.stifness"),
    ],
)
def test_a_nested_key_it_does_not_read_is_refused_by_its_full_path(config, path):
    (error,) = _errors(**config)
    assert error.startswith(f"'{path}' is not a key of ")


def test_a_mujoco_gain_name_is_refused_once_with_its_guidance():
    (error,) = _errors(actuators={"control": "position", "kp": 2000.0})
    assert error.startswith("'actuators.kp' is not a key of 'actuators' -- that is MuJoCo's")
    assert "use 'p' (control: position) or 'stiffness' (control: impedance)" in error
    (error,) = _errors(actuators={"each": {"wrist_3": {"kv": 5.0}}})
    assert error.startswith("'actuators.each.wrist_3.kv' is not a key of")


def test_the_rail_s_rules_between_keys_are_each_reported_once():
    assert _errors(rail={"axis": [0, 0, 0], "range": [1.0, -1.0]}) == [
        "'rail.axis' must be a non-zero [x, y, z] direction",
        "'rail.range' must be [min, max] with min < max, in metres",
    ]
    assert _errors(rail={"home": 3.0}) == ["'rail.home' must lie within 'rail.range'"]
    assert _errors(rail={"axis": [1, 0]}) == ["'rail.axis' must have exactly 3 entries, got 2"]


def test_a_required_name_left_out_or_empty_is_refused():
    assert _errors(mount={"body": "base_link"}, prefix="a_") == [
        "'mount.robot' is required -- entity name of a spawn_robot declared before this arm"
    ]
    assert _errors(end_effector={"model": ""}) == [
        "'end_effector.model' is empty; it names what to spawn or weld to"
    ]


def test_the_catalog_publishes_the_schema_as_strict_with_the_actuator_map():
    details = get_plugin_details("spawn_arm")
    assert details["strict_keys"] is True
    assert [f["name"] for f in details["schema"]] == list(SpawnArmPlugin.CONFIG_SCHEMA)
    actuators = next(f for f in details["schema"] if f["name"] == "actuators")
    each = next(f for f in actuators["fields"] if f["name"] == "each")
    assert "p" in [f["name"] for f in each["values"]["fields"]]
    assert each["values"]["hints"]["kp"].startswith("that is MuJoCo's spelling")
    names = [p["name"] for p in details["parameters"]]
    assert {"rail.kp", "mount.robot", "end_effector.replaces", "actuators.each.<name>.p"} <= set(
        names
    )


# -- every shipped world that spawns an arm -------------------------------------------------------
SHIPPED = sorted(
    path
    for path in REPO.glob("*/src/*/worlds/**/*.yaml")
    if "spawn_arm:" in path.read_text(encoding="utf-8")
)


def test_the_shipped_worlds_with_an_arm_are_found():
    assert len(SHIPPED) >= 6, SHIPPED


@pytest.mark.parametrize("world", SHIPPED, ids=lambda p: p.stem)
def test_a_shipped_world_with_an_arm_loads(world):
    plugins = instantiate_plugins(load_config(str(world)))
    assert any(isinstance(p, SpawnArmPlugin) for p in plugins)
