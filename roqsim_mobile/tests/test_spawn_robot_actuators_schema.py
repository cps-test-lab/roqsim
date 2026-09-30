"""spawn_robot declares its ``actuators:`` block as the shared declaration, as spawn_arm does."""

from __future__ import annotations

from roqsim.introspection import get_plugin_details
from roqsim_mobile.plugins.spawn_robot import SpawnRobotPlugin


def _errors(actuators) -> list[str]:
    config = {"model": "husky_a200", "actuators": actuators}
    return SpawnRobotPlugin(config).config_errors(config)


def test_an_unknown_key_of_an_actuator_entry_is_refused_by_its_full_path():
    (error,) = _errors({"each": {"wheel_left_motor": {"dd": 1.0}}})
    assert error.startswith("'actuators.each.wheel_left_motor.dd' is not a key of")


def test_a_mujoco_gain_name_is_refused_once_with_its_guidance():
    (error,) = _errors({"control": "velocity", "kv": 1.0})
    assert error.startswith("'actuators.kv' is not a key of 'actuators' -- that is MuJoCo's")


def test_the_catalog_publishes_the_actuator_map():
    schema = get_plugin_details("spawn_robot")["schema"]
    actuators = next(f for f in schema if f["name"] == "actuators")
    each = next(f for f in actuators["fields"] if f["name"] == "each")
    assert [f["name"] for f in each["values"]["fields"]][:3] == ["control", "p", "d"]
