"""The asset plugins' endpoints: typed, on the entity each registers, and applied once per step."""

from __future__ import annotations

from pathlib import Path

import pytest

from roqsim.config import load_config_from_dict
from roqsim.engine import Engine
from roqsim.introspection import get_plugin_details


def _engine(*components):
    engine = Engine(
        load_config_from_dict({"sim": {}, "components": list(components)}, base_dir=Path("."))
    )
    engine.setup()
    engine.reset()
    return engine


def _endpoints(engine):
    return {(e.owner, e.name): e for e in engine.ctx.interface.all() if e.owner != "sim"}


def test_conveyor_speed_is_a_stream_and_the_package_pose_is_the_packages():
    engine = _engine({"conveyor": {"namespace": "belt"}, "name": "conveyor"})
    try:
        eps = _endpoints(engine)
        speed = eps[("conveyor", "speed")]
        assert speed.namespace == "belt"
        assert [(p.name, p.type.unit) for p in speed.params] == [("data", "m/s")]
        assert eps[("package", "package_pose")].namespace == ""
        conveyor = engine.ctx.blackboard.get("conveyor:conveyor")
        speed.write({"data": -0.3})
        assert conveyor.get_speed() != -0.3, "a stream is applied on the physics thread"
        engine.step()
        assert conveyor.get_speed() == -0.3
    finally:
        engine.shutdown()


def test_door_cmd_is_a_stream_and_door_a_command():
    engine = _engine({"door": {}, "name": "door"})
    try:
        eps = _endpoints(engine)
        door = engine.ctx.blackboard.get("door:door")
        eps[("door", "cmd")].write({"data": 0.4})
        engine.step()
        assert door.get_openness() == pytest.approx(0.4)
        future = eps[("door", "door")].write({"position": 1.5})
        engine.step()
        assert future.result(0) is None
        assert door.get_openness() == 1.0, "clamped to fully open"
        assert isinstance(eps[("door", "state")].read(), float)
    finally:
        engine.shutdown()


@pytest.mark.parametrize(
    ("plugin", "name", "units"),
    [
        ("prop_trajectory", "stage_progress", ["m", "m", None]),
        ("conveyor", "package_pose", None),
    ],
)
def test_the_out_endpoints_describe_their_payload(plugin, name, units):
    rows = {row["name"]: row for row in get_plugin_details(plugin)["endpoints"]}
    result = rows[name]["result"]
    if units is not None:
        assert [i.get("unit") for i in result["items"]] == units
    else:
        (bone,) = result["items"]
        assert [i.get("unit") for i in bone["items"]] == [None, "m", None]
