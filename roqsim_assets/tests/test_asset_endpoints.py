"""The asset plugins' endpoints: typed, on the entity each registers, and applied once per step."""

from __future__ import annotations

from pathlib import Path

import pytest

from roqsim.config import load_config_from_dict
from roqsim.engine import Engine
from roqsim.introspection import get_plugin_details
from roqsim.types import Transform


def _engine(*components):
    engine = Engine(
        load_config_from_dict({"sim": {}, "components": list(components)}, base_dir=Path("."))
    )
    engine.setup()
    engine.reset()
    return engine


def _endpoints(engine):
    return {(e.owner, e.name): e for e in engine.ctx.interface.all() if e.owner != "sim"}


def test_conveyor_speed_is_a_stream_on_the_conveyor():
    engine = _engine({"conveyor": {"namespace": "belt"}, "name": "conveyor"})
    try:
        eps = _endpoints(engine)
        speed = eps[("conveyor", "speed")]
        assert speed.namespace == "belt"
        assert [(p.name, p.type.unit) for p in speed.params] == [("data", "m/s")]
        conveyor = engine.ctx.blackboard.get("conveyor:conveyor")
        speed.write({"data": -0.3})
        assert conveyor.get_speed() != -0.3, "a stream is applied on the physics thread"
        engine.step()
        assert conveyor.get_speed() == -0.3
    finally:
        engine.shutdown()


def test_the_package_pose_is_the_packages_own_transform():
    engine = _engine({"conveyor": {"namespace": "belt", "object_name": "box"}, "name": "conveyor"})
    try:
        ep = _endpoints(engine)[("box", "package_pose")]
        assert ep.namespace == ""
        pose = ep.read()
        assert isinstance(pose, Transform) and pose.child == "package" and pose.parent == ""
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


@pytest.mark.parametrize("config", [{"controllable": False}, {"leaf": False}])
def test_a_passive_or_leafless_door_declares_no_endpoints(config):
    engine = _engine({"door": config, "name": "door"})
    try:
        assert not {n for (o, n) in _endpoints(engine) if o == "door"} & {"cmd", "state", "door"}
    finally:
        engine.shutdown()


def test_stage_progress_publishes_the_distance_travelled():
    rows = {row["name"]: row for row in get_plugin_details("prop_trajectory")["endpoints"]}
    fields = rows["stage_progress"]["result"]["fields"]
    assert [(f["name"], f.get("unit")) for f in fields] == [
        ("s", "m"),
        ("total", "m"),
        ("done", None),
    ]
    assert rows["stage_progress"]["payload"] == "Progress"
