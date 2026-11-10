"""The asset plugins' endpoints: typed, on the entity each registers, and applied once per step."""

from __future__ import annotations

import math
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
        conveyor = engine.ctx.blackboard.get("machine:conveyor")
        speed.write({"data": -0.3})
        assert conveyor.value("speed") != -0.3, "a stream is applied on the physics thread"
        engine.step()
        assert conveyor.value("speed") == -0.3
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


def test_door_openness_is_a_stream_and_openness_goal_a_command():
    engine = _engine({"door": {}, "name": "door"})
    try:
        eps = _endpoints(engine)
        door = engine.ctx.blackboard.get("machine:door")
        span = math.radians(120)  # the default max_angle
        eps[("door", "openness")].write({"openness": 0.4})
        engine.step()
        assert door.value("hinge_pos") == pytest.approx(0.4 * span)
        future = eps[("door", "openness_goal")].write({"openness": 1.5})
        engine.step()
        assert future.result(0) is None
        assert door.value("hinge_pos") == pytest.approx(span), "clamped to fully open"
        assert isinstance(eps[("door", "openness_state")].read(), float)
    finally:
        engine.shutdown()


@pytest.mark.parametrize("config", [{"controllable": False}, {"leaf": False}])
def test_a_passive_or_leafless_door_declares_no_endpoints(config):
    engine = _engine({"door": config, "name": "door"})
    try:
        assert not {n for (o, n) in _endpoints(engine) if o == "door"} & {
            "openness",
            "openness_state",
            "openness_goal",
        }
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


@pytest.mark.parametrize(
    ("name", "programs"), [("door", {"open", "close", "move_to"}), ("conveyor", {"run", "stop"})]
)
def test_describe_lists_a_machines_programs(name, programs):
    details = get_plugin_details(name)
    assert {p["name"] for p in details["programs"]} == programs
    assert {"start", "status", "cancel"} <= {row["name"] for row in details["endpoints"]}


def test_a_door_and_a_conveyor_run_by_their_start_status_and_cancel_endpoints():
    """What ``machine_run`` does with any machine: start a program by its endpoint with a token,
    follow the status, cancel that start by its token."""
    engine = _engine(
        {"door": {"speed": 0.5}, "name": "door"}, {"conveyor": {"ramp": 0.5}, "name": "belt"}
    )
    try:
        eps = _endpoints(engine)
        future = eps[("door", "start")].write(
            {"program": "move_to", "args": {"openness": 0.5}, "token": "t1"}
        )
        engine.step()
        seq = future.result(1.0)
        for _ in range(4000):
            engine.step()
            status = eps[("door", "status")].read()
            if status.seq == seq and status.finished:
                break
        assert status.error == "" and eps[("door", "openness_state")].read() == pytest.approx(
            0.5, abs=0.03
        )
        future = eps[("belt", "start")].write(
            {"program": "run", "args": {"speed": 0.4}, "token": "t2"}
        )
        engine.step()
        seq = future.result(1.0)
        cancelled = eps[("belt", "cancel")].write({"token": "t2"})
        engine.step()
        assert cancelled.result(1.0) == seq
        engine.step()
        assert eps[("belt", "status")].read().error == "cancelled"
    finally:
        engine.shutdown()
