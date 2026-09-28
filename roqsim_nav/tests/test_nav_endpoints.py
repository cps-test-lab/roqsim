"""The navigator's goal endpoints: a family of typed commands, applied on the physics thread."""

from __future__ import annotations

from pathlib import Path

import pytest

from roqsim.config import load_config_from_dict
from roqsim.endpoint import ParameterError
from roqsim.engine import Engine
from roqsim.introspection import get_plugin_details


def _engine(**navigator):
    world = {
        "sim": {},
        "components": [
            {
                "spawn_robot": {"model": "makerspet_mini"},
                "name": "bot",
                "components": [{"navigator": {"speed": 0.3, **navigator}}],
            }
        ],
    }
    engine = Engine(load_config_from_dict(world, base_dir=Path(".")))
    engine.setup()
    engine.reset()
    return engine


def _goal_endpoints(engine):
    return {
        e.name: e for e in engine.ctx.interface.all() if e.owner == "bot" and e.direction == "in"
    }


def test_the_goal_endpoints_are_one_family_described_once():
    rows = {row["name"]: row for row in get_plugin_details("navigator")["endpoints"]}
    assert rows["{item}"]["family"] and rows["{item}"]["kind"] == "command"
    assert [p["name"] for p in rows["{item}"]["params"]] == ["poses"]
    assert rows["start_route"]["kind"] == "command" and rows["start_route"]["params"] == []


def test_a_goal_command_replaces_the_route_in_order():
    engine = _engine(goals=[[1.0, 0.0]], autostart=False)
    try:
        eps = _goal_endpoints(engine)
        nav = engine.ctx.blackboard.get("nav:bot")
        first = eps["navigate_to_pose"].write({"poses": [(0.5, 0.2, 0.0)]})
        second = eps["navigate_through_poses"].write({"poses": [(0.5, 0.0), (0.8, 0.4)]})
        engine.step()
        assert first.result(0) is None and second.result(0) is None
        assert nav.started
        assert nav._state.waypoints[1:].tolist() == [[0.5, 0.0], [0.8, 0.4]]
    finally:
        engine.shutdown()


def test_an_empty_route_is_refused_into_the_future():
    engine = _engine(goals=[[1.0, 0.0]])
    try:
        future = _goal_endpoints(engine)["navigate_through_poses"].write({"poses": []})
        engine.step()
        with pytest.raises(ValueError, match="at least one pose"):
            future.result(0)
        refused = _goal_endpoints(engine)["navigate_to_pose"].write({"pose": [(1.0, 0.0)]})
        with pytest.raises(ParameterError, match="'poses'"):
            refused.result(0)
    finally:
        engine.shutdown()


def test_start_route_releases_a_held_route():
    engine = _engine(goals=[[1.0, 0.0]], autostart=False)
    try:
        nav = engine.ctx.blackboard.get("nav:bot")
        assert not nav.started
        future = _goal_endpoints(engine)["start_route"].write(None)
        engine.step()
        assert future.result(0) is None and nav.started
    finally:
        engine.shutdown()
