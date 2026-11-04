# Copyright (C) 2026 Frederik Pasch
#
# SPDX-License-Identifier: Apache-2.0

"""Python scenarios: a generator on simulated time that drives entities through their handles."""

from __future__ import annotations

import pytest

from roqsim.config import load_config_from_dict
from roqsim.engine import Engine
from roqsim.scenario import ScenarioError

PRESS = {"machine_fixtures:Press": {"prefix": "p1_"}, "name": "press"}


def _engine(*components):
    engine = Engine(
        load_config_from_dict({"sim": {"gravity": [0, 0, 0]}, "components": list(components)})
    )
    engine.ctx.seed = 1
    engine.setup()
    engine.reset()
    return engine


def _events(engine, source):
    return [e for e in engine.ctx.events.all() if e["source"] == source]


def _until_done(engine, scenario="scene", limit=30.0):
    done = engine.ctx.interface.find(scenario, "done")
    while not done.read():
        assert engine.ctx.sim_time < limit, "the scenario never ended"
        engine.step()


def test_a_scenario_runs_a_machine_twice_and_says_when_it_is_done():
    engine = _engine(PRESS, {"scenario_fixtures:TwoStrokes": {"pause": 0.2}, "name": "scene"})
    try:
        _until_done(engine)
        press = [e["event"] for e in _events(engine, "press")]
        assert press.count("program_start") == 2 and press.count("program_end") == 2
        scene = _events(engine, "scene")
        assert [e["event"] for e in scene] == ["first", "second", "scenario_end"]
        assert scene[0]["seq"] == 1
        ends = [e["t"] for e in _events(engine, "press") if e["event"] == "program_end"]
        starts = [e["t"] for e in _events(engine, "press") if e["event"] == "program_start"]
        assert starts[1] - ends[0] == pytest.approx(0.2, abs=2 * engine.ctx.dt)
    finally:
        engine.shutdown()


def test_a_machine_error_raises_in_the_program_and_stops_the_run_naming_both():
    engine = _engine(PRESS, {"scenario_fixtures:Jammed": {}, "name": "scene"})
    try:
        with pytest.raises(ScenarioError, match=r"scenario scene: press.jam \(seq 1\): jammed"):
            for _ in range(1000):
                engine.step()
    finally:
        engine.shutdown()


def test_a_program_that_asks_for_the_status_decides_itself():
    engine = _engine(PRESS, {"scenario_fixtures:Jammed": {"check": False}, "name": "scene"})
    try:
        _until_done(engine)
        assert _events(engine, "scene")[0]["error"] == "jammed"
    finally:
        engine.shutdown()


def test_tasks_run_beside_each_other_join_and_race():
    engine = _engine(
        PRESS,
        {"machine_fixtures:Spinner": {"ramp": 0.5}, "name": "spin"},
        {"scenario_fixtures:Concurrent": {}, "name": "scene"},
    )
    try:
        _until_done(engine)
        scene = {e["event"]: e for e in _events(engine, "scene")}
        # The press dwells 0.3 s, the spinner ramps 0.3 m/s at 0.5 m/s^2 (0.6 s): both started in
        # the same step, so the join is the longer of the two.
        starts = [e["t"] for e in engine.ctx.events.all() if e["event"] == "program_start"]
        assert starts[0] == starts[1]
        assert scene["joined"]["t"] - starts[0] == pytest.approx(0.6, abs=3 * engine.ctx.dt)
        assert scene["won"]["by"] == "short" and scene["won"]["long_cancelled"]
        assert "short_done" in scene and "long_done" not in scene
    finally:
        engine.shutdown()


def test_a_route_is_waited_on_by_its_own_number():
    engine = _engine(
        {"scenario_fixtures:FakeWalker": {"route_s": 0.5, "error": "blocked"}, "name": "worker"},
        {"scenario_fixtures:Walk": {}, "name": "scene"},
    )
    try:
        _until_done(engine)
        arrived = _events(engine, "scene")[0]
        assert arrived["event"] == "arrived" and arrived["error"] == "blocked"
        assert arrived["t"] == pytest.approx(0.5, abs=3 * engine.ctx.dt)
    finally:
        engine.shutdown()


def test_a_handle_nothing_published_is_refused_at_reset_naming_the_key():
    with pytest.raises(ScenarioError, match="scenario scene: needs 'machine:press'"):
        _engine({"scenario_fixtures:TwoStrokes": {}, "name": "scene"})


def test_needs_may_rename_a_handle_and_refuses_one_the_scenario_does_not_use():
    renamed = {"machine_fixtures:Press": {"prefix": "p2_"}, "name": "press_2"}
    engine = _engine(
        renamed,
        {"scenario_fixtures:TwoStrokes": {"needs": {"press": "machine:press_2"}}, "name": "scene"},
    )
    try:
        _until_done(engine)
        assert [e["event"] for e in _events(engine, "press_2")].count("program_end") == 2
    finally:
        engine.shutdown()
    with pytest.raises(Exception, match="'needs' names \\['gripper'\\], which it does not use"):
        _engine(
            PRESS,
            {"scenario_fixtures:TwoStrokes": {"needs": {"gripper": "machine:x"}}, "name": "scene"},
        )


def test_stop_at_end_ends_a_standalone_run_and_a_reset_starts_the_scene_again():
    engine = _engine(
        PRESS, {"scenario_fixtures:TwoStrokes": {"stop_at_end": True}, "name": "scene"}
    )
    try:
        _until_done(engine)
        assert engine.ctx.stop_requested
        engine.reset()
        assert not engine.ctx.interface.find("scene", "done").read()
        _until_done(engine)
        assert [e["event"] for e in _events(engine, "scene")].count("scenario_end") == 2
    finally:
        engine.shutdown()


def test_two_runs_of_a_scene_give_the_same_timeline():
    def run():
        engine = _engine(PRESS, {"scenario_fixtures:TwoStrokes": {}, "name": "scene"})
        try:
            _until_done(engine)
            return [(e["t"], e["source"], e["event"]) for e in engine.ctx.events.all()]
        finally:
            engine.shutdown()

    assert run() == run()
