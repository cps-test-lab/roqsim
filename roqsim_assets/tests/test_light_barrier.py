"""A light barrier at a conveyor's end: a zone_trigger reports the package, and the scenario stops the
belt while the package is in it, takes it off the belt as a worker would, and runs the belt again."""

from __future__ import annotations

from pathlib import Path

import pytest

import roqsim_assets
from roqsim.config import load_config
from roqsim.engine import Engine

WORLD = Path(roqsim_assets.__file__).parent / "worlds" / "light_barrier_demo.yaml"


def _engine(**scene):
    overrides = {"components": {"barrier_stop": scene}} if scene else None
    engine = Engine(load_config(WORLD, overrides=overrides))
    engine.ctx.seed = 1
    engine.setup()
    engine.reset()
    return engine


def _run(engine, seconds):
    end = engine.ctx.sim_time + seconds
    while engine.ctx.sim_time < end - 1e-9:
        engine.step()


def _package_x(engine):
    return float(engine.ctx.data.xipos[engine.ctx.model.body("package").id][0])


def _events(engine):
    return [
        (e["source"], e["event"])
        for e in engine.ctx.events.all()
        if e["source"] in ("belt_end", "barrier_stop") or e["event"] == "program_start"
    ]


def test_the_belt_stops_while_the_package_is_in_the_barrier_and_runs_once_it_is_taken():
    engine = _engine(hold=2.0)
    try:
        belt = engine.ctx.blackboard.require("machine:belt")
        barrier = engine.ctx.blackboard.require("zone:belt_end")
        _run(engine, 2.5)
        assert barrier.inside() == ["package"] and belt.value("speed") == 0.0
        entered = next(e["t"] for e in engine.ctx.events.all() if e["event"] == "entered")
        stopped = next(e["t"] for e in engine.ctx.events.all() if e["event"] == "stopped")
        # The stop ramps 0.2 m/s down at 0.5 m/s^2: 0.4 s after the step that applied it.
        assert stopped - entered == pytest.approx(0.4, abs=0.02)
        held = _package_x(engine)
        _run(engine, 0.5)
        assert _package_x(engine) == pytest.approx(held, abs=0.002), "the belt stays stopped"
        _run(engine, 2.0)  # the worker takes the package 2 s after the belt stopped
        assert barrier.inside() == [] and belt.value("speed") == pytest.approx(0.2)
        assert _events(engine) == [
            ("belt_end", "entered"),
            ("belt", "program_start"),
            ("barrier_stop", "held"),
            ("belt_end", "left"),
            ("belt", "program_start"),
            ("barrier_stop", "released"),
            ("barrier_stop", "scenario_end"),  # one cycle
        ]
        assert engine.ctx.stop_requested, "and the run with it: roqsim sim exits"
    finally:
        engine.shutdown()


def test_without_a_worker_the_belt_stays_stopped_until_someone_takes_the_package():
    from roqsim.entity_control import set_state

    engine = _engine(take_to={})
    try:
        belt = engine.ctx.blackboard.require("machine:belt")
        _run(engine, 5.0)
        assert belt.value("speed") == 0.0, "nobody took it"
        set_state(engine.ctx, "package", [-0.5, -1.0, 0.1], [1.0, 0.0, 0.0, 0.0])
        _run(engine, 1.0)
        assert belt.value("speed") == pytest.approx(0.2)
    finally:
        engine.shutdown()


def test_the_belt_comes_back_at_the_speed_it_was_re_speeded_to_between_two_packages():
    from roqsim.entity_control import set_state

    engine = _engine(cycles=2)
    try:
        belt = engine.ctx.blackboard.require("machine:belt")
        start = engine.ctx.data.xipos[engine.ctx.model.body("package").id].copy()

        def released():
            return [e for e in engine.ctx.events.all() if e["event"] == "released"]

        while not released():
            assert engine.ctx.sim_time < 10.0
            engine.step()
        _run(engine, 1.0)
        assert belt.value("speed") == pytest.approx(0.2)
        # Something else re-speeds the belt; then the next package arrives before the barrier.
        seq = belt.start("run", speed=0.3)
        while not (belt.status().seq == seq and belt.status().finished):
            engine.step()
        set_state(engine.ctx, "package", start.tolist(), [1.0, 0.0, 0.0, 0.0])
        while len(released()) < 2:
            assert engine.ctx.sim_time < 25.0
            engine.step()
        _run(engine, 1.0)
        assert belt.value("speed") == pytest.approx(0.3), "back at the speed it had, not 0.2"
        assert [e["event"] for e in engine.ctx.events.all() if e["event"] == "held"] == [
            "held",
            "held",
        ]
        assert engine.ctx.stop_requested
    finally:
        engine.shutdown()


def _released(engine):
    return [e for e in engine.ctx.events.all() if e["event"] == "released"]


def test_a_belt_stopped_while_ramping_comes_back_at_the_speed_it_was_ramping_to():
    engine = Engine(
        load_config(
            WORLD,
            overrides={"components": {"belt": {"ramp": 0.2}, "barrier_stop": {"cycles": 1}}},
        )
    )
    engine.ctx.seed = 1
    engine.setup()
    engine.reset()
    try:
        belt = engine.ctx.blackboard.require("machine:belt")
        _run(engine, 0.2)
        belt.start("run", speed=1.0)  # a 4 s ramp: the package reaches the barrier long before
        while not _released(engine):
            assert engine.ctx.sim_time < 15.0
            engine.step()
        runs = [
            e
            for e in engine.ctx.events.all()
            if e["source"] == "belt" and e["event"] == "program_start" and e["program"] == "run"
        ]
        # The re-speed was cancelled for the stop, and the scene ran the belt on to where it was
        # going, not to the speed it had reached when the part arrived.
        assert [r["args"]["speed"] for r in runs] == [1.0, 1.0]
        assert belt.value("speed") == pytest.approx(1.0)
    finally:
        engine.shutdown()


def test_a_belt_that_stood_still_when_the_part_arrived_stays_still():
    from roqsim.entity_control import set_state

    engine = _engine(cycles=1)
    try:
        belt = engine.ctx.blackboard.require("machine:belt")
        seq = belt.start("stop")
        while not (belt.status().seq == seq and belt.status().finished):
            engine.step()
        set_state(engine.ctx, "package", [-0.9, 0.6, 1.0], [1.0, 0.0, 0.0, 0.0])
        while not _released(engine):
            assert engine.ctx.sim_time < 10.0
            engine.step()
        _run(engine, 1.0)
        assert belt.goal("speed") == 0.0 and belt.value("speed") == 0.0, "nobody started it"
    finally:
        engine.shutdown()
