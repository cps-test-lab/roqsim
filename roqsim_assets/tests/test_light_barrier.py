"""A light barrier at a conveyor's end: a zone_trigger reports the package, and the scenario stops the
belt while the package is in it and runs it again once the package is taken away."""

from __future__ import annotations

from pathlib import Path

import pytest

import roqsim_assets
from roqsim.config import load_config
from roqsim.engine import Engine

WORLD = Path(roqsim_assets.__file__).parent / "worlds" / "light_barrier_demo.yaml"


def _engine():
    # The package starts 0.4 m before the barrier rather than at the feed end, so it arrives in 2 s.
    engine = Engine(
        load_config(
            WORLD,
            overrides={
                "components": {
                    "belt": {"package_pose": {"position": {"x": -0.5, "y": 0.6, "z": 0.996}}}
                }
            },
        )
    )
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


def test_the_belt_stops_while_the_package_is_in_the_barrier_and_runs_once_it_is_gone():
    engine = _engine()
    try:
        belt = engine.ctx.blackboard.require("machine:belt")
        barrier = engine.ctx.blackboard.require("zone:belt_end")
        _run(engine, 4.0)
        assert barrier.inside() == ["package"]
        assert belt.value("speed") == 0.0
        entered = next(e["t"] for e in engine.ctx.events.all() if e["event"] == "entered")
        stopped = next(e["t"] for e in engine.ctx.events.all() if e["event"] == "stopped")
        # The stop ramps 0.2 m/s down at 0.5 m/s^2: 0.4 s after the step that applied it.
        assert stopped - entered == pytest.approx(0.4, abs=0.02)
        held = _package_x(engine)
        _run(engine, 2.0)
        assert _package_x(engine) == pytest.approx(held, abs=0.002), "the belt stays stopped"
        # A worker takes the package away.
        from roqsim.entity_control import set_state

        set_state(engine.ctx, "package", [-0.5, -1.0, 0.1], [1.0, 0.0, 0.0, 0.0])
        _run(engine, 1.0)
        assert barrier.inside() == [] and belt.value("speed") == pytest.approx(0.2)
        events = [
            (e["source"], e["event"])
            for e in engine.ctx.events.all()
            if e["source"] in ("belt_end", "barrier_stop") or e["event"] == "program_start"
        ]
        assert events == [
            ("belt_end", "entered"),
            ("belt", "program_start"),
            ("barrier_stop", "held"),
            ("belt_end", "left"),
            ("belt", "program_start"),
            ("barrier_stop", "released"),
        ]
    finally:
        engine.shutdown()
