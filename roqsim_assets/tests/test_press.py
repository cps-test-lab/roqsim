"""The example press: a machine whose ram strokes only behind its closed guard door, changing the
state of the part in its die at the stroke's bottom; and the scenario that places a part and starts it."""

from __future__ import annotations

from pathlib import Path

import pytest

import roqsim_assets
from roqsim.config import load_config, load_config_from_dict
from roqsim.engine import Engine
from roqsim.machine import MachineError, MachineStatus

WORLDS = Path(roqsim_assets.__file__).parent / "worlds"
STRIP = str(WORLDS / "press_demo_strip.xml")
DIE = {"position": {"x": 0.0, "y": 0.0, "z": 0.945}}


def _engine(*components, strip=True, part_state=True):
    parts = [
        {"press": {"prefix": "p1_"}, "name": "press_1"},
        *(
            [
                {
                    "spawn_model": {
                        "model": STRIP,
                        "prefix": "s1_",
                        "pose": {"position": {"x": 0.0, "y": 0.0, "z": 0.945}},
                    },
                    "name": "strip_1",
                }
            ]
            if strip
            else []
        ),
        *([{"part_state": {}}] if part_state else []),
        *components,
    ]
    engine = Engine(load_config_from_dict({"sim": {}, "components": parts}))
    engine.ctx.seed = 1
    engine.setup()
    engine.reset()
    return engine


def _joint(engine, name):
    m = engine.ctx.model
    return float(engine.ctx.data.qpos[m.jnt_qposadr[m.joint(name).id]])


def _finish(engine, handle, seq, limit=15.0):
    t0 = engine.ctx.sim_time
    while not (handle.status().seq == seq and handle.status().finished):
        assert engine.ctx.sim_time - t0 < limit, handle.status()
        engine.step()
    return handle.status()


def _events(engine, source="press_1"):
    return [e for e in engine.ctx.events.all() if e["source"] == source]


def test_nothing_moves_before_a_start_and_a_second_start_is_refused():
    engine = _engine()
    try:
        press = engine.ctx.blackboard.require("machine:press_1")
        for _ in range(300):
            engine.step()
        assert press.ready
        assert _joint(engine, "p1_ram_slide") == pytest.approx(0.0, abs=1e-3)
        assert _joint(engine, "p1_door_slide") == pytest.approx(0.0, abs=1e-3)
        seq = press.start("stroke")
        with pytest.raises(MachineError, match="busy running 'stroke'"):
            press.start("stroke")
        assert _finish(engine, press, seq) == MachineStatus(seq, "stroke", True, "")
        assert press.ready
    finally:
        engine.shutdown()


def test_the_ram_moves_only_behind_the_closed_guard_each_phase_at_its_speed():
    engine = _engine()
    try:
        press = engine.ctx.blackboard.require("machine:press_1")
        m, d = engine.ctx.model, engine.ctx.data
        ram_dof = m.jnt_dofadr[m.joint("p1_ram_slide").id]
        seq = press.start("stroke")
        fast = slow = 0.0
        t0 = engine.ctx.sim_time
        while not (press.status().seq == seq and press.status().finished):
            engine.step()
            v = abs(float(d.qvel[ram_dof]))
            if v > 1e-3:
                assert _joint(engine, "p1_door_slide") == pytest.approx(0.5, abs=0.003), (
                    "the ram moved with the guard open"
                )
            ram = _joint(engine, "p1_ram_slide")
            if float(d.qvel[ram_dof]) > 0:
                if ram < 0.185:
                    fast = max(fast, v)
                elif ram > 0.197:
                    slow = max(slow, v)
        assert fast == pytest.approx(0.2, rel=0.1)
        assert slow == pytest.approx(0.03, rel=0.15)
        # Two guard moves of 2 s, approach 0.98 s, power 0.67 s, return 1.08 s, plus settling.
        assert 6.7 < engine.ctx.sim_time - t0 < 7.3
    finally:
        engine.shutdown()


def test_the_part_changes_at_the_strokes_bottom_and_every_step_is_an_event():
    engine = _engine()
    try:
        press = engine.ctx.blackboard.require("machine:press_1")
        states = engine.ctx.blackboard.require("part_state")
        for _ in range(100):
            engine.step()
        seq = press.start("stroke", show="hole")
        while not any(e["event"] == "power" for e in _events(engine)):
            engine.step()
        assert states.shown("strip_1") == [], "not before the bottom"
        _finish(engine, press, seq)
        assert states.shown("strip_1") == ["hole"]
        events = _events(engine)
        assert [e["event"] for e in events] == [
            "program_start",
            "door_close",
            "approach",
            "power",
            "bottom",
            "return",
            "door_open",
            "program_end",
        ]
        bottom = next(e for e in events if e["event"] == "bottom")
        shown = next(e for e in engine.ctx.events.all() if e["event"] == "shown")
        assert bottom["part"] == "strip_1" and shown["t"] == bottom["t"]
    finally:
        engine.shutdown()


def test_a_stroke_that_changes_a_part_with_an_empty_die_fails_and_the_run_goes_on():
    engine = _engine(strip=False)
    try:
        press = engine.ctx.blackboard.require("machine:press_1")
        status = _finish(engine, press, press.start("stroke", show="hole"))
        assert (
            status.error == "a stroke that changes a part needs one part in the die; it holds none"
        )
        assert press.ready
        assert _finish(engine, press, press.start("stroke")).error == "", "an empty stroke is fine"
    finally:
        engine.shutdown()


def test_a_stroke_that_changes_a_part_without_part_state_stops_the_run_naming_it():
    engine = _engine(part_state=False)
    try:
        press = engine.ctx.blackboard.require("machine:press_1")
        press.start("stroke", show="hole")
        with pytest.raises(RuntimeError, match="needs a part_state entry in the world"):
            for _ in range(10):
                engine.step()
    finally:
        engine.shutdown()


def test_cancel_holds_the_ram_and_a_reset_puts_the_press_home():
    engine = _engine()
    try:
        press = engine.ctx.blackboard.require("machine:press_1")
        seq = press.start("stroke")
        while not any(e["event"] == "approach" for e in _events(engine)):
            engine.step()
        for _ in range(100):
            engine.step()
        assert press.cancel() == seq
        engine.step()
        assert press.status().error == "cancelled" and not press.ready
        held = press.value("ram")
        for _ in range(300):
            engine.step()
        assert _joint(engine, "p1_ram_slide") == pytest.approx(held, abs=0.003)
        engine.reset()
        assert press.status() == MachineStatus() and press.ready
        assert _joint(engine, "p1_ram_slide") == 0.0 and _joint(engine, "p1_door_slide") == 0.0
    finally:
        engine.shutdown()


def test_the_guard_programs_and_bad_config():
    engine = _engine()
    try:
        press = engine.ctx.blackboard.require("machine:press_1")
        _finish(engine, press, press.start("close_guard"))
        assert _joint(engine, "p1_door_slide") == pytest.approx(0.5, abs=0.003)
        _finish(engine, press, press.start("open_guard"))
        assert _joint(engine, "p1_door_slide") == pytest.approx(0.0, abs=0.003)
    finally:
        engine.shutdown()
    with pytest.raises(Exception, match="0 < ram_contact < ram_bottom"):
        Engine(load_config_from_dict({"components": [{"press": {"ram_contact": 0.3}}]}))


def test_the_demo_world_places_the_strip_punches_it_and_ends():
    engine = Engine(load_config(WORLDS / "press_demo.yaml"))
    engine.ctx.seed = 1
    engine.setup()
    engine.reset()
    try:
        while not engine.ctx.stop_requested:
            assert engine.ctx.sim_time < 15.0, "the demo never ended"
            engine.step()
        assert engine.ctx.blackboard.require("part_state").shown("strip_1") == ["hole"]
        demo = [e["event"] for e in _events(engine, "demo")]
        assert demo == ["punched", "scenario_end"]
    finally:
        engine.shutdown()


def test_the_ram_starts_down_only_once_the_guard_has_come_to_rest():
    """The interlock: closed is not enough, the guard must have stopped moving."""
    engine = _engine()
    try:
        press = engine.ctx.blackboard.require("machine:press_1")
        m, d = engine.ctx.model, engine.ctx.data
        door_dof = m.jnt_dofadr[m.joint("p1_door_slide").id]
        press.start("stroke")
        door_speed_at_approach = None
        for _ in range(4000):
            engine.step()
            if any(e["event"] == "approach" for e in _events(engine)):
                door_speed_at_approach = abs(float(d.qvel[door_dof]))
                break
        assert door_speed_at_approach is not None
        assert door_speed_at_approach <= press._machine.STOPPED_SPEED
    finally:
        engine.shutdown()


def test_open_guard_after_a_cancelled_stroke_returns_the_ram_home_before_the_guard_moves():
    engine = _engine()
    try:
        press = engine.ctx.blackboard.require("machine:press_1")
        press.start("stroke")
        while not any(e["event"] == "power" for e in _events(engine)):
            engine.step()
        for _ in range(100):
            engine.step()
        press.cancel()
        engine.step()
        assert _joint(engine, "p1_ram_slide") > 0.15, "cancelled with the ram down"
        seq = press.start("open_guard")
        ram_when_guard_moved = None
        door_closed = _joint(engine, "p1_door_slide")
        while not (press.status().seq == seq and press.status().finished):
            engine.step()
            if (
                ram_when_guard_moved is None
                and _joint(engine, "p1_door_slide") < door_closed - 0.005
            ):
                ram_when_guard_moved = _joint(engine, "p1_ram_slide")
        assert press.status().error == ""
        assert ram_when_guard_moved is not None and ram_when_guard_moved == pytest.approx(
            0.0, abs=0.003
        )
        assert _joint(engine, "p1_door_slide") == pytest.approx(0.0, abs=0.003)
    finally:
        engine.shutdown()
