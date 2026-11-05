# Copyright (C) 2026 Frederik Pasch
#
# SPDX-License-Identifier: Apache-2.0

"""The machine base: programs on simulated time, ramps, waits, events and the one handle.

The press here has a guard door and a ram, each a position servo, and its stroke is a program: the
ram moves only behind the closed door, each phase at its speed, every step an event. The handle's
contract is checked beside it: a sequence number per start, a refused start while busy, a status per
start. The one-axis machine checks the base's refusals and waits on an actuator the world states.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest
from machine_fixtures import APPROACH, BOTTOM, CONTACT, DOOR_CLOSED, DOOR_SPEED, POWER

from roqsim.config import load_config_from_dict
from roqsim.engine import Engine
from roqsim.introspection import undeclared_config_reads
from roqsim.machine import MachineError, MachineStatus


def _engine(components, gravity=True):
    sim = {} if gravity else {"gravity": [0, 0, 0]}
    engine = Engine(load_config_from_dict({"sim": sim, "components": components}))
    engine.ctx.seed = 1
    engine.setup()
    engine.reset()
    return engine


def _press(**config):
    return _engine(
        [{"machine_fixtures:Press": {"prefix": "p1_", **config}, "name": "press"}], gravity=False
    )


def _joint(engine, name):
    m = engine.ctx.model
    return float(engine.ctx.data.qpos[m.jnt_qposadr[m.joint(name).id]])


def _run_until_done(engine, handle, seq, limit=20.0):
    t0 = engine.ctx.sim_time
    while not (handle.status().seq == seq and handle.status().finished):
        assert engine.ctx.sim_time - t0 < limit, f"seq {seq} never finished: {handle.status()}"
        engine.step()
    return handle.status()


def _events(engine, source="press"):
    return [e for e in engine.ctx.events.all() if e["source"] == source]


def test_a_start_returns_its_number_runs_the_program_and_refuses_a_second_while_busy():
    engine = _press()
    try:
        press = engine.ctx.blackboard.require("machine:press")
        for _ in range(200):
            engine.step()
        assert press.ready and _joint(engine, "p1_door_joint") == pytest.approx(0, abs=1e-4)
        seq = press.start("stroke")
        assert seq == 1 and press.busy and not press.ready
        with pytest.raises(MachineError, match="busy running 'stroke'"):
            press.start("stroke")
        t0 = engine.ctx.sim_time
        status = _run_until_done(engine, press, seq)
        assert status == MachineStatus(seq=1, program="stroke", finished=True, error="")
        assert not press.busy and press.ready
        # Two door moves of 2 s, the stroke's three phases (0.47 s, 0.2 s, 0.5 s), plus settling.
        assert 5.1 < engine.ctx.sim_time - t0 < 5.6
        assert press.start("stroke") == 2
    finally:
        engine.shutdown()


def test_the_ram_moves_only_while_the_door_is_closed_and_each_phase_at_its_speed():
    engine = _press()
    try:
        press = engine.ctx.blackboard.require("machine:press")
        seq = press.start("stroke")
        m, d = engine.ctx.model, engine.ctx.data
        door_dof = m.jnt_dofadr[m.joint("p1_door_joint").id]
        ram_dof = m.jnt_dofadr[m.joint("p1_ram_joint").id]
        door_peak = ram_fast = ram_slow = 0.0
        while not (press.status().seq == seq and press.status().finished):
            engine.step()
            door, ram = _joint(engine, "p1_door_joint"), _joint(engine, "p1_ram_joint")
            v_ram = abs(float(d.qvel[ram_dof]))
            if v_ram > 1e-3:
                assert door == pytest.approx(DOOR_CLOSED, abs=0.003), (
                    "the ram moved with the door open"
                )
            door_peak = max(door_peak, abs(float(d.qvel[door_dof])))
            down = float(d.qvel[ram_dof]) > 0
            if down and ram < CONTACT - 0.01:
                ram_fast = max(ram_fast, v_ram)
            elif down and ram > CONTACT + 0.002:
                ram_slow = max(ram_slow, v_ram)
        assert door_peak == pytest.approx(DOOR_SPEED, rel=0.1)
        assert ram_fast == pytest.approx(APPROACH, rel=0.1)
        assert ram_slow == pytest.approx(POWER, rel=0.15)
    finally:
        engine.shutdown()


def test_each_step_is_an_event_between_a_start_and_an_end_carrying_the_sequence_number():
    engine = _press()
    try:
        press = engine.ctx.blackboard.require("machine:press")
        seq = press.start("stroke", depth=BOTTOM)
        _run_until_done(engine, press, seq)
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
        assert all(e["program"] == "stroke" and e["seq"] == seq for e in events)
        assert events[0]["args"] == {"depth": BOTTOM} and events[-1]["error"] == ""
        assert [e["t"] for e in events] == sorted(e["t"] for e in events)
    finally:
        engine.shutdown()


def test_an_unknown_program_or_argument_is_refused_at_the_call_naming_the_machine():
    engine = _press()
    try:
        press = engine.ctx.blackboard.require("machine:press")
        with pytest.raises(
            MachineError, match="machine press: no program 'bend'; it has dwell, early, jam, stroke"
        ):
            press.start("bend")
        with pytest.raises(MachineError, match="program 'stroke'.*depht"):
            press.start("stroke", depht=0.1)
        with pytest.raises(MachineError, match="program 'dwell'.*seconds"):
            press.start("dwell")
        with pytest.raises(MachineError, match="depth is float, not 'deep'"):
            press.start("stroke", depth="deep")
        assert not press.busy and press.status().seq == 0
    finally:
        engine.shutdown()


def test_a_failure_is_the_programs_outcome_in_its_status_and_the_run_goes_on():
    engine = _press()
    try:
        press = engine.ctx.blackboard.require("machine:press")
        status = _run_until_done(engine, press, press.start("jam"))
        assert status.finished and status.error == "jammed"
        assert _events(engine)[-1]["error"] == "jammed"
        assert press.ready
    finally:
        engine.shutdown()


def test_a_program_that_raises_stops_the_run_naming_it_unless_told_to_report():
    engine = _press(fail_at="power")
    try:
        press = engine.ctx.blackboard.require("machine:press")
        press.start("stroke")
        with pytest.raises(
            RuntimeError, match=r"machine press: program 'stroke' \(seq 1\) raised: a die"
        ):
            for _ in range(5000):
                engine.step()
        assert press.status().error == "ValueError: a die that is not there"
    finally:
        engine.shutdown()
    engine = _press(fail_at="power", on_error="report")
    try:
        press = engine.ctx.blackboard.require("machine:press")
        status = _run_until_done(engine, press, press.start("stroke"))
        assert status.error == "ValueError: a die that is not there"
    finally:
        engine.shutdown()


def test_a_program_raising_before_its_first_yield_stops_the_run_too():
    # Its first segment runs where the start is applied, in the command drain, which only logs an
    # exception: the base carries it to the next pre_step.
    engine = _press()
    try:
        press = engine.ctx.blackboard.require("machine:press")
        press.start("early")
        with pytest.raises(RuntimeError, match=r"program 'early' \(seq 1\) raised: at once"):
            engine.step()
            engine.step()
    finally:
        engine.shutdown()


def test_cancel_ends_the_program_where_it_is_and_the_machine_holds():
    engine = _press()
    try:
        press = engine.ctx.blackboard.require("machine:press")
        seq = press.start("stroke")
        for _ in range(250):
            engine.step()
        assert press.cancel() == seq
        assert not press.busy, "a cancel frees the machine at the call"
        engine.step()
        assert press.status() == MachineStatus(seq, "stroke", True, "cancelled")
        held = press.value("door")
        assert 0.0 < held < DOOR_CLOSED
        for _ in range(200):
            engine.step()
        assert _joint(engine, "p1_door_joint") == pytest.approx(held, abs=0.003)
        assert _events(engine)[-1]["event"] == "program_end"
        assert press.cancel() == 0
    finally:
        engine.shutdown()


def test_a_reset_drops_the_program_and_puts_the_machine_at_home():
    engine = _press(home={"ram": 0.05})
    try:
        press = engine.ctx.blackboard.require("machine:press")
        assert _joint(engine, "p1_ram_joint") == pytest.approx(0.05)
        seq = press.start("stroke")
        for _ in range(300):
            engine.step()
        engine.reset()
        assert press.status() == MachineStatus() and not press.busy
        assert _joint(engine, "p1_ram_joint") == pytest.approx(0.05)
        assert _joint(engine, "p1_door_joint") == pytest.approx(0.0)
        assert press.start("stroke") == seq + 1, (
            "numbers keep counting, so a stale one never matches"
        )
    finally:
        engine.shutdown()


@pytest.mark.parametrize(
    "home, said",
    [
        ({"rma": 0.05}, "home names 'rma', which it does not drive"),
    ],
)
def test_a_home_the_machine_does_not_drive_is_refused_when_the_world_loads(home, said):
    with pytest.raises(Exception, match=said):
        _engine([{"machine_fixtures:Press": {"prefix": "p1_", "home": home}, "name": "press"}])


def test_a_home_outside_the_actuators_range_is_refused_naming_the_machine():
    with pytest.raises(MachineError, match=r"machine press: home 'ram' is 5.0, outside its range"):
        _engine(
            [{"machine_fixtures:Press": {"prefix": "p1_", "home": {"ram": 5.0}}, "name": "press"}]
        )


def test_an_actuator_the_model_lacks_is_refused_naming_what_it_has():
    from machine_fixtures import Press

    class Lacking(Press):
        ACTUATORS = ("door", "ram", "clamp")

    import machine_fixtures

    machine_fixtures.Lacking = Lacking
    try:
        with pytest.raises(
            MachineError,
            match=r"it drives 'clamp' \('p1_clamp'\), which the model "
            r"lacks \(its actuators: p1_door, p1_ram\)",
        ):
            _engine([{"machine_fixtures:Lacking": {"prefix": "p1_"}, "name": "press"}])
    finally:
        del machine_fixtures.Lacking


def test_a_setpoint_ramps_at_its_rate_and_the_plugin_applies_it():
    engine = _engine([{"machine_fixtures:Spinner": {"ramp": 0.5}, "name": "spin"}])
    try:
        spin = engine.ctx.blackboard.require("machine:spin")
        plugin = spin._machine
        assert spin.value("speed") == pytest.approx(0.1)
        t0 = engine.ctx.sim_time
        status = _run_until_done(engine, spin, spin.start("run", speed=0.6))
        assert status.error == ""
        assert engine.ctx.sim_time - t0 == pytest.approx(1.0, abs=0.02)
        assert spin.value("speed") == pytest.approx(0.6)
        assert plugin.applied[-1] == pytest.approx(0.6)
        steps = [
            b - a for a, b in zip(plugin.applied[:-1], plugin.applied[1:], strict=True) if b != a
        ]
        assert max(steps) == pytest.approx(0.5 * engine.ctx.dt, rel=1e-6)
    finally:
        engine.shutdown()


def test_manual_control_leaves_the_actuators_to_the_viewer():
    engine = _press()
    try:
        engine.ctx.manual_control = True
        press = engine.ctx.blackboard.require("machine:press")
        press.start("dwell", seconds=0.5)
        engine.ctx.data.ctrl[:] = 0.03
        for _ in range(20):
            engine.step()
        assert list(engine.ctx.data.ctrl) == [0.03, 0.03]
    finally:
        engine.shutdown()


def test_a_start_from_another_thread_is_applied_at_the_next_step():
    engine = _press()
    try:
        press = engine.ctx.blackboard.require("machine:press")
        got = []
        thread = threading.Thread(target=lambda: got.append(press.start("dwell", seconds=0.1)))
        thread.start()
        thread.join()
        assert got == [1] and press.busy
        assert press.status().seq == 0, "not applied before the physics thread drains it"
        engine.step()
        assert press.status() == MachineStatus(1, "dwell", False, "")
    finally:
        engine.shutdown()


def test_two_runs_give_the_same_timeline():
    def run():
        engine = _press()
        try:
            press = engine.ctx.blackboard.require("machine:press")
            _run_until_done(engine, press, press.start("stroke"))
            return [(e["t"], e["event"]) for e in _events(engine)], list(engine.ctx.data.qpos)
        finally:
            engine.shutdown()

    assert run() == run()


def test_the_endpoints_start_report_and_cancel_through_the_interface():
    engine = _press()
    try:
        start = engine.ctx.interface.find("press", "start")
        status = engine.ctx.interface.find("press", "status")
        future = start.write({"program": "dwell", "args": {"seconds": 0.05}})
        engine.step()
        assert future.result(1.0) == 1
        for _ in range(40):
            engine.step()
        assert status.read() == MachineStatus(1, "dwell", True, "")
        refused = start.write({"program": "bend"})
        engine.step()
        with pytest.raises(MachineError, match="no program 'bend'"):
            refused.result(1.0)
    finally:
        engine.shutdown()


def test_describe_publishes_the_programs_with_their_parameters():
    programs = {p["name"]: p for p in __import__("machine_fixtures").Press.describe_programs()}
    assert set(programs) == {"stroke", "jam", "dwell", "early"}
    assert programs["stroke"]["doc"].startswith("Close the guard")
    assert programs["stroke"]["parameters"] == [
        {"name": "depth", "type": "float", "default": BOTTOM}
    ]
    assert programs["dwell"]["parameters"] == [{"name": "seconds", "type": "float"}]
    assert undeclared_config_reads(__import__("machine_fixtures").Press) == []


# -- handles a machine needs, and preemption -----------------------------------------------------
def test_a_handle_the_machine_needs_is_refused_by_name_when_nothing_published_it():
    with pytest.raises(MachineError, match="machine press: needs 'machine:feeder', which no entry"):
        _engine([{"machine_fixtures:FedPress": {"prefix": "p1_"}, "name": "press"}], gravity=False)
    engine = _engine(
        [
            {"machine_fixtures:Press": {"prefix": "f_"}, "name": "feeder"},
            {"machine_fixtures:FedPress": {"prefix": "p1_"}, "name": "press"},
        ],
        gravity=False,
    )
    try:
        press = engine.ctx.blackboard.require("machine:press")._machine
        assert press.feeder is engine.ctx.blackboard.require("machine:feeder")
    finally:
        engine.shutdown()


def test_preempt_cancels_what_runs_and_starts_the_new_program_in_the_same_step():
    engine = _press()
    try:
        press = engine.ctx.blackboard.require("machine:press")
        first = press.start("dwell", seconds=1.0)
        for _ in range(10):
            engine.step()
        t = engine.ctx.sim_time
        second = press._machine.preempt("dwell", seconds=0.1)
        assert second == first + 1
        assert press.status() == MachineStatus(second, "dwell", False, "")
        events = [(e["event"], e["seq"], e.get("error")) for e in _events(engine)][-2:]
        assert events == [("program_end", first, "cancelled"), ("program_start", second, None)]
        assert all(e["t"] == t for e in _events(engine)[-2:])
        assert _run_until_done(engine, press, second).error == ""
        with pytest.raises(MachineError, match="no program 'bend'"):
            press._machine.preempt("bend")
    finally:
        engine.shutdown()


# -- a controller with no model of its own, under the spawn that placed the model -----------------
def test_a_controller_nested_under_spawn_model_drives_that_entity_and_speaks_for_it(tmp_path):
    from machine_fixtures import PRESS

    (tmp_path / "press.xml").write_text(PRESS)
    spawn = {
        "spawn_model": {
            "model": str(tmp_path / "press.xml"),
            "prefix": "p1_",
            "motion": "static",
            "pose": {"position": {"x": 0.0, "y": 0.0, "z": 0.0}},
        },
        "name": "press_1",
        "components": [{"machine_fixtures:PressController": {}}],
    }
    engine = _engine([spawn], gravity=False)
    try:
        handle = engine.ctx.blackboard.require("machine:press_1")
        assert engine.ctx.blackboard.get("machine:press_1.PressController") is None
        assert engine.ctx.interface.find("press_1", "start") is not None
        assert _run_until_done(engine, handle, handle.start("close")).error == ""
        assert _joint(engine, "p1_door_joint") == pytest.approx(DOOR_CLOSED, abs=0.003)
        assert [e["source"] for e in engine.ctx.events.all() if e["event"] == "program_end"] == [
            "press_1"
        ]
    finally:
        engine.shutdown()


def test_plugins_describe_publishes_a_machines_programs(monkeypatch):
    import machine_fixtures

    from roqsim import introspection

    fake = SimpleNamespace(
        name="test_press", value="machine_fixtures:Press", load=lambda: machine_fixtures.Press
    )
    monkeypatch.setattr(introspection, "entry_points", lambda group: [fake])
    details = introspection.get_plugin_details("test_press")
    assert {p["name"] for p in details["programs"]} == {"stroke", "jam", "dwell", "early"}
    assert {"start", "status", "cancel"} <= {row["name"] for row in details["endpoints"]}


# -- the actuator a machine drives ---------------------------------------------------------------
def _axis(**config):
    return _engine([{"machine_fixtures:Axis": config, "name": "axis"}], gravity=False)


@pytest.mark.parametrize(
    ("actuator", "said"),
    [
        ('<motor name="x" joint="x_joint" ctrlrange="-1 1"/>', "no affine bias"),
        ('<velocity name="x" joint="x_joint" kv="100" ctrlrange="-1 1"/>', "bias stiffness"),
        (
            '<position name="x" joint="x_joint" kp="2000" kv="100" gear="2" ctrlrange="-1 1"/>',
            "its gear is 2, not 1",
        ),
    ],
)
def test_an_actuator_that_is_no_plain_position_servo_is_refused_naming_the_machine(actuator, said):
    with pytest.raises(
        MachineError, match=f"machine axis: actuator 'x' is no plain position servo.*{said}"
    ):
        _axis(actuator=actuator)


def test_a_target_outside_the_actuators_range_fails_the_program_and_moves_nothing():
    engine = _axis()
    try:
        axis = engine.ctx.blackboard.require("machine:axis")
        for speed in (0.0, 0.5):
            status = _run_until_done(engine, axis, axis.start("go", x=5.0, speed=speed))
            assert status.error == "machine axis: target 5 for 'x' is outside its range [-1, 1]"
        assert axis.value("x") == 0.0
        assert _joint(engine, "x_joint") == pytest.approx(0.0, abs=1e-6)
        assert not [e for e in _events(engine, "axis") if e["event"] == "go"], "no step began"
    finally:
        engine.shutdown()


# -- failure holds, and a move does not hang ----------------------------------------------------
@pytest.mark.parametrize(
    ("program", "config", "error"),
    [
        ("go_then_fail", {}, "guard open"),
        ("go_then_raise", {"on_error": "report"}, "ValueError: broken"),
    ],
)
def test_a_program_that_fails_mid_ramp_leaves_the_machine_holding_as_a_cancel_does(
    program, config, error
):
    engine = _axis(**config)
    try:
        axis = engine.ctx.blackboard.require("machine:axis")
        status = _run_until_done(engine, axis, axis.start(program, x=0.8, speed=0.2, after=0.5))
        assert status.error == error
        held = axis.value("x")
        assert held == pytest.approx(0.1, abs=0.002), "the ramp stopped where the program ended"
        for _ in range(1000):
            engine.step()
        assert axis.value("x") == held
        assert _joint(engine, "x_joint") == pytest.approx(held, abs=0.003)
    finally:
        engine.shutdown()
