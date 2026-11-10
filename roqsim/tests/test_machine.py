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

import math
import threading
from types import SimpleNamespace

import pytest
from machine_fixtures import APPROACH, BOTTOM, CONTACT, DOOR_CLOSED, DOOR_SPEED, POWER

from roqsim import machine as machine_module
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


def test_a_numbered_cancel_of_an_ended_start_leaves_a_later_start_running():
    engine = _press()
    try:
        press = engine.ctx.blackboard.require("machine:press")
        first = press.start("dwell", seconds=0.01)
        _run_until_done(engine, press, first)
        second = press.start("dwell", seconds=1.0)
        engine.step()
        assert press.cancel(first) == 0
        assert press.cancel(second + 1) == 0, "a start the machine never made"
        engine.step()
        assert press.busy and press.status() == MachineStatus(second, "dwell", False, "")
    finally:
        engine.shutdown()


def test_a_numbered_cancel_of_a_pending_start_cancels_it():
    engine = _press()
    try:
        press = engine.ctx.blackboard.require("machine:press")
        seq = press.start("dwell", seconds=1.0)
        assert press.cancel(seq) == seq
        assert not press.busy
        engine.step()
        assert press.status() == MachineStatus(seq, "dwell", True, "cancelled")
        assert not [e for e in _events(engine) if e["event"] == "program_start"]
    finally:
        engine.shutdown()


def test_a_numbered_cancel_of_a_running_start_cancels_it():
    engine = _press()
    try:
        press = engine.ctx.blackboard.require("machine:press")
        seq = press.start("dwell", seconds=1.0)
        engine.step()
        assert press.cancel(seq) == seq
        assert press.cancel(seq) == 0, "it is being cancelled already"
        engine.step()
        assert press.status() == MachineStatus(seq, "dwell", True, "cancelled")
        ends = [(e["seq"], e["error"]) for e in _events(engine) if e["event"] == "program_end"]
        assert ends == [(seq, "cancelled")]
    finally:
        engine.shutdown()


def test_an_unnumbered_cancel_ends_whatever_is_pending_or_runs():
    engine = _press()
    try:
        press = engine.ctx.blackboard.require("machine:press")
        pending = press.start("dwell", seconds=1.0)
        assert press.cancel() == pending
        running = press.start("dwell", seconds=1.0)
        engine.step()
        assert press.cancel() == running
        engine.step()
        assert press.cancel() == 0
        assert press.status() == MachineStatus(running, "dwell", True, "cancelled")
    finally:
        engine.shutdown()


def test_the_cancel_endpoint_takes_the_start_it_cancels():
    engine = _press()
    try:
        start = engine.ctx.interface.find("press", "start")
        cancel = engine.ctx.interface.find("press", "cancel")
        first = start.write({"program": "dwell", "args": {"seconds": 0.01}})
        for _ in range(20):
            engine.step()
        second = start.write({"program": "dwell", "args": {"seconds": 1.0}})
        engine.step()
        stale = cancel.write({"seq": first.result(1.0)})
        engine.step()
        assert stale.result(1.0) == 0
        own = cancel.write({"seq": second.result(1.0)})
        engine.step()
        assert own.result(1.0) == second.result(1.0)
    finally:
        engine.shutdown()


def test_a_cancel_by_token_ends_only_the_start_that_carries_it():
    engine = _press()
    try:
        press = engine.ctx.blackboard.require("machine:press")
        pending = press.start("dwell", seconds=1.0, token="a")
        assert press.cancel(token="b") == 0, "no start carries it"
        assert press.cancel(token="a") == pending
        engine.step()
        assert press.status() == MachineStatus(pending, "dwell", True, "cancelled")
        running = press.start("dwell", seconds=1.0, token="c")
        engine.step()
        assert press.cancel(token="a") == 0, "that start has ended"
        assert press.cancel(token="c") == running
        assert press.cancel(token="c") == 0, "it is being cancelled already"
        engine.step()
        assert press.status() == MachineStatus(running, "dwell", True, "cancelled")
        other = press.start("dwell", seconds=1.0)
        engine.step()
        assert press.cancel(token="c") == 0
        assert press.busy and press.status() == MachineStatus(other, "dwell", False, "")
        with pytest.raises(MachineError, match="a seq or a token, not both"):
            press.cancel(other, token="c")
    finally:
        engine.shutdown()


def test_a_cancel_sent_behind_its_start_through_the_endpoints_reaches_it():
    engine = _press()
    try:
        start = engine.ctx.interface.find("press", "start")
        cancel = engine.ctx.interface.find("press", "cancel")
        started = start.write({"program": "dwell", "args": {"seconds": 1.0}, "token": "t"})
        cancelled = cancel.write({"token": "t"})
        engine.step()
        assert cancelled.result(1.0) == started.result(1.0)
        press = engine.ctx.blackboard.require("machine:press")
        assert not press.busy and press.status().error == "cancelled"
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


def test_goal_is_where_a_ramp_ends_and_otherwise_the_current_target():
    engine = _engine([{"machine_fixtures:Spinner": {"ramp": 0.5}, "name": "spin"}])
    try:
        spin = engine.ctx.blackboard.require("machine:spin")
        assert spin.goal("speed") == spin.value("speed") == pytest.approx(0.1)
        seq = spin.start("run", speed=0.6)
        for _ in range(20):
            engine.step()
        assert spin.goal("speed") == pytest.approx(0.6), "the end of the ramp"
        assert 0.1 < spin.value("speed") < 0.6, "the target, part of the way"
        spin.cancel(seq)
        _run_until_done(engine, spin, seq)
        assert spin.goal("speed") == spin.value("speed") < 0.6, "a cancel holds where the ramp was"
        _run_until_done(engine, spin, spin.start("run", speed=0.3))
        engine.reset()
        assert spin.goal("speed") == spin.value("speed") == pytest.approx(0.1), "home again"
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


def test_a_start_is_told_to_the_machine_when_it_is_accepted_not_when_it_is_applied():
    """``start_accepted`` sees a handle start before the physics thread applies it, a preempt at
    once, and no start that was refused."""
    engine = _press()
    try:
        press = engine.ctx.blackboard.require("machine:press")
        seen = []
        press._machine.start_accepted = lambda program, args: seen.append(
            (program, args, press._machine.status_now.seq)
        )
        first = press.start("dwell", seconds=1.0)
        assert seen == [("dwell", {"seconds": 1.0}, 0)], "told before it was applied"
        with pytest.raises(MachineError, match="busy"):
            press.start("dwell", seconds=2.0)
        with pytest.raises(MachineError, match="no program"):
            press.start("bend")
        assert len(seen) == 1, "a refused start is not told"
        engine.step()
        press._machine.preempt("dwell", seconds=0.1)
        assert seen[1:] == [("dwell", {"seconds": 0.1}, first)]
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
        (
            '<intvelocity name="x" joint="x_joint" kp="2000" actrange="-1 1" ctrlrange="-1 1"/>',
            "its dynamics make its control no position",
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
        # The ramp's target was at 0.1 when the program ended; the machine stops where the joint
        # stood, a little behind it.
        assert held == pytest.approx(0.1, abs=0.015), "it stopped where the program ended"
        for _ in range(1000):
            engine.step()
        assert axis.value("x") == held
        assert _joint(engine, "x_joint") == pytest.approx(held, abs=0.003)
    finally:
        engine.shutdown()


def test_a_cleanup_that_goes_wrong_still_stops_the_machine_and_is_reported():
    engine = _axis()
    try:
        axis = engine.ctx.blackboard.require("machine:axis")
        seq = axis.start("go_then_linger", x=0.8, speed=0.2)
        for _ in range(200):
            engine.step()
        axis.cancel()
        engine.step()
        status = axis.status()
        assert status.seq == seq and status.finished
        assert status.error.startswith("cancelled") and "cleanup raised" in status.error
        assert _events(engine, "axis")[-1]["event"] == "program_end"
        held = axis.value("x")
        for _ in range(1000):
            engine.step()
        assert axis.value("x") == held, "nothing carries on toward the program's target"
    finally:
        engine.shutdown()


@pytest.mark.parametrize(("config", "limit"), [({"timeout": 1.0}, 1.0), ({}, 5.0)])
def test_an_obstructed_joint_fails_its_move_after_the_timeout_naming_machine_and_actuator(
    config, limit
):
    engine = _axis(obstacle=True, **config)
    try:
        axis = engine.ctx.blackboard.require("machine:axis")
        t0 = engine.ctx.sim_time
        status = _run_until_done(engine, axis, axis.start("go", x=0.8))
        assert status.error.startswith(
            f"machine axis: 'x' did not reach 0.8 within {limit:g} s of its target (it is at 0.3"
        )
        assert engine.ctx.sim_time - t0 == pytest.approx(limit, abs=0.01)
    finally:
        engine.shutdown()


def test_until_with_a_timeout_fails_the_program_when_it_runs_out():
    engine = _axis()
    try:
        axis = engine.ctx.blackboard.require("machine:axis")
        t0 = engine.ctx.sim_time
        status = _run_until_done(engine, axis, axis.start("wait_for", seconds=0.3))
        assert status.error == "machine axis: the feeder not true after 0.3 s"
        assert engine.ctx.sim_time - t0 == pytest.approx(0.3, abs=0.01)
    finally:
        engine.shutdown()


# -- a cancelled or failed program stops the machine where it is -------------------------------
def test_a_cancel_during_an_unramped_move_stops_the_joint_where_it_is():
    engine = _axis()
    try:
        axis = engine.ctx.blackboard.require("machine:axis")
        axis.start("go", x=0.8)
        for _ in range(25):
            engine.step()
        axis.cancel()
        engine.step()
        stopped = axis.value("x")
        assert 0.0 < stopped < 0.5, "cancelled on its way"
        for _ in range(1000):
            engine.step()
        assert axis.value("x") == stopped
        assert _joint(engine, "x_joint") == pytest.approx(stopped, abs=0.02)
    finally:
        engine.shutdown()


def test_an_obstructed_move_that_timed_out_stays_when_the_obstacle_is_taken_away():
    engine = _axis(obstacle=True, timeout=1.0)
    try:
        axis = engine.ctx.blackboard.require("machine:axis")
        status = _run_until_done(engine, axis, axis.start("go", x=0.8))
        assert status.error.startswith("machine axis: 'x' did not reach 0.8")
        m = engine.ctx.model
        engine.ctx.data.mocap_pos[m.body_mocapid[m.body("wall").id]] = [5.0, 0.0, 0.5]
        for _ in range(1000):
            engine.step()
        assert _joint(engine, "x_joint") == pytest.approx(0.3, abs=0.01), "it did not move on"
    finally:
        engine.shutdown()


def test_an_optional_argument_takes_none():
    engine = _axis()
    try:
        axis = engine.ctx.blackboard.require("machine:axis")
        assert _run_until_done(engine, axis, axis.start("nudge", x=0.2, speed=None)).error == ""
        assert _run_until_done(engine, axis, axis.start("nudge", x=0.1, speed=0.5)).error == ""
        with pytest.raises(MachineError, match="speed is float . None, not 'fast'"):
            axis.start("nudge", x=0.1, speed="fast")
        with pytest.raises(MachineError, match="x is float, not None"):
            axis.start("nudge", x=None)
    finally:
        engine.shutdown()


# -- starts and cancels from any thread, in order ------------------------------------------------
def test_a_status_never_goes_back_to_an_older_start():
    engine = _press()
    try:
        press = engine.ctx.blackboard.require("machine:press")
        first = press.start("dwell", seconds=1.0)
        engine.step()
        press.cancel()
        second = press.start("dwell", seconds=1.0)
        press.cancel()
        assert press.status() == MachineStatus(second, "dwell", True, "cancelled")
        seen = []
        for _ in range(5):
            engine.step()
            seen.append(press.status().seq)
        assert seen == [second] * 5, "the cancel of the first start applied after the second's"
        ends = [(e["seq"], e["error"]) for e in _events(engine) if e["event"] == "program_end"]
        assert ends == [(first, "cancelled")]
    finally:
        engine.shutdown()


def test_a_start_from_another_thread_during_a_cancel_waits_for_the_cancel_to_be_posted():
    engine = _press()
    try:
        press = engine.ctx.blackboard.require("machine:press")
        first = press.start("dwell", seconds=1.0)
        engine.step()
        post = engine.ctx.post
        other: list = []
        racer: list = []

        def post_once_raced(command):
            # The other thread's start is let loose while the cancel is being posted.
            if not racer:
                thread = threading.Thread(
                    target=lambda: other.append(press.start("dwell", seconds=0.1))
                )
                racer.append(thread)
                thread.start()
                thread.join(0.3)
            post(command)

        engine.ctx.post = post_once_raced
        assert press.cancel() == first
        engine.ctx.post = post
        racer[0].join()
        second = other[0]
        engine.step()
        events = [(e["event"], e["seq"]) for e in _events(engine)][-2:]
        assert events == [("program_end", first), ("program_start", second)]
        assert press.status() == MachineStatus(second, "dwell", False, "")
    finally:
        engine.shutdown()


def test_busy_is_true_while_a_start_is_being_applied():
    engine = _press()
    try:
        press = engine.ctx.blackboard.require("machine:press")
        plugin = press._machine
        press.start("dwell", seconds=1.0)
        bind = plugin.bind
        read: list = []
        readers: list = []

        def bind_and_look(name, args):
            # Another thread asks `busy` while the start is being applied.
            thread = threading.Thread(target=lambda: read.append(press.busy))
            readers.append(thread)
            thread.start()
            thread.join(0.3)
            return bind(name, args)

        plugin.bind = bind_and_look
        engine.step()
        plugin.bind = bind
        readers[0].join()
        assert read == [True]
    finally:
        engine.shutdown()


# -- annotations, needs, and holding under load ----------------------------------------------------
def test_a_real_union_annotation_takes_none_and_refuses_what_fits_no_part():
    engine = _engine(
        [{"machine_fixtures_runtime_annotations:TypedPress": {"prefix": "p1_"}, "name": "press"}],
        gravity=False,
    )
    try:
        press = engine.ctx.blackboard.require("machine:press")
        assert _run_until_done(engine, press, press.start("nudge", depth=None)).error == ""
        assert _run_until_done(engine, press, press.start("nudge", depth=0.1)).error == ""
        with pytest.raises(MachineError, match="depth is float . None, not 'abc'"):
            press.start("nudge", depth="abc")
    finally:
        engine.shutdown()


def test_a_handle_named_like_a_member_of_the_machine_is_refused():
    with pytest.raises(Exception, match=r"NEEDS names \['home'\], which the machine already has"):
        _engine([{"machine_fixtures:HomeClash": {"prefix": "p1_"}, "name": "press"}], gravity=False)


def test_a_loaded_joint_does_not_sink_at_every_hold():
    engine = _engine([{"machine_fixtures:Lift": {}, "name": "lift"}])
    try:
        lift = engine.ctx.blackboard.require("machine:lift")
        for _ in range(1000):
            engine.step()
        standing = _joint(engine, "z_joint")
        assert standing == pytest.approx(-9.81 / 1000 * 5 / 5, abs=0.001), "load/kp below home"
        for _ in range(5):
            lift.start("dwell", seconds=10.0)
            engine.step()
            lift.cancel()
            for _ in range(300):
                engine.step()
        assert _joint(engine, "z_joint") == pytest.approx(standing, abs=0.0005), "five holds"
        status = _run_until_done(engine, lift, lift.start("go", z=5.0))
        assert "outside its range" in status.error
        for _ in range(300):
            engine.step()
        assert _joint(engine, "z_joint") == pytest.approx(standing, abs=0.0005), "nothing moved"
    finally:
        engine.shutdown()


def test_a_loaded_joint_arrives_at_its_target_less_its_standing_offset():
    engine = _engine([{"machine_fixtures:Lift": {}, "name": "lift"}])
    try:
        lift = engine.ctx.blackboard.require("machine:lift")
        for _ in range(1000):
            engine.step()
        assert lift.ready and lift._machine.at_home(), "standing at home under its load"
        status = _run_until_done(engine, lift, lift.start("go", z=0.1))
        assert status.error == ""
        assert _joint(engine, "z_joint") == pytest.approx(0.1 - 0.00981, abs=0.0025)
    finally:
        engine.shutdown()


def test_a_program_named_like_what_every_machine_has_is_refused():
    with pytest.raises(Exception, match=r"program\(s\) \['start'\]"):
        _engine([{"machine_fixtures:ProgramClash": {}, "name": "lift"}])


def test_a_program_argument_named_token_is_refused():
    with pytest.raises(Exception, match=r"program\(s\) \['lift'\] take an argument 'token'"):
        _engine([{"machine_fixtures:TokenClash": {}, "name": "lift"}])


def test_a_target_or_home_that_is_no_number_is_refused():
    engine = _axis()
    try:
        axis = engine.ctx.blackboard.require("machine:axis")
        status = _run_until_done(engine, axis, axis.start("go", x=float("nan")))
        assert "is no number" in status.error
        assert not math.isnan(float(engine.ctx.data.ctrl[0]))
    finally:
        engine.shutdown()
    with pytest.raises(Exception, match="home 'x' is a number"):
        _axis(home={"x": float("inf")})


def test_without_a_control_range_the_joints_range_bounds_a_target():
    servo = '<position name="x" joint="x_joint" kp="400" kv="40"/>'
    engine = _axis(actuator=servo)
    try:
        axis = engine.ctx.blackboard.require("machine:axis")
        status = _run_until_done(engine, axis, axis.start("go", x=50.0))
        assert "outside its range" in status.error
    finally:
        engine.shutdown()


def test_a_start_another_thread_got_in_during_a_preempt_is_followed_by_a_larger_number():
    engine = _axis()
    try:
        axis = engine.ctx.blackboard.require("machine:axis")
        machine = axis._machine
        axis.start("go_then_linger", x=0.8, speed=0.2)
        engine.step()
        finish, other = machine._finish, []

        def finish_and_let_another_start(run, error):
            finish(run, error)
            other.append(machine.request_start("nudge", {"x": 0.1}))

        machine._finish = finish_and_let_another_start
        machine.preempt("nudge", x=0.2)
        machine._finish = finish
        assert other
        assert machine.status_now.seq > other[0]
        assert axis.status().seq > other[0], "a larger number says another start followed"
    finally:
        engine.shutdown()


@pytest.mark.parametrize(
    ("annotation", "value", "fits"),
    [
        ("Optional[float]", None, True),
        ("Optional[float]", "fast", False),
        ("Optional[float]", 1.0, True),
        ("typing.Optional[str]", "a", True),
        ("Any", None, True),
        ("float", None, False),
        ("float | None", None, True),
    ],
)
def test_a_parameters_annotation_is_read_through_its_forms(annotation, value, fits):
    assert machine_module._fits(annotation, value) is fits


def test_a_moving_damped_joint_has_not_arrived_until_it_has():
    engine = _axis(damping=400.0)
    try:
        axis = engine.ctx.blackboard.require("machine:axis")
        status = _run_until_done(engine, axis, axis.start("go", x=0.4))
        assert status.error == ""
        assert _joint(engine, "x_joint") == pytest.approx(0.4, abs=0.0025)
    finally:
        engine.shutdown()


def test_an_obstructed_damped_joint_fails_its_move():
    engine = _axis(damping=400.0, obstacle=True, timeout=1.0)
    try:
        axis = engine.ctx.blackboard.require("machine:axis")
        status = _run_until_done(engine, axis, axis.start("go", x=0.4))
        assert "did not reach" in status.error
    finally:
        engine.shutdown()


def test_a_cancel_stops_a_damped_joint_where_it_is():
    engine = _axis(damping=400.0)
    try:
        axis = engine.ctx.blackboard.require("machine:axis")
        axis.start("go", x=0.4)
        for _ in range(200):
            engine.step()
        moving = _joint(engine, "x_joint")
        assert 0.01 < moving < 0.39
        axis.cancel()
        for _ in range(1500):
            engine.step()
        assert _joint(engine, "x_joint") == pytest.approx(moving, abs=0.01), "not on to the target"
    finally:
        engine.shutdown()


def test_a_reset_lets_a_programs_cleanup_read_the_machine():
    engine = _axis()
    try:
        axis = engine.ctx.blackboard.require("machine:axis")
        axis.start("cleanup_reads_the_machine")
        engine.step()
        engine.reset()
        assert axis._machine.seen_busy is False
    finally:
        engine.shutdown()


def test_a_timeout_that_is_no_number_fails_the_program():
    engine = _axis()
    try:
        axis = engine.ctx.blackboard.require("machine:axis")
        status = _run_until_done(engine, axis, axis.start("wait_nan"))
        assert "a timeout is a number" in status.error
    finally:
        engine.shutdown()


def test_a_filtered_servo_starts_an_episode_at_home():
    engine = _engine(
        [{"machine_fixtures:FilteredLift": {"home": {"z": 0.3}}, "name": "lift"}], gravity=False
    )
    try:
        lowest = min(_joint(engine, "z_joint") for _ in _steps(engine, 100))
        assert lowest == pytest.approx(0.3, abs=0.001)
    finally:
        engine.shutdown()


def test_arriving_with_a_timeout_that_is_no_number_fails_the_program():
    engine = _axis()
    try:
        axis = engine.ctx.blackboard.require("machine:axis")
        status = _run_until_done(engine, axis, axis.start("arrive_nan"))
        assert "a timeout is a number" in status.error
    finally:
        engine.shutdown()


def test_a_program_may_have_a_parameter_named_program():
    engine = _axis()
    try:
        axis = engine.ctx.blackboard.require("machine:axis")
        status = _run_until_done(engine, axis, axis.start("takes_a_program", program="x"))
        assert status.error == ""
    finally:
        engine.shutdown()


def test_a_bare_yield_of_a_helper_stops_the_run_rather_than_doing_nothing():
    engine = _axis()
    try:
        axis = engine.ctx.blackboard.require("machine:axis")
        axis.start("yields_a_helper", x=0.4)
        with pytest.raises(RuntimeError, match="yield from"):
            for _ in range(5):
                engine.step()
    finally:
        engine.shutdown()


def _steps(engine, count):
    for _ in range(count):
        engine.step()
        yield


def test_a_machine_publishes_the_handle_its_class_names_once():
    engine = _engine(
        [{"machine_fixtures:HandledPress": {"prefix": "p1_"}, "name": "press"}], gravity=False
    )
    try:
        handle = engine.ctx.blackboard.get("machine:press")
        plugin = next(p for p in engine.plugins if p.address == "press")
        assert type(handle).__name__ == "PressHandle" and plugin.handle is handle
        assert _run_until_done(engine, handle, handle.stroke()).error == ""
    finally:
        engine.shutdown()


def test_a_machine_names_its_start_state_in_its_own_quantity():
    """A machine with HOME_KEYS takes ``home:`` in them and maps them onto what it drives."""
    engine = _engine(
        [{"machine_fixtures:FractionLift": {"home": {"raised": 0.75}}, "name": "lift"}]
    )
    try:
        assert _joint(engine, "z_joint") == pytest.approx(0.25)
    finally:
        engine.shutdown()


@pytest.mark.parametrize(
    "config, said",
    [
        (
            {"home": {"z": 0.1}},
            r"home names 'z', which it does not take \(its home names \['raised'\]\)",
        ),
        ({"start_raised": 1.0}, r"'start_raised' is not read -- .* 'home': home: \{raised"),
    ],
    ids=["an-actuator-in-home", "a-start-key"],
)
def test_a_start_state_outside_the_machines_home_keys_is_refused(config, said):
    with pytest.raises(Exception, match=said):
        _engine([{"machine_fixtures:FractionLift": config, "name": "lift"}])
