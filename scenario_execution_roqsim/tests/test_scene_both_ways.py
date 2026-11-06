"""One scene in Python and in OpenSCENARIO gives one timeline: the two forms stay one vocabulary."""

from __future__ import annotations

import pytest

pytest.importorskip("scenario_execution", reason="the OpenSCENARIO half of the comparison")

import py_trees  # noqa: E402
from test_osc_library import _build  # noqa: E402

from roqsim.config import load_config_from_dict  # noqa: E402
from roqsim.engine import Engine  # noqa: E402
from roqsim.machine import MachineStatus  # noqa: E402

GATE = {
    "zone_trigger": {
        "pose": {"position": {"x": 0.0, "y": 0.0, "z": 0.5}},
        "size": [0.2, 0.3, 0.2],
        "dwell": 0.1,
    },
    "name": "gate",
}
WORLD = [
    {"scene_fixtures:Press": {}, "name": "press"},
    {"scene_fixtures:Box": {}, "name": "box"},
    GATE,
]

SCENE = """
import osc.roqsim
scenario scene:
    do serial:
        zone_entered(zone: 'gate')
        machine_run(machine: 'press', program: 'stroke', args: 'seconds=0.1')
        zone_left(zone: 'gate')
"""

JAMS = """
import osc.roqsim
scenario jams:
    do serial:
        machine_run(machine: 'press', program: 'jam')
"""

RACE = """
import osc.roqsim
scenario race:
    do serial:
        one_of:
            machine_run(machine: 'press', program: 'stroke', args: 'seconds=10.0')
            zone_entered(zone: 'gate')
        machine_run(machine: 'press', program: 'stroke', args: 'seconds=0.05')
"""


def _engine(*components):
    engine = Engine(
        load_config_from_dict({"sim": {"gravity": [0, 0, 0]}, "components": [*WORLD, *components]})
    )
    engine.ctx.seed = 1
    engine.setup()
    engine.reset()
    return engine


class _Sim:
    def __init__(self, ctx):
        self.context = ctx


class _Clock:
    def __init__(self, ctx):
        self._ctx = ctx

    def now(self):
        return self._ctx.sim_time


def _run_osc(engine, text, seconds=3.0):
    """Step the engine and tick the tree between steps, as the stepped runner does."""
    root = _build(text)
    tree = py_trees.trees.BehaviourTree(root)
    tree.setup(simulation=_Sim(engine.ctx), clock=_Clock(engine.ctx))
    while engine.ctx.sim_time < seconds:
        engine.step()
        tree.tick()
        if root.status != py_trees.common.Status.RUNNING:
            break
    return root.status


def _timeline(engine):
    return [
        (e["source"], e["event"], e.get("error"), e.get("entity"))
        for e in engine.ctx.events.all()
        if e["source"] in ("press", "gate")
    ]


def _times(engine):
    return [e["t"] for e in engine.ctx.events.all() if e["source"] in ("press", "gate")]


def test_a_scene_in_either_form_gives_the_same_timeline():
    python = _engine({"scene_fixtures:Scene": {}, "name": "scene"})
    try:
        done = python.ctx.interface.find("scene", "done")
        while not done.read() and python.ctx.sim_time < 3.0:
            python.step()
        assert done.read()
        expected, expected_times = _timeline(python), _times(python)
    finally:
        python.shutdown()
    osc = _engine()
    try:
        assert _run_osc(osc, SCENE) == py_trees.common.Status.SUCCESS
        assert _timeline(osc) == expected
        # The tree ticks after a step and a scenario's post_step within one, so a step or two apart.
        assert _times(osc) == pytest.approx(expected_times, abs=3 * osc.ctx.dt)
    finally:
        osc.shutdown()


def test_a_machines_error_fails_the_trial_as_it_stops_a_python_scene():
    engine = _engine()
    try:
        assert _run_osc(engine, JAMS) == py_trees.common.Status.FAILURE
    finally:
        engine.shutdown()


def test_an_abandoned_machine_run_cancels_the_program():
    engine = _engine()
    try:
        assert _run_osc(engine, RACE) == py_trees.common.Status.SUCCESS
        ends = [e["error"] for e in engine.ctx.events.all() if e["event"] == "program_end"]
        assert ends == ["cancelled", ""]
    finally:
        engine.shutdown()


def _machine_run(engine, **args):
    from scenario_execution_roqsim.actions.machine_run import MachineRun

    action = MachineRun()
    action.setup(simulation=_Sim(engine.ctx), clock=_Clock(engine.ctx))
    action.execute(machine="press", program="stroke", **args)
    return action


def _abandon(action):
    action.terminate(py_trees.common.Status.INVALID)


def test_a_machine_run_abandoned_in_flight_cancels_its_own_start_and_no_other():
    engine = _engine()
    try:
        press = engine.ctx.blackboard.require("machine:press")
        # Accepted: its start is cancelled once its number is known.
        action = _machine_run(engine, args="seconds=10.0")
        assert action.update() == py_trees.common.Status.RUNNING
        _abandon(action)
        engine.step()
        engine.step()
        own = press.status()
        assert own.finished and own.error == "cancelled" and not press.busy
        # Refused, as another client's program runs: that program is left alone.
        other = press.start("stroke", seconds=10.0)
        engine.step()
        action = _machine_run(engine, args="seconds=0.1")
        assert action.update() == py_trees.common.Status.RUNNING
        _abandon(action)
        for _ in range(5):
            engine.step()
        assert press.busy and press.status() == MachineStatus(other, "stroke", False, "")
    finally:
        engine.shutdown()


def test_a_machine_run_abandoned_after_its_program_ended_leaves_the_next_start_alone():
    engine = _engine()
    try:
        press = engine.ctx.blackboard.require("machine:press")
        action = _machine_run(engine, args="seconds=0.01")
        action.update()
        engine.step()
        assert action.update() == py_trees.common.Status.RUNNING, "it knows its number now"
        while press.busy:
            engine.step()
        other = press.start("stroke", seconds=10.0)
        _abandon(action)
        for _ in range(5):
            engine.step()
        assert press.busy and press.status().seq == other and not press.status().finished
    finally:
        engine.shutdown()


def test_since_finds_who_came_and_went_and_who_is_inside_now():
    engine = _engine()
    try:
        text = """
import osc.roqsim
scenario late:
    do serial:
        wait elapsed(2s)
        zone_entered(zone: 'gate', since: 0.0)
        zone_dwelled(zone: 'gate', since: 0.0)
"""
        assert _run_osc(engine, text, seconds=3.0) == py_trees.common.Status.SUCCESS
    finally:
        engine.shutdown()
