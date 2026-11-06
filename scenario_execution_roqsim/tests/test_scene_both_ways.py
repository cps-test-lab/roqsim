"""One scene in Python and in OpenSCENARIO gives one timeline: the two forms stay one vocabulary."""

from __future__ import annotations

import pytest

pytest.importorskip("scenario_execution", reason="the OpenSCENARIO half of the comparison")

import py_trees  # noqa: E402
from scenario_execution.actions.base_action import ActionError  # noqa: E402
from test_osc_library import _build  # noqa: E402

from roqsim.config import load_config_from_dict  # noqa: E402
from roqsim.engine import Engine  # noqa: E402
from scenario_execution_roqsim.actions.machine_run import MachineRun  # noqa: E402
from scenario_execution_roqsim.actions.zone import (  # noqa: E402
    ZoneDwelled,
    ZoneEntered,
    ZoneLeft,
)

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


FIRST_TICK = """
import osc.roqsim
scenario first_tick:
    do serial:
        wait elapsed(1s)
        one_of:
            machine_run(machine: 'press', program: 'stroke', args: 'seconds=10.0')
            zone_entered(zone: 'gate')
        machine_run(machine: 'press', program: 'stroke', args: 'seconds=0.05')
"""


def test_a_machine_run_abandoned_on_its_first_tick_still_cancels_the_program():
    engine = _engine()
    try:
        assert _run_osc(engine, FIRST_TICK) == py_trees.common.Status.SUCCESS
        ends = [e["error"] for e in engine.ctx.events.all() if e["event"] == "program_end"]
        assert ends[-1] == ""
    finally:
        engine.shutdown()


def test_a_second_cancel_of_an_abandoned_run_leaves_a_later_start_alone():
    engine = _engine()
    try:
        action = MachineRun()
        action.setup(simulation=_Sim(engine.ctx), clock=_Clock(engine.ctx))
        action.execute("press", "stroke", "seconds=10.0")
        action.update()
        assert action.request_cancel()
        engine.step()
        later = engine.ctx.blackboard.require("machine:press").start("stroke", seconds=10.0)
        assert action.request_cancel()
        engine.step()
        status = engine.ctx.blackboard.require("machine:press").status()
        assert status.seq == later and not status.finished
    finally:
        engine.shutdown()


def test_machine_run_waits_for_the_program_to_end():
    engine = _engine()
    try:
        text = """
import osc.roqsim
scenario waits:
    do serial:
        machine_run(machine: 'press', program: 'stroke', args: 'seconds=0.2')
"""
        assert _run_osc(engine, text) == py_trees.common.Status.SUCCESS
        assert engine.ctx.sim_time == pytest.approx(0.2, abs=3 * engine.ctx.dt)
    finally:
        engine.shutdown()


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        ("", {}),
        ("a=1", {"a": 1}),
        ("a=1, b=hole", {"a": 1, "b": "hole"}),
        ("a=[1, 2], b=true", {"a": [1, 2], "b": True}),
        ("a=x=y", {"a": "x=y"}),
    ],
)
def test_args_are_name_value_pairs(args, expected):
    action = MachineRun()
    action.execute("press", "stroke", args)
    assert action._args == expected


@pytest.mark.parametrize("args", ["a", "=1", "a=[1, b=2"])
def test_args_that_are_no_pairs_are_refused(args):
    with pytest.raises(ActionError):
        MachineRun().execute("press", "stroke", args)


def _zone_engine(dwell=0.1):
    gate = {**GATE, "zone_trigger": {**GATE["zone_trigger"], "dwell": dwell}}
    engine = Engine(
        load_config_from_dict({"sim": {"gravity": [0, 0, 0]}, "components": [*WORLD[:2], gate]})
    )
    engine.ctx.seed = 1
    engine.setup()
    engine.reset()
    return engine


def _run_action(engine, action, seconds=3.0):
    action.setup(simulation=_Sim(engine.ctx), clock=_Clock(engine.ctx))
    while engine.ctx.sim_time < seconds:
        engine.step()
        if action.update() != py_trees.common.Status.RUNNING:
            return engine.ctx.sim_time
    return None


def test_zone_dwelled_waits_for_the_dwell():
    engine = _zone_engine(dwell=0.1)
    try:
        action = ZoneDwelled()
        action.execute("gate")
        # The box enters at 0.8 s and dwells 0.1 s.
        assert _run_action(engine, action) == pytest.approx(0.9, abs=3 * engine.ctx.dt)
    finally:
        engine.shutdown()


def test_zone_dwelled_on_a_zone_with_no_dwell_is_refused():
    engine = _zone_engine(dwell=0.0)
    try:
        action = ZoneDwelled()
        action.execute("gate")
        with pytest.raises(ActionError, match="has dwell 0"):
            _run_action(engine, action)
    finally:
        engine.shutdown()


def test_a_pattern_narrows_who_counts():
    engine = _zone_engine()
    try:
        action = ZoneEntered()
        action.execute("gate", pattern="other_*")
        assert _run_action(engine, action, seconds=1.5) is None, "the box is not other_*"
        action = ZoneLeft()
        action.execute("gate", pattern="other_*")
        assert _run_action(engine, action, seconds=1.6) is not None, "and is not held to leave"
    finally:
        engine.shutdown()
