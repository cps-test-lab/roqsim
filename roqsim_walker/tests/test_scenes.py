"""The handling scenes, each in Python and in OpenSCENARIO: the same timeline, and what physically
happened -- a button pushed, a drawer pulled, totes carried and set down by the worker's holds.

Each runs the shipped worlds (``roqsim_walker/worlds/handling_*.yaml`` and their ``*_osc`` twins).
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import pytest

from roqsim.config import load_config
from roqsim.engine import Engine

WORLDS = Path(__file__).resolve().parents[1] / "src" / "roqsim_walker" / "worlds"


def _run(world: str, seconds: float, watch=None):
    engine = Engine(load_config(WORLDS / f"{world}.yaml"))
    engine.ctx.seed = 1
    engine.setup()
    engine.reset()
    scene = next(p for p in engine.plugins if p.address in ("scene", "scene_osc"))
    while engine.ctx.sim_time < seconds and not scene.done():
        engine.step()
        if watch is not None:
            watch(engine)
    assert scene.done(), f"{world}: the scene did not end in {seconds} s"
    return engine


def _timeline(engine):
    return [
        (e["source"], e["event"], e.get("program"), e.get("error"))
        for e in engine.ctx.events.all()
        if not e["source"].startswith("scene")
    ]


def _times(engine):
    return [e["t"] for e in engine.ctx.events.all() if not e["source"].startswith("scene")]


def _both(world: str, seconds: float, watch=None):
    """The scene's Python form, checked by ``watch``, and its timeline against the OSC form's."""
    python = _run(world, seconds, watch)
    try:
        expected, times = _timeline(python), _times(python)
        yield python
    finally:
        python.shutdown()
    osc = _run(f"{world}_osc", seconds)
    try:
        assert _timeline(osc) == expected
        assert _times(osc) == pytest.approx(times, abs=0.05)
    finally:
        osc.shutdown()


def _q(engine, joint: str) -> float:
    m = engine.ctx.model
    return float(engine.ctx.data.qpos[m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, joint)]])


def test_scene_1_a_worker_presses_a_machines_start_button_and_it_runs():
    for engine in _both("handling_start_machine", 20.0):
        events = [(e["source"], e["event"]) for e in engine.ctx.events.all()]
        assert engine.ctx.blackboard.get("button:start").presses == 1
        # pressed by the fingertip, then the machine's program: the scenario's decision
        assert events.index(("start", "pressed")) < events.index(("press_1", "program_start"))
        steps = [e for (src, e) in events if src == "press_1"]
        assert steps.index("door_close") < steps.index("power") < steps.index("door_open")
        assert all(not e.get("error") for e in engine.ctx.events.all() if e["event"] == "program_end")


def test_scene_2_a_drawer_opened_a_tote_carried_to_a_bench_and_set_down():
    heights = []

    def watch(engine):
        heights.append(float(engine.ctx.data.body("tote_1").xpos[2]))

    for engine in _both("handling_drawer_to_bench", 90.0, watch):
        d = engine.ctx.data
        assert abs(_q(engine, "cabinet/drawer_1/slide") - 0.44) < 0.02  # pulled by the hold
        tote = d.body("tote_1").xpos
        assert abs(tote[2] - 0.75) < 0.02  # standing on the bench top
        assert abs(tote[0]) < 0.4 and abs(tote[1] - 2.5) < 0.3
        # carried, never dropped: from leaving the drawer to the bench it was never below the drawer
        lifted = next(i for i, z in enumerate(heights) if z > 0.6)
        assert min(heights[lifted:]) > 0.6


def test_scene_3_a_full_tote_swapped_for_an_empty_one_at_a_cells_drawer():
    deepest = []

    def watch(engine):
        # the worker stands clear of the rack it takes the empty tote from: no joint of its legs or
        # pelvis inside the rack's box (> 0: inside, by that much)
        m, d = engine.ctx.model, engine.ctx.data
        rack = d.geom("rack_box")
        size = m.geom_size[rack.id]
        rot = rack.xmat.reshape(3, 3)
        for joint in ("pelvis", "hip_l", "hip_r", "knee_l", "knee_r", "ankle_l", "ankle_r", "toe_l", "toe_r"):
            p = rot.T @ (d.body(f"worker/{joint}").xpos - rack.xpos)
            deepest.append(float(np.min(size - np.abs(p))))

    for engine in _both("handling_box_swap", 150.0, watch):
        assert max(deepest) < 0.0, f"a leg is {max(deepest):.3f} m inside the rack"
        d = engine.ctx.data
        full, empty = d.body("tote_full").xpos, d.body("tote_empty").xpos
        assert abs(full[2] - 0.70) < 0.02 and np.hypot(full[0] + 1.2, full[1] - 1.0) < 0.45  # on the trolley
        assert abs(empty[2] - 0.495) < 0.02  # in the drawer
        assert _q(engine, "cell/drawer_1/slide") < 0.03  # the drawer pushed shut
        assert np.hypot(empty[0] - 2.28, empty[1] + 1.0) < 0.1  # inside the closed drawer
        assert engine.ctx.blackboard.get("part_state").is_shown("tote_full", "full")


def test_the_pedestrian_world_is_the_walker_as_it_was():
    engine = Engine(load_config(WORLDS / "walker_pedestrians.yaml"), preview=True)
    engine.setup()
    try:
        m = engine.ctx.model
        assert m.nmocap == 10 * 17  # no finger bones, no proxies: hands off
        assert all(engine.ctx.blackboard.get(f"humanoid_body:p{i}") is None for i in range(10))
    finally:
        engine.shutdown()
