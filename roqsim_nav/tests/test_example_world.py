"""The shipped example: one navigator, three embodiments, in one scene.

This is the artefact the package exists to make possible, so it is worth a test that says what it
demonstrates rather than merely that it loads. If any of these stops holding, the example has
stopped being an example.
"""

from __future__ import annotations

import pathlib

import mujoco
import numpy as np
import pytest

import roqsim_nav
from roqsim.config import load_config
from roqsim.engine import Engine

WORLD = pathlib.Path(roqsim_nav.WORLDS_DIR) / "nav_opponents.yaml"


@pytest.fixture(scope="module")
def sim():
    engine = Engine(load_config(str(WORLD)))
    engine.setup()
    engine.reset()
    yield engine
    engine.shutdown()


def _xy(engine, name):
    model, data = engine.ctx.model, engine.ctx.data
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, engine.ctx.entities.get(name).body)
    mid = int(model.body_mocapid[bid])
    return (data.mocap_pos[mid][:2] if mid >= 0 else data.xpos[bid][:2]).copy()


def _navigators(engine):
    return {p.entity: p for p in engine.plugins if type(p).__name__ == "NavigatorPlugin"}


def _run(engine, seconds):
    for _ in range(int(seconds / engine.ctx.dt)):
        engine.step()


def test_one_plugin_serves_three_embodiments(sim):
    """The claim the whole package makes, in one assertion.

    Asserted as the SET of embodiments rather than a mover-by-mover map, so adding another
    encounter to the world does not have to be mirrored here -- what matters is that one plugin is
    driving a pedestrian, a robot and a written pose in the same room.
    """
    outputs = {type(nav._output).__name__ for nav in _navigators(sim).values()}
    assert outputs == {"WalkerOutput", "DriveOutput", "MocapOutput"}


def test_every_opponent_actually_moves(sim):
    starts = {n: _xy(sim, n) for n in _navigators(sim)}
    _run(sim, 10.0)
    for name, start in starts.items():
        assert np.linalg.norm(_xy(sim, name) - start) > 0.5, f"{name} never set off"


def test_the_omni_pair_passes_each_other():
    """Two omnidirectional bases head-on, both giving way.

    A different base geometry from the wheeled pair: an omni can move sideways without turning
    first, so the way it takes round an encounter is not a differential base's. One navigator drives
    both, and the control law comes from what each drive declares itself to be.
    """
    engine = Engine(load_config(str(WORLD)))
    engine.setup()
    engine.reset()
    try:
        starts = {n: _xy(engine, n) for n in ("omni_a", "omni_b")}
        closest = float("inf")
        for _ in range(int(35.0 / engine.ctx.dt)):
            engine.step()
            closest = min(
                closest, float(np.linalg.norm(_xy(engine, "omni_a") - _xy(engine, "omni_b")))
            )
        assert closest > 0.5, f"they came within {closest:.2f} m"
        for a, b in (("omni_a", "omni_b"), ("omni_b", "omni_a")):
            assert np.linalg.norm(_xy(engine, a) - starts[b]) < 0.7, f"{a} did not get across"
    finally:
        engine.shutdown()


def test_one_robot_stops_and_the_other_goes_around_it():
    """The asymmetry the pair exists to show: both LOOK, only one GIVES WAY.

    `stopper` names no avoidance model, so it holds its line and stops; `dodger` names one, so it
    steers around. It is also the regression for a mover that opts out of yielding being left out of
    the model altogether -- it was then invisible, and the two bumped.
    """
    engine = Engine(load_config(str(WORLD)))
    engine.setup()
    engine.reset()
    try:
        starts = {n: _xy(engine, n) for n in ("stopper", "dodger")}
        lateral = {"stopper": 0.0, "dodger": 0.0}
        closest = float("inf")
        for _ in range(int(50.0 / engine.ctx.dt)):
            engine.step()
            closest = min(
                closest, float(np.linalg.norm(_xy(engine, "stopper") - _xy(engine, "dodger")))
            )
            for name in lateral:
                lateral[name] = max(lateral[name], abs(_xy(engine, name)[1] - starts[name][1]))

        assert closest > 0.4, f"they bumped ({closest:.2f} m apart at closest)"
        assert lateral["stopper"] < 0.1, "the one with no avoidance model still stepped aside"
        assert lateral["dodger"] > 0.2, "the one with an avoidance model never went around"
    finally:
        engine.shutdown()


def test_every_prop_sits_on_the_floor(sim):
    """A spawn places the body ORIGIN, so a centred box needs its half-height as `pos.z`.

    Getting it wrong buries half the prop, which looks like a rendering fault rather than a world
    that asked for it -- and a half-sunk obstacle is a different obstacle to everything that senses
    it.
    """
    model, data = sim.ctx.model, sim.ctx.data
    body = sim.ctx.entities.get("crate").body
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
    for g in range(model.ngeom):
        if int(model.geom_bodyid[g]) == bid:
            bottom = float(data.geom_xpos[g][2] - model.geom_size[g][2])
            assert bottom > -1e-6, f"the crate is {-bottom:.3f} m into the floor"


def test_every_navigator_publishes_a_goal_handle(sim):
    """So a scenario can command any of them, over either transport, by entity name."""
    for name in _navigators(sim):
        assert sim.ctx.blackboard.get(f"nav:{name}:handle") is not None


def test_the_head_on_pair_gets_past_each_other():
    """The encounter the world exists to show, and the one nothing else in it can resolve.

    A robot and a walker five metres apart, each driving to where the other started. The planner
    knows nothing about either -- its grid holds walls -- so if the avoidance model were absent or
    inert they would stop nose to nose and stay there for the rest of the run.

    Its own engine rather than the module's: the encounter only happens once, and by the time this
    runs the shared world's pair has long since finished and is standing at the far end.
    """
    engine = Engine(load_config(str(WORLD)))
    engine.setup()
    engine.reset()
    try:
        starts = {n: _xy(engine, n) for n in ("runner", "pedestrian")}
        closest = float("inf")
        for _ in range(int(30.0 / engine.ctx.dt)):
            engine.step()
            closest = min(
                closest, float(np.linalg.norm(_xy(engine, "runner") - _xy(engine, "pedestrian")))
            )

        assert closest > 0.25, f"they came within {closest:.2f} m -- a collision, not a pass"
        for name, other in (("runner", "pedestrian"), ("pedestrian", "runner")):
            travelled = float(np.linalg.norm(_xy(engine, name) - starts[name]))
            assert travelled > 3.5, f"{name} did not get through (moved {travelled:.2f} m)"
            towards = float(np.linalg.norm(_xy(engine, name) - starts[other]))
            assert towards < 0.6, f"{name} did not reach where {other} started"
    finally:
        engine.shutdown()
