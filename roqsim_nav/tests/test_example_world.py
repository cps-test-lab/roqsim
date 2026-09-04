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
    """The claim the whole package makes, in one assertion."""
    outputs = {name: type(nav._output).__name__ for name, nav in _navigators(sim).items()}
    assert outputs == {
        "visitor": "WalkerOutput",  # a pedestrian
        "cart": "DriveOutput",  # a robot whose wheels turn
        "pallet": "MocapOutput",  # a prop whose pose is written
    }


def test_the_subject_is_not_driven_by_anything(sim):
    """`robot` has no navigator: it is the thing an experiment measures, not apparatus."""
    assert "robot" not in _navigators(sim)
    start = _xy(sim, "robot")
    _run(sim, 6.0)
    assert np.linalg.norm(_xy(sim, "robot") - start) < 0.05, "something is driving the subject"


def test_the_opponents_that_should_be_moving_are(sim):
    starts = {n: _xy(sim, n) for n in ("visitor", "pallet")}
    _run(sim, 8.0)
    for name, start in starts.items():
        assert np.linalg.norm(_xy(sim, name) - start) > 1.0, f"{name} never set off"


def test_the_held_opponent_waits_for_its_trigger_then_runs(sim):
    """`autostart: false` -- the route is in the world, the timing belongs to the trial."""
    cart = _navigators(sim)["cart"]
    start = _xy(sim, "cart")
    assert not cart.started
    _run(sim, 5.0)
    assert np.linalg.norm(_xy(sim, "cart") - start) < 0.05, "it moved without being started"

    cart.start()
    _run(sim, 12.0)
    moved = _xy(sim, "cart")
    assert moved[0] - start[0] > 1.5, "it never set off after being started"
    # `route_mode: exact`: the polyline is a straight run at constant y, and nothing planned it.
    assert abs(moved[1] - start[1]) < 0.35, "it left the exact polyline it was given"


def test_every_navigator_publishes_a_goal_handle(sim):
    """So a scenario can command any of them, over either transport, by entity name."""
    for name in _navigators(sim):
        assert sim.ctx.blackboard.get(f"nav:{name}:handle") is not None
