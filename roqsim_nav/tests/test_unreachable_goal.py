"""A goal the mover cannot reach ends its route as a failure, never as an arrival.

The room is sealed by a full-height divider, so the planner finds no path from one half to the other.
A mocap body has no collision response: sent along a straight leg it walks through the divider and
reports the goal reached. The navigator must instead stop where it is and say which goal it could not
reach, so that a scenario fails the action and a nav2 client sees the goal aborted.
"""

from __future__ import annotations

import numpy as np
import pytest

from roqsim.config import load_config_from_dict
from roqsim.engine import Engine
from roqsim_nav.behavior import NavCore, NavParams
from roqsim_nav.state import NavState

SEALED = """<mujoco model="sealed_room">
  <worldbody>
    <geom name="floor" type="plane" size="8 8 .05"/>
    <geom name="wall_n" type="box" pos="0  4 1" size="4 .1 1"/>
    <geom name="wall_s" type="box" pos="0 -4 1" size="4 .1 1"/>
    <geom name="wall_e" type="box" pos=" 4 0 1" size=".1 4 1"/>
    <geom name="wall_w" type="box" pos="-4 0 1" size=".1 4 1"/>
    <geom name="divider" type="box" pos="0 0 1" size=".15 4 1"/>
  </worldbody>
</mujoco>"""
CRATE = """<mujoco model="crate">
  <worldbody><body name="crate"><geom name="g" type="box" size=".25 .25 .25"/></body></worldbody>
</mujoco>"""
START = (-2.5, 0.0)
BEYOND = (2.5, 0.0)  # the other half of the room
NEAR = (-2.5, 2.0)  # the same half


@pytest.fixture
def sim(tmp_path):
    (tmp_path / "sealed.xml").write_text(SEALED)
    (tmp_path / "crate.xml").write_text(CRATE)
    cfg = load_config_from_dict(
        {
            "sim": {"pacing": "asap", "world": str(tmp_path / "sealed.xml")},
            "components": [
                {
                    "spawn_model": {
                        "model": str(tmp_path / "crate.xml"),
                        "pose": {"position": {"x": START[0], "y": START[1], "z": 0.25}},
                        "motion": "driven",
                    },
                    "name": "cart",
                    "components": [
                        {
                            "navigator": {
                                "speed": 0.8,
                                "goals": [list(BEYOND)],
                                "obstacle_height": [0.05, 0.6],
                            }
                        }
                    ],
                }
            ],
        },
        base_dir=tmp_path,
    )
    engine = Engine(cfg)
    engine.setup()
    engine.reset()
    yield engine
    engine.shutdown()


def _run(engine, seconds: float) -> None:
    for _ in range(int(round(seconds / engine.ctx.dt))):
        engine.step()


def _handle(engine):
    return engine.ctx.blackboard.get("nav:cart:handle")


def test_a_configured_goal_beyond_a_sealed_wall_fails_and_the_mover_stays_put(sim):
    _run(sim, 10.0)
    x, y, _ = _handle(sim).pose()
    assert x < -0.15, f"the mover crossed the divider to x = {x:.2f}"
    assert np.hypot(x - START[0], y - START[1]) < 1e-6, "it moved toward a goal it cannot reach"
    _seq, finished, goals_left, _dist, error = _handle(sim).status()
    assert finished and goals_left == 0
    assert "unreachable" in error and "(2.50, 0.00)" in error, error


def test_a_commanded_unreachable_goal_fails_and_a_reachable_one_then_arrives(sim):
    handle = _handle(sim)
    failing = handle.send_goals([BEYOND])
    _run(sim, 1.0)
    seq, finished, _goals, _dist, error = handle.status()
    assert (seq, finished) == (failing, True) and "unreachable" in error

    reaching = handle.send_goals([NEAR])
    _run(sim, 6.0)
    assert handle.status() == (reaching, True, 0, 0.0, ""), "a new route inherited the old error"


def _core(waypoints, max_recovery=1):
    st = NavState(name="mover", waypoints=np.asarray(waypoints, dtype=float), speed=0.5)
    return NavCore(st, None, NavParams(max_recovery=max_recovery), uniform=lambda lo, hi: lo)


def test_a_goal_given_up_on_after_recoveries_is_recorded_as_the_routes_failure():
    core = _core([(0.0, 0.0), (5.0, 0.0)])
    core.observe(0.0, (0.0, 0.0), None)
    core.start_recovery()
    assert core.failure == ""
    core.start_recovery()  # one more than max_recovery: the goal is abandoned
    assert core.st.done
    assert "gave up on goal 1 at (5.00, 0.00)" in core.failure

    core.reset()
    assert core.failure == "", "a new route must not inherit the last one's failure"
