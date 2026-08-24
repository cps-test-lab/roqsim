"""End-to-end plugin tests: patrol, the goal-route interface, and reset -- driven through the engine.

These are the in-process equivalent of what the ROS 2 ``NavigateThroughPoses`` handler does: send a
route via the blackboard :class:`~roqsim_walker.plugins.walker.WalkerHandle`, poll ``status()``
until it reports finished, and check the walker actually got there.
"""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from roqsim.config import load_config_from_dict
from roqsim.engine import Engine

WAYPOINTS = [[-2.0, -2.0], [2.0, -2.0], [2.0, 2.0]]


def _world(*, name="pedestrian", **walker_overrides):
    walker = {
        "walker": "MaleVisitorWalk",
        "speed": 1.2,
        "loop": True,
        "arrival_radius": 0.25,
        "avoidance": False,
        "skin": False,  # capsules: keeps the test fast (no 5 MB OBJ load / skin rig)
        "waypoints": WAYPOINTS,
    }
    walker.update(walker_overrides)
    return load_config_from_dict(
        {
            "sim": {"headless": True, "pacing": "asap"},
            "components": [{"walker": walker, "name": name}],
        }
    )


@pytest.fixture
def sim():
    engine = Engine(_world())
    engine.setup()
    engine.reset()
    yield engine
    engine.shutdown()


def _xy(engine):
    model, data = engine.ctx.model, engine.ctx.data
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pedestrian/pelvis")
    return data.mocap_pos[model.body_mocapid[bid]][:2].copy()


def _run(engine, seconds):
    for _ in range(max(1, int(seconds / engine.ctx.dt))):
        engine.step()


def _run_until(engine, predicate, timeout=40.0, tick=0.02):
    """Step until ``predicate()`` or ``timeout`` sim-seconds elapse. Returns whether it fired."""
    deadline = engine.ctx.sim_time + timeout
    while engine.ctx.sim_time < deadline:
        _run(engine, tick)
        if predicate():
            return True
    return False


# -- registration ------------------------------------------------------------------------------
def test_plugin_registers_entity_handle_and_goal_endpoint(sim):
    ctx = sim.ctx
    entity = ctx.entities.get("pedestrian")
    assert entity is not None and entity.kind == "pedestrian"
    assert entity.body == "pedestrian/pelvis"

    assert ctx.blackboard.get("walker:pedestrian") is not None

    # The walker registers two endpoints: the body_poses TF stream (out) and the goal action (in).
    endpoints = {e.name: e for e in ctx.interface.all() if e.owner == "pedestrian"}
    assert set(endpoints) == {"body_poses", "navigate_through_poses"}
    assert endpoints["body_poses"].direction == "out"
    endpoint = endpoints["navigate_through_poses"]
    assert endpoint.direction == "in"
    hints = endpoint.backend["ros2"]
    assert hints["action"] == "nav2_msgs.action.NavigateThroughPoses"
    assert hints["name"] == "navigate_through_poses"


def test_walker_spawns_at_the_first_waypoint(sim):
    np.testing.assert_allclose(_xy(sim), WAYPOINTS[0], atol=1e-6)


# -- patrol ------------------------------------------------------------------------------------
def test_walker_patrols_toward_its_next_waypoint(sim):
    _run(sim, 2.0)
    pos = _xy(sim)
    assert pos[0] > -1.0, "should have walked east along the first leg"
    assert pos[1] == pytest.approx(-2.0, abs=0.2)


def test_reset_returns_the_walker_to_the_route_start(sim):
    _run(sim, 2.0)
    assert _xy(sim)[0] > -1.5
    sim.reset()
    np.testing.assert_allclose(_xy(sim), WAYPOINTS[0], atol=1e-6)


# -- goal route (what the NavigateThroughPoses handler drives) ---------------------------------
def test_send_route_overrides_patrol_and_reports_arrival(sim):
    handle = sim.ctx.blackboard.get("walker:pedestrian")
    goals = [(0.0, 2.0), (-2.0, 0.0)]

    seq = handle.send_route(goals)
    assert seq >= 1
    _run(sim, 0.05)  # let the posted command reach the physics thread
    applied_seq, finished, goals_left, _ = handle.status()
    assert (applied_seq, finished, goals_left) == (seq, False, 2)

    arrived = _run_until(sim, lambda: handle.status()[1])
    assert arrived, "walker never reported finishing its route"

    applied_seq, finished, goals_left, dist = handle.status()
    assert (applied_seq, finished, goals_left, dist) == (seq, True, 0, 0.0)
    # It really is at the final pose (within its arrival radius).
    assert np.linalg.norm(_xy(sim) - np.array(goals[-1])) < 0.26


def test_route_feedback_counts_goals_down(sim):
    handle = sim.ctx.blackboard.get("walker:pedestrian")
    handle.send_route([(0.0, 2.0), (-2.0, 0.0)])
    _run(sim, 0.05)

    seen = []
    _run_until(sim, lambda: seen.append(handle.status()[2]) or handle.status()[1])
    assert seen[0] == 2
    assert 1 in seen, "the first goal should be retired before the second"
    assert seen[-1] == 0


def test_patrol_resumes_after_the_route_completes(sim):
    handle = sim.ctx.blackboard.get("walker:pedestrian")
    handle.send_route([(0.0, 0.0)])
    _run(sim, 0.05)
    assert _run_until(sim, lambda: handle.status()[1]), "route never finished"

    at_goal = _xy(sim)
    _run(sim, 2.0)
    assert np.linalg.norm(_xy(sim) - at_goal) > 0.5, "walker should resume patrolling, not stand"


def test_cancel_route_stops_the_walker(sim):
    handle = sim.ctx.blackboard.get("walker:pedestrian")
    handle.send_route([(2.0, 2.0)])
    _run(sim, 1.0)

    seq = handle.cancel_route()
    _run(sim, 0.3)
    applied_seq, finished, _, _ = handle.status()
    assert (applied_seq, finished) == (seq, True)

    stopped_at = _xy(sim)
    _run(sim, 1.5)
    assert np.linalg.norm(_xy(sim) - stopped_at) < 0.05, "walker kept moving after cancel"


def test_a_newer_route_supersedes_an_older_one(sim):
    handle = sim.ctx.blackboard.get("walker:pedestrian")
    first = handle.send_route([(2.0, 2.0)])
    _run(sim, 0.5)
    second = handle.send_route([(-2.0, -2.0)])
    _run(sim, 0.05)

    applied_seq, finished, _, _ = handle.status()
    assert applied_seq == second > first
    assert not finished
    # The handler for `first` sees a larger seq and aborts its goal.


# -- goal-driven only (no patrol) --------------------------------------------------------------
def test_walker_without_waypoints_stands_at_pos_until_commanded():
    engine = Engine(_world(waypoints=[], pos=[1.0, 1.0]))
    engine.setup()
    engine.reset()
    try:
        np.testing.assert_allclose(_xy(engine), [1.0, 1.0], atol=1e-6)
        _run(engine, 2.0)
        np.testing.assert_allclose(_xy(engine), [1.0, 1.0], atol=0.02)  # stands

        handle = engine.ctx.blackboard.get("walker:pedestrian")
        handle.send_route([(-1.0, 1.0)])
        _run(engine, 0.05)
        assert _run_until(engine, lambda: handle.status()[1]), "route never finished"
        assert np.linalg.norm(_xy(engine) - np.array([-1.0, 1.0])) < 0.26
    finally:
        engine.shutdown()
