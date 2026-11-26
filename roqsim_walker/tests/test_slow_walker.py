"""A walker slower than its gait walks its route at its speed, and one at 0 m/s stands.

Below the speed at which it covers its navigator's ``stuck_eps`` in ``stuck_time`` (0.067 m/s by
default), a walker walking freely made less progress than that in every window, was read as
wedged, and each recovery backed it off against its heading: it walked backwards off its route,
through the room's wall. Driven through the engine, as a world runs it, with recovery on.
"""

from __future__ import annotations

import mujoco
import pytest

from roqsim.config import load_config_from_dict
from roqsim.engine import Engine

SECONDS = 20.0
SPEEDS = (0.0, 0.02, 0.044, 0.055, 0.5)


def _world():
    walkers = []
    for i, speed in enumerate(SPEEDS):
        y = -2.0 + i
        walkers.append(
            {
                # capsules: keeps the test fast (no OBJ load / skin rig)
                "walker": {
                    "walker": "MaleVisitorWalk",
                    "skin": False,
                    "pose": {"position": {"x": -4.0, "y": y}, "orientation": {"yaw": 0.0}},
                },
                "name": f"w{i}",
                "components": [
                    {
                        "navigator": {
                            "output": "walker",
                            "speed": speed,
                            "goals": [[4.0, y]],
                            "radius": 0.26,
                            "avoidance": {"steer": "none", "stop": True},
                        }
                    }
                ],
            }
        )
    return load_config_from_dict({"sim": {"pacing": "asap"}, "components": walkers})


@pytest.fixture(scope="module")
def ends():
    """Each walker's root (x, y) after ``SECONDS``, by speed."""
    engine = Engine(_world())
    engine.setup()
    engine.reset()
    try:
        for _ in range(int(SECONDS / engine.ctx.dt)):
            engine.step()
        model, data = engine.ctx.model, engine.ctx.data
        out = {}
        for i, speed in enumerate(SPEEDS):
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"w{i}/pelvis")
            out[speed] = (data.mocap_pos[model.body_mocapid[bid]][:2].copy(), -2.0 + i)
        return out
    finally:
        engine.shutdown()


@pytest.mark.parametrize("speed", [0.02, 0.044, 0.055])
def test_a_walker_slower_than_its_gait_walks_its_route_at_its_speed(ends, speed):
    (x, y), route_y = ends[speed]
    assert y == pytest.approx(route_y, abs=1e-6), f"at {speed} m/s it left its route's line"
    assert x - -4.0 == pytest.approx(speed * SECONDS, abs=0.05), (
        f"at {speed} m/s it is at x={x:.2f} after {SECONDS} s"
    )


def test_a_walker_at_0_m_s_stands_where_it_starts(ends):
    (x, y), route_y = ends[0.0]
    assert (x, y) == pytest.approx((-4.0, route_y), abs=1e-6)


def test_a_walker_at_a_usual_speed_reaches_its_goal(ends):
    (x, y), route_y = ends[0.5]
    assert x > 3.5 and y == pytest.approx(route_y, abs=1e-6)
