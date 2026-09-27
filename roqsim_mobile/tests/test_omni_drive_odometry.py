"""``omni_drive``'s odometry is in the odom frame, which is the pose the base was spawned at.

A holonomic base spawned facing +y and driven forward moves along the world's +y; its odometry must
say it went forward (+x in its own frame), not that it strafed. Driven on the Ridgeback, a mecanum
base whose manifest carries no camera.
"""

from __future__ import annotations

# `roqsim` selects MuJoCo's GL backend on import, so it comes first (see test_wheels_roll.py).
import roqsim  # noqa: F401, I001
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from roqsim.config import load_config_from_dict  # noqa: E402
from roqsim.engine import Engine  # noqa: E402

MODEL = "ridgeback"
SPEED = 0.3
STEPS = 1500  # 3 s at 2 ms


def _drive(yaw: float):
    """Spawn at *yaw*, drive forward for STEPS; return (odometry pose, the base's true displacement)."""
    world = {
        "sim": {"timestep": 0.002},
        "components": [
            {
                "spawn_robot": {
                    "model": MODEL,
                    "prefix": "r_",
                    "pose": {"position": {"x": 0.0, "y": 0.0}, "orientation": {"yaw": yaw}},
                },
                "name": "r",
            }
        ],
    }
    engine = Engine(load_config_from_dict(world, base_dir=Path(".")))
    engine.ctx.seed = 7
    engine.setup()
    engine.reset()
    try:
        handle = engine.ctx.blackboard.get("robot:r")
        start = engine.ctx.data.body("r_base_link").xpos.copy()
        handle.drive(SPEED, 0.0, 0.0)
        for _ in range(STEPS):
            engine.step()
        truth = engine.ctx.data.body("r_base_link").xpos.copy() - start
        return np.array(handle.read_odom()[:3]), truth
    finally:
        engine.shutdown()


@pytest.mark.parametrize("yaw", [0.0, np.pi / 2])
def test_driving_forward_reads_as_forward_whatever_the_spawn_heading(yaw):
    odom, truth = _drive(yaw)
    moved = float(np.hypot(truth[0], truth[1]))
    assert moved > 0.3, f"the base did not move ({truth})"
    # The body went along its own +x, which at this spawn heading is the world direction (cos, sin).
    assert np.allclose(truth[:2] / moved, [np.cos(yaw), np.sin(yaw)], atol=0.05)
    # Odometry reports that same motion in the odom frame: forward, no strafe, no turn.
    assert odom[0] == pytest.approx(moved, abs=0.02)
    assert abs(odom[1]) < 0.02
    assert abs(odom[2]) < 0.02
