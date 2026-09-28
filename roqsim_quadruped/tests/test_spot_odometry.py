"""``spot_locomotion``'s odometry starts at the spawn pose, and a stale command stops the robot.

The promises every velocity-commanded plugin keeps (docs/plugins.rst, "A velocity command"), checked
on a Spot spawned away from the origin and turned, so a world-frame odometry cannot pass.
"""

from __future__ import annotations

# `roqsim` selects MuJoCo's GL backend on import, so it comes first.
import roqsim  # noqa: F401, I001
import math  # noqa: E402
import os  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from roqsim.config import load_config_from_dict  # noqa: E402
from roqsim.engine import Engine  # noqa: E402
from roqsim_quadruped.policy import DEFAULT_POLICY  # noqa: E402

POLICY = os.environ.get("SPOT_POLICY_PATH") or str(DEFAULT_POLICY)
pytestmark = pytest.mark.skipif(
    not Path(POLICY).exists(),
    reason=f"no Spot policy at {POLICY}: python -m roqsim_quadruped.policy.fetch_policy",
)

X0, Y0, YAW0 = 1.5, -1.0, 2.0


def _engine(**loco):
    world = {
        "sim": {"timestep": 0.002},
        "components": [
            {
                "spawn_robot": {
                    "model": "spot",
                    "prefix": "r_",
                    "pose": {"position": {"x": X0, "y": Y0}, "orientation": {"yaw": YAW0}},
                    "default_plugins": False,
                },
                "name": "r",
                "components": [{"spot_locomotion": loco}],
            }
        ],
    }
    engine = Engine(load_config_from_dict(world, base_dir=Path(".")))
    engine.ctx.seed = 1
    engine.setup()
    engine.reset()
    return engine


def _run(engine, seconds):
    for _ in range(int(seconds / engine.ctx.dt)):
        engine.step()


def _speed(handle):
    return math.hypot(*handle.read_odom()[3:5])


def test_odometry_starts_at_zero_at_the_spawn_pose_and_reads_forward_as_x():
    engine = _engine()
    try:
        handle = engine.ctx.blackboard.get("robot:r")
        assert handle.read_odom()[:3] == pytest.approx((0.0, 0.0, 0.0), abs=1e-6)
        start = engine.ctx.data.body("r_base_link").xpos.copy()
        handle.drive(0.5, 0.0, 0.0)
        _run(engine, 3.0)
        x, y, yaw = handle.read_odom()[:3]
        moved = engine.ctx.data.body("r_base_link").xpos - start
        c, s = math.cos(YAW0), math.sin(YAW0)
        # The true displacement, turned into the spawn heading, is what odometry reports.
        assert (x, y) == pytest.approx((c * moved[0] + s * moved[1], -s * moved[0] + c * moved[1]))
        assert x > 0.5 and abs(y) < 0.5 * x, (x, y, yaw)
    finally:
        engine.shutdown()


@pytest.mark.parametrize(("timeout", "moving"), [(0.5, False), (0.0, True)])
def test_a_stale_command_stops_the_robot_only_with_the_timeout_on(timeout, moving):
    engine = _engine(cmd_vel_timeout=timeout)
    try:
        handle = engine.ctx.blackboard.get("robot:r")
        handle.drive(0.5, 0.0, 0.0)  # once, then silence
        _run(engine, 4.0)
        assert (_speed(handle) > 0.3) is moving, _speed(handle)
    finally:
        engine.shutdown()


def test_reset_clears_the_command_and_the_odometry():
    engine = _engine()
    try:
        handle = engine.ctx.blackboard.get("robot:r")
        handle.drive(0.5, 0.0, 0.0)
        _run(engine, 2.0)
        assert _speed(handle) > 0.3
        engine.reset()
        assert handle.read_odom()[:3] == pytest.approx((0.0, 0.0, 0.0), abs=1e-6)
        _run(engine, 3.0)
        assert _speed(handle) < 0.15
        assert np.hypot(*handle.read_odom()[:2]) < 0.3
    finally:
        engine.shutdown()
