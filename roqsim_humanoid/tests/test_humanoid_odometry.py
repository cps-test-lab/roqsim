"""``g1_locomotion`` and ``oli_locomotion``: odometry starts at the spawn pose, a stale command stops.

The promises every velocity-commanded plugin keeps (docs/plugins.rst, "A velocity command"), checked
on a humanoid spawned away from the origin and turned, so a world-frame odometry cannot pass.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from roqsim.config import load_config_from_dict
from roqsim.engine import Engine

X0, Y0, YAW0 = 1.5, -1.0, 2.0

#: model, its locomotion plugin, the step its PD loop is tuned for, a forward speed it walks well at
CASES = {
    "g1": ("unitree_g1", "g1_locomotion", 0.002, 0.4),
    "oli": ("oli", "oli_locomotion", 0.001, 0.3),
}


def _engine(case, **loco):
    model, plugin, timestep, _ = CASES[case]
    world = {
        "sim": {"timestep": timestep},
        "components": [
            {
                "spawn_robot": {
                    "model": model,
                    "prefix": "r_",
                    "pose": {"position": {"x": X0, "y": Y0}, "orientation": {"yaw": YAW0}},
                    "default_plugins": False,
                },
                "name": "r",
                "components": [{plugin: loco}],
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


def _speed(engine, handle):
    """Mean speed over the next second, from odometry: a walking base's speed swings within a step."""
    x0, y0 = handle.read_odom()[:2]
    _run(engine, 1.0)
    x1, y1 = handle.read_odom()[:2]
    return math.hypot(x1 - x0, y1 - y0)


@pytest.mark.parametrize("case", list(CASES))
def test_odometry_starts_at_zero_at_the_spawn_pose_and_reads_forward_as_x(case):
    engine = _engine(case)
    try:
        handle = engine.ctx.blackboard.get("robot:r")
        assert handle.read_odom()[:3] == pytest.approx((0.0, 0.0, 0.0), abs=1e-6)
        start = engine.ctx.data.body("r_base_link").xpos.copy()
        handle.drive(CASES[case][3], 0.0, 0.0)
        _run(engine, 3.0)
        x, y, _ = handle.read_odom()[:3]
        moved = engine.ctx.data.body("r_base_link").xpos - start
        c, s = math.cos(YAW0), math.sin(YAW0)
        # The true displacement, turned into the spawn heading, is what odometry reports.
        assert (x, y) == pytest.approx((c * moved[0] + s * moved[1], -s * moved[0] + c * moved[1]))
        assert x > 0.4 and abs(y) < 0.5 * x, (x, y)
        assert handle.read_odom()[6] > 0.5, "z is the pelvis height"
    finally:
        engine.shutdown()


@pytest.mark.parametrize("case", list(CASES))
@pytest.mark.parametrize(("timeout", "moving"), [(0.5, False), (0.0, True)])
def test_a_stale_command_stops_the_robot_only_with_the_timeout_on(case, timeout, moving):
    engine = _engine(case, cmd_vel_timeout=timeout)
    try:
        handle = engine.ctx.blackboard.get("robot:r")
        speed = CASES[case][3]
        handle.drive(speed, 0.0, 0.0)  # once, then silence
        _run(engine, 3.0)
        got = _speed(engine, handle)
        # Standing, a policy still drifts a few cm/s, so "stopped" is well under the command.
        assert got > 0.5 * speed if moving else got < 0.35 * speed, got
    finally:
        engine.shutdown()


@pytest.mark.parametrize("case", list(CASES))
def test_reset_clears_the_command_and_the_odometry(case):
    engine = _engine(case)
    try:
        handle = engine.ctx.blackboard.get("robot:r")
        speed = CASES[case][3]
        handle.drive(speed, 0.0, 0.0)
        _run(engine, 1.0)
        assert _speed(engine, handle) > 0.5 * speed
        engine.reset()
        assert handle.read_odom()[:3] == pytest.approx((0.0, 0.0, 0.0), abs=1e-6)
        _run(engine, 2.0)
        assert _speed(engine, handle) < 0.35 * speed
    finally:
        engine.shutdown()
