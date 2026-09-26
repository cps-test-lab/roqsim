"""Every base stops when its commands stop, once ``cmd_vel_timeout`` is set -- not only diff_drive.

A stack that dies mid-run must leave a stationary robot, not one driving at its last velocity into
a wall. ``omni_drive`` and ``ackermann_drive`` had no watchdog, so a holonomic base or a car drove
on for ever.
"""

from __future__ import annotations

from pathlib import Path  # noqa: E402

import pytest  # noqa: E402

# `roqsim` selects MuJoCo's GL backend on import, so it comes first (see test_wheels_roll.py).
import roqsim  # noqa: F401, I001
from roqsim.config import load_config_from_dict  # noqa: E402
from roqsim.engine import Engine  # noqa: E402


def _ridgeback(timeout):
    world = {
        "sim": {"timestep": 0.002},
        "components": [
            {
                "spawn_robot": {"model": "ridgeback", "prefix": "r_"},
                "name": "r",
                "components": [{"omni_drive": {"cmd_vel_timeout": timeout}}],
            }
        ],
    }
    engine = Engine(load_config_from_dict(world, base_dir=Path(".")))
    engine.ctx.seed = 7
    engine.setup()
    engine.reset()
    return engine


@pytest.mark.parametrize(("timeout", "moving"), [(0.5, False), (0.0, True)])
def test_omni_drive_stops_when_commands_stop(timeout, moving):
    engine = _ridgeback(timeout)
    try:
        handle = engine.ctx.blackboard.get("robot:r")
        handle.drive(0.3, 0.2, 0.0)  # once, then silence
        for _ in range(1500):  # 3 s
            engine.step()
        vx, vy = handle.read_odom()[3:5]
        assert bool(abs(vx) + abs(vy) > 0.2) is moving, (vx, vy)
    finally:
        engine.shutdown()


@pytest.mark.parametrize(("timeout", "moving"), [(0.5, False), (0.0, True)])
def test_ackermann_drive_stops_when_commands_stop(timeout, moving):
    from test_ackermann_drive import _engine, _plugin

    engine = _engine(cmd_vel_timeout=timeout)
    try:
        plugin = _plugin(engine)
        plugin.drive(0.8, 0.0, 0.0)  # once, then silence
        for _ in range(1500):
            engine.step()
        assert bool(abs(plugin.read_odom()[3]) > 0.2) is moving
    finally:
        engine.shutdown()
