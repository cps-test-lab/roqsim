"""``quadrotor_controller``: odometry in the spawn frame, and a stale velocity command ends in a hover.

A Crazyflie spawned away from the origin and turned, holding 1 m above its spawn point, so a
world-frame odometry cannot pass.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from roqsim.config import load_config_from_dict
from roqsim.engine import Engine

X0, Y0, YAW0 = 1.5, -1.0, 2.0
HOVER = [X0, Y0, 1.0]


def _flown(**controller):
    world = {
        "sim": {"integrator": "rk4", "density": 1.225, "viscosity": 1.8e-5},
        "components": [
            {
                "spawn_robot": {
                    "model": "crazyflie_2",
                    "prefix": "cf2_",
                    "pose": {"position": {"x": X0, "y": Y0}, "orientation": {"yaw": YAW0}},
                    "default_plugins": False,
                },
                "name": "drone",
                "components": [
                    {"quadrotor_controller": {"target": HOVER, "yaw": YAW0, **controller}}
                ],
            }
        ],
    }
    engine = Engine(load_config_from_dict(world, base_dir=Path(".")))
    engine.ctx.seed = 1
    engine.setup()
    engine.reset()
    controller = next(p for p in engine.plugins if type(p).__name__ == "QuadrotorControllerPlugin")
    return engine, controller


def _fly(engine, seconds):
    for _ in range(int(seconds / engine.ctx.dt)):
        engine.step()


def _xyz(controller):
    o = controller.read_odom6()
    return np.array([o["x"], o["y"], o["z"]])


def test_odometry_starts_at_zero_at_the_spawn_pose_and_reads_forward_as_x():
    engine, controller = _flown()
    try:
        spawn_z = controller.read_state()[2]
        assert _xyz(controller) == pytest.approx([0.0, 0.0, spawn_z], abs=1e-6)
        assert controller.read_odom()[2] == pytest.approx(0.0, abs=1e-6)
        _fly(engine, 5.0)
        # Hovering over the spawn point: x and y still zero, z the altitude.
        assert _xyz(controller) == pytest.approx([0.0, 0.0, 1.0], abs=0.03)
        controller.drive(0.5, 0.0, 0.0)
        _fly(engine, 2.0)
        x, y, _ = _xyz(controller)
        assert x > 0.5 and abs(y) < 0.1, (x, y)
        vx, vy, vz = (controller.read_odom6()[k] for k in ("vx", "vy", "vz"))
        assert vx > 0.3 and abs(vy) < 0.1 and abs(vz) < 0.1, "the twist is in the body frame"
    finally:
        engine.shutdown()


@pytest.mark.parametrize(("timeout", "hovering"), [(0.5, True), (0.0, False)])
def test_a_stale_velocity_command_ends_in_a_hover_only_with_the_timeout_on(timeout, hovering):
    engine, controller = _flown(cmd_vel_timeout=timeout)
    try:
        _fly(engine, 5.0)
        controller.drive(0.5, 0.0, 0.0)  # once, then silence
        _fly(engine, 4.0)
        before = _xyz(controller)
        _fly(engine, 1.0)
        drift = float(np.linalg.norm(_xyz(controller) - before))
        if hovering:
            assert drift < 0.03, f"moved {drift:.3f} m in 1 s after the command went stale"
            assert _xyz(controller)[2] == pytest.approx(1.0, abs=0.05)
        else:
            assert drift > 0.3, f"moved only {drift:.3f} m in 1 s on a held command"
    finally:
        engine.shutdown()


def test_reset_clears_the_velocity_command():
    engine, controller = _flown()
    try:
        _fly(engine, 3.0)
        controller.drive(0.5, 0.0, 0.3)
        _fly(engine, 1.0)
        engine.reset()
        _fly(engine, 6.0)
        assert _xyz(controller) == pytest.approx([0.0, 0.0, 1.0], abs=0.05)
        assert controller.read_odom()[2] == pytest.approx(0.0, abs=0.05)
        assert math.hypot(*controller.read_odom()[3:5]) < 0.05
    finally:
        engine.shutdown()
