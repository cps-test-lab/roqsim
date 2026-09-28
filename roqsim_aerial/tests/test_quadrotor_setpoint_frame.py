"""``quadrotor_controller``: a position setpoint is read in the frame it names.

The drone is spawned away from the origin and turned, so reading an ``odom`` setpoint as a world
position, or the reverse, flies it somewhere else. The setpoints go in through ``cmd_pos``'s write,
with the ``(position, quaternion, frame_id)`` payload the bridge decodes a ``PoseStamped`` to.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from roqsim.config import load_config_from_dict
from roqsim.engine import Engine

X0, Y0, YAW0 = 1.5, -1.0, 2.0


def _flown():
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
                "components": [{"quadrotor_controller": {"target": [X0, Y0, 1.0], "yaw": YAW0}}],
            }
        ],
    }
    engine = Engine(load_config_from_dict(world, base_dir=Path(".")))
    engine.ctx.seed = 1
    engine.setup()
    engine.reset()
    controller = next(p for p in engine.plugins if type(p).__name__ == "QuadrotorControllerPlugin")
    return engine, controller, engine.ctx.interface.find("drone", "cmd_pos")


def _fly(engine, seconds):
    for _ in range(int(seconds / engine.ctx.dt)):
        engine.step()


def _yaw_quat(yaw):
    return (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))


def _heading_error(a, b):
    return abs(math.atan2(math.sin(a - b), math.cos(a - b)))


def test_an_odom_setpoint_flies_to_its_point_in_the_spawn_frame():
    engine, controller, cmd_pos = _flown()
    try:
        _fly(engine, 3.0)
        cmd_pos.write(((1.0, 0.0, 1.2), _yaw_quat(math.pi / 2), "odom"))
        _fly(engine, 6.0)
        x, y, z = controller.read_state()[:3]
        # One metre ahead of the spawn pose, along the spawn heading.
        expected = (X0 + math.cos(YAW0), Y0 + math.sin(YAW0), 1.2)
        assert (x, y, z) == pytest.approx(expected, abs=0.05)
        assert _heading_error(controller.read_state()[6], YAW0 + math.pi / 2) < 0.05
        o = controller.read_odom6()
        assert (o["x"], o["y"], o["z"]) == pytest.approx((1.0, 0.0, 1.2), abs=0.05)
    finally:
        engine.shutdown()


@pytest.mark.parametrize("frame", ["world", "map", ""])
def test_a_world_setpoint_flies_to_its_world_point(frame):
    engine, controller, cmd_pos = _flown()
    try:
        _fly(engine, 3.0)
        cmd_pos.write(((0.5, 0.5, 1.2), _yaw_quat(0.3), frame))
        _fly(engine, 6.0)
        assert controller.read_state()[:3] == pytest.approx((0.5, 0.5, 1.2), abs=0.05)
        assert _heading_error(controller.read_state()[6], 0.3) < 0.05
    finally:
        engine.shutdown()


def test_a_setpoint_in_an_unknown_frame_is_refused_by_name():
    engine, controller, cmd_pos = _flown()
    try:
        before = controller._target.copy()
        with pytest.raises(ValueError, match="'base_link'"):
            cmd_pos.write(((1.0, 0.0, 1.0), _yaw_quat(0.0), "base_link"))
        assert list(controller._target) == list(before)
    finally:
        engine.shutdown()
