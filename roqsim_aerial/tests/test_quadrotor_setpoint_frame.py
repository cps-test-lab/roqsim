"""``quadrotor_controller``: a position setpoint is read in the frame it names.

The drone is spawned away from the origin and turned, so reading an ``odom`` setpoint as a world
position, or the reverse, flies it somewhere else. The setpoints go in through ``cmd_pos``'s write,
with the ``position``, ``orientation`` and ``frame_id`` of a ``Pose``, as the bridge decodes a
``PoseStamped``.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import pytest

from roqsim.config import load_config_from_dict
from roqsim.engine import Engine
from roqsim.types import Odometry

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


def _pose(position, yaw, frame_id):
    return {"position": position, "orientation": _yaw_quat(yaw), "frame_id": frame_id}


def _heading_error(a, b):
    return abs(math.atan2(math.sin(a - b), math.cos(a - b)))


def test_an_odom_setpoint_flies_to_its_point_in_the_spawn_frame():
    engine, controller, cmd_pos = _flown()
    try:
        _fly(engine, 3.0)
        cmd_pos.write(_pose((1.0, 0.0, 1.2), math.pi / 2, "odom"))
        _fly(engine, 6.0)
        x, y, z = controller.read_state()[:3]
        # One metre ahead of the spawn pose, along the spawn heading.
        expected = (X0 + math.cos(YAW0), Y0 + math.sin(YAW0), 1.2)
        assert (x, y, z) == pytest.approx(expected, abs=0.05)
        assert _heading_error(controller.read_state()[6], YAW0 + math.pi / 2) < 0.05
        o = controller.read_odom6()
        assert (o["x"], o["y"], o["z"]) == pytest.approx((1.0, 0.0, 1.2), abs=0.05)
        odom = engine.ctx.interface.find("drone", "odom").read()
        assert isinstance(odom, Odometry)
        assert odom.position.tolist() == pytest.approx([o["x"], o["y"], o["z"]])
        assert odom.orientation.tolist() == pytest.approx([o["qw"], o["qx"], o["qy"], o["qz"]])
    finally:
        engine.shutdown()


@pytest.mark.parametrize("frame", ["world", "map", ""])
def test_a_world_setpoint_flies_to_its_world_point(frame):
    engine, controller, cmd_pos = _flown()
    try:
        _fly(engine, 3.0)
        cmd_pos.write(_pose((0.5, 0.5, 1.2), 0.3, frame))
        _fly(engine, 6.0)
        assert controller.read_state()[:3] == pytest.approx((0.5, 0.5, 1.2), abs=0.05)
        assert _heading_error(controller.read_state()[6], 0.3) < 0.05
    finally:
        engine.shutdown()


def test_a_setpoint_in_an_unknown_frame_is_refused_by_name(caplog):
    engine, controller, cmd_pos = _flown()
    try:
        before = controller._target.copy()
        cmd_pos.write(_pose((1.0, 0.0, 1.0), 0.0, "base_link"))
        with caplog.at_level(logging.ERROR):
            engine.step()
        assert "'base_link'" in caplog.text
        assert list(controller._target) == list(before)
    finally:
        engine.shutdown()


def test_a_setpoint_without_an_orientation_keeps_the_heading():
    engine, controller, cmd_pos = _flown()
    try:
        cmd_pos.write({"position": (X0, Y0, 1.1)})
        engine.step()
        assert controller._target.tolist() == pytest.approx([X0, Y0, 1.1])
        assert controller._yaw == pytest.approx(YAW0)
    finally:
        engine.shutdown()
