# Copyright (C) 2026 Frederik Pasch
#
# SPDX-License-Identifier: Apache-2.0

"""The typed endpoints of ``ackermann_drive``, ``omni_drive`` and ``spawn_robot``.

Each endpoint's ROS hints, owner, namespace and rate are what the plugin declared before its methods
carried them, and a named write reaches the drive once per step.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from roqsim.config import load_config_from_dict
from roqsim.endpoint import ParameterError
from roqsim.engine import Engine
from roqsim_mobile.plugins.ackermann_drive import AckermannDrivePlugin
from roqsim_mobile.plugins.spawn_robot import SpawnRobotPlugin


def _engine(model: str, **spawn) -> Engine:
    world = {
        "sim": {},
        "components": [{"spawn_robot": {"model": model, **spawn}, "name": "bot"}],
    }
    engine = Engine(load_config_from_dict(world, base_dir=Path(".")), preview=True)
    engine.setup()
    engine.reset()
    return engine


def _endpoints(engine: Engine) -> dict:
    return {e.name: e for e in engine.ctx.interface.all() if e.owner == "bot"}


def _plugin(engine: Engine, cls: type):
    return next(p for p in engine.plugins if isinstance(p, cls))


def test_ackermann_endpoints_keep_their_ros_interface():
    eps = _endpoints(_engine("piracer", namespace="car"))
    assert {n: (e.direction, e.namespace, e.rate_hz) for n, e in eps.items()} == {
        "cmd_vel": ("in", "car", 0.0),
        "ackermann_cmd": ("in", "car", 0.0),
        "odom": ("out", "car", 50.0),
        "joint_states": ("out", "car", 50.0),
    }
    assert eps["cmd_vel"].backend == {
        "ros2": {"type": "geometry_msgs.msg.Twist", "topic": "cmd_vel"}
    }
    assert eps["ackermann_cmd"].backend == {
        "ros2": {"type": "ackermann_msgs.msg.AckermannDriveStamped", "topic": "drive"}
    }
    assert eps["odom"].backend["ros2"]["child_frame_id"] == "base_link"
    assert [p.name for p in eps["ackermann_cmd"].params] == ["steering_angle", "speed"]
    assert eps["ackermann_cmd"].params[0].type.unit == "rad"


def test_an_ackermann_command_by_name_turns_the_wheels_at_rest():
    engine = _engine("piracer")
    eps = _endpoints(engine)
    eps["ackermann_cmd"].write({"steering_angle": 0.3, "speed": 0.0})
    for _ in range(200):
        engine.step()
    drive = _plugin(engine, AckermannDrivePlugin)
    assert drive._steer == pytest.approx(0.3, abs=1e-6)
    names, positions, _ = eps["joint_states"].read()
    assert list(names[:2]) == drive.steer_joint_names
    assert all(abs(p) > 0.1 for p in positions[:2]), "the steer joints follow the command"


def test_a_twist_by_name_drives_the_car_and_a_misfit_is_refused():
    engine = _engine("piracer")
    eps = _endpoints(engine)
    with pytest.raises(ParameterError, match="speed"):
        eps["ackermann_cmd"].write({"steering_angle": 0.1, "sped": 0.2})
    eps["cmd_vel"].write({"vx": 0.5})
    for _ in range(300):
        engine.step()
    assert eps["odom"].read()[3] > 0.2


def test_omni_endpoints_keep_their_ros_interface_and_strafe_by_name():
    engine = _engine("lgdxrobot2")
    eps = _endpoints(engine)
    assert {n: (e.direction, e.rate_hz) for n, e in eps.items() if n != "frames"} == {
        "cmd_vel": ("in", 0.0),
        "odom": ("out", 50.0),
        "joint_states": ("out", 50.0),
    }
    assert eps["odom"].backend["ros2"]["child_frame_id"] == "base_footprint"
    eps["cmd_vel"].write({"vx": 0.0, "vy": 0.3})
    for _ in range(500):
        engine.step()
    assert eps["odom"].read()[4] > 0.1, "a holonomic base honours vy"


def test_spawn_robot_frames_belong_to_the_robot_and_its_namespace():
    engine = _engine("turtlebot4", namespace="tb")
    frames = next(e for e in engine.ctx.interface.all() if e.name == "frames" and e.owner == "bot")
    spawn = _plugin(engine, SpawnRobotPlugin)
    assert frames.namespace == "tb" and frames.direction == "out"
    assert frames.backend == {
        "ros2": {
            "type": "tf2_msgs.msg.TFMessage",
            "topic": "tf",
            "frame_id": spawn.frame_transforms[0]["parent"],
            "static_tf": spawn.frame_transforms,
        }
    }
    assert frames.read() is None


def test_a_robot_without_frames_has_no_frames_endpoint():
    engine = _engine("piracer")
    assert "frames" not in _endpoints(engine)
