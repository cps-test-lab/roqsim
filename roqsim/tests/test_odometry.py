"""The command watchdog and the spawn-pose odom frame every velocity-commanded controller shares."""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from roqsim.odometry import CommandWatchdog, SpawnFrame, planar_odom
from roqsim.pose import rpy_to_quat


def _at(t):
    return SimpleNamespace(sim_time=t)


def test_the_watchdog_expires_a_command_after_its_timeout():
    dog = CommandWatchdog.from_config({"cmd_vel_timeout": 0.5})
    dog.stamp(_at(1.0))
    assert not dog.expired(_at(1.5))
    assert dog.expired(_at(1.51))


def test_a_zero_timeout_never_expires():
    dog = CommandWatchdog.from_config({})
    dog.stamp(_at(0.0))
    assert not dog.expired(_at(1e6))


def test_a_cleared_watchdog_holds_no_command():
    dog = CommandWatchdog(0.5)
    dog.stamp(_at(1.0))
    dog.clear()
    assert dog.expired(_at(0.0))


def test_a_negative_timeout_is_refused():
    assert CommandWatchdog.validate({"cmd_vel_timeout": -1}) != []
    assert CommandWatchdog.validate({"cmd_vel_timeout": 0.5}) == []


def _frame(x, y, yaw):
    frame = SpawnFrame()
    frame.capture((x, y, 0.7), rpy_to_quat(0.0, 0.0, yaw))
    return frame


def test_the_spawn_pose_is_the_origin_and_heights_are_kept():
    frame = _frame(3.0, -2.0, 2.0)
    assert frame.position((3.0, -2.0, 0.7)) == pytest.approx((0.0, 0.0, 0.7))
    assert frame.yaw(rpy_to_quat(0.0, 0.0, 2.0)) == pytest.approx(0.0, abs=1e-12)


def test_ahead_of_the_spawn_heading_is_odom_x():
    frame = _frame(1.0, 1.0, math.pi / 2)
    x, y, _ = frame.position((1.0, 3.0, 0.0))  # 2 m along world +y, which the robot faced
    assert (x, y) == pytest.approx((2.0, 0.0))
    assert frame.yaw(rpy_to_quat(0.0, 0.0, math.pi)) == pytest.approx(math.pi / 2)


def test_orientation_keeps_tilt_and_takes_off_the_spawn_heading():
    frame = _frame(0.0, 0.0, 1.0)
    got = frame.orientation(rpy_to_quat(0.2, -0.1, 1.3))
    want = rpy_to_quat(0.2, -0.1, 0.3)
    assert abs(float(np.dot(got, want))) == pytest.approx(1.0, abs=1e-9)


def test_planar_odom_reports_the_body_twist():
    yaw = 0.8
    data = SimpleNamespace(
        xpos=np.array([[0.0, 0.0, 0.0], [2.0, 1.0, 0.5]]),
        xquat=np.array([[1.0, 0.0, 0.0, 0.0], rpy_to_quat(0.0, 0.0, yaw)]),
        # world-frame linear velocity 0.4 m/s along the heading, 0.3 rad/s body yaw rate
        qvel=np.array([0.4 * math.cos(yaw), 0.4 * math.sin(yaw), 0.0, 0.0, 0.0, 0.3]),
    )
    x, y, th, vx, vy, w, z = planar_odom(_frame(2.0, 1.0, yaw), data, body=1, dof=0)
    assert (x, y, th, z) == pytest.approx((0.0, 0.0, 0.0, 0.5), abs=1e-12)
    assert (vx, vy, w) == pytest.approx((0.4, 0.0, 0.3))
