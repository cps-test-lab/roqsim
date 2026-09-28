"""``diff_drive``'s contract with a real stack: the watchdog, the odom rate, one joint_states.

Three keys a world running a robot's own software stack sets, each checked on a real base (the
TurtleBot 3 Waffle, a true differential drive): a command expires after ``cmd_vel_timeout`` and
the base stops through its ramp; ``odom_rate_hz`` is the rate the odom and joint_states endpoints
declare; and ``publish_joint_states: false`` leaves the topic to a ``joint_state_publisher`` that
carries every joint in one message.
"""

from __future__ import annotations

from pathlib import Path

import pytest

# `roqsim` selects MuJoCo's GL backend on import, so it comes before anything that imports mujoco
# (see test_wheels_roll.py).
import roqsim  # noqa: F401, I001
from roqsim.config import load_config_from_dict
from roqsim.engine import Engine
from roqsim_mobile.plugins.diff_drive import DiffDrivePlugin

MODEL = "turtlebot3_waffle"


def _engine(drive_cfg=None, extra=()):
    components = [{"diff_drive": dict(drive_cfg or {})}, *extra]
    world = {
        "sim": {"timestep": 0.002},
        "components": [
            {"spawn_robot": {"model": MODEL, "prefix": "z_"}, "name": "z", "components": components}
        ],
    }
    engine = Engine(load_config_from_dict(world, base_dir=Path(".")))
    engine.ctx.seed = 7
    engine.setup()
    engine.reset()
    return engine


def _speed(engine) -> float:
    return float(engine.ctx.blackboard.get("robot:z").read_odom()[3])


def _endpoints(engine, name):
    return [e for e in engine.ctx.interface.all() if e.name == name]


def test_without_a_watchdog_one_command_holds_forever():
    """The default, for an in-process driver that sets a twist once and steps."""
    engine = _engine()
    try:
        engine.ctx.blackboard.get("robot:z").drive(0.2, 0.0, 0.0)
        for _ in range(1500):  # 3 s
            engine.step()
        assert _speed(engine) == pytest.approx(0.2, abs=0.03)
    finally:
        engine.shutdown()


def test_the_watchdog_stops_the_base_after_the_timeout():
    """A stack that dies mid-run leaves a stationary robot, not one driving on its last command."""
    engine = _engine({"cmd_vel_timeout": 0.5})
    try:
        handle = engine.ctx.blackboard.get("robot:z")
        handle.drive(0.2, 0.0, 0.0)
        for _ in range(200):  # 0.4 s: still within the timeout
            engine.step()
        assert _speed(engine) > 0.1, "the command should still be live at 0.4 s"
        for _ in range(600):  # to 1.6 s: expired at 0.5 s, then the ramp down
            engine.step()
        assert _speed(engine) == pytest.approx(0.0, abs=0.01)
        # A fresh command restarts it: the watchdog expires commands, not the base.
        handle.drive(0.2, 0.0, 0.0)
        for _ in range(200):
            engine.step()
        assert _speed(engine) > 0.1
    finally:
        engine.shutdown()


def test_a_command_refreshed_within_the_timeout_keeps_driving():
    """What a live stack does: republish at a rate, each command extending the deadline."""
    engine = _engine({"cmd_vel_timeout": 0.5})
    try:
        handle = engine.ctx.blackboard.get("robot:z")
        for _ in range(15):  # 3 s, a command every 0.2 s
            handle.drive(0.2, 0.0, 0.0)
            for _ in range(100):
                engine.step()
        assert _speed(engine) == pytest.approx(0.2, abs=0.03)
    finally:
        engine.shutdown()


def test_odom_rate_is_declared_on_odom_and_joint_states():
    engine = _engine({"odom_rate_hz": 62.0})
    try:
        (odom,) = _endpoints(engine, "odom")
        (joints,) = _endpoints(engine, "joint_states")
        assert odom.rate_hz == 62.0
        assert joints.rate_hz == 62.0
    finally:
        engine.shutdown()


def test_publish_joint_states_false_leaves_the_topic_to_a_joint_state_publisher():
    """One message with every joint: the base's own is off, the generic publisher carries the
    wheels beside every other hinge and slide joint of the robot."""
    engine = _engine({"publish_joint_states": False}, extra=[{"joint_state_publisher": {}}])
    try:
        (joints,) = _endpoints(engine, "joint_states")
        names, pos, vel, eff = joints.read()
        assert "left_wheel_joint" in names and "right_wheel_joint" in names
        assert len(pos) == len(vel) == len(eff) == len(names)
        drive = next(p for p in engine.plugins if isinstance(p, DiffDrivePlugin))
        assert drive.publish_joint_states is False
        # And it reads the wheels turning.
        engine.ctx.blackboard.get("robot:z").drive(0.2, 0.0, 0.0)
        for _ in range(500):
            engine.step()
        i = names.index("left_wheel_joint")
        assert abs(float(joints.read()[2][i])) > 1.0
    finally:
        engine.shutdown()


def test_by_default_the_base_publishes_its_own_joint_states():
    engine = _engine()
    try:
        assert len(_endpoints(engine, "joint_states")) == 1
    finally:
        engine.shutdown()


@pytest.mark.parametrize("bad", [{"cmd_vel_timeout": -1}, {"odom_rate_hz": 0}])
def test_bad_values_are_reported(bad):
    assert DiffDrivePlugin(bad, entity="z").validate_config(bad) != []


def test_the_watchdog_ramps_rather_than_cuts():
    """The stop is a command to zero, through wheel_accel_limit, so the base does not lurch."""
    engine = _engine({"cmd_vel_timeout": 0.6, "wheel_accel_limit": 0.5})
    try:
        engine.ctx.blackboard.get("robot:z").drive(0.2, 0.0, 0.0)
        speeds = []
        for _ in range(650):  # 1.3 s
            engine.step()
            speeds.append(_speed(engine))
        # Up to 0.2 m/s by 0.4 s; expired at 0.6 s; at 0.5 m/s^2 it takes 0.4 s to come down, so
        # at 0.7 s the base is still moving and at 1.2 s it is not.
        assert speeds[349] > 0.1
        assert speeds[599] == pytest.approx(0.0, abs=0.02)
    finally:
        engine.shutdown()
