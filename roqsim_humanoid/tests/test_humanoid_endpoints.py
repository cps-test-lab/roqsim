"""The humanoid plugins' endpoints: declared on their methods, typed, and wired to the controller."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from roqsim.config import load_config_from_dict
from roqsim.engine import Engine
from roqsim.introspection import get_plugin_details
from roqsim.types import JointState, Odometry

#: plugin -> its model and the step its controller is tuned for
CASES = {
    "g1_locomotion": ("unitree_g1", 0.002),
    "oli_locomotion": ("oli", 0.001),
    "agibot_g2_controller": ("agibot_g2", 0.002),
}


def _engine(plugin, **config):
    model, timestep = CASES[plugin]
    world = {
        "sim": {"timestep": timestep},
        "components": [
            {
                "spawn_robot": {"model": model, "prefix": "r_", "default_plugins": False},
                "name": "r",
                "components": [{plugin: config}],
            }
        ],
    }
    engine = Engine(load_config_from_dict(world, base_dir=Path(".")))
    engine.ctx.seed = 1
    engine.setup()
    engine.reset()
    return engine


def _plugin(engine):
    """The plugin under test: the one component the robot carries."""
    return next(p for p in engine.plugins if p.entity == "r")


def _endpoints(engine):
    return {e.name: e for e in engine.ctx.interface.all() if e.owner == "r"}


@pytest.mark.parametrize("plugin", ["g1_locomotion", "oli_locomotion"])
def test_cmd_vel_is_a_twist_stream_the_policy_receives(plugin):
    engine = _engine(plugin)
    try:
        cmd_vel = _endpoints(engine)["cmd_vel"]
        assert [(p.name, p.type.unit, p.required) for p in cmd_vel.params] == [
            ("vx", "m/s", True),
            ("vy", "m/s", False),
            ("wz", "rad/s", False),
        ]
        loco = _plugin(engine)
        cmd_vel.write({"vx": 0.2, "wz": 0.1})
        assert not loco._cmd.any(), "a stream is applied on the physics thread, not by write"
        engine.step()
        assert loco._cmd.tolist() == pytest.approx([0.2, 0.0, 0.1])
    finally:
        engine.shutdown()


@pytest.mark.parametrize("plugin", ["g1_locomotion", "oli_locomotion"])
def test_odom_and_joint_states_are_neutral_types(plugin):
    rows = {row["name"]: row for row in get_plugin_details(plugin)["endpoints"]}
    assert set(rows) == {"cmd_vel", "odom", "joint_states"}
    assert rows["cmd_vel"]["payload"] == "Twist"
    assert rows["odom"]["payload"] == "Odometry" and rows["odom"]["rate_hz"] == 50.0
    assert rows["joint_states"]["payload"] == "JointState"
    assert rows["joint_states"]["rate_hz"] == 50.0

    engine = _engine(plugin)
    try:
        eps = _endpoints(engine)
        odom = eps["odom"].read()
        assert isinstance(odom, Odometry)
        assert odom.position[2] > 0.5, "the pelvis height is carried as z"
        joints = eps["joint_states"].read()
        assert isinstance(joints, JointState)
        assert len(joints.names) == len(joints.positions) == len(joints.velocities) > 0
    finally:
        engine.shutdown()


def test_agibot_joint_command_is_a_command_applied_in_order():
    engine = _engine("agibot_g2_controller")
    try:
        eps = _endpoints(engine)
        command = eps["joint_command"]
        assert command.payload_type.cls.__name__ == "JointPositions"
        first, second = eps["joint_states"].read().names[:2]
        # Two partial commands within one step: a stream would keep only the second.
        a = command.write({"names": [first], "positions": [0.3]})
        b = command.write({"names": [second], "positions": [-0.2]})
        engine.step()
        assert a.result(0) is None and b.result(0) is None
        controller = _plugin(engine)
        target = dict(zip(controller._jnames, controller._target, strict=True))
        assert target[first] == pytest.approx(0.3) and target[second] == pytest.approx(-0.2)
    finally:
        engine.shutdown()


def test_agibot_joint_states_reads_the_driven_joints():
    engine = _engine("agibot_g2_controller")
    try:
        joints = _endpoints(engine)["joint_states"].read()
        assert isinstance(joints, JointState)
        assert len(joints.names) == len(joints.positions) == len(joints.velocities) > 0
        assert not any(n.startswith("r_") for n in joints.names)
        assert isinstance(joints.positions, np.ndarray)
    finally:
        engine.shutdown()
