"""arm_controller's JointVelocities on ROS: the mapping its package registers through the
``roqsim.ros2_types`` entry point."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("roqsim_manipulation")
pytest.importorskip("rclpy")

from builtin_interfaces.msg import Time  # noqa: E402
from roqsim_manipulation.plugins.arm_controller import (  # noqa: E402
    ArmControllerPlugin,
    JointVelocities,
)
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint  # noqa: E402

from roqsim.config import load_config_from_dict  # noqa: E402
from roqsim.engine import Engine  # noqa: E402
from roqsim_ros_bridge.registry import resolve_type  # noqa: E402
from roqsim_ros_bridge.typemap import resolve  # noqa: E402


def test_a_joint_velocity_trajectory_reaches_the_arm_with_its_positions_as_velocities():
    """arm_controller's JointVelocities is mapped by its package's roqsim.ros2_types entry: a
    JointTrajectory whose last point's positions carry the velocities, both ways."""
    world = {
        "sim": {"timestep": 0.001},
        "components": [
            {
                "spawn_arm": {"model": "ur5e", "prefix": "ur5e_", "namespace": "ur5e"},
                "name": "ur5e",
                "components": [{"arm_controller": {"velocity_commands": True}}],
            }
        ],
    }
    engine = Engine(load_config_from_dict(world, base_dir=Path(".")))
    engine.setup()
    engine.reset()
    try:
        ep = next(e for e in engine.ctx.interface.all() if e.name == "joint_velocity")
        binding = resolve(ep)
        assert binding.hints["type"] == "trajectory_msgs.msg.JointTrajectory"
        assert binding.hints["topic"] == "arm_controller/joint_velocity"
        msg = JointTrajectory(
            joint_names=["shoulder_pan_joint"], points=[JointTrajectoryPoint(positions=[0.1])]
        )
        ep.write(binding.decode(msg))
        engine.step()
        arm = next(p for p in engine.plugins if isinstance(p, ArmControllerPlugin))
        assert arm._vel_cmd == {"shoulder_pan_joint": pytest.approx(0.1)}

        out = resolve_type(binding.hints["type"])()
        binding.fill(out, JointVelocities(["a"], [0.5]), Time(sec=1), binding.hints)
        assert list(out.joint_names) == ["a"] and list(out.points[0].positions) == [0.5]
    finally:
        engine.shutdown()
