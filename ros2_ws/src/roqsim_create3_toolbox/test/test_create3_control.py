"""create3_control: the reference diffdrive_controller on mock hardware beside the simulator."""

import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
import yaml

CONFIG = Path(__file__).resolve().parents[1] / "config"


def _overrides():
    with open(CONFIG / "create3_control.yaml", encoding="utf-8") as handle:
        return yaml.safe_load(handle)["/**"]


def test_the_mock_description_gives_the_controller_its_two_wheels():
    robot = ET.parse(CONFIG / "create3_control.urdf").getroot()
    (control,) = robot.findall("ros2_control")
    assert control.find("hardware/plugin").text == "mock_components/GenericSystem"
    joints = {j.get("name"): j for j in control.findall("joint")}
    assert sorted(joints) == ["left_wheel_joint", "right_wheel_joint"]
    for joint in joints.values():
        assert [c.get("name") for c in joint.findall("command_interface")] == ["velocity"]
        states = sorted(s.get("name") for s in joint.findall("state_interface"))
        assert states == ["position", "velocity"]


def _released_control_share() -> Path:
    """irobot_create_control's share directory; skips the test where it is not installed."""
    ament = pytest.importorskip("ament_index_python.packages")
    try:
        return Path(ament.get_package_share_directory("irobot_create_control"))
    except ament.PackageNotFoundError:
        pytest.skip("irobot_create_control is not installed")


def test_the_wheels_are_the_ones_the_released_control_yaml_names():
    share = _released_control_share()
    with open(share / "config" / "control.yaml", encoding="utf-8") as handle:
        params = yaml.safe_load(handle)["/**"]["diffdrive_controller"]["ros__parameters"]
    robot = ET.parse(CONFIG / "create3_control.urdf").getroot()
    mocked = sorted(j.get("name") for j in robot.findall("ros2_control/joint"))
    assert mocked == sorted(params["left_wheel_names"] + params["right_wheel_names"])


def test_the_overrides_publish_the_limited_command_and_leave_odometry_to_the_simulator():
    controller = _overrides()["diffdrive_controller"]["ros__parameters"]
    assert controller["publish_limited_velocity"] is True
    assert controller["enable_odom_tf"] is False
    assert _overrides()["controller_manager"]["ros__parameters"]["update_rate"] == 62


def test_the_launch_file_spawns_the_controller_and_not_the_joint_state_broadcaster():
    from launch_ros.actions import Node
    from roqsim_create3_toolbox_launch import create3_control

    _released_control_share()  # the launch file reads its control.yaml
    nodes = [
        a for a in create3_control.generate_launch_description().entities if isinstance(a, Node)
    ]
    executables = sorted(n._Node__node_executable for n in nodes)
    assert executables == ["robot_state_publisher", "ros2_control_node", "spawner"]
    (spawner,) = [n for n in nodes if n._Node__node_executable == "spawner"]
    spawned = [str(a) for a in spawner._Node__arguments]
    assert "diffdrive_controller" in spawned
    assert "joint_state_broadcaster" not in spawned


def test_the_node_graphs_offer_the_controller_off_by_default():
    from launch.actions import DeclareLaunchArgument
    from roqsim_create3_toolbox_launch import create3_nodes, turtlebot4_nodes

    for module in (create3_nodes, turtlebot4_nodes):
        declared = {
            a.name: a.default_value
            for a in module.generate_launch_description().entities
            if isinstance(a, DeclareLaunchArgument)
        }
        assert "ros2_control" in declared
        assert "".join(s.text for s in declared["ros2_control"]) == "false"
