"""The bumper adapter: a pressed zone is a BUMP event framed by the zone, and only while pressed.

Run with the ROS overlay sourced (`irobot_create_msgs` and `rclpy` are ROS packages); the node is
driven directly rather than through a launch, since what is under test is the conversion.
"""

import pytest

rclpy = pytest.importorskip("rclpy")
pytest.importorskip("irobot_create_msgs")

from irobot_create_msgs.msg import HazardDetection  # noqa: E402
from roqsim_create3_toolbox.bumper_hazard import DEFAULT_ZONES, BumperHazard  # noqa: E402
from std_msgs.msg import Bool  # noqa: E402


@pytest.fixture
def node():
    rclpy.init()
    n = BumperHazard()
    yield n
    n.destroy_node()
    rclpy.shutdown()


def _press(node, zone, pressed=True):
    node._make_callback(zone)(Bool(data=pressed))


def test_nothing_pressed_is_no_event(node):
    assert node.hazards() == []


def test_a_pressed_zone_is_one_bump_event_framed_by_the_zone(node):
    _press(node, "bump_front_center")
    (msg,) = node.hazards()
    assert msg.type == HazardDetection.BUMP
    assert msg.header.frame_id == "bump_front_center"


def test_two_pressed_zones_are_two_events_and_a_release_ends_one(node):
    _press(node, "bump_left")
    _press(node, "bump_front_left")
    assert {m.header.frame_id for m in node.hazards()} == {"bump_left", "bump_front_left"}
    _press(node, "bump_left", pressed=False)
    assert [m.header.frame_id for m in node.hazards()] == ["bump_front_left"]


def test_the_default_zones_are_the_create3s_five(node):
    assert DEFAULT_ZONES == [
        "bump_left",
        "bump_front_left",
        "bump_front_center",
        "bump_front_right",
        "bump_right",
    ]
    assert [s.topic_name for s in node._subs] == [f"/bumper/{z}" for z in DEFAULT_ZONES]
    assert node._pub.topic_name == "/_internal/bumper/event"


def test_the_launch_files_describe_the_reference_node_graph():
    """Both launch descriptions build, and the Create 3 one lists the reference simulator's nodes."""
    from launch_ros.actions import Node
    from roqsim_create3_toolbox_launch import create3_nodes, turtlebot4_nodes

    nodes = [a for a in create3_nodes.generate_launch_description().entities if isinstance(a, Node)]
    executables = sorted(n._Node__node_executable for n in nodes)
    assert executables == sorted(
        [
            "pose_republisher_node",
            "sensors_node",
            "interface_buttons_node",
            "bumper_hazard",
            "hazards_vector_publisher",
            "ir_intensity_vector_publisher",
            "motion_control",
            "wheel_status_publisher",
            "mock_publisher",
            "robot_state",
            "kidnap_estimator_publisher",
            "ui_mgr",
        ]
    )
    tb4 = turtlebot4_nodes.generate_launch_description().entities
    assert any(isinstance(a, Node) and a._Node__node_executable == "turtlebot4_node" for a in tb4)
