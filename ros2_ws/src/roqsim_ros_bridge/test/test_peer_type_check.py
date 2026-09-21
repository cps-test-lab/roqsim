# SPDX-License-Identifier: Apache-2.0
"""A peer of another type on one of the bridge's topics fails the run instead of going silent.

A ROS 2 topic is one name and one type: a stack publishing a plain ``Twist`` on ``cmd_vel`` to a
base subscribing a ``TwistStamped`` is not a degraded connection but none, and neither side logs
it. The symptom is a robot that never moves, which in a campaign reads as a planner failure.

The filter is tested ROS-free; the end-to-end test stands up the bridge, publishes the wrong type
at it from a second node and expects the stepping thread to raise, naming both sides. It skips
without ROS, like the other bridge tests that need a graph.
"""

from __future__ import annotations

import shutil
import subprocess
import threading
import time
from dataclasses import dataclass

import pytest

from roqsim_ros_bridge.ros2_bridge import _foreign_types, _ros_type_name


@dataclass
class _Peer:
    node_name: str
    node_namespace: str
    topic_type: str


def test_the_graph_spelling_of_a_type():
    assert _ros_type_name("geometry_msgs.msg.TwistStamped") == "geometry_msgs/msg/TwistStamped"


def test_only_a_peer_of_another_type_is_foreign_and_our_own_node_never_is():
    peers = [
        _Peer("velocity_smoother", "/", "geometry_msgs/msg/Twist"),
        _Peer("teleop", "/ops", "geometry_msgs/msg/TwistStamped"),
        _Peer("roqsim_bridge", "/", "geometry_msgs/msg/Twist"),  # our own other endpoint
    ]
    assert _foreign_types(peers, "geometry_msgs/msg/TwistStamped", "roqsim_bridge") == [
        ("/velocity_smoother", "geometry_msgs/msg/Twist"),
    ]
    assert _foreign_types(peers, "geometry_msgs/msg/Twist", "roqsim_bridge") == [
        ("/ops/teleop", "geometry_msgs/msg/TwistStamped"),
    ]
    assert _foreign_types([], "geometry_msgs/msg/Twist", "roqsim_bridge") == []


def test_a_twist_published_at_a_stamped_base_fails_the_run(tmp_path):
    roqsim = pytest.importorskip("roqsim")  # noqa: F841  selects the GL backend first
    pytest.importorskip("rclpy")
    ros2 = shutil.which("ros2")
    if ros2 is None:
        pytest.skip("ros2 CLI not on PATH (ROS not sourced)")
    from roqsim.config import load_config_from_dict, with_transport
    from roqsim.engine import Engine

    raw = {
        "sim": {"pacing": "asap"},
        "components": [
            {
                "spawn_robot": {"model": "turtlebot4"},
                "name": "robot",
                "components": [
                    {"diff_drive": {"stamped_cmd_vel": True}},
                    {
                        "spawn_sensor": {},
                        "name": "oakd",
                        "components": [{"oakd_camera": {}, "enabled": False}],
                    },
                ],
            }
        ],
    }
    # The stack in its own process: one DDS participant refuses a second type on a topic it
    # already carries, which is exactly the situation two processes get into without a word.
    stack = subprocess.Popen(
        [ros2, "topic", "pub", "-r", "10", "/cmd_vel", "geometry_msgs/msg/Twist", "{}"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    engine = Engine(load_config_from_dict(with_transport(raw, ros=True), base_dir=tmp_path))
    engine.setup()
    engine.reset()
    failure: list[BaseException] = []
    stop = threading.Event()

    def run():
        try:
            while not stop.is_set():
                engine.step()
        except BaseException as exc:  # noqa: BLE001  the failure is what the test reads
            failure.append(exc)

    threading.Thread(target=run, daemon=True).start()
    try:
        deadline = time.monotonic() + 30.0
        while not failure and time.monotonic() < deadline:
            time.sleep(0.1)
    finally:
        stop.set()
        stack.terminate()
        stack.wait(timeout=10)
        engine.shutdown()
    assert failure, "the wrong type on cmd_vel went unnoticed"
    message = str(failure[0])
    assert isinstance(failure[0], RuntimeError)
    assert "'/cmd_vel'" in message
    assert "geometry_msgs/msg/TwistStamped" in message and "geometry_msgs/msg/Twist " in message
    assert " publishes geometry_msgs/msg/Twist on it" in message
    assert "stamped_cmd_vel" in message
