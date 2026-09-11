# Copyright (C) 2026 Frederik Pasch
# SPDX-License-Identifier: Apache-2.0
"""A transition is readable by whoever asks, whenever they ask.

A controller's transition is a fact about a moment that has already passed, and everything that
wants it subscribes afterwards: a trial that switches and then asks when the hand-over happened, a
recorder attached after the run began, a scenario waiting on the event after its service call
returned. Published with volatile durability, every one of those receives nothing -- the topic
exists, the publisher is there, and the record is silently unreadable.
"""

from __future__ import annotations

import pytest
from rclpy.qos import DurabilityPolicy

from roqsim.controllers import ACTIVE, Controller, registry_for


class _Blackboard:
    def __init__(self):
        self._d = {}

    def get(self, key, default=None):
        return self._d.get(key, default)

    def set(self, key, value):
        self._d[key] = value


class _Ctx:
    def __init__(self):
        self.blackboard = _Blackboard()
        self.sim_time = 0.0


class _FakeNode:
    """Records what was advertised, so the QoS is inspectable without a ROS graph."""

    def __init__(self):
        self.publishers: dict[str, object] = {}
        self.services: list[str] = []

    def create_publisher(self, msg_type, topic, qos):
        self.publishers[topic] = qos
        return object()

    def create_service(self, srv_type, name, handler):
        self.services.append(name)
        return object()


def _served():
    from roqsim_ros_bridge.controller_manager import serve

    ctx = _Ctx()
    registry_for(ctx).register(
        Controller(
            name="arm_controller",
            type="joint_trajectory_controller/JointTrajectoryController",
            claims=("shoulder_pan_joint/position",),
            state=ACTIVE,
            namespace="ur5e",
            owner="ur5e",
        )
    )
    node = _FakeNode()
    serve(node, ctx)
    return node


def test_a_transition_event_is_latched_for_a_late_subscriber():
    node = _served()
    topic = "/ur5e/arm_controller/transition_event"
    assert topic in node.publishers, f"advertised: {sorted(node.publishers)}"
    assert node.publishers[topic].durability == DurabilityPolicy.TRANSIENT_LOCAL


def test_the_history_keeps_more_than_the_last_transition():
    """A trial may switch several times; a depth of one would leave only the final hand-over
    readable, and the interesting one is usually not the last."""
    node = _served()
    assert node.publishers["/ur5e/arm_controller/transition_event"].depth > 1


@pytest.mark.parametrize(
    "service",
    [
        "list_controllers",
        "switch_controller",
        "load_controller",
        "configure_controller",
        "unload_controller",
        "list_hardware_interfaces",
    ],
)
def test_the_manager_serves_what_the_standard_tooling_calls(service):
    """`ros2 run controller_manager spawner` and the `ros2 control` CLI call these by name; a
    missing one is not a smaller surface, it is a tool that does not work."""
    assert f"/ur5e/controller_manager/{service}" in _served().services


def test_a_robot_with_no_controllers_is_served_no_manager():
    """Nothing in a world opts into this, so the absence has to be the answer for a world that has
    no controllers rather than an empty manager answering for a robot that is not there."""
    from roqsim_ros_bridge.controller_manager import serve

    node = _FakeNode()
    assert serve(node, _Ctx()) == []
    assert node.services == []
