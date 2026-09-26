# SPDX-License-Identifier: Apache-2.0
"""The bridge says what it publishes, and a scenario reads a report through what it says.

A scenario addresses a plugin's report the way the world names it -- entity and endpoint -- while
the bridge publishes it on a topic made from the endpoint's namespace, a ``topics:`` rename, a
stripped namespace and the node's own namespace. The latched map is the one place the two meet, so
these tests stand up a real bridge on an isolated domain and check that the map names exactly the
topics the bridge's publishers are on, and that ``osc.roqsim``'s ROS access, reading that map from
a second node, receives the value the plugin reports. They skip without ROS, like the other bridge
tests that need a graph.
"""

from __future__ import annotations

import json
import os
import time

import pytest

pytest.importorskip("roqsim")  # selects the GL backend before mujoco is imported
rclpy = pytest.importorskip("rclpy")

import mujoco  # noqa: E402
from rclpy.executors import SingleThreadedExecutor  # noqa: E402
from rclpy.node import Node  # noqa: E402
from rclpy.qos import DurabilityPolicy, QoSProfile  # noqa: E402
from std_msgs.msg import String  # noqa: E402

from roqsim.bridge import ENDPOINT_MAP  # noqa: E402
from roqsim.context import Endpoint, Entity, SimContext  # noqa: E402
from roqsim.plugins.contact_monitor import ContactMonitorPlugin  # noqa: E402
from roqsim_ros_bridge.ros2_bridge import Ros2Bridge  # noqa: E402

SCENE = """
<mujoco>
  <option timestep="0.002"/>
  <worldbody>
    <geom name="floor" type="plane" size="2 2 0.1"/>
    <body name="crate" pos="0 0 0.2"><freejoint/><geom name="crate" size="0.05" mass="1"/></body>
    <body name="crate_b" pos="1 0 0.2"><freejoint/><geom name="crate_b" size="0.05" mass="1"/></body>
  </worldbody>
</mujoco>
"""


def _world():
    """Two watched crates -- one namespaced, one renamed and stripped -- and an absolute topic."""
    ctx = SimContext(config={})
    ctx.model = mujoco.MjModel.from_xml_string(SCENE)
    ctx.data = mujoco.MjData(ctx.model)
    mujoco.mj_forward(ctx.model, ctx.data)
    ctx.entities.add(Entity(name="parcel", kind="object", body="crate", meta={"namespace": "p"}))
    ctx.entities.add(Entity(name="spare", kind="object", body="crate_b", meta={"namespace": "b"}))
    monitors = [
        ContactMonitorPlugin({"ignore": [], "rate_hz": 500.0}, entity="parcel"),
        ContactMonitorPlugin(
            {"ignore": [], "rate_hz": 500.0, "topics": {"contact": "bump"}}, entity="spare"
        ),
    ]
    for monitor in monitors:
        monitor.configure(ctx)
        monitor.on_reset(ctx)
    ctx.interface.add(
        Endpoint(
            name="clock_seconds",
            direction="out",
            owner="parcel",
            namespace="p",
            read=lambda: float(ctx.data.time),
            backend={"ros2": {"type": "std_msgs.msg.Float64", "topic": "/hw/seconds"}},
        )
    )
    return ctx, monitors


@pytest.fixture
def bridge():
    ctx, monitors = _world()
    # An isolated domain per process, so a parallel run of this suite cannot hear this bridge.
    domain = 100 + os.getpid() % 100
    bridge = Ros2Bridge(
        {"namespace": "sim1", "domain_id": domain, "strip_namespace": "b", "clock_rate_hz": 0}
    )
    bridge.configure(ctx)
    try:
        yield bridge, ctx, monitors
    finally:
        bridge.shutdown(ctx)


def _receive_map(bridge) -> dict:
    """The map as a late subscriber gets it: over the latched topic, after the bridge sent it."""
    node = Node("reader", namespace="sim1", context=bridge._context)
    got: list[str] = []
    qos = QoSProfile(depth=10, durability=DurabilityPolicy.TRANSIENT_LOCAL)
    node.create_subscription(String, ENDPOINT_MAP, lambda m: got.append(m.data), qos)
    executor = SingleThreadedExecutor(context=bridge._context)
    executor.add_node(node)
    deadline = time.monotonic() + 10.0
    while not got and time.monotonic() < deadline:
        executor.spin_once(timeout_sec=0.1)
    executor.shutdown()
    node.destroy_node()
    assert got, f"nothing arrived on the latched {ENDPOINT_MAP!r}"
    return json.loads(got[-1])


def test_the_map_names_the_topics_the_bridge_publishes_on(bridge):
    bridge, _ctx, _monitors = bridge
    emap = _receive_map(bridge)
    topics = {(e["owner"], e["name"]): e["topic"] for e in emap["endpoints"]}

    assert topics == {
        ("parcel", "contact"): "/sim1/p/collision",  # node namespace + endpoint namespace
        ("spare", "contact"): "/sim1/bump",  # a `topics:` rename, its namespace stripped
        ("parcel", "clock_seconds"): "/hw/seconds",  # absolute: verbatim
    }
    published = {
        name
        for name, _types in bridge._node.get_publisher_names_and_types_by_node(
            bridge._node.get_name(), bridge._node.get_namespace()
        )
    }
    assert set(topics.values()) <= published, "every mapped topic is one this bridge publishes"
    contact = next(e for e in emap["endpoints"] if e["name"] == "contact")
    assert contact["type"] == "std_msgs.msg.Bool"
    assert contact["field"] == "in_contact"
    assert emap["owners"] is None


def test_a_scenario_reads_a_report_through_the_map(bridge):
    """``osc.roqsim``'s ROS access, from a node of its own: map, then the renamed topic, then the
    value the plugin reports once the crate lands."""
    access_mod = pytest.importorskip("scenario_execution_roqsim.access.ros")
    bridge, ctx, monitors = bridge
    node = Node("scenario", namespace="sim1", context=bridge._context)
    executor = SingleThreadedExecutor(context=bridge._context)
    executor.add_node(node)
    access = access_mod.RosAccess(node)
    try:
        call = access.entity_report("spare", "contact", "")
        reading = None
        deadline = time.monotonic() + 20.0
        while time.monotonic() < deadline:
            mujoco.mj_step(ctx.model, ctx.data)
            for monitor in monitors:
                monitor.post_step(ctx)
            bridge.post_step(ctx)
            executor.spin_once(timeout_sec=0.01)
            reading = call.poll()
            if reading is not None and reading.value is True:
                break
        assert reading is not None, call.pending_reason()
        assert reading.value is True, "the crate landed, so the report says in contact"
        assert reading.field == "in_contact"
        assert reading.source == "/sim1/bump"
    finally:
        access.teardown()
        executor.shutdown()
        node.destroy_node()
