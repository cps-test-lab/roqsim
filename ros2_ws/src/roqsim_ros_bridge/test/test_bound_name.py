# SPDX-License-Identifier: Apache-2.0
"""The bridge says what each endpoint is called on ROS, as its publisher and servers resolved it.

Another transport describing an endpoint (the control socket's ``describe``) asks
``bound_name``, so the name it reports is the one after the node namespace, the endpoint's own
namespace, a ``topics:`` rename, a stripped namespace and an absolute topic -- not a re-derivation --
with the QoS the bridge used.
"""

from __future__ import annotations

import os

import pytest

pytest.importorskip("roqsim")  # selects the GL backend before mujoco is imported
pytest.importorskip("rclpy")

import mujoco  # noqa: E402

from roqsim.context import Endpoint, Entity, SimContext  # noqa: E402
from roqsim.endpoint import qos_profile  # noqa: E402
from roqsim.plugins.contact_monitor import ContactMonitorPlugin  # noqa: E402
from roqsim_ros_bridge.ros2_bridge import Ros2Bridge  # noqa: E402

SCENE = """
<mujoco>
  <worldbody>
    <geom name="floor" type="plane" size="2 2 0.1"/>
    <body name="crate" pos="0 0 0.2"><freejoint/><geom name="crate" size="0.05" mass="1"/></body>
    <body name="crate_b" pos="1 0 0.2"><freejoint/><geom name="crate_b" size="0.05" mass="1"/></body>
  </worldbody>
</mujoco>
"""


def test_bound_names_are_the_resolved_ros_names():
    ctx = SimContext(config={})
    ctx.model = mujoco.MjModel.from_xml_string(SCENE)
    ctx.data = mujoco.MjData(ctx.model)
    ctx.entities.add(Entity(name="parcel", kind="object", body="crate", meta={"namespace": "p"}))
    ctx.entities.add(Entity(name="spare", kind="object", body="crate_b", meta={"namespace": "b"}))
    for config, entity in (
        ({"ignore": []}, "parcel"),
        ({"ignore": [], "topics": {"contact": "bump"}}, "spare"),
    ):
        monitor = ContactMonitorPlugin(config, entity=entity)
        monitor.configure(ctx)
        monitor.register_endpoints(ctx)
    absolute = Endpoint(
        name="seconds",
        direction="out",
        owner="parcel",
        namespace="p",
        read=lambda: 0.0,
        backend={"ros2": {"type": "std_msgs.msg.Float64", "topic": "/hw/seconds"}},
    )
    ctx.interface.add(absolute)
    domain = 100 + os.getpid() % 100
    bridge = Ros2Bridge(
        {"namespace": "sim1", "domain_id": domain, "strip_namespace": "b", "clock_rate_hz": 0}
    )
    bridge.configure(ctx)
    try:
        named = {
            (ep.owner, ep.name): bridge.bound_name(ep)["topic"]
            for ep in ctx.interface.all()
            if ep.direction == "out"
        }
        assert named == {
            ("parcel", "contact"): "/sim1/p/collision",  # node namespace + endpoint namespace
            ("spare", "contact"): "/sim1/bump",  # a `topics:` rename, its namespace stripped
            ("parcel", "seconds"): "/hw/seconds",  # absolute: verbatim
        }
        assert bridge.bound_name(absolute)["type"] == "std_msgs.msg.Float64"
        assert bridge.bound_name(absolute)["qos"] == qos_profile("default")
        assert bridge.bound_name(Endpoint(name="unbound", direction="out")) is None
    finally:
        bridge.shutdown(ctx)
