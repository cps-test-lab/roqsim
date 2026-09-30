"""Transforms as payloads: ``Transform`` and ``Transforms`` travel as a ``tf2_msgs/TFMessage``.

Stamped with sim time, parent from the value or else the ``frame_id`` hint (namespaced), child
verbatim; with ``static: true`` the first value goes once to the latched static broadcaster, parent
and child namespaced, as a ``static_tf`` hint's transforms do.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("roqsim")  # selects the GL backend before mujoco is imported
pytest.importorskip("rclpy")

from builtin_interfaces.msg import Time  # noqa: E402
from tf2_msgs.msg import TFMessage  # noqa: E402

from roqsim import endpoint  # noqa: E402
from roqsim import types as T  # noqa: E402
from roqsim.context import SimContext  # noqa: E402
from roqsim.endpoint import QOS_PRESETS, build  # noqa: E402
from roqsim.plugin import Plugin  # noqa: E402
from roqsim_ros_bridge import typemap  # noqa: E402
from roqsim_ros_bridge.ros2_bridge import Ros2Bridge  # noqa: E402

STAMP = Time(sec=3, nanosec=250)
Q = np.array([0.5, 0.5, 0.5, 0.5])
MOUSE = T.Transform("base_link", "mouse", np.array([0.1, 0.0, 0.02]), Q)
CHAIN = T.Transforms(
    [
        T.Transform("base_link", "shell_link", np.array([0.0, 0.0, 0.1])),
        T.Transform("shell_link", "rplidar_link", np.array([0.0, 0.0, 0.2]), Q),
        T.Transform("", "pelvis", np.array([1.0, 2.0, 0.9])),  # the hint's parent
    ]
)


class Frames(Plugin):
    @endpoint.out(ros2={"topic": "pose"})
    def one(self) -> T.Transform:
        return MOUSE

    @endpoint.out(ros2={"topic": "/tf", "frame_id": "odom"})
    def several(self) -> T.Transforms:
        return CHAIN

    @endpoint.out(ros2={"static": True, "frame_id": "base_link"})
    def mounts(self) -> T.Transforms:
        return CHAIN

    @endpoint.out(ros2={"static": True})
    def mount(self) -> T.Transform:
        return MOUSE

    @endpoint.out(ros2={"static": True})
    def not_a_transform(self) -> T.Pose:
        return T.Pose()


@pytest.fixture(scope="module")
def eps():
    return {e.name: e for e in build(Frames({}, label="f"), SimContext(config={}))}


def _same(a: T.Transform, b: T.Transform) -> bool:
    return (
        (a.parent, a.child) == (b.parent, b.child)
        and np.allclose(a.translation, b.translation)
        and np.allclose(a.rotation, b.rotation)
    )


def _fill(binding, value, hints=None) -> TFMessage:
    msg = TFMessage()
    binding.fill(msg, value, STAMP, {**binding.hints, **(hints or {})})
    return msg


def test_one_transform_round_trips_as_a_one_transform_tfmessage(eps):
    binding = typemap.resolve(eps["one"])
    assert (binding.hints["type"], binding.hints["topic"]) == ("tf2_msgs.msg.TFMessage", "pose")
    msg = _fill(binding, MOUSE)
    (tf,) = msg.transforms
    assert (tf.header.stamp, tf.header.frame_id, tf.child_frame_id) == (STAMP, "base_link", "mouse")
    assert (tf.transform.rotation.w, tf.transform.rotation.x) == (0.5, 0.5)
    assert _same(typemap.lookup(T.Transform).wires[0].decode(msg), MOUSE)


def test_one_transform_refuses_a_message_of_several():
    msg = TFMessage()
    typemap.lookup(T.Transforms).wires[0].fill(msg, CHAIN, STAMP, {})
    with pytest.raises(ValueError, match="one transform, got 3"):
        typemap.lookup(T.Transform).wires[0].decode(msg)


def test_several_transforms_round_trip_in_one_tfmessage(eps):
    binding = typemap.resolve(eps["several"])
    assert binding.hints["topic"] == "/tf"
    msg = _fill(binding, CHAIN)
    assert [tf.header.stamp for tf in msg.transforms] == [STAMP] * 3
    back = typemap.lookup(T.Transforms).wires[0].decode(msg)
    expected = [
        *CHAIN.transforms[:2],
        T.Transform("odom", "pelvis", CHAIN.transforms[2].translation),
    ]
    assert len(back.transforms) == 3
    assert all(_same(a, b) for a, b in zip(back.transforms, expected, strict=True))


def test_the_parent_is_namespaced_and_the_child_is_verbatim(eps):
    binding = typemap.resolve(eps["several"])
    msg = _fill(binding, CHAIN, {"frame_prefix": "tb"})
    assert [(tf.header.frame_id, tf.child_frame_id) for tf in msg.transforms] == [
        ("tb/base_link", "shell_link"),
        ("tb/shell_link", "rplidar_link"),
        ("tb/odom", "pelvis"),
    ]
    unhinted = TFMessage()
    typemap.lookup(T.Transform).wires[0].fill(unhinted, T.Transform("", "x"), STAMP, {})
    assert unhinted.transforms[0].header.frame_id == "map"  # a global frame stays bare


def test_a_static_endpoint_is_latched_on_tf_static(eps):
    binding = typemap.resolve(eps["mounts"])
    assert binding.hints["topic"] == "/tf_static"
    assert binding.hints["qos"] == QOS_PRESETS["latched"]
    assert typemap.describe(eps["mount"])["topic"] == "/tf_static"


def test_static_needs_a_transform_payload_and_no_topic_of_its_own(eps):
    with pytest.raises(ValueError, match="needs an out endpoint returning Transform or Transforms"):
        typemap.resolve(eps["not_a_transform"])
    from dataclasses import replace

    renamed = replace(eps["mount"], topic="/elsewhere")
    with pytest.raises(ValueError, match="takes no topic or qos of its own"):
        typemap.resolve(renamed)


# -- through the bridge ------------------------------------------------------------------------------
class _FakeStatic:
    def __init__(self):
        self.sent = []

    def sendTransform(self, tf):  # noqa: N802 -- tf2_ros's spelling
        self.sent.append(tf if isinstance(tf, list) else [tf])


class _FakePublisher:
    topic_name = "/fake"

    def __init__(self):
        self.published = []

    def get_subscription_count(self):
        return 0

    def publish(self, msg):
        self.published.append(msg)


class _FakeNode:
    def __init__(self):
        self.publishers = []

    def create_publisher(self, msg_type, topic, qos):
        self.publishers.append(_FakePublisher())
        return self.publishers[-1]


def _bridge(**config) -> Ros2Bridge:
    bridge = Ros2Bridge({"reuse_messages": False, **config})
    bridge._node = _FakeNode()
    bridge._shutting_down = lambda: False
    bridge._static_tf = _FakeStatic()
    return bridge


def _with_namespace(ep, namespace="tb"):
    from dataclasses import replace

    return replace(ep, namespace=namespace)


def test_a_static_endpoint_sends_its_first_value_once_both_frames_namespaced(eps):
    bridge = _bridge()
    ep = _with_namespace(eps["mounts"])
    handle = bridge._make_output(ep, {})
    assert handle.publisher is None and bridge._node.publishers == []
    assert not bridge._skip_unsubscribed(ep)
    bridge._publish(handle, CHAIN, STAMP)
    assert bridge._skip_unsubscribed(ep)  # sent: never read again
    bridge._publish(handle, CHAIN, STAMP)
    (sent,) = bridge._static_tf.sent
    assert [(t.header.frame_id, t.child_frame_id) for t in sent] == [
        ("tb/base_link", "tb/shell_link"),
        ("tb/shell_link", "tb/rplidar_link"),
        ("tb/base_link", "tb/pelvis"),  # empty parent: the frame_id hint
    ]
    assert all(t.header.stamp == Time() for t in sent)
    assert (sent[1].transform.translation.z, sent[1].transform.rotation.w) == (0.2, 0.5)


def test_a_static_transform_equals_the_static_tf_hint_path(eps):
    """The same mount through ``static: true`` and through a ``static_tf`` hint: one transform."""
    from roqsim.context import Endpoint

    typed = _bridge()
    typed._publish(typed._make_output(_with_namespace(eps["mount"]), {}), MOUSE, STAMP)
    hinted = _bridge()
    legacy = Endpoint(
        name="frames",
        direction="out",
        owner="robot",
        namespace="tb",
        read=lambda: None,
        backend={
            "ros2": {
                "type": "tf2_msgs.msg.TFMessage",
                "topic": "tf",
                "frame_id": "mouse",
                "static_tf": {
                    "parent": "base_link",
                    "translation": list(MOUSE.translation),
                    "rotation": list(MOUSE.rotation),
                },
            }
        },
    )
    hinted._make_output(legacy, {})
    assert typed._static_tf.sent == hinted._static_tf.sent


def test_a_static_endpoint_is_named_by_the_tf_static_topic_it_is_sent_on(eps):
    bridge = _bridge(tf_namespace="fleet")
    bridge._make_output(eps["mounts"], {})
    named = bridge._names[id(eps["mounts"])]
    assert (named["topic"], named["type"]) == ("/fleet/tf_static", "tf2_msgs.msg.TFMessage")


def test_publish_static_tf_false_turns_a_static_endpoint_off(eps):
    bridge = _bridge(publish_static_tf=False)
    bridge._static_tf = None
    handle = bridge._make_output(eps["mounts"], {})
    assert bridge._skip_unsubscribed(eps["mounts"])  # never read
    bridge._publish(handle, CHAIN, STAMP)  # nothing to send to, and nothing sent
    assert bridge._static_tf is None


def test_a_dynamic_endpoint_publishes_its_tfmessage_each_time_stamped(eps):
    bridge = _bridge()
    handle = bridge._make_output(_with_namespace(eps["several"]), {})
    bridge._publish(handle, CHAIN, STAMP)
    bridge._publish(handle, CHAIN, Time(sec=4))
    (pub,) = bridge._node.publishers
    assert [m.transforms[0].header.stamp for m in pub.published] == [STAMP, Time(sec=4)]
    assert pub.published[0].transforms[0].header.frame_id == "tb/base_link"
    assert bridge._static_tf.sent == []
