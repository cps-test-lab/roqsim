"""A producer's ``static_tf`` hint: one transform, or a list of them, published once and namespaced.

A mount's fixed-link chain (``base_link -> shell_link -> rplidar_link``) is several transforms on one
endpoint that carries no payload of its own, so the hint names each child; the single-transform form
takes its child from the endpoint's ``frame_id``. ``publish_static_tf: false`` turns both off, for a
world whose robot_state_publisher owns these frames.
"""

from __future__ import annotations

from roqsim.context import Endpoint
from roqsim_ros_bridge.ros2_bridge import Ros2Bridge


class _FakeStatic:
    def __init__(self):
        self.sent = []

    def sendTransform(self, tf):  # noqa: N802 -- tf2_ros's spelling
        self.sent.append(
            (
                tf.header.frame_id,
                tf.child_frame_id,
                tf.transform.translation.z,
                tf.transform.rotation.w,
            )
        )


class _FakePublisher:
    def get_subscription_count(self):
        return 0


class _FakeNode:
    def create_publisher(self, msg_type, topic, qos):
        return _FakePublisher()


def _bridge(**config) -> Ros2Bridge:
    bridge = Ros2Bridge(config)
    bridge._node = _FakeNode()
    bridge._shutting_down = lambda: False
    bridge._static_tf = _FakeStatic()
    return bridge


def _endpoint(static_tf, namespace="tb", frame_id="rplidar_link") -> Endpoint:
    return Endpoint(
        name="frames",
        direction="out",
        owner="robot",
        namespace=namespace,
        read=lambda: None,
        backend={
            "ros2": {
                "type": "tf2_msgs.msg.TFMessage",
                "topic": "tf",
                "frame_id": frame_id,
                "static_tf": static_tf,
            }
        },
    )


CHAIN = [
    {
        "parent": "base_link",
        "child": "shell_link",
        "translation": [0, 0, 0.1],
        "rotation": [1, 0, 0, 0],
    },
    {
        "parent": "shell_link",
        "child": "rplidar_link",
        "translation": [0, 0, 0.2],
        "rotation": [0, 1, 0, 0],
    },
]


def test_a_list_publishes_every_link_under_the_endpoints_namespace():
    bridge = _bridge()
    ep = _endpoint(CHAIN)
    bridge._make_output(ep, ep.backend["ros2"])
    assert bridge._static_tf.sent == [
        ("tb/base_link", "tb/shell_link", 0.1, 1.0),
        ("tb/shell_link", "tb/rplidar_link", 0.2, 0.0),
    ]


def test_a_single_transform_still_takes_its_child_from_frame_id():
    bridge = _bridge()
    ep = _endpoint({"parent": "base_link", "translation": [0, 0, 0.3], "rotation": [1, 0, 0, 0]})
    bridge._make_output(ep, ep.backend["ros2"])
    assert bridge._static_tf.sent == [("tb/base_link", "tb/rplidar_link", 0.3, 1.0)]


def test_publish_static_tf_false_publishes_neither_form():
    bridge = _bridge(publish_static_tf=False)
    for st in (CHAIN, CHAIN[0]):
        ep = _endpoint(st)
        bridge._make_output(ep, ep.backend["ros2"])
    assert bridge._static_tf.sent == []
