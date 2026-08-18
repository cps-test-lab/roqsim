"""The ``ros2_bridge`` plugin's ``tf_namespace``: TF on ``/<ns>/tf``, frames untouched.

Guards the failure that motivated the option. tf2_ros's broadcasters hardwire the absolute ``/tf``, so
the bridge could only ever publish to the global tree. A namespaced Nav2 follows the multi-robot
convention (``/tf -> tf`` remap) and lives on ``/<robot>/tf``, and scenario_execution's
``NamespacedTransformListener`` subscribes ``<namespace>/tf`` — so the bridge's transforms were
invisible to both, and a scenario hung forever on "Waiting for transform map -> base_link" while the
stack looked healthy.
"""

from rclpy.qos import DurabilityPolicy

from roqsim_ros_bridge.ros2_bridge import _NamespacedTfPublisher


class _FakePub:
    def __init__(self):
        self.published = []

    def publish(self, msg):
        self.published.append([t.child_frame_id for t in msg.transforms])


class _FakeNode:
    """Records the topic/QoS a publisher is created with; no rclpy context needed."""

    def __init__(self):
        self.pub = _FakePub()
        self.topics = []

    def create_publisher(self, msg_type, topic, qos):
        self.topics.append((topic, qos.depth, qos.durability))
        return self.pub


class _Tf:
    def __init__(self, child):
        self.child_frame_id = child


def test_dynamic_tf_goes_to_the_namespaced_topic():
    node = _FakeNode()
    pub = _NamespacedTfPublisher(node, "/a200_0000/tf", static=False)
    assert node.topics[0][0] == "/a200_0000/tf"
    pub.sendTransform(_Tf("base_link"))
    assert node.pub.published == [["base_link"]]


def test_static_tf_topic_is_latched():
    """A consumer that subscribes after the mount transforms were sent must still get them."""
    node = _FakeNode()
    _NamespacedTfPublisher(node, "/a200_0000/tf_static", static=True)
    topic, _depth, durability = node.topics[0]
    assert topic == "/a200_0000/tf_static"
    assert durability == DurabilityPolicy.TRANSIENT_LOCAL


def test_static_tf_republishes_the_whole_accumulated_set():
    """StaticTransformBroadcaster's accumulate-and-republish semantics. Publishing only the newest
    transform on a depth-1 latched topic would silently drop every earlier mount frame."""
    node = _FakeNode()
    pub = _NamespacedTfPublisher(node, "/a200_0000/tf_static", static=True)
    pub.sendTransform(_Tf("lidar"))
    pub.sendTransform(_Tf("camera"))
    assert node.pub.published[-1] == ["lidar", "camera"]


def test_static_tf_replaces_a_resent_frame_rather_than_duplicating():
    node = _FakeNode()
    pub = _NamespacedTfPublisher(node, "/a200_0000/tf_static", static=True)
    pub.sendTransform(_Tf("lidar"))
    pub.sendTransform(_Tf("lidar"))
    assert node.pub.published[-1] == ["lidar"]


def test_a_list_of_transforms_is_accepted():
    """tf2_ros's sendTransform takes a single transform or a list; ours must too."""
    node = _FakeNode()
    pub = _NamespacedTfPublisher(node, "/x/tf", static=False)
    pub.sendTransform([_Tf("a"), _Tf("b")])
    assert node.pub.published == [["a", "b"]]
