"""Merged ``/joint_states``: one ``robot_description`` implies one merged joint-state stream.

Guards the gap this closed. A world with two arms declares one ``joint_states`` endpoint PER arm
(``arm_controller``), each scoped under its own namespace so their individual topics don't collide --
e.g. ``/r1/joint_states`` and ``/r2/joint_states``. Nothing published the unqualified, unnamespaced
``/joint_states`` that ``robot_state_publisher`` and MoveIt's planning-scene/current-state monitor
actually subscribe to, so a world combining two arms into one ``robot_description`` (see
``roqsim.export_moveit``, which documents the joint names it emits as "the ones that reach
``/joint_states``") got no TF for either arm and a ``move_group`` that never learned a current state --
silently: it still logged "You can start planning now!".

The merge logic itself (``_merge_joint_state_payloads``) is pure, so it is tested without any ROS
node or executor; the wiring test below uses a fake node standing in for ``rclpy.node.Node``, and
fakes out ``_shutting_down`` (which otherwise reads a real rclpy context this test never creates) to
exercise the real ``sensor_msgs.msg.JointState`` converter end to end.
"""

from __future__ import annotations

from roqsim.context import Endpoint, SimContext
from roqsim_ros_bridge.ros2_bridge import Ros2Bridge, _merge_joint_state_payloads


class _FakePublisher:
    def __init__(self):
        self.published = []

    def publish(self, msg):
        self.published.append(msg)


class _FakeNode:
    """Records the (topic, msg_type) a publisher is created with; no rclpy context needed."""

    def __init__(self):
        self.created: list[tuple[str, object]] = []

    def create_publisher(self, msg_type, topic, qos):
        self.created.append((topic, msg_type))
        return _FakePublisher()


def _joint_endpoint(
    owner: str, namespace: str, names: list[str], positions: list[float]
) -> Endpoint:
    return Endpoint(
        name="joint_states",
        direction="out",
        owner=owner,
        namespace=namespace,
        read=lambda: (names, positions, [0.0] * len(names)),
        rate_hz=50.0,
        backend={"ros2": {"type": "sensor_msgs.msg.JointState", "topic": "joint_states"}},
    )


def _bridge_with_fake_node(**config) -> Ros2Bridge:
    bridge = Ros2Bridge(config)
    bridge._node = _FakeNode()
    bridge._shutting_down = lambda: False  # no real rclpy context in this test
    return bridge


# -- the merge itself, no bridge/node involved ---------------------------------------------------


def test_merge_concatenates_in_argument_order():
    payload = _merge_joint_state_payloads(
        [(["a", "b"], [1.0, 2.0], [0.0, 0.0], [0.1, 0.2]), (["c"], [3.0], [0.0])]
    )
    names, positions, velocities, efforts = payload
    assert names == ["a", "b", "c"]
    assert positions == [1.0, 2.0, 3.0]
    assert efforts == [0.1, 0.2, 0.0], "a source with no effort is padded with 0.0, not dropped"


def test_merge_omits_effort_when_no_source_reports_it():
    payload = _merge_joint_state_payloads([(["a"], [1.0], [0.0]), (["b"], [2.0], [0.0])])
    assert len(payload) == 3, "no source reported effort, so the merged payload carries none either"


# -- wiring: a bridge serving two joint-state endpoints --------------------------------------------


def test_two_arms_get_one_merged_publisher_in_registration_order():
    ctx = SimContext(config={})
    ctx.interface.add(_joint_endpoint("r1", "r1", ["r1_shoulder", "r1_elbow"], [0.1, 0.2]))
    ctx.interface.add(_joint_endpoint("r2", "r2", ["r2_shoulder", "r2_elbow"], [0.3, 0.4]))

    bridge = _bridge_with_fake_node()
    bridge._setup_merged_joint_states(ctx)

    assert bridge._merged_joint_states is not None, "two arms must get a merged /joint_states"
    assert [topic for topic, _ in bridge._node.created] == ["joint_states"]
    assert [ep.owner for ep in bridge._joint_state_sources] == ["r1", "r2"]

    bridge._publish_merged_joint_states(stamp=0)
    published = bridge._merged_joint_states.publisher.published
    assert len(published) == 1
    msg = published[0]
    assert list(msg.name) == ["r1_shoulder", "r1_elbow", "r2_shoulder", "r2_elbow"], (
        "the merged message must carry every arm's joint names, in the endpoints' declared order -- "
        "the same order the exporter reads them in"
    )
    assert list(msg.position) == [0.1, 0.2, 0.3, 0.4]


def test_owner_filter_keeps_a_scoped_bridge_from_merging_a_robot_it_does_not_serve():
    """The multi-robot pattern runs one bridge instance PER robot, scoped by ``owner``. That instance
    must not merge in an endpoint belonging to a robot it was never asked to serve."""
    ctx = SimContext(config={})
    ctx.interface.add(_joint_endpoint("r1", "r1", ["r1_j1"], [0.1]))
    ctx.interface.add(_joint_endpoint("r2", "r2", ["r2_j1"], [0.2]))

    bridge = _bridge_with_fake_node(owner="r1")
    bridge._setup_merged_joint_states(ctx)

    assert bridge._merged_joint_states is None
    assert bridge._node.created == []


# -- single-arm regression: the whole feature must be a no-op there --------------------------------


def test_single_arm_world_gets_no_merged_publisher():
    ctx = SimContext(config={})
    ctx.interface.add(_joint_endpoint("only_arm", "", ["shoulder", "elbow"], [0.1, 0.2]))

    bridge = _bridge_with_fake_node()
    bridge._setup_merged_joint_states(ctx)

    assert bridge._merged_joint_states is None, (
        "a single joint-state endpoint's own topic already IS /joint_states in the common "
        "unnamespaced case -- a second, merged publisher there would be a pointless duplicate"
    )
    assert bridge._joint_state_sources == []
    assert bridge._node.created == [], "no publisher beyond the endpoint's own must be created"
