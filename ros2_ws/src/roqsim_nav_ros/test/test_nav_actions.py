# SPDX-License-Identifier: Apache-2.0
"""Both nav2 actions, end to end against a running simulator.

Not a unit test of the handlers: it stands up the bridge, waits for a real ``ActionServer`` to
appear, sends a real goal from a real ``ActionClient``, and checks the prop actually moved. The
handlers are thirty lines each -- what is worth testing is the chain they sit in, where every part
that can be wrong is somewhere else: the endpoint the navigator declares, the action *type string*
the bridge resolves, the entry point that gets this module imported at all, and the namespace the
action name is built from.

Skips cleanly without ROS, like ``roqsim_nav2_example``'s goal test, so ``make test`` stays green in
a pip-only checkout and gains this when a workspace is sourced.
"""

from __future__ import annotations

import threading
import time

import pytest

roqsim = pytest.importorskip("roqsim")  # selects the GL backend before mujoco is imported
rclpy = pytest.importorskip("rclpy")
pytest.importorskip("nav2_msgs")
pytest.importorskip("roqsim_ros_bridge")

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from geometry_msgs.msg import PoseStamped  # noqa: E402
from nav2_msgs.action import NavigateThroughPoses, NavigateToPose  # noqa: E402
from rclpy.action import ActionClient  # noqa: E402
from rclpy.node import Node  # noqa: E402

from roqsim.config import load_config_from_dict, with_transport  # noqa: E402
from roqsim.engine import Engine  # noqa: E402

#: 4 == ``action_msgs/GoalStatus.STATUS_SUCCEEDED``. Named, because a bare 4 in an assertion is the
#: kind of thing that gets "fixed" to whatever the code happens to return.
STATUS_SUCCEEDED = 4

CRATE = """<mujoco model="crate"><worldbody><body name="crate">
  <geom name="g" type="box" size=".25 .25 .25"/></body></worldbody></mujoco>"""


def _pose(x: float, y: float) -> PoseStamped:
    pose = PoseStamped()
    pose.header.frame_id = "map"
    pose.pose.position.x, pose.pose.position.y = float(x), float(y)
    pose.pose.orientation.w = 1.0
    return pose


@pytest.fixture(scope="module")
def sim(tmp_path_factory):
    """A navigated prop with the bridge injected, stepping on its own thread.

    ``with_transport`` rather than a ``ros2_bridge`` entry in the world: a checked-in world is
    ROS-free and the transport is injected, which is the same thing ``roqsim sim --ros`` does.
    """
    tmp = tmp_path_factory.mktemp("nav_ros")
    (tmp / "crate.xml").write_text(CRATE)
    raw = {
        "sim": {"pacing": "asap"},
        "components": [
            {
                "spawn_model": {
                    "model": str(tmp / "crate.xml"),
                    "pos": [0.0, 0.0, 0.25],
                    "mocap": True,
                },
                "name": "cart",
                "components": [
                    {
                        "navigator": {
                            "speed": 1.0,
                            "namespace": "cart",
                            "caution": {"enabled": False},
                        }
                    }
                ],
            }
        ],
    }
    engine = Engine(load_config_from_dict(with_transport(raw, ros=True), base_dir=tmp))
    engine.setup()
    engine.reset()
    stop = threading.Event()

    def run():
        while not stop.is_set():
            engine.step()

    threading.Thread(target=run, daemon=True).start()
    # Nothing here calls `engine.reset()`: the stepping thread owns the physics, and resetting from
    # the test thread races it -- `on_reset` rewinds the route sequence under a goal that is already
    # in flight, which surfaces as an aborted goal and an exception in the stepper. Each test sends
    # ABSOLUTE goals instead, so they are independent of each other and of their order.
    if not rclpy.ok():  # the bridge initialises rclpy itself; a second init raises
        rclpy.init()
    node = Node("roqsim_nav_ros_test")
    try:
        yield engine, node
    finally:
        stop.set()
        time.sleep(0.2)
        node.destroy_node()
        engine.shutdown()


def _xy(engine):
    model, data = engine.ctx.model, engine.ctx.data
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, engine.ctx.entities.get("cart").body)
    return data.mocap_pos[int(model.body_mocapid[bid])][:2].copy()


def _send(node, client, goal, timeout=90.0):
    """Send ``goal`` and return ``(status, feedback list)``. Never blocks the executor."""
    assert client.wait_for_server(timeout_sec=30.0), "no action server appeared"
    feedback = []
    sent = client.send_goal_async(goal, feedback_callback=feedback.append)
    rclpy.spin_until_future_complete(node, sent, timeout_sec=30.0)
    handle = sent.result()
    assert handle.accepted, "the navigator rejected the goal"
    result = handle.get_result_async()
    rclpy.spin_until_future_complete(node, result, timeout_sec=timeout)
    return result.result().status, feedback


def test_navigate_to_pose_drives_the_mover_there(sim):
    engine, node = sim
    client = ActionClient(node, NavigateToPose, "/cart/navigate_to_pose")
    goal = NavigateToPose.Goal()
    goal.pose = _pose(3.0, 0.0)
    try:
        status, _ = _send(node, client, goal)
    finally:
        client.destroy()
    assert status == STATUS_SUCCEEDED
    assert np.linalg.norm(_xy(engine) - np.asarray([3.0, 0.0])) < 0.3


def test_navigate_through_poses_visits_them_in_order(sim):
    """The action name, the namespace and the second handler, in one goal.

    Two legs rather than one, because a single pose would pass equally well if the second goal were
    ignored -- and the corner is what distinguishes driving a route from driving at a point.
    """
    engine, node = sim
    client = ActionClient(node, NavigateThroughPoses, "/cart/navigate_through_poses")
    goal = NavigateThroughPoses.Goal()
    goal.poses = [_pose(2.0, 0.0), _pose(2.0, 2.0)]
    try:
        status, _ = _send(node, client, goal)
    finally:
        client.destroy()
    assert status == STATUS_SUCCEEDED
    assert np.linalg.norm(_xy(engine) - np.asarray([2.0, 2.0])) < 0.4


def test_an_empty_route_is_rejected_rather_than_reported_as_arrival(sim):
    """A goal with nowhere to go must not succeed: the caller would read that as "already there"."""
    engine, node = sim
    client = ActionClient(node, NavigateThroughPoses, "/cart/navigate_through_poses")
    try:
        assert client.wait_for_server(timeout_sec=30.0)
        sent = client.send_goal_async(NavigateThroughPoses.Goal())
        rclpy.spin_until_future_complete(node, sent, timeout_sec=30.0)
        handle = sent.result()
        assert handle.accepted
        result = handle.get_result_async()
        rclpy.spin_until_future_complete(node, result, timeout_sec=30.0)
        assert result.result().status != STATUS_SUCCEEDED
    finally:
        client.destroy()
