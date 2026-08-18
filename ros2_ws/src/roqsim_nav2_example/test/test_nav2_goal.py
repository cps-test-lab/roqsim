"""Headless nav2 goal-reaching integration test.

Launches the minimal nav2 + roqsim TurtleBot stack, sends one goal via
``nav2_simple_commander.BasicNavigator``, and asserts the robot reaches within a loose radius under a
generous timeout. Skips cleanly if ROS / nav2 is not available (so `pytest` without ROS is a no-op).

The whole launch tree is started with the *current* interpreter (the venv Python that has
``roqsim``), by running the ``ros2`` CLI via ``sys.executable`` — otherwise the bridge subprocess
would use ``/usr/bin/python3`` and fail to import ``roqsim``.
"""

from __future__ import annotations

import math
import os
import shutil
import signal
import subprocess
import sys
import time

import pytest

rclpy = pytest.importorskip("rclpy")
pytest.importorskip("nav2_simple_commander")

from geometry_msgs.msg import PoseStamped  # noqa: E402
from lifecycle_msgs.srv import GetState  # noqa: E402
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult  # noqa: E402

GOAL_X, GOAL_Y = 1.5, 0.0
GOAL_RADIUS = 0.5  # loose tolerance
SERVER_TIMEOUT = 90.0  # wait for nav2 to come up
GOAL_TIMEOUT = 120.0  # wait for the robot to reach the goal


def _dist(px, py):
    return math.hypot(px - GOAL_X, py - GOAL_Y)


def _wait_active(nav, node_names, timeout):
    """Poll each node's lifecycle get_state until all report 'active' (single-threaded spin)."""
    deadline = time.time() + timeout
    for node in node_names:
        cli = nav.create_client(GetState, f"{node}/get_state")
        while True:
            if time.time() >= deadline:
                return False
            if cli.wait_for_service(timeout_sec=1.0):
                fut = cli.call_async(GetState.Request())
                rclpy.spin_until_future_complete(nav, fut, timeout_sec=2.0)
                res = fut.result()
                if res is not None and res.current_state.label == "active":
                    break
            time.sleep(0.5)
    return True


@pytest.fixture
def nav2_stack():
    ros2 = shutil.which("ros2")
    if ros2 is None:
        pytest.skip("ros2 CLI not on PATH (ROS not sourced)")
    env = dict(os.environ)
    env.setdefault("MUJOCO_GL", "egl")
    # Run the launch tree under THIS interpreter so ExecuteProcess(sys.executable, ...) has roqsim.
    proc = subprocess.Popen(
        [sys.executable, ros2, "launch", "roqsim_nav2_example", "nav2_turtlebot.launch.py"],
        env=env,
        start_new_session=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        yield proc
    finally:
        # Terminate the whole process group (launch + all its children).
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGINT)
            proc.wait(timeout=15)
        except Exception:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except Exception:
                pass


def test_robot_reaches_goal(nav2_stack):
    rclpy.init()
    nav = BasicNavigator()
    try:
        # Wait until nav2 is actually ACTIVE (the lifecycle manager activates the servers). Sending
        # a goal to a merely-configured (inactive) bt_navigator gets rejected.
        assert _wait_active(
            nav,
            ["map_server", "planner_server", "controller_server", "bt_navigator"],
            SERVER_TIMEOUT,
        ), "nav2 did not become active in time"
        time.sleep(2.0)  # let costmaps ingest the first scan / TF settle

        goal = PoseStamped()
        goal.header.frame_id = "map"
        goal.header.stamp = nav.get_clock().now().to_msg()
        goal.pose.position.x = GOAL_X
        goal.pose.position.y = GOAL_Y
        goal.pose.orientation.w = 1.0

        nav.goToPose(goal)
        best = float("inf")
        t0 = time.time()
        while not nav.isTaskComplete():
            fb = nav.getFeedback()
            if fb is not None:
                p = fb.current_pose.pose.position
                best = min(best, _dist(p.x, p.y))
                if best <= GOAL_RADIUS:
                    nav.cancelTask()
                    break
            assert time.time() - t0 < GOAL_TIMEOUT, (
                f"goal not reached within {GOAL_TIMEOUT}s (closest {best:.2f} m)"
            )
            time.sleep(0.5)

        result = nav.getResult()
        assert best <= GOAL_RADIUS or result == TaskResult.SUCCEEDED, (
            f"robot did not reach goal: closest={best:.2f} m, result={result}"
        )
    finally:
        try:
            nav.destroy_node()
        finally:
            rclpy.shutdown()
