"""Headless Nav2 goal test for car-like robots: a turn, then a reverse, judged on ground truth.

For each robot in :data:`ROBOTS`, launches ``nav2_carlike.launch.py robot:=<name>`` and sends two
goals through ``nav2_simple_commander``:

1. ahead and to the left, facing left -- reached on a curve, with the steering over to one side;
2. straight behind that pose with the same heading -- reached by reversing, since a forward loop
   round to it is far longer than the reverse penalty makes the straight line.

Each goal is judged on GROUND TRUTH, not on the odometry Nav2 steers by: the robot entity's world
pose from the simulator's ``simulation_interfaces/GetEntityState`` service (the ``sim_interfaces``
plugin in the world), read once Nav2 reports the goal done. The world's origin is the map's, so the
pose compares with the goal as it is. Alongside, the test records what makes the run a car-like
one: the measured steering angle from ``/joint_states`` swings over on the turn, and the odometry
reports backwards travel on the reverse.

Each launch runs on a ROS domain of its own, discovered on localhost only: a second simulator's
``/clock`` on the same domain -- another test, another checkout's run -- interleaves with this one's
and every sim-time wait in Nav2 goes wrong. ``ROS_DOMAIN_ID`` set by the caller is used as given.

Skips cleanly when ROS or Nav2 is not available. ``CARLIKE_NAV_LOG=<dir>`` keeps each launch's
output as ``<dir>/<robot>.log``.

To test another car-like robot, add its entry to :data:`ROBOTS` with goals that fit its turning
radius inside the 10 m room, after adding its world and per-robot params (see the launch file).
"""

from __future__ import annotations

import math
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass

import pytest

rclpy = pytest.importorskip("rclpy")
pytest.importorskip("nav2_simple_commander")

from geometry_msgs.msg import PoseStamped  # noqa: E402
from lifecycle_msgs.srv import GetState  # noqa: E402
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult  # noqa: E402
from nav_msgs.msg import Odometry  # noqa: E402
from rclpy.executors import SingleThreadedExecutor  # noqa: E402
from rclpy.node import Node  # noqa: E402
from sensor_msgs.msg import JointState  # noqa: E402
from simulation_interfaces.msg import Result  # noqa: E402
from simulation_interfaces.srv import GetEntityState  # noqa: E402


@dataclass(frozen=True)
class Robot:
    steer_joint: str  # a joint in /joint_states whose position is the steering angle
    turn_goal: tuple[float, float, float]  # (x, y, yaw) in map == world
    reverse_goal: tuple[float, float, float]
    xy_tolerance: float  # m, on ground truth
    min_reverse_speed: float  # m/s the odometry must report on the reverse leg (negative)


ROBOTS = {
    # Minimum planning radius 0.5 m (params/carlike_piracer.yaml).
    "piracer": Robot(
        steer_joint="left_steer_joint",
        turn_goal=(1.5, 1.2, math.pi / 2),
        reverse_goal=(1.5, 0.3, math.pi / 2),
        xy_tolerance=0.25,
        min_reverse_speed=-0.05,
    ),
}
YAW_TOLERANCE = 0.5  # rad, on ground truth; the goal checker's own is 0.35 on the odometry
MIN_STEER = 0.15  # rad the steering must reach on the turn leg
SERVER_TIMEOUT = 120.0
GOAL_TIMEOUT = 180.0
LIFECYCLE_NODES = ["map_server", "planner_server", "controller_server", "bt_navigator"]
ENTITY = "robot"  # the spawn_robot entry's name in worlds/<robot>_nav2.yaml


class _Recorder(Node):
    """Odometry speed and the steering angle as they arrive, and ground truth on request."""

    def __init__(self, steer_joint: str):
        super().__init__("carlike_nav_recorder")
        self.steer_joint = steer_joint
        self.v = 0.0
        self.steer = 0.0
        self.rows: list[tuple] = []  # (v, steer), one per odometry message
        self.create_subscription(Odometry, "/odom", self._on_odom, 50)
        self.create_subscription(JointState, "/joint_states", self._on_joints, 50)
        self._truth = self.create_client(GetEntityState, "get_entity_state")

    def _on_odom(self, msg):
        self.v = msg.twist.twist.linear.x
        self.rows.append((self.v, self.steer))

    def _on_joints(self, msg):
        if self.steer_joint in msg.name:
            self.steer = msg.position[msg.name.index(self.steer_joint)]

    def ground_truth(self, entity: str, timeout: float = 10.0) -> tuple[float, float, float]:
        """*entity*'s true (x, y, yaw) in the world; raises when the simulator does not answer."""
        if not self._truth.wait_for_service(timeout_sec=timeout):
            raise AssertionError("no get_entity_state service (sim_interfaces not in the world?)")
        fut = self._truth.call_async(GetEntityState.Request(entity=entity))
        deadline = time.time() + timeout
        while not fut.done():  # the executor thread spins this node and completes the future
            if time.time() >= deadline:
                raise AssertionError(f"get_entity_state({entity!r}) did not answer in {timeout}s")
            time.sleep(0.05)
        resp = fut.result()
        if resp.result.result != Result.RESULT_OK:
            raise AssertionError(
                f"get_entity_state({entity!r}): {resp.result.result} {resp.result.error_message}"
            )
        p, q = resp.state.pose.position, resp.state.pose.orientation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y**2 + q.z**2))
        return p.x, p.y, yaw


def _wait_active(nav, node_names, timeout):
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


def _pose(x, y, yaw):
    goal = PoseStamped()
    goal.header.frame_id = "map"
    # Stamp left at zero, "the latest transform": this node runs on wall time and the stack on sim
    # time, so a wall-clock stamp would be decades ahead of every transform Nav2 could look up.
    goal.pose.position.x = x
    goal.pose.position.y = y
    goal.pose.orientation.z = math.sin(yaw / 2)
    goal.pose.orientation.w = math.cos(yaw / 2)
    return goal


def _drive_to(nav, rec, goal):
    """Send one goal and wait for Nav2 to finish it.

    Returns (result, ground truth when it finished, rows recorded meanwhile).
    """
    start = len(rec.rows)
    nav.goToPose(_pose(*goal))
    t0 = time.time()
    while not nav.isTaskComplete():
        if time.time() - t0 > GOAL_TIMEOUT:
            nav.cancelTask()
            pytest.fail(
                f"goal {goal} not reached within {GOAL_TIMEOUT}s; truth {rec.ground_truth(ENTITY)}"
            )
        time.sleep(0.5)
    return nav.getResult(), rec.ground_truth(ENTITY), rec.rows[start:]


def _error(gt, goal):
    dyaw = (gt[2] - goal[2] + math.pi) % (2 * math.pi) - math.pi
    return math.hypot(gt[0] - goal[0], gt[1] - goal[1]), abs(dyaw)


def _domain_id() -> int:
    """The caller's ROS_DOMAIN_ID, else one derived from this process (1..100, never 0)."""
    given = os.environ.get("ROS_DOMAIN_ID")
    return int(given) if given else 1 + os.getpid() % 100


@pytest.fixture
def nav2_stack(request):
    robot = request.param
    ros2 = shutil.which("ros2")
    if ros2 is None:
        pytest.skip("ros2 CLI not on PATH (ROS not sourced)")
    domain = _domain_id()
    env = dict(os.environ)
    env.setdefault("MUJOCO_GL", "egl")
    env["ROS_DOMAIN_ID"] = str(domain)
    env["ROS_AUTOMATIC_DISCOVERY_RANGE"] = "LOCALHOST"
    log_dir = os.environ.get("CARLIKE_NAV_LOG")
    log = open(os.path.join(log_dir, f"{robot}.log"), "w") if log_dir else subprocess.DEVNULL  # noqa: SIM115 -- closed below
    # The whole launch tree under THIS interpreter, so the simulator has roqsim.
    proc = subprocess.Popen(
        [
            sys.executable,
            ros2,
            "launch",
            "roqsim_nav2_example",
            "nav2_carlike.launch.py",
            f"robot:={robot}",
        ],
        env=env,
        start_new_session=True,
        stdout=log,
        stderr=subprocess.STDOUT,
    )
    try:
        yield robot, domain
    finally:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGINT)
            proc.wait(timeout=20)
        except Exception:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except Exception:
                pass
        if log is not subprocess.DEVNULL:
            log.close()


@pytest.mark.parametrize("nav2_stack", list(ROBOTS), indirect=True)
def test_carlike_robot_turns_to_a_goal_and_reverses_to_the_next(nav2_stack, monkeypatch):
    name, domain = nav2_stack
    robot = ROBOTS[name]
    # The test's own node joins the launch's domain, on localhost only, like the launch.
    monkeypatch.setenv("ROS_AUTOMATIC_DISCOVERY_RANGE", "LOCALHOST")
    rclpy.init(domain_id=domain)
    nav = BasicNavigator()
    rec = _Recorder(robot.steer_joint)
    executor = SingleThreadedExecutor()
    executor.add_node(rec)
    spinner = threading.Thread(target=executor.spin, daemon=True)
    spinner.start()
    try:
        assert _wait_active(nav, LIFECYCLE_NODES, SERVER_TIMEOUT), "nav2 did not become active"
        time.sleep(3.0)  # the costmaps take the map and their first scans
        rec.ground_truth(ENTITY)  # answers before the first goal, or the run fails here

        result, truth, turn = _drive_to(nav, rec, robot.turn_goal)
        dist, dyaw = _error(truth, robot.turn_goal)
        steer = max(abs(r[1]) for r in turn)
        print(
            f"{name} turn goal: {result}, error {dist:.3f} m / {dyaw:.3f} rad, "
            f"max |steer| {steer:.3f} rad"
        )
        assert result == TaskResult.SUCCEEDED, f"turn goal: {result}"
        assert dist <= robot.xy_tolerance and dyaw <= YAW_TOLERANCE, (dist, dyaw)
        assert steer > MIN_STEER, "reached a goal a quarter turn round without steering"

        result, truth, back = _drive_to(nav, rec, robot.reverse_goal)
        dist, dyaw = _error(truth, robot.reverse_goal)
        slowest = min(r[0] for r in back)
        print(
            f"{name} reverse goal: {result}, error {dist:.3f} m / {dyaw:.3f} rad, "
            f"min odom speed {slowest:.3f} m/s"
        )
        assert result == TaskResult.SUCCEEDED, f"reverse goal: {result}"
        assert dist <= robot.xy_tolerance and dyaw <= YAW_TOLERANCE, (dist, dyaw)
        assert slowest < robot.min_reverse_speed, "the robot never drove backwards"
    finally:
        executor.shutdown()
        rec.destroy_node()
        nav.destroy_node()
        rclpy.shutdown()
