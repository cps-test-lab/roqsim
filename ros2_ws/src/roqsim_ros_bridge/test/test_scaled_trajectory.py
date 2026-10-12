"""A ``FollowJointTrajectory`` goal runs on trajectory time advanced at the producer's time scale.

The UR driver's scaled trajectory controller is the plain one whose trajectory time advances at the
robot's speed scaling. A stack drives it through the action, so the action's handler -- which feeds
the waypoints itself -- is where the scaling has to apply: at 0.5 the same path takes twice the
simulated time. The stand-in arm and physics loop are the reduced ones the cancel tests use.
"""

from __future__ import annotations

import threading

import pytest
from builtin_interfaces.msg import Duration as DurationMsg
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint

from roqsim.context import Endpoint, SimContext
from roqsim_ros_bridge.actions import follow_joint_trajectory

JOINT = "shoulder_pan_joint"
DT = 0.005


class _Clock:
    def __init__(self) -> None:
        self.time = 0.0


class _Arm:
    """Takes every commanded setpoint at once and records it with the sim time it was applied at."""

    def __init__(self, ctx, scale) -> None:
        self._ctx = ctx
        self.position = 0.0
        self.applied: list[tuple[float, float]] = []
        self.scale = scale
        if scale is not None:
            self.trajectory_time_scale = lambda: self.scale

    def set_targets(self, names, positions) -> None:
        self.position = float(dict(zip(names, positions, strict=False))[JOINT])
        self.applied.append((self._ctx.sim_time, self.position))

    def read_state(self):
        return ([JOINT], [self.position], [0.0], [0.0])


class _GoalHandle:
    def __init__(self, request) -> None:
        self.request = request
        self.is_cancel_requested = False
        self.status = "executing"

    def succeed(self) -> None:
        self.status = "succeeded"

    def abort(self) -> None:
        self.status = "aborted"

    def canceled(self) -> None:
        self.status = "canceled"

    def publish_feedback(self, _feedback) -> None:
        pass


def _run(scale, *, change_at: float | None = None, to: float | None = None):
    """Execute a 2 s straight line of one joint from 0 to 2 rad; return the arm and the end time."""
    ctx = SimContext(config={})
    ctx.data = _Clock()
    arm = _Arm(ctx, scale)
    ctx.blackboard.set("arm:ur5e", arm)
    endpoint = Endpoint(
        name="follow_joint_trajectory",
        direction="in",
        owner="ur5e",
        write=lambda wp: arm.set_targets(*wp),
        backend={
            "ros2": {
                "action": "control_msgs.action.FollowJointTrajectory",
                "name": "scaled_joint_trajectory_controller/follow_joint_trajectory",
                "arm_state_key": "arm:ur5e",
            }
        },
    )
    goal = FollowJointTrajectory.Goal()
    goal.trajectory.joint_names = [JOINT]
    start, end = JointTrajectoryPoint(), JointTrajectoryPoint()
    start.positions, end.positions = [0.0], [2.0]
    end.time_from_start = DurationMsg(sec=2)
    goal.trajectory.points = [start, end]

    stop = threading.Event()

    def physics():
        while not stop.is_set():
            ctx.drain_commands()
            ctx.data.time += DT
            if change_at is not None and ctx.data.time >= change_at:
                arm.scale = to
            stop.wait(0.001)

    thread = threading.Thread(target=physics, daemon=True)
    thread.start()
    handle = _GoalHandle(goal)
    try:
        result = follow_joint_trajectory(
            handle, ctx, lambda p: ctx.post(lambda _c, p=p: endpoint.write(p)), endpoint
        )
        finished = ctx.sim_time
    finally:
        stop.set()
        thread.join(timeout=2.0)
    assert handle.status == "succeeded"
    assert result.error_code == FollowJointTrajectory.Result.SUCCESSFUL
    return arm, finished


def _at(arm: _Arm, sim_time: float) -> float:
    """The last setpoint applied at or before *sim_time*."""
    return [q for t, q in arm.applied if t <= sim_time][-1]


@pytest.mark.parametrize("scale", [None, 1.0])
def test_without_a_scale_the_trajectory_runs_on_sim_time(scale):
    arm, finished = _run(scale)
    assert _at(arm, 1.0) == pytest.approx(1.0, abs=0.05)
    assert finished == pytest.approx(2.0, abs=0.05)


def test_at_half_scale_the_trajectory_progresses_half_as_far_per_sim_second():
    arm, finished = _run(0.5)
    assert _at(arm, 1.0) == pytest.approx(0.5, abs=0.05)
    assert _at(arm, 2.0) == pytest.approx(1.0, abs=0.05)
    assert finished == pytest.approx(4.0, abs=0.05)


def test_a_scale_change_mid_goal_keeps_the_time_travelled():
    arm, finished = _run(1.0, change_at=1.0, to=0.25)
    assert _at(arm, 1.0) == pytest.approx(1.0, abs=0.05)
    assert _at(arm, 3.0) == pytest.approx(1.5, abs=0.05)
    assert finished == pytest.approx(5.0, abs=0.05)
