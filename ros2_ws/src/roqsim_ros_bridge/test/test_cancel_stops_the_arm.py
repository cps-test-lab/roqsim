"""Cancelling a goal stops the MOTION, not only the stream of setpoints.

An executor whose only output is the setpoint it posts each step does not stop anything by falling
silent: the producer holds the last target it was given, every tick, for as long as nobody writes
another. A cancel that merely stops feeding therefore leaves the arm converging on an interpolated
point of a path the caller has withdrawn -- so a caller that cancels and reads the joints reads a
moving arm, and reads it moving to a pose nothing asked for. The cancel paths below assert the hold
that makes the stop real, and the result that tells a cut-short execution from a completed one.

The producers here are stand-ins that behave the way ``ArmControllerPlugin`` does in the one respect
that matters -- ``data.ctrl`` is written from the last commanded target every step -- so the test
needs no model, no physics and no action server, only a thread draining the context's command queue
the way the engine drains it in ``pre_step``.
"""

from __future__ import annotations

import threading

import pytest
from builtin_interfaces.msg import Duration as DurationMsg
from control_msgs.action import FollowJointTrajectory, GripperCommand
from trajectory_msgs.msg import JointTrajectoryPoint

from roqsim.context import Endpoint, SimContext
from roqsim_ros_bridge.actions import follow_joint_trajectory, gripper_command

JOINTS = ["shoulder_pan_joint", "elbow_joint"]
#: Sim seconds one pass of the stand-in physics thread advances the clock by.
DT = 0.005


class _Clock:
    """Stands in for ``MjData``: the one member ``ctx.sim_time`` reads."""

    def __init__(self) -> None:
        self.time = 0.0


class _Arm:
    """A stand-in ``ArmControllerPlugin``: holds a target, and the joints converge on it.

    ``converge`` is the fraction of the remaining error the joints close each step, so the measured
    pose LAGS the commanded setpoint -- which is what makes "the arm was still moving when the
    cancel arrived" true here as it is in the simulator.
    """

    def __init__(self, start, converge: float = 0.05) -> None:
        self.joint_names = list(JOINTS)
        self.position = list(start)
        self.target = list(start)
        self.converge = converge
        self.commands: list[list[float]] = []

    def set_targets(self, names, positions) -> None:
        by_name = dict(zip(names, positions, strict=False))
        self.target = [
            float(by_name.get(n, t)) for n, t in zip(self.joint_names, self.target, strict=True)
        ]
        self.commands.append(list(self.target))

    def step(self) -> None:
        self.position = [
            p + (t - p) * self.converge for p, t in zip(self.position, self.target, strict=True)
        ]

    def read_state(self):
        pos = list(self.position)
        return (list(self.joint_names), pos, [0.0] * len(pos), [0.0] * len(pos))


class _Fingers:
    """A stand-in 1-DOF hand: the same hold, and the ``() -> (position, velocity)`` reader."""

    def __init__(self, start: float, converge: float = 0.05) -> None:
        self.position = start
        self.target = start
        self.velocity = 0.0
        self.converge = converge
        self.commands: list[float] = []

    def set_gripper(self, position) -> None:
        self.target = float(position)
        self.commands.append(self.target)

    def step(self) -> None:
        moved = (self.target - self.position) * self.converge
        self.position += moved
        self.velocity = moved / DT

    def read_state(self):
        return (self.position, self.velocity)


class _GoalHandle:
    """Stands in for rclpy's ServerGoalHandle: the members a handler touches.

    The cancel arrives at a sim TIME rather than after a number of polls, because that is what the
    handler's loops are paced by and it keeps the moment of the cancel the same however fast the
    machine running the test happens to be.
    """

    def __init__(self, request, ctx, cancel_at: float | None = None) -> None:
        self.request = request
        self._ctx = ctx
        self._cancel_at = cancel_at
        self.status = "executing"
        self.feedback: list = []

    @property
    def is_cancel_requested(self) -> bool:
        return self._cancel_at is not None and self._ctx.sim_time >= self._cancel_at

    def canceled(self) -> None:
        self.status = "canceled"

    def succeed(self) -> None:
        self.status = "succeeded"

    def abort(self) -> None:
        self.status = "aborted"

    def publish_feedback(self, feedback) -> None:
        self.feedback.append(feedback)


class _Physics:
    """The engine's loop, reduced to what a handler can observe: drain, step, advance the clock."""

    def __init__(self, ctx: SimContext, producer) -> None:
        self._ctx = ctx
        self._producer = producer
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            self._ctx.drain_commands()  # the engine drains at the start of every pre_step
            self._producer.step()
            self._ctx.data.time += DT
            self._stop.wait(0.001)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)

    def run_for(self, sim_seconds: float) -> None:
        """Let the stand-in arm settle, so a hold can be told from a motion that merely slowed."""
        until = self._ctx.sim_time + sim_seconds
        deadline = threading.Event()
        while self._ctx.sim_time < until and not deadline.wait(0.001):
            pass


def _wire(state, write, *, name: str, state_key: str, backend: dict):
    ctx = SimContext(config={})
    ctx.data = _Clock()
    ctx.blackboard.set(state_key, state)
    endpoint = Endpoint(
        name=name,
        direction="in",
        owner="ur5e",
        write=write,
        backend={"ros2": backend},
    )

    # What the bridge hands a handler: the payload marshalled onto the physics thread via ctx.post.
    def on_payload(payload):
        ctx.post(lambda _c, p=payload: endpoint.write(p))

    return ctx, endpoint, on_payload


def _trajectory_goal(waypoint, seconds: float, *, joints=JOINTS):
    goal = FollowJointTrajectory.Goal()
    goal.trajectory.joint_names = list(joints)
    point = JointTrajectoryPoint()
    point.positions = [float(p) for p in waypoint]
    point.time_from_start = DurationMsg(sec=int(seconds), nanosec=int((seconds % 1.0) * 1e9))
    goal.trajectory.points = [point]
    goal.goal_time_tolerance = DurationMsg(sec=0, nanosec=500_000_000)
    return goal


def _arm_setup(start, *, converge=0.05, goal_tolerance=0.5):
    arm = _Arm(start, converge=converge)
    ctx, endpoint, on_payload = _wire(
        arm,
        lambda wp: arm.set_targets(*wp),
        name="follow_joint_trajectory",
        state_key="arm:ur5e",
        backend={
            "action": "control_msgs.action.FollowJointTrajectory",
            "name": "arm_controller/follow_joint_trajectory",
            "arm_state_key": "arm:ur5e",
            "goal_tolerance": goal_tolerance,
            "goal_time_tolerance": 0.5,
        },
    )
    return arm, ctx, endpoint, on_payload


def test_a_cancel_mid_trajectory_holds_the_arm_where_it_is():
    """The hold is the MEASURED pose, and the arm stops there instead of finishing the path."""
    arm, ctx, endpoint, on_payload = _arm_setup([0.0, 0.0])
    goal = _trajectory_goal([2.0, -1.0], 2.0)
    handle = _GoalHandle(goal, ctx, cancel_at=0.5)

    with _Physics(ctx, arm) as physics:
        follow_joint_trajectory(handle, ctx, on_payload, endpoint)
        physics.run_for(0.05)  # the hold is posted onto the queue; let the stand-in drain it
        commanded_on_cancel = list(arm.commands[-1])
        physics.run_for(2.0)  # long enough for a chased setpoint to arrive

    assert handle.status == "canceled"
    setpoint_in_flight = arm.commands[-2]
    assert commanded_on_cancel[0] < setpoint_in_flight[0], (
        "the hold must be the pose the arm is IN, which lags the setpoint it was chasing; "
        "re-posting the setpoint would leave the cancel converging on the withdrawn path"
    )
    assert arm.commands[-1] == commanded_on_cancel, "nothing may command the arm after the stop"
    assert arm.position == pytest.approx(commanded_on_cancel, abs=1e-3), (
        "the arm must come to rest at the hold, not carry on toward the last waypoint"
    )
    assert abs(arm.position[0] - 2.0) > 0.5, "an arm that reached the goal did not stop on cancel"


def test_a_cancelled_execution_does_not_read_like_a_completed_one():
    """``CANCELED`` is the status; the code says the joints are not at the last waypoint."""
    arm, ctx, endpoint, on_payload = _arm_setup([0.0, 0.0])
    handle = _GoalHandle(_trajectory_goal([2.0, -1.0], 2.0), ctx, cancel_at=0.5)

    with _Physics(ctx, arm):
        result = follow_joint_trajectory(handle, ctx, on_payload, endpoint)

    assert handle.status == "canceled"
    assert result.error_code == FollowJointTrajectory.Result.GOAL_TOLERANCE_VIOLATED, (
        "a goal cut short mid-path must not return the code a goal that ran to its end returns"
    )
    assert "cancelled" in result.error_string
    assert "shoulder_pan_joint" in result.error_string, "the result names the joint it graded"


def test_a_cancel_with_the_arm_at_the_goal_is_not_an_error():
    """The grading is on the joints, so a cancel that arrives after arrival reports SUCCESSFUL."""
    arm, ctx, endpoint, on_payload = _arm_setup([1.0, 0.25])
    # The waypoint the arm already sits at, due far enough out that the cancel lands while waiting.
    handle = _GoalHandle(_trajectory_goal([1.0, 0.25], 5.0), ctx, cancel_at=0.5)

    with _Physics(ctx, arm):
        result = follow_joint_trajectory(handle, ctx, on_payload, endpoint)

    assert handle.status == "canceled"
    assert result.error_code == FollowJointTrajectory.Result.SUCCESSFUL
    assert "cancelled" in result.error_string, (
        "the status carries the cancellation, and the result says so too"
    )


def test_a_trajectory_that_runs_to_its_end_still_succeeds():
    """The cancel path must not have moved the ordinary ending."""
    arm, ctx, endpoint, on_payload = _arm_setup([0.0, 0.0], converge=0.5)
    handle = _GoalHandle(_trajectory_goal([0.4, -0.2], 1.0), ctx)

    with _Physics(ctx, arm):
        result = follow_joint_trajectory(handle, ctx, on_payload, endpoint)

    assert handle.status == "succeeded"
    assert result.error_code == FollowJointTrajectory.Result.SUCCESSFUL
    assert result.error_string == ""
    assert arm.position == pytest.approx([0.4, -0.2], abs=1e-2)
    assert handle.feedback, "a completed execution still reports feedback per waypoint"


def test_an_arm_that_never_arrives_is_still_aborted():
    """The other ending the result distinguishes: a blocked arm, with no cancel in sight."""
    arm, ctx, endpoint, on_payload = _arm_setup([0.0, 0.0], converge=0.0)
    handle = _GoalHandle(_trajectory_goal([2.0, -1.0], 0.5), ctx)

    with _Physics(ctx, arm):
        result = follow_joint_trajectory(handle, ctx, on_payload, endpoint)

    assert handle.status == "aborted"
    assert result.error_code == FollowJointTrajectory.Result.GOAL_TOLERANCE_VIOLATED
    assert "cancelled" not in result.error_string


def _gripper_setup(start: float, *, converge=0.05):
    fingers = _Fingers(start, converge=converge)
    ctx, endpoint, on_payload = _wire(
        fingers.read_state,
        lambda position: fingers.set_gripper(position),
        name="gripper_cmd",
        state_key="gripper:ur5e",
        backend={
            "action": "control_msgs.action.GripperCommand",
            "name": "gripper_controller/gripper_cmd",
            "state_key": "gripper:ur5e",
        },
    )
    return fingers, ctx, endpoint, on_payload


def _gripper_goal(position: float):
    goal = GripperCommand.Goal()
    goal.command.position = float(position)
    goal.command.max_effort = 0.0
    return goal


def test_a_cancelled_gripper_command_stops_the_fingers():
    """The hand holds where it is rather than going on closing on the withdrawn position."""
    fingers, ctx, endpoint, on_payload = _gripper_setup(0.0)
    handle = _GoalHandle(_gripper_goal(0.8), ctx, cancel_at=0.3)

    with _Physics(ctx, fingers) as physics:
        result = gripper_command(handle, ctx, on_payload, endpoint)
        physics.run_for(0.05)  # the hold is posted onto the queue; let the stand-in drain it
        commanded_on_cancel = fingers.commands[-1]
        physics.run_for(2.0)

    assert handle.status == "canceled"
    assert commanded_on_cancel < 0.8, "the hold is where the fingers are, not where they were sent"
    assert fingers.commands[-1] == commanded_on_cancel, (
        "nothing may command the hand after the stop"
    )
    assert fingers.position == pytest.approx(commanded_on_cancel, abs=1e-3)
    assert result.position == pytest.approx(commanded_on_cancel, abs=1e-3), (
        "the result reports the measured position the hand was left at"
    )
    assert result.reached_goal is False
    assert result.stalled is False


def test_a_gripper_command_that_reaches_its_position_still_succeeds():
    fingers, ctx, endpoint, on_payload = _gripper_setup(0.0, converge=0.5)
    handle = _GoalHandle(_gripper_goal(0.4), ctx)

    with _Physics(ctx, fingers):
        result = gripper_command(handle, ctx, on_payload, endpoint)

    assert handle.status == "succeeded"
    assert result.reached_goal is True
    assert result.position == pytest.approx(0.4, abs=5e-3)
