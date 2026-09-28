"""The action handlers hand a typed endpoint its named parameters.

The producers are declared with :mod:`roqsim.endpoint` and served as the bridge serves them: the
handler's ``on_payload`` is the endpoint's own marshalled ``write``. The stand-in arm, hand and
physics loop are the ones ``test_cancel_stops_the_arm`` drives the untyped endpoints with, so the
same goal must end the same way through either.
"""

from __future__ import annotations

import pytest
from control_msgs.action import FollowJointTrajectory
from test_cancel_stops_the_arm import (
    _Arm,
    _Clock,
    _Fingers,
    _GoalHandle,
    _gripper_goal,
    _Physics,
    _trajectory_goal,
)

from roqsim import endpoint
from roqsim.context import SimContext
from roqsim.endpoint import build
from roqsim.plugin import Plugin
from roqsim_ros_bridge.actions import follow_joint_trajectory, gripper_command

FJT_HINTS = {
    "action": "control_msgs.action.FollowJointTrajectory",
    "name": "arm_controller/follow_joint_trajectory",
    "arm_state_key": "arm:ur5e",
    "goal_tolerance": 0.5,
    "goal_time_tolerance": 0.5,
}
GRIPPER_HINTS = {
    "action": "control_msgs.action.GripperCommand",
    "name": "gripper_controller/gripper_cmd",
    "state_key": "gripper:ur5e",
}


class _TypedArm(Plugin):
    """Takes a waypoint by the names the handler sends."""

    def __init__(self, arm: _Arm) -> None:
        super().__init__({}, label="ur5e")
        self.arm = arm

    @endpoint.command("follow_joint_trajectory", ros2=FJT_HINTS)
    def follow(self, names: list[str], positions: list[float]) -> None:
        self.arm.set_targets(names, positions)


class _MisnamedArm(_TypedArm):
    """Declares parameters the handler does not send."""

    @endpoint.command("follow_joint_trajectory", ros2=FJT_HINTS)
    def follow(self, joints: list[str], targets: list[float]) -> None:
        self.arm.set_targets(joints, targets)


class _TypedHand(Plugin):
    def __init__(self, fingers: _Fingers) -> None:
        super().__init__({}, label="ur5e")
        self.fingers = fingers

    @endpoint.command("gripper_cmd", ros2=GRIPPER_HINTS)
    def grip(self, position: float) -> None:
        self.fingers.set_gripper(position)


class _MisnamedHand(_TypedHand):
    @endpoint.command("gripper_cmd", ros2=GRIPPER_HINTS)
    def grip(self, openness: float) -> None:
        self.fingers.set_gripper(openness)


def _served(producer: Plugin, state_key: str, state):
    ctx = SimContext(config={})
    ctx.data = _Clock()
    ctx.blackboard.set(state_key, state)
    (ep,) = build(producer, ctx)
    assert ep.params is not None and ep.marshalled
    return ctx, ep, ep.write  # what BridgeBase._inbound hands a marshalled endpoint's handler


def test_a_typed_arm_follows_a_trajectory_to_its_end():
    arm = _Arm([0.0, 0.0], converge=0.5)
    ctx, ep, on_payload = _served(_TypedArm(arm), "arm:ur5e", arm)
    handle = _GoalHandle(_trajectory_goal([0.4, -0.2], 1.0), ctx)

    with _Physics(ctx, arm):
        result = follow_joint_trajectory(handle, ctx, on_payload, ep)

    assert handle.status == "succeeded"
    assert result.error_code == FollowJointTrajectory.Result.SUCCESSFUL
    assert arm.position == pytest.approx([0.4, -0.2], abs=1e-2)
    assert len(arm.commands) > 2, "every interpolated setpoint reached the arm, in order"


def test_a_cancel_holds_a_typed_arm_where_it_is():
    arm = _Arm([0.0, 0.0])
    ctx, ep, on_payload = _served(_TypedArm(arm), "arm:ur5e", arm)
    handle = _GoalHandle(_trajectory_goal([2.0, -1.0], 2.0), ctx, cancel_at=0.5)

    with _Physics(ctx, arm) as physics:
        result = follow_joint_trajectory(handle, ctx, on_payload, ep)
        physics.run_for(0.05)
        held = list(arm.commands[-1])
        physics.run_for(1.0)

    assert handle.status == "canceled"
    assert result.error_code == FollowJointTrajectory.Result.GOAL_TOLERANCE_VIOLATED
    assert arm.commands[-1] == held
    assert arm.position == pytest.approx(held, abs=1e-3)


def test_a_typed_arm_that_would_refuse_the_waypoints_fails_the_goal_unmoved():
    arm = _Arm([0.0, 0.0])
    ctx, ep, on_payload = _served(_MisnamedArm(arm), "arm:ur5e", arm)
    handle = _GoalHandle(_trajectory_goal([0.4, -0.2], 1.0), ctx)

    with _Physics(ctx, arm):
        result = follow_joint_trajectory(handle, ctx, on_payload, ep)

    assert handle.status == "aborted"
    assert result.error_code == FollowJointTrajectory.Result.INVALID_GOAL
    assert "names" in result.error_string and "joints" in result.error_string
    assert arm.commands == [], "nothing was sent"


def test_a_typed_hand_reaches_its_position():
    fingers = _Fingers(0.0, converge=0.5)
    ctx, ep, on_payload = _served(_TypedHand(fingers), "gripper:ur5e", fingers.read_state)
    handle = _GoalHandle(_gripper_goal(0.4), ctx)

    with _Physics(ctx, fingers):
        result = gripper_command(handle, ctx, on_payload, ep)

    assert handle.status == "succeeded"
    assert result.reached_goal is True
    assert result.position == pytest.approx(0.4, abs=5e-3)


def test_a_typed_hand_that_would_refuse_the_position_aborts_the_goal_unmoved():
    fingers = _Fingers(0.0)
    ctx, ep, on_payload = _served(_MisnamedHand(fingers), "gripper:ur5e", fingers.read_state)
    handle = _GoalHandle(_gripper_goal(0.4), ctx)

    with _Physics(ctx, fingers):
        result = gripper_command(handle, ctx, on_payload, ep)

    assert handle.status == "aborted"
    assert result.reached_goal is False
    assert fingers.commands == []
