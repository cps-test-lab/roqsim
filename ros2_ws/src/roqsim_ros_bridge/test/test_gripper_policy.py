"""What the GripperCommand handler decides, asked of the producer and tested without ROS.

An arm's gripper and an automatic door share the handler, and differ in two things only they know:
how near counts as reached, and whether a goal still moving at its timeout has gone wrong. Each
states them in its endpoint's hints; nothing here looks at a name or a kind.
"""

from __future__ import annotations

import math

import pytest

from roqsim_ros_bridge.gripper_policy import (
    ABORT,
    DEFAULT_GRIPPER_TIMEOUT,
    DEFAULT_REACH_TOLERANCE,
    SUCCEED,
    GripperPolicy,
)

#: The hints a door states (roqsim_assets' door plugin): its own arrival band, and an abort.
DOOR = {
    "state_key": "machine:door:openness",
    "timeout": 12.0,
    "reach_tolerance": 0.02,
    "on_timeout": "abort",
}
#: The hints an arm's gripper states: none of the two.
ARM_GRIPPER = {"state_key": "gripper:ur5e", "effort_key": "gripper_effort:ur5e"}


def test_a_gripper_that_states_nothing_keeps_its_defaults():
    policy = GripperPolicy.from_hints(ARM_GRIPPER)
    assert policy == GripperPolicy(DEFAULT_GRIPPER_TIMEOUT, DEFAULT_REACH_TOLERANCE, "succeed")
    assert DEFAULT_REACH_TOLERANCE == 0.005


def test_a_gripper_goal_still_moving_at_its_timeout_succeeds():
    """Neither reached nor stalled: a slow hand comes back succeeded, as MoveIt expects."""
    assert GripperPolicy.from_hints(ARM_GRIPPER).verdict(reached=False, stalled=False) == SUCCEED


def test_a_door_goal_still_moving_at_its_timeout_is_aborted():
    assert GripperPolicy.from_hints(DOOR).verdict(reached=False, stalled=False) == ABORT


@pytest.mark.parametrize("hints", [ARM_GRIPPER, DOOR], ids=["arm-gripper", "door"])
@pytest.mark.parametrize("reached, stalled", [(True, False), (False, True)])
def test_a_goal_that_reached_or_stalled_succeeds_whoever_produced_it(hints, reached, stalled):
    assert GripperPolicy.from_hints(hints).verdict(reached, stalled) == SUCCEED


def test_a_door_stopped_inside_its_arrival_band_has_reached_its_goal():
    """The door counts itself arrived within 0.02 of its openness, and ROS says the same."""
    door = GripperPolicy.from_hints(DOOR)
    assert door.reached(0.985, 1.0)
    assert not door.reached(0.975, 1.0)


def test_a_gripper_keeps_its_tight_tolerance():
    gripper = GripperPolicy.from_hints(ARM_GRIPPER)
    assert gripper.reached(0.404, 0.4)
    assert not gripper.reached(0.415, 0.4)


@pytest.mark.parametrize(
    "hints, said",
    [
        ({"on_timeout": "fail"}, "on_timeout"),
        ({"on_timeout": True}, "on_timeout"),
        ({"reach_tolerance": 0}, "reach_tolerance"),
        ({"reach_tolerance": -0.1}, "reach_tolerance"),
        ({"reach_tolerance": "0.02"}, "reach_tolerance"),
        ({"reach_tolerance": math.nan}, "reach_tolerance"),
        ({"timeout": -1.0}, "timeout"),
        ({"timeout": None}, "timeout"),
        ({"timeout": True}, "timeout"),
    ],
)
def test_a_hint_it_cannot_use_is_refused_naming_it(hints, said):
    with pytest.raises(ValueError, match=f"`{said}`"):
        GripperPolicy.from_hints(hints)
