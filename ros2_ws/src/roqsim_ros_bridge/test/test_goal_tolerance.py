"""Grading a trajectory on where the JOINTS ended, not on the trajectory's clock running out.

Guards the gap this closed. ``follow_joint_trajectory`` used to call ``goal_handle.succeed()`` as
soon as the last waypoint had been fed, so an arm that never got there -- blocked by a collision,
saturated, or following a plan that went through the furniture -- returned ``SUCCESSFUL``. MoveIt
forwards that verdict, so the caller saw a clean execution and a scene that had not changed.
Measured on a manipulation cell that planned through its own bench: the arm stalled 5.59 rad from
the last waypoint and this action still reported success.

The policy is deliberately pure (`_goal_tolerances`, `_worst_violation`) so it is testable without
an action server, a live ROS graph or a stepping simulation -- which is what makes it *tested*
rather than exercised only by a cell that happens to run an arm.
"""

from __future__ import annotations

import pytest

from roqsim_ros_bridge.actions import (
    DEFAULT_GOAL_TOLERANCE,
    _goal_tolerances,
    _worst_violation,
)

JOINTS = ["shoulder_pan_joint", "elbow_joint"]


class _Tol:
    """Stand-in for control_msgs' JointTolerance: the two fields this policy reads."""

    def __init__(self, name: str, position: float) -> None:
        self.name = name
        self.position = position


class _Request:
    def __init__(self, goal_tolerance=()) -> None:
        self.goal_tolerance = list(goal_tolerance)


def test_the_default_applies_when_nobody_asks():
    """A goal with no tolerance and a controller with no config still gets graded."""
    tol = _goal_tolerances(_Request(), {}, JOINTS)
    assert tol == dict.fromkeys(JOINTS, DEFAULT_GOAL_TOLERANCE), (
        "an unconfigured arm must still be graded, or the silent-success bug is back"
    )


def test_the_controller_config_overrides_the_default():
    assert _goal_tolerances(_Request(), {"goal_tolerance": 0.25}, JOINTS) == dict.fromkeys(
        JOINTS, 0.25
    )


def test_a_per_joint_config_sets_them_apart():
    hints = {"goal_tolerance": {"elbow_joint": 0.1}}
    tol = _goal_tolerances(_Request(), hints, JOINTS)
    assert tol == {"elbow_joint": 0.1}, (
        "a joint the per-joint map omits is unconstrained, not defaulted -- naming one joint is "
        "how an author says the others do not matter"
    )


def test_the_goal_wins_over_the_controller():
    """A caller that cares can ask for more than the controller's default."""
    req = _Request([_Tol("elbow_joint", 0.01)])
    tol = _goal_tolerances(req, {"goal_tolerance": 0.25}, JOINTS)
    assert tol["elbow_joint"] == 0.01
    assert tol["shoulder_pan_joint"] == 0.25


def test_zero_disables_the_check():
    """The action's own convention for 'no constraint', and ros2_control's."""
    assert _goal_tolerances(_Request(), {"goal_tolerance": 0.0}, JOINTS) == {}
    req = _Request([_Tol("elbow_joint", 0.0)])
    assert "elbow_joint" not in _goal_tolerances(req, {"goal_tolerance": 0.25}, JOINTS)


def test_a_tolerance_for_a_joint_we_do_not_own_is_ignored():
    """It cannot be applied, and silently widening some other joint would be worse."""
    req = _Request([_Tol("wrist_3_joint", 0.001)])
    assert set(_goal_tolerances(req, {}, JOINTS)) == set(JOINTS)


def test_an_arm_that_arrived_reports_no_violation():
    tol = dict.fromkeys(JOINTS, 0.5)
    assert _worst_violation(JOINTS, [1.0, 2.0], [1.02, 1.97], tol) is None


def test_the_stalled_arm_is_caught():
    """The measured case: 5.59 rad short, previously reported SUCCESSFUL."""
    tol = dict.fromkeys(JOINTS, DEFAULT_GOAL_TOLERANCE)
    worst = _worst_violation(JOINTS, [5.004, 1.698], [-0.581, 1.663], tol)
    assert worst is not None, "a 5.59 rad miss must not read as a completed trajectory"
    joint, err, allowed = worst
    assert joint == "shoulder_pan_joint"
    assert err == pytest.approx(5.585, abs=1e-3)
    assert allowed == DEFAULT_GOAL_TOLERANCE


def test_the_worst_offender_is_the_one_reported():
    """By how far each joint EXCEEDS its own tolerance, since they need not share one."""
    tol = {"shoulder_pan_joint": 1.0, "elbow_joint": 0.01}
    # pan is 0.5 past its tolerance, elbow 0.09 past its own: pan is the worse breach.
    worst = _worst_violation(JOINTS, [0.0, 0.0], [1.5, 0.1], tol)
    assert worst[0] == "shoulder_pan_joint"


def test_an_unconstrained_joint_cannot_fail_the_goal():
    tol = {"elbow_joint": 0.5}
    assert _worst_violation(JOINTS, [0.0, 0.0], [99.0, 0.1], tol) is None
