"""What ``arm_controller`` accepts as a goal tolerance, and what it refuses.

The tolerance is what turns "the trajectory's clock ran out" into "the joints arrived", so a value
that silently fails to apply is the worst outcome available: the author reads the config as grading
the execution, and nothing is graded. These pin the two ways that can happen.

The policy those numbers feed is tested next door, in ``roqsim_ros_bridge``'s
``test_goal_tolerance`` -- this is only about the config surface.
"""

from __future__ import annotations

import pytest

from roqsim_manipulation.plugins.arm_controller import ArmControllerPlugin

JOINTS = ["shoulder_pan_joint", "elbow_joint"]


def _errors(**config) -> list[str]:
    cfg = {"arm": "ur5e", **config}
    return ArmControllerPlugin(cfg).validate_config(cfg)


def test_a_plain_tolerance_is_accepted():
    assert _errors(goal_tolerance=0.25) == []
    assert _errors(goal_tolerance=0.0) == [], "zero is the action's own 'no constraint'"


def test_a_per_joint_tolerance_is_accepted():
    assert _errors(joints=JOINTS, goal_tolerance={"elbow_joint": 0.1}) == []


@pytest.mark.parametrize("value", [-0.1, "loose", -1])
def test_a_negative_or_non_numeric_tolerance_is_refused(value):
    errors = _errors(goal_tolerance=value)
    assert any("non-negative number" in e for e in errors), errors


def test_a_negative_time_tolerance_is_refused():
    errors = _errors(goal_time_tolerance=-1.0)
    assert any("goal_time_tolerance" in e for e in errors), errors


def test_a_tolerance_for_an_unowned_joint_is_refused():
    """The silent case: it would be dropped, so the author's tolerance was never in force."""
    errors = _errors(joints=JOINTS, goal_tolerance={"wrist_3_joint": 0.01})
    assert any("does not list" in e for e in errors), errors


def test_an_unowned_joint_is_only_checkable_against_an_explicit_joint_list():
    """Without ``joints:`` the plugin claims joints by prefix scan, which needs the compiled model.

    Refusing here would reject every valid per-joint tolerance on an arm that lets the scan do the
    work, so this stays quiet rather than guessing.
    """
    assert _errors(goal_tolerance={"wrist_3_joint": 0.01}) == []
