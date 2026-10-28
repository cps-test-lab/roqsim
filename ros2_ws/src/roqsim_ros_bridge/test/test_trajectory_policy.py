"""What a JointTrajectoryController refuses and grades, decided without ROS (trajectory_policy)."""

from __future__ import annotations

import threading
from types import SimpleNamespace

from roqsim_ros_bridge.trajectory_policy import (
    PREEMPTED,
    CurrentGoal,
    path_tolerances,
    rejection,
    worst_violation,
)

JOINTS = [f"J{i}" for i in range(1, 7)]
STRICT = {
    "joint_names": JOINTS,
    "allow_partial_joints_goal": False,
    "allow_nonzero_velocity_at_trajectory_end": False,
}


def test_a_full_goal_ending_at_rest_is_taken():
    assert rejection(JOINTS, [0.0] * 6, STRICT) is None


def test_a_partial_goal_is_rejected_where_partial_goals_are_not_allowed():
    assert "don't match" in rejection(JOINTS[:3], [0.0] * 3, STRICT)
    assert rejection(JOINTS[:3], [0.0] * 3, {"joint_names": JOINTS}) is None


def test_a_foreign_joint_is_rejected_by_a_strict_controller():
    assert "J7" in rejection([*JOINTS[:5], "J7"], [0.0] * 6, STRICT)


def test_a_moving_end_is_rejected_where_it_is_not_allowed():
    assert "not zero" in rejection(JOINTS, [0.0] * 5 + [0.1], STRICT)
    assert rejection(JOINTS, [0.0] * 5 + [0.1], {"joint_names": JOINTS}) is None


def test_a_goal_without_velocities_is_at_rest():
    assert rejection(JOINTS, [], STRICT) is None


def test_path_tolerances_come_from_the_controller_and_the_goal_tightens_them():
    tol = path_tolerances((), {"path_tolerance": 0.25}, JOINTS)
    assert tol == dict.fromkeys(JOINTS, 0.25)
    entry = SimpleNamespace(name="J2", position=0.1)
    off = SimpleNamespace(name="J3", position=0.0)
    tol = path_tolerances([entry, off], {"path_tolerance": 0.25}, JOINTS)
    assert tol["J2"] == 0.1 and "J3" not in tol
    assert path_tolerances((), {}, JOINTS) == {}


def test_the_worst_violation_names_the_joint_furthest_out():
    tol = dict.fromkeys(JOINTS[:2], 0.25)
    assert worst_violation(JOINTS[:2], [0.0, 0.0], [0.3, 0.5], tol)[0] == "J2"
    assert worst_violation(JOINTS[:2], [0.0, 0.0], [0.1, 0.2], tol) is None


def test_a_new_goal_replaces_the_running_one():
    goals = CurrentGoal()
    first = goals.start("arm")
    second = goals.start("arm")
    assert not goals.is_current("arm", first)
    assert goals.is_current("arm", second)
    goals.finish("arm", first)  # the replaced goal ending does not clear the new one
    assert goals.is_current("arm", second)
    goals.finish("arm", second)
    assert not goals.is_current("arm", second)
    assert "new incoming action" in PREEMPTED


def test_goals_on_two_controllers_do_not_replace_each_other():
    goals = CurrentGoal()
    left, right = goals.start("left"), goals.start("right")
    assert goals.is_current("left", left) and goals.is_current("right", right)


def test_starting_goals_from_many_threads_leaves_exactly_one_current():
    goals = CurrentGoal()
    tokens = []
    threads = [threading.Thread(target=lambda: tokens.append(goals.start("arm"))) for _ in range(50)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sum(goals.is_current("arm", t) for t in tokens) == 1
