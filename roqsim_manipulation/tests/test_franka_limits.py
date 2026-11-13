"""The limits the simulated FR3 enforces are Franka's, read from the pinned files.

``rate_limiting.h`` (libfranka): the joint acceleration, jerk and torque-rate limits and the
Cartesian and elbow limits; ``joint_limits.yaml`` (franka_description): ranges, velocities, efforts and
the position-based velocity limits; ``src/joint_velocity_limits.cpp``: their formula.
"""

from __future__ import annotations

import numpy as np
import pytest
from _franka_upstream import LIBFRANKA, eval_constexpr, joint_limits, strip_comments, text
from roqsim_manipulation.cli.franka_upstream import check
from roqsim_manipulation.franka import limits as L


def test_the_vendored_files_are_the_pinned_ones():
    from _franka_upstream import UPSTREAM

    assert check(UPSTREAM) == []


RATE = strip_comments(text(LIBFRANKA / "include/franka/rate_limiting.h"))


def _env() -> dict:
    env = {}
    for name in (
        "kDeltaT",
        "kLimitEps",
        "kTolNumberPacketsLost",
        "kMaxJointAcceleration",
        "kMaxTranslationalAcceleration",
        "kMaxRotationalAcceleration",
        "kMaxElbowAcceleration",
    ):
        env[name] = eval_constexpr(RATE, name, env)
    return env


@pytest.mark.parametrize(
    "name, ours",
    [
        ("kDeltaT", L.DELTA_T),
        ("kLimitEps", L.LIMIT_EPS),
        ("kTolNumberPacketsLost", L.TOL_NUMBER_PACKETS_LOST),
        ("kMaxTorqueRate", L.MAX_TORQUE_RATE),
        ("kMaxJointJerk", L.MAX_JOINT_JERK),
        ("kMaxJointAcceleration", L.MAX_JOINT_ACCELERATION),
        ("kJointVelocityLimitsTolerance", L.JOINT_VELOCITY_LIMITS_TOLERANCE),
        ("kMaxTranslationalJerk", L.MAX_TRANSLATIONAL_JERK),
        ("kMaxTranslationalAcceleration", L.MAX_TRANSLATIONAL_ACCELERATION),
        ("kMaxTranslationalVelocity", L.MAX_TRANSLATIONAL_VELOCITY),
        ("kMaxRotationalJerk", L.MAX_ROTATIONAL_JERK),
        ("kMaxRotationalAcceleration", L.MAX_ROTATIONAL_ACCELERATION),
        ("kMaxRotationalVelocity", L.MAX_ROTATIONAL_VELOCITY),
        ("kMaxElbowJerk", L.MAX_ELBOW_JERK),
        ("kMaxElbowAcceleration", L.MAX_ELBOW_ACCELERATION),
        ("kMaxElbowVelocity", L.MAX_ELBOW_VELOCITY),
    ],
)
def test_rate_limiting_constants(name, ours):
    assert np.allclose(eval_constexpr(RATE, name, _env()), ours, rtol=0, atol=1e-12)


def test_joint_limits_are_franka_descriptions():
    jl = joint_limits()
    rows = [jl[f"joint{i}"] for i in range(1, 8)]
    assert np.array_equal(L.Q_LOWER, [r["limit"]["lower"] for r in rows])
    assert np.array_equal(L.Q_UPPER, [r["limit"]["upper"] for r in rows])
    assert np.array_equal(L.DQ_MAX, [r["limit"]["velocity"] for r in rows])
    assert np.array_equal(L.TAU_MAX, [r["limit"]["effort"] for r in rows])
    assert np.array_equal(
        L.VELOCITY_OFFSET, [r["position_based_velocity_limits"]["velocity_offset"] for r in rows]
    )
    assert np.array_equal(
        L.DECELERATION_LIMIT,
        [r["position_based_velocity_limits"]["deceleration_limit"] for r in rows],
    )


def test_the_velocity_limit_formula_is_libfrankas():
    """The two expressions of ``getUpper/LowerJointVelocityLimits``, evaluated against ours."""
    src = strip_comments(text(LIBFRANKA / "src/joint_velocity_limits.cpp"))
    flat = " ".join(src.split())
    assert (
        "std::min(params.max_velocity, std::max(0.0, -params.velocity_offset + std::sqrt(std::max(0.0, "
        "2.0 * params.deceleration_limit * (params.upper_position_limit - joint_positions[i]))))) - "
        "kJointVelocityLimitsTolerance[i]"
    ) in flat
    assert (
        "std::max(-params.max_velocity, std::min(0.0, params.velocity_offset - std::sqrt(std::max(0.0, "
        "2.0 * params.deceleration_limit * (-params.lower_position_limit + joint_positions[i]))))) + "
        "kJointVelocityLimitsTolerance[i]"
    ) in flat
    rng = np.random.default_rng(0)
    for _ in range(50):
        q = rng.uniform(L.Q_LOWER, L.Q_UPPER)
        up = (
            np.minimum(
                L.DQ_MAX,
                np.maximum(
                    0.0,
                    -L.VELOCITY_OFFSET
                    + np.sqrt(np.maximum(0.0, 2 * L.DECELERATION_LIMIT * (L.Q_UPPER - q))),
                ),
            )
            - L.JOINT_VELOCITY_LIMITS_TOLERANCE
        )
        assert np.allclose(L.upper_joint_velocity_limits(q), up)
        assert np.all(L.lower_joint_velocity_limits(q) <= 0.0 + L.JOINT_VELOCITY_LIMITS_TOLERANCE)


def test_impedance_ranges_are_the_documented_ones():
    import re

    robot_h = text(LIBFRANKA / "include/franka/robot.h").replace(" ", "")
    ranges = [tuple(float(x) for x in m) for m in re.findall(r"\\in\[(\d+),(\d+)\]", robot_h)]
    assert ranges[:3] == [
        L.JOINT_IMPEDANCE_RANGE,
        L.CARTESIAN_TRANSLATIONAL_STIFFNESS_RANGE,
        L.CARTESIAN_ROTATIONAL_STIFFNESS_RANGE,
    ]
    assert L.JOINT_IMPEDANCE_RANGE == (0.0, 14250.0)
    assert L.CARTESIAN_TRANSLATIONAL_STIFFNESS_RANGE == (10.0, 3000.0)
    assert L.CARTESIAN_ROTATIONAL_STIFFNESS_RANGE == (1.0, 300.0)
