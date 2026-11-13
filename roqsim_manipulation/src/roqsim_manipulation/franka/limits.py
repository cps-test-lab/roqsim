"""The limits a Franka Research 3 enforces, as Franka publishes them.

Each value names the file it is from, at the commits ``tests/upstream/franka/SOURCES.toml`` pins;
``tests/test_franka_limits.py`` parses those files and fails when a value here differs.
"""

from __future__ import annotations

import math

import numpy as np

#: libfranka ``rate_limiting.h``: the control period and the margin every limit is reduced by.
DELTA_T = 1e-3
LIMIT_EPS = 1e-3
TOL_NUMBER_PACKETS_LOST = 0.0

#: ``rate_limiting.h``: per joint.
MAX_TORQUE_RATE = np.full(7, 1000.0 - LIMIT_EPS)
MAX_JOINT_JERK = np.full(7, 5000.0 - LIMIT_EPS)
MAX_JOINT_ACCELERATION = np.full(7, 10.0 - LIMIT_EPS)
JOINT_VELOCITY_LIMITS_TOLERANCE = (
    LIMIT_EPS + TOL_NUMBER_PACKETS_LOST * DELTA_T * MAX_JOINT_ACCELERATION
)

#: ``rate_limiting.h``: the end effector.
MAX_TRANSLATIONAL_JERK = 4500.0 - LIMIT_EPS
MAX_TRANSLATIONAL_ACCELERATION = 9.0 - LIMIT_EPS
MAX_TRANSLATIONAL_VELOCITY = (
    3.0 - LIMIT_EPS - TOL_NUMBER_PACKETS_LOST * DELTA_T * MAX_TRANSLATIONAL_ACCELERATION
)
MAX_ROTATIONAL_JERK = 8500.0 - LIMIT_EPS
MAX_ROTATIONAL_ACCELERATION = 17.0 - LIMIT_EPS
MAX_ROTATIONAL_VELOCITY = (
    2.5 - LIMIT_EPS - TOL_NUMBER_PACKETS_LOST * DELTA_T * MAX_ROTATIONAL_ACCELERATION
)
MAX_ELBOW_JERK = 5000.0 - LIMIT_EPS
MAX_ELBOW_ACCELERATION = 10.0 - LIMIT_EPS
MAX_ELBOW_VELOCITY = 1.5 - LIMIT_EPS - TOL_NUMBER_PACKETS_LOST * DELTA_T * MAX_ELBOW_ACCELERATION

#: franka_description ``robots/fr3/joint_limits.yaml``.
Q_LOWER = np.array([-2.9007, -1.8361, -2.9007, -3.0770, -2.8763, 0.4398, -3.0508])
Q_UPPER = np.array([2.9007, 1.8361, 2.9007, -0.1169, 2.8763, 4.6216, 3.0508])
DQ_MAX = np.array([2.62, 2.62, 2.62, 2.62, 5.26, 4.18, 5.26])
TAU_MAX = np.array([87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0])
VELOCITY_OFFSET = np.array([0.6520, 0.2500, 0.2005, 0.3542, 0.5738, 0.4885, 0.4592])
DECELERATION_LIMIT = np.array([6.0, 2.585, 3.5, 4.0, 17.0, 5.5, 17.0])

#: libfranka ``robot.h``: the ranges ``setJointImpedance`` and ``setCartesianImpedance`` document.
JOINT_IMPEDANCE_RANGE = (0.0, 14250.0)
CARTESIAN_TRANSLATIONAL_STIFFNESS_RANGE = (10.0, 3000.0)
CARTESIAN_ROTATIONAL_STIFFNESS_RANGE = (1.0, 300.0)


def upper_joint_velocity_limits(q) -> np.ndarray:
    """libfranka ``JointVelocityLimitsConfig::getUpperJointVelocityLimits`` (``src/joint_velocity_limits.cpp``)."""
    q = np.asarray(q, float)
    root = np.sqrt(np.maximum(0.0, 2.0 * DECELERATION_LIMIT * (Q_UPPER - q)))
    return (
        np.minimum(DQ_MAX, np.maximum(0.0, -VELOCITY_OFFSET + root))
        - JOINT_VELOCITY_LIMITS_TOLERANCE
    )


def lower_joint_velocity_limits(q) -> np.ndarray:
    """libfranka ``JointVelocityLimitsConfig::getLowerJointVelocityLimits``."""
    q = np.asarray(q, float)
    root = np.sqrt(np.maximum(0.0, 2.0 * DECELERATION_LIMIT * (q - Q_LOWER)))
    return (
        np.maximum(-DQ_MAX, np.minimum(0.0, VELOCITY_OFFSET - root))
        + JOINT_VELOCITY_LIMITS_TOLERANCE
    )


assert math.isclose(DELTA_T, 1e-3)
