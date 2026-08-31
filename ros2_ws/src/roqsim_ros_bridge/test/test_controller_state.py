"""``JointTrajectoryControllerState``: the sign of ``error``.

The message defines it for us -- "The error of the controlled value, essentially reference -
feedback (for a regular PID implementation)" -- and ros2_control's own JointTrajectoryController
publishes that. This converter published the negation, so an arm lagging behind its setpoint
reported as leading it, and anything reading the field as "how much further to go" got the
direction of travel wrong while the magnitude looked right.
"""

import pytest

from roqsim_ros_bridge.registry import get_converter


class _Point:
    def __init__(self):
        self.positions = []
        self.velocities = []


class _Header:
    def __init__(self):
        self.stamp = None


class _StateMsg:
    """Stand-in for control_msgs.msg.JointTrajectoryControllerState (Jazzy field names)."""

    def __init__(self):
        self.header = _Header()
        self.joint_names = []
        self.reference = _Point()
        self.feedback = _Point()
        self.error = _Point()


def _fill(desired, actual, velocities=None):
    msg = _StateMsg()
    payload = (["j1", "j2"], desired, actual, velocities or [0.0] * len(desired))
    get_converter("control_msgs.msg.JointTrajectoryControllerState")(msg, payload, None, {})
    return msg


def test_error_is_reference_minus_feedback():
    msg = _fill(desired=[1.0, -0.5], actual=[0.7, -0.2])
    assert list(msg.reference.positions) == pytest.approx([1.0, -0.5])
    assert list(msg.feedback.positions) == pytest.approx([0.7, -0.2])
    # Positive where the joint still has further to go in the positive direction, and negative
    # where it has overshot -- not the other way round.
    assert list(msg.error.positions) == pytest.approx([0.3, -0.3])


def test_a_joint_that_has_arrived_reports_no_error():
    msg = _fill(desired=[0.4, 0.4], actual=[0.4, 0.4])
    assert list(msg.error.positions) == pytest.approx([0.0, 0.0])
