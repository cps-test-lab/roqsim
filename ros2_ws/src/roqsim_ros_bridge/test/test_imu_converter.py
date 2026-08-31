"""The ``sensor_msgs/Imu`` converter: the two conventions a consumer of this message relies on.

Both are easy to get subtly wrong and impossible to notice downstream. Gravity belongs *in* the
acceleration (REP 145) -- a filter fed coordinate acceleration sees a robot accelerating upward at
1 g forever and tilts its estimate to match. And a device that does not measure attitude must say so
with ``orientation_covariance[0] = -1``: an identity quaternion published instead is read as a
perfectly level robot, which is the one attitude a consumer will not question.
"""

from dataclasses import dataclass, field

import pytest

from roqsim_ros_bridge.registry import get_converter, to_time_msg


@dataclass
class _Reading:
    """Stand-in for roqsim_sensors.plugins.imu.ImuReading (the bridge never imports the producer)."""

    orientation: list = field(default_factory=lambda: [1.0, 0.0, 0.0, 0.0])
    angular_velocity: list = field(default_factory=lambda: [0.0, 0.0, 0.0])
    linear_acceleration: list = field(default_factory=lambda: [0.0, 0.0, 9.81])
    orientation_valid: bool = True
    orientation_variance: float = 0.0
    angular_velocity_variance: float = 0.0
    linear_acceleration_variance: float = 0.0


def _fill(payload, hints=None):
    from sensor_msgs.msg import Imu

    msg = Imu()
    get_converter("sensor_msgs.msg.Imu")(msg, payload, to_time_msg(1.5), hints or {})
    return msg


def test_the_reading_lands_in_the_right_fields():
    msg = _fill(
        _Reading(
            orientation=[0.7071067811865476, 0.7071067811865476, 0.0, 0.0],
            angular_velocity=[0.1, 0.2, 0.3],
            linear_acceleration=[0.0, 9.81, 0.0],
        ),
        {"frame_id": "imu_link"},
    )
    assert msg.header.frame_id == "imu_link"
    assert msg.header.stamp.sec == 1 and msg.header.stamp.nanosec == 500000000
    # (w, x, y, z) in, and the message's own field order is (x, y, z, w) -- a swap here would publish
    # a 90 deg roll as a 90 deg... something else, silently.
    assert msg.orientation.w == pytest.approx(0.7071067811865476)
    assert msg.orientation.x == pytest.approx(0.7071067811865476)
    assert (msg.angular_velocity.x, msg.angular_velocity.y) == (0.1, 0.2)
    assert msg.linear_acceleration.y == pytest.approx(9.81)


def test_gravity_is_passed_through_not_removed():
    """A level, resting robot publishes +9.81 on z. Nothing here compensates it."""
    msg = _fill(_Reading())
    assert msg.linear_acceleration.z == pytest.approx(9.81)


def test_variances_become_isotropic_diagonals():
    msg = _fill(
        _Reading(
            orientation_variance=0.01,
            angular_velocity_variance=1e-4,
            linear_acceleration_variance=0.04,
        )
    )
    assert list(msg.orientation_covariance) == [0.01, 0, 0, 0, 0.01, 0, 0, 0, 0.01]
    assert list(msg.angular_velocity_covariance)[0] == pytest.approx(1e-4)
    assert list(msg.linear_acceleration_covariance)[8] == pytest.approx(0.04)


def test_a_rate_only_imu_marks_the_attitude_channel_absent():
    msg = _fill(_Reading(orientation_valid=False, angular_velocity=[0.0, 0.0, 0.5]))
    assert msg.orientation_covariance[0] == -1.0
    # The rates still arrive: only the attitude is withheld.
    assert msg.angular_velocity.z == pytest.approx(0.5)


def test_the_frame_id_is_namespaced_like_every_other_endpoint():
    msg = _fill(_Reading(), {"frame_id": "imu_link", "frame_prefix": "robot_b"})
    assert msg.header.frame_id == "robot_b/imu_link"
