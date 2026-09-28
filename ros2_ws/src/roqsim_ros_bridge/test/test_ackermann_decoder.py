"""The ``ackermann_msgs`` decoders: a steering angle is not a yaw rate, and must not become one.

The tempting shape is a decoder that turns an AckermannDrive into the twist every other base here
consumes. It is wrong twice over. Converting needs the WHEELBASE -- robot geometry, which this module
does not know and is documented not to know -- and `w = v * tan(delta) / L` collapses at v = 0, which
is the one case the message exists to carry: a steering angle still says where the wheels point when
the car is stopped, while a curvature says nothing at all.

So the decoders pass the two numbers through untouched, and the consumer -- which knows its own
wheelbase -- decides what they mean.
"""

from dataclasses import dataclass, field

from roqsim_ros_bridge.registry import DECODERS


@dataclass
class _Drive:
    """Stand-in for ackermann_msgs.msg.AckermannDrive."""

    steering_angle: float = 0.0
    steering_angle_velocity: float = 0.0
    speed: float = 0.0
    acceleration: float = 0.0
    jerk: float = 0.0


@dataclass
class _Stamped:
    """Stand-in for ackermann_msgs.msg.AckermannDriveStamped."""

    drive: _Drive = field(default_factory=_Drive)


def test_the_two_numbers_pass_through_in_order():
    decode = DECODERS["ackermann_msgs.msg.AckermannDrive"]
    assert decode(_Drive(steering_angle=0.35, speed=1.25)) == (0.35, 1.25)


def test_a_stopped_car_still_states_an_angle():
    """The case a twist cannot express, and the reason not to convert here."""
    decode = DECODERS["ackermann_msgs.msg.AckermannDrive"]
    assert decode(_Drive(steering_angle=0.42, speed=0.0)) == (0.42, 0.0)


def test_the_stamped_form_unwraps_to_the_same_pair():
    decode = DECODERS["ackermann_msgs.msg.AckermannDriveStamped"]
    assert decode(_Stamped(_Drive(steering_angle=-0.2, speed=-0.5))) == (-0.2, -0.5)


def test_the_decoder_reads_no_geometry_and_no_derivatives():
    """It must not reach for a wheelbase it does not have, nor silently drop what it cannot use.

    `steering_angle_velocity`, `acceleration` and `jerk` are rate LIMITS the consumer configures for
    itself (`steer_rate`, `accel_limit`); a decoder that folded them into the payload would be
    handing per-message overrides to a plugin that treats them as vehicle properties.
    """
    decode = DECODERS["ackermann_msgs.msg.AckermannDrive"]
    payload = decode(_Drive(steering_angle=0.1, speed=2.0, acceleration=99.0, jerk=99.0))
    assert payload == (0.1, 2.0)
