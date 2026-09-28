"""The ``sensor_msgs/LaserScan`` converter publishes every range exactly as the sensor produced it.

A scan's special values carry meaning a consumer acts on. REP 117 reserves ``-inf`` for a return too
close to measure, ``+inf`` for no return and NaN for an erroneous one, and a device model may
publish its driver's own constants instead (``urg_node``'s 0.004 and 65.533). A converter that
clamped, filtered or replaced any of them would turn a blind spot into an obstacle.
"""

import math
from dataclasses import dataclass

import numpy as np

from roqsim_ros_bridge.registry import get_converter, to_time_msg


@dataclass
class _Scan:
    """Stand-in for roqsim_sensors.plugins.payloads.LaserScan (the bridge never imports the producer)."""

    ranges: np.ndarray
    angle_min: float = -2.356194490
    angle_max: float = 2.356194490
    angle_increment: float = 4.71238898 / 4
    range_min: float = 0.02
    range_max: float = 30.0


def _fill(payload, hints=None):
    from sensor_msgs.msg import LaserScan

    msg = LaserScan()
    get_converter("sensor_msgs.msg.LaserScan")(msg, payload, to_time_msg(2.25), hints or {})
    return msg


def test_special_values_and_driver_constants_pass_through_unchanged():
    ranges = np.array([-math.inf, math.inf, math.nan, 0.004, 65.533, 1.5], dtype=np.float64)
    msg = _fill(_Scan(ranges=ranges), {"frame_id": "laser"})
    out = np.asarray(msg.ranges, dtype=np.float64)
    assert len(out) == len(ranges)
    assert out[0] == -math.inf
    assert out[1] == math.inf
    assert math.isnan(out[2])
    # float32 on the wire: the constants arrive as the nearest float32, not clamped to range_min or
    # range_max and not replaced by inf.
    np.testing.assert_array_equal(out[3:], ranges[3:].astype(np.float32))


def test_the_header_is_the_payloads():
    msg = _fill(_Scan(ranges=np.ones(5)), {"frame_id": "laser"})
    assert msg.header.frame_id == "laser"
    assert msg.header.stamp.sec == 2 and msg.header.stamp.nanosec == 250000000
    assert (msg.range_min, msg.range_max) == (np.float32(0.02), np.float32(30.0))
    assert msg.angle_min == np.float32(-2.356194490)
    assert msg.angle_max == np.float32(2.356194490)
