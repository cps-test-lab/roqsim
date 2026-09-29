# Copyright (C) 2026 Frederik Pasch
# SPDX-License-Identifier: Apache-2.0
"""A GNSS fix across the wire: the status is the field a consumer reads first.

The failure behind this: a plugin that declares ``NavSatFix`` with no converter makes the reflective
fallback raise at the first publish, so the receiver a world configured never reaches ROS at all.
"""

from __future__ import annotations

from roqsim_ros_bridge.registry import get_converter


class _Header:
    def __init__(self):
        self.stamp = None
        self.frame_id = ""


class _Status:
    def __init__(self):
        self.status = 0
        self.service = 0


class _NavSatFixMsg:
    """Stand-in for sensor_msgs.msg.NavSatFix."""

    def __init__(self):
        self.header, self.status = _Header(), _Status()
        self.latitude = self.longitude = self.altitude = 0.0
        self.position_covariance = [0.0] * 9
        self.position_covariance_type = 0


def _fix(**over):
    fix = {
        "lat": 47.397742,
        "lon": 8.545594,
        "alt": 488.0,
        "eph": 0.5,
        "epv": 1.0,
        "fix_type": 3,
        "satellites": 12,
        "valid": True,
    }
    fix.update(over)
    return fix


def test_a_fix_has_a_converter_at_all():
    fill = get_converter("sensor_msgs.msg.NavSatFix")
    msg = _NavSatFixMsg()
    fill(msg, _fix(), None, {})
    assert (msg.latitude, msg.longitude, msg.altitude) == (47.397742, 8.545594, 488.0)
    assert msg.status.status == 0  # STATUS_FIX
    assert msg.status.service == 1  # SERVICE_GPS
    assert msg.header.frame_id == "gnss_link"


def test_the_covariance_is_the_declared_noise_squared_on_the_diagonal():
    fill = get_converter("sensor_msgs.msg.NavSatFix")
    msg = _NavSatFixMsg()
    fill(msg, _fix(eph=0.5, epv=2.0), None, {})
    assert msg.position_covariance == [0.25, 0.0, 0.0, 0.0, 0.25, 0.0, 0.0, 0.0, 4.0]
    assert msg.position_covariance_type == 2  # COVARIANCE_TYPE_DIAGONAL_KNOWN


def test_no_fix_is_published_as_no_fix_not_dropped():
    """Denied or not yet locked, the receiver still reports -- with the status a consumer must read
    before trusting a position, and no covariance claimed for one."""
    fill = get_converter("sensor_msgs.msg.NavSatFix")
    msg = _NavSatFixMsg()
    fill(msg, _fix(lat=0.0, lon=0.0, alt=0.0, eph=0.0, epv=0.0, fix_type=0, valid=False), None, {})
    assert msg.status.status == -1  # STATUS_NO_FIX
    assert msg.position_covariance_type == 0  # COVARIANCE_TYPE_UNKNOWN
    assert msg.position_covariance == [0.0] * 9


def test_the_frame_takes_the_bridge_prefix():
    fill = get_converter("sensor_msgs.msg.NavSatFix")
    msg = _NavSatFixMsg()
    fill(msg, _fix(), None, {"frame_prefix": "drone", "frame_id": "gps"})
    assert msg.header.frame_id == "drone/gps"
