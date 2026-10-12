"""roqsim.ros_messages encodes what the bridge publishes: each message, decoded by rclpy, equals the
one the bridge's converter fills from the same payload.

Compared as messages rather than bytes, since rclpy leaves a message's alignment padding
uninitialised; the bridge's message goes through rclpy's own serialisation first, which rounds a
float32 field as the wire does.
"""

from __future__ import annotations

import numpy as np
import pytest
from rclpy.serialization import deserialize_message, serialize_message
from sensor_msgs.msg import CameraInfo, CompressedImage, Image, LaserScan, PointCloud2

from roqsim import ros_messages as rm
from roqsim import types as T
from roqsim_ros_bridge import registry as reg

HINTS = {"frame_id": "cam", "frame_prefix": "r1"}
T_SIM = 12.345678901

CASES = [
    (
        Image,
        reg.fill_image,
        np.arange(24, dtype=np.uint8).reshape(2, 4, 3),
        {**HINTS, "encoding": "rgb8"},
    ),
    (Image, reg.fill_image, np.full((3, 5), 1.5, np.float32), {**HINTS, "encoding": "32FC1"}),
    (
        CompressedImage,
        reg.fill_compressed_image,
        np.arange(48, dtype=np.uint8).reshape(4, 4, 3),
        {**HINTS, "encoding": "rgb8"},
    ),
    (
        CompressedImage,
        reg.fill_compressed_image,
        np.arange(12, dtype=np.uint16).reshape(3, 4),
        {**HINTS, "encoding": "16UC1", "format": "rvl"},
    ),
    (
        CameraInfo,
        reg.fill_camera_info,
        T.CameraInfo(64, 48, 50.0, 51.0, 31.5, 23.5, [0.1, 0, 0, 0, 0]),
        HINTS,
    ),
    (
        PointCloud2,
        reg.fill_pointcloud,
        T.PointCloud(np.random.default_rng(0).random((7, 3), np.float32)),
        HINTS,
    ),
    (
        LaserScan,
        reg.fill_scan,
        T.LaserScan(np.array([1, np.inf, 2], np.float32), -1.0, 1.0, 1.0, 0.1, 10.0),
        {},
    ),
]


@pytest.mark.parametrize("cls, fill, payload, hints", CASES)
def test_the_offline_message_is_the_bridges(cls, fill, payload, hints):
    published = cls()
    fill(published, payload, reg.to_time_msg(T_SIM), hints)
    name = f"sensor_msgs/msg/{cls.__name__}"
    wire = deserialize_message(serialize_message(published), cls)
    assert deserialize_message(rm.encode(name, payload, T_SIM, hints), cls) == wire
