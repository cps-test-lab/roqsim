"""roqsim.ros_messages: the ROS 2 sensor messages, encoded without ROS.

Byte-level agreement with ``rclpy`` is checked where ROS is sourced
(``roqsim_ros_bridge/test/test_ros_messages_match_rclpy.py``); these pin the layout itself.
"""

from __future__ import annotations

import struct

import numpy as np
import pytest

from roqsim import ros_messages as rm
from roqsim import types as T


def test_an_image_is_header_then_fields_in_cdr():
    pixels = np.arange(24, dtype=np.uint8).reshape(2, 4, 3)
    raw = rm.encode(
        rm.IMAGE, T.Image(pixels, "rgb8"), 12.5, {"frame_id": "cam", "frame_prefix": "r1"}
    )
    assert raw[:4] == b"\x00\x01\x00\x00"
    sec, nanosec, n = struct.unpack_from("<iII", raw, 4)
    assert (sec, nanosec) == (12, 500_000_000)
    assert raw[16 : 16 + n] == b"r1/cam\x00"
    assert raw.endswith(struct.pack("<I", 24) + pixels.tobytes())


def test_a_global_frame_takes_no_namespace():
    raw = rm.encode(
        rm.LASER_SCAN,
        T.LaserScan(np.ones(3, np.float32), -1, 1, 1, 0.1, 5),
        0.0,
        {"frame_id": "map", "frame_prefix": "r1"},
    )
    assert b"map\x00" in raw and b"r1/map" not in raw


def test_the_encoding_must_fit_the_pixels():
    with pytest.raises(ValueError, match="needs 2 bytes per pixel"):
        rm.encode(rm.IMAGE, T.Image(np.zeros((2, 2), np.float32), "16UC1"), 0.0, {})


def test_compressed_depth_is_16_bit_only_and_names_its_codec():
    depth = np.full((4, 4), 1234, np.uint16)
    fields = rm.compressed_image_fields(depth, "16UC1", {"format": "png"})
    assert fields["format"] == "16UC1; compressedDepth png"
    with pytest.raises(TypeError, match="cannot compress encoding '32FC1'"):
        rm.compressed_image_fields(depth.astype(np.float32), "32FC1", {})
    assert rm.is_lossless_format("png") and rm.is_lossless_format("rvl")
    assert not rm.is_lossless_format("jpeg")


def test_an_endpoints_message_is_its_hint_else_its_payloads_default():
    assert rm.message_type(T.Image, {}) == rm.IMAGE
    assert (
        rm.message_type(T.Image, {"type": "sensor_msgs.msg.CompressedImage"}) == rm.COMPRESSED_IMAGE
    )
    assert rm.message_type(T.PointCloud, {}) == rm.POINT_CLOUD2
    assert rm.message_type(None, {"type": "vision_msgs.msg.Detection2DArray"}) is None


def test_every_schema_carries_its_dependencies():
    for name, schema in rm.SCHEMAS.items():
        assert schema.splitlines()[0].startswith("std_msgs/Header header"), name
        assert "MSG: builtin_interfaces/Time" in schema
    assert "MSG: sensor_msgs/PointField" in rm.SCHEMAS[rm.POINT_CLOUD2]
    assert "MSG: sensor_msgs/RegionOfInterest" in rm.SCHEMAS[rm.CAMERA_INFO]


def test_a_bgr8_frame_compresses_to_the_colour_an_rgb8_one_does():
    """The codec reads RGB, so a bgr8 frame is put back in that order: both decode to one colour."""
    rgb = np.zeros((8, 8, 3), np.uint8)
    rgb[..., 0] = 200  # red
    as_rgb = rm.compressed_image_fields(rgb, "rgb8", {"format": "png"})
    as_bgr = rm.compressed_image_fields(
        np.ascontiguousarray(rgb[..., ::-1]), "bgr8", {"format": "png"}
    )
    assert as_bgr["data"] == as_rgb["data"]
    assert as_bgr["format"] == "bgr8; png compressed bgr8"
    assert rm.image_fields(rgb[..., ::-1], "bgr8")["step"] == 24
