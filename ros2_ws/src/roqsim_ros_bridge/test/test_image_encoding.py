"""The image converters: `step` per encoding, the mismatches they refuse, and which codec
serves which stream.

Needs ROS on the path (the registry resolves real message types), unlike its `image_codec` sibling.
"""

from __future__ import annotations

import numpy as np
import pytest
from sensor_msgs.msg import CompressedImage, Image

from roqsim_ros_bridge.image_codec import encode_depth
from roqsim_ros_bridge.registry import fill_compressed_image, fill_image

STAMP = None  # the converter only assigns it; no field of it is read here


def test_step_and_data_follow_the_encoding():
    for encoding, payload in (
        ("rgb8", np.zeros((48, 64, 3), np.uint8)),
        ("mono8", np.zeros((48, 64), np.uint8)),
        ("16UC1", np.zeros((48, 64), np.uint16)),
        ("32FC1", np.zeros((48, 64), np.float32)),
    ):
        msg = Image()
        fill_image(msg, payload, STAMP, {"encoding": encoding})
        assert (msg.height, msg.width, msg.encoding) == (48, 64, encoding)
        assert msg.step == 64 * payload.itemsize * (payload.shape[2] if payload.ndim > 2 else 1)
        assert len(msg.data) == 48 * msg.step


def test_a_dtype_that_disagrees_with_the_encoding_is_refused():
    """The failure this guards is silent otherwise: `step` right for the encoding, `data` holding a
    different number of bytes, and a garbled or truncated image at the far end. Depth is where the
    two can drift -- float32 metres under a 16UC1 hint, or the reverse."""
    for encoding, payload in (
        ("16UC1", np.zeros((4, 4), np.float32)),
        ("32FC1", np.zeros((4, 4), np.uint16)),
        ("rgb8", np.zeros((4, 4), np.uint8)),
    ):
        with pytest.raises(ValueError, match="bytes per pixel"):
            fill_image(Image(), payload, STAMP, {"encoding": encoding})


def test_an_unknown_encoding_names_the_ones_that_exist():
    with pytest.raises(ValueError, match="unsupported image encoding"):
        fill_image(Image(), np.zeros((4, 4), np.uint8), STAMP, {"encoding": "bayer_rggb8"})


# -- compressedDepth, and the pairings the converter refuses -------------------------------------
def test_compressed_depth_names_the_transport_image_transport_expects():
    """`<encoding>; compressedDepth <codec>` -- the string cv_bridge and `image_transport republish`
    parse. A consumer picks the codec out of it, so our own spelling would decode as nothing."""
    frame = np.arange(1, 17, dtype=np.uint16).reshape(4, 4) * 100
    msg = CompressedImage()
    fill_compressed_image(msg, frame, STAMP, {"encoding": "16UC1", "format": "rvl"})
    assert msg.format == "16UC1; compressedDepth rvl"
    assert bytes(msg.data) == encode_depth(frame)


def test_colour_and_depth_are_told_apart_by_the_codec_not_the_message_type():
    """Both are CompressedImage, and converters are keyed on the type -- so the `format` hint is
    what decides, and each half refuses the other's encoding by name."""
    colour = CompressedImage()
    fill_compressed_image(colour, np.zeros((4, 4, 3), np.uint8), STAMP, {"encoding": "rgb8"})
    assert colour.format == "rgb8; jpeg compressed bgr8"

    with pytest.raises(TypeError, match="16-bit only"):  # colour pixels, depth codec
        fill_compressed_image(
            CompressedImage(),
            np.zeros((4, 4, 3), np.uint8),
            STAMP,
            {"encoding": "rgb8", "format": "rvl"},
        )
    with pytest.raises(TypeError, match="compressedDepth"):  # depth pixels, colour codec
        fill_compressed_image(
            CompressedImage(),
            np.zeros((4, 4), np.uint16),
            STAMP,
            {"encoding": "16UC1", "format": "jpeg"},
        )


def test_float_depth_is_pointed_at_the_encoding_that_can_carry_it():
    """32FC1 compressedDepth is a quantised-PNG pipeline, and lossy; the answer is to publish the
    depth as 16UC1, which is what the hardware does."""
    with pytest.raises(TypeError, match="Publish the depth as 16UC1"):
        fill_compressed_image(
            CompressedImage(),
            np.zeros((4, 4), np.float32),
            STAMP,
            {"encoding": "32FC1", "format": "rvl"},
        )
