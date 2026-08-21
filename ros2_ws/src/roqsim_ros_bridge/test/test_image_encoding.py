"""fill_image: `step` per encoding, and the dtype/encoding mismatch it refuses.

Needs ROS on the path (the registry resolves real message types), unlike its `image_codec` sibling.
"""

from __future__ import annotations

import numpy as np
import pytest
from sensor_msgs.msg import Image

from roqsim_ros_bridge.registry import fill_image

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
