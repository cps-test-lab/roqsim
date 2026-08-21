"""image_codec: channel order, byte-parity with OpenCV, and what it refuses.

No ROS imports here -- the module under test has none either, which is the point of splitting it out
of the registry: these run in a plain venv.
"""

from __future__ import annotations

import struct

import numpy as np
import pytest

from roqsim_ros_bridge import rvl
from roqsim_ros_bridge.image_codec import (
    DEFAULT_DEPTH_MAX_M,
    DEFAULT_JPEG_QUALITY,
    encode,
    encode_depth,
)


def frame(h=120, w=160):
    """A camera-like frame: smooth gradients plus the hard black/white edges a marker has."""
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    img = np.stack([40 + 180 * x / w, 40 + 180 * y / h, 128 + 100 * np.sin(6 * np.pi * x / w)], -1)
    img[20:50, 20:50] = 0
    img[70:100, 100:140] = 255
    return np.clip(img, 0, 255).astype(np.uint8)


def test_rgb_frame_decodes_to_the_right_colours_without_a_channel_swap():
    """The regression test for "but cv_bridge wants BGR, so swap first".

    It does not: JPEG stores YCbCr, so Pillow's RGB->YCbCr and OpenCV's BGR->YCbCr agree and
    ``cv2.imdecode`` (which yields BGR) already gets the right pixels. Pre-swapping puts the error
    around 55 rather than the ~1 of JPEG loss, which is what this bounds.
    """
    cv2 = pytest.importorskip("cv2")
    rgb = frame()
    decoded = cv2.imdecode(np.frombuffer(encode(rgb), np.uint8), cv2.IMREAD_ANYCOLOR)
    err = np.abs(decoded.astype(int) - rgb[..., ::-1].astype(int)).mean()
    assert err < 3.0, f"mean abs error {err:.2f} -- channels look swapped"


def test_jpeg_is_byte_identical_to_opencv():
    """What ``image_transport`` and real camera drivers put on the wire, so a simulated stream is
    comparable to a recorded one at the same quality.

    Compares the two encoders in THIS environment rather than against a golden hash, so it survives a
    libjpeg-turbo change while still failing if the pinned ``subsampling``/``optimize`` are dropped.
    """
    cv2 = pytest.importorskip("cv2")
    rgb = frame(480, 640)
    ours = encode(rgb, quality=DEFAULT_JPEG_QUALITY)
    ok, theirs = cv2.imencode(
        ".jpg",
        cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
        [cv2.IMWRITE_JPEG_QUALITY, DEFAULT_JPEG_QUALITY],
    )
    assert ok
    assert ours == theirs.tobytes()


def test_quality_controls_size():
    rgb = frame(480, 640)
    assert len(encode(rgb, quality=50)) < len(encode(rgb, quality=95))


def test_mono_frame_stays_single_channel():
    cv2 = pytest.importorskip("cv2")
    mono = frame()[..., 0].copy()
    decoded = cv2.imdecode(np.frombuffer(encode(mono), np.uint8), cv2.IMREAD_ANYCOLOR)
    assert decoded.shape == mono.shape


def test_png_is_lossless():
    cv2 = pytest.importorskip("cv2")
    rgb = frame()
    decoded = cv2.imdecode(np.frombuffer(encode(rgb, fmt="png"), np.uint8), cv2.IMREAD_ANYCOLOR)
    assert np.array_equal(decoded, rgb[..., ::-1])


@pytest.mark.parametrize(
    "array",
    [
        np.zeros((8, 8), np.float32),  # 32FC1 depth
        np.zeros((8, 8), np.uint16),  # 16UC1 depth
    ],
    ids=["32FC1", "16UC1"],
)
def test_depth_frames_are_refused_loudly(array):
    with pytest.raises(TypeError, match="8-bit"):
        encode(array)


def test_unknown_format_is_refused_loudly():
    with pytest.raises(ValueError, match="unsupported compressed format"):
        encode(frame(), fmt="tiff")


def test_wrong_channel_count_is_refused_loudly():
    with pytest.raises(ValueError, match=r"\(H, W\)"):
        encode(np.zeros((8, 8, 4), np.uint8))


# -- compressedDepth -----------------------------------------------------------------------------
def depth_frame(h=16, w=24):
    """Millimetre depth with a no-return band, as a 16UC1 camera publishes it."""
    y, x = np.mgrid[0:h, 0:w]
    frame = (600 + 40 * x + 3 * y).astype(np.uint16)
    frame[y > 0.75 * h] = 0
    return frame


def test_compressed_depth_frames_the_stream_the_way_the_reference_reads_it():
    """A 12-byte ConfigHeader, then uint32 columns and rows, then the codec's stream -- the layout
    `compressed_depth_image_transport`'s decoder parses (it reads the dimensions at bytes 12..20 and
    hands everything after them to the codec)."""
    frame = depth_frame()
    data = encode_depth(frame)
    assert len(data) == 20 + len(rvl.compress(frame))
    assert struct.unpack_from("<iff", data, 0) == (0, 0.0, 0.0)  # INV_DEPTH, unused quantisation
    assert struct.unpack_from("<II", data, 12) == (frame.shape[1], frame.shape[0])
    assert np.array_equal(rvl.decompress(data[20:], frame.size), frame.reshape(-1))


def test_compressed_depth_mirrors_the_encoder_s_max_depth_filter():
    """image_transport's encoder zeroes returns past its `depth_max` before compressing. We do the
    same, so the bytes match a driver's for the same pixels -- a camera whose range reaches past it
    is refused at load time by the sensor plugin, so this is a backstop."""
    frame = np.array([[1000, 20000], [DEFAULT_DEPTH_MAX_M * 1000, 0]], np.uint16)
    decoded = rvl.decompress(encode_depth(frame)[20:], frame.size)
    assert decoded.tolist() == [1000, 0, int(DEFAULT_DEPTH_MAX_M * 1000), 0]


def test_compressed_depth_refuses_a_frame_that_is_not_16_bit_depth():
    with pytest.raises(TypeError, match="16-bit"):
        encode_depth(np.zeros((4, 4), np.float32))
    with pytest.raises(TypeError, match="16-bit"):
        encode_depth(np.zeros((4, 4, 3), np.uint8))


def test_the_other_compressed_depth_codec_is_named_rather_than_guessed():
    """PNG over an inverse-depth quantisation of 32FC1 is the format's other half, and lossy; saying
    so beats writing an rvl stream under a png label."""
    with pytest.raises(ValueError, match="expected 'rvl'"):
        encode_depth(depth_frame(), fmt="png")
