"""image_codec: channel order, byte-parity with OpenCV, and what it refuses.

No ROS imports here -- the module under test has none either, which is the point of splitting it out
of the registry: these run in a plain venv.
"""

from __future__ import annotations

import io
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


def test_compressed_depth_rvl_frames_the_stream_the_way_the_reference_reads_it():
    """A 12-byte ConfigHeader, then uint32 columns and rows, then the codec's stream -- the layout
    `compressed_depth_image_transport`'s decoder parses (it reads the dimensions at bytes 12..20 and
    hands everything after them to the codec)."""
    frame = depth_frame()
    data = encode_depth(frame, fmt="rvl")
    assert len(data) == 20 + len(rvl.compress(frame))
    assert struct.unpack_from("<iff", data, 0) == (0, 0.0, 0.0)  # INV_DEPTH, unused quantisation
    assert struct.unpack_from("<II", data, 12) == (frame.shape[1], frame.shape[0])
    assert np.array_equal(rvl.decompress(data[20:], frame.size), frame.reshape(-1))


def test_compressed_depth_png_is_the_header_and_a_16_bit_png():
    """PNG carries its own dimensions, so nothing sits between the ConfigHeader and the image -- the
    reference hands everything after the header to `cv::imdecode`."""
    frame = depth_frame()
    data = encode_depth(frame)  # png is the default
    assert struct.unpack_from("<iff", data, 0) == (0, 0.0, 0.0)
    assert data[12:16] == b"\x89PNG"


def test_the_png_depth_bytes_read_back_as_the_same_uint16_pixels():
    """The decoder calls `cv::imdecode(..., IMREAD_UNCHANGED)`; PNG stores 16-bit samples big-endian
    while the frame is little-endian in memory, so a byte-order slip here would be invisible until
    something read the depth as metres. Pillow is asked to read it back when OpenCV is absent -- the
    property under test is the file, not which library opens it."""
    frame = depth_frame()
    payload = encode_depth(frame)[12:]
    try:
        import cv2

        back = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_UNCHANGED)
    except ImportError:
        from PIL import Image

        back = np.array(Image.open(io.BytesIO(payload)))
    assert back.dtype in (np.uint16, np.int32)  # cv2 gives uint16, Pillow widens to int32
    assert np.array_equal(back.astype(np.uint16), frame)


def test_both_depth_codecs_are_lossless():
    """Which codec a bag carries changes its size and nothing else."""
    frame = depth_frame()
    from PIL import Image

    png = np.array(Image.open(io.BytesIO(encode_depth(frame, fmt="png")[12:]))).astype(np.uint16)
    assert np.array_equal(png, frame)
    assert np.array_equal(
        rvl.decompress(encode_depth(frame, fmt="rvl")[20:], frame.size), frame.reshape(-1)
    )


def test_compressed_depth_mirrors_the_encoder_s_max_depth_filter():
    """image_transport's encoder zeroes returns past its `depth_max` before compressing. We do the
    same, so the bytes match a driver's for the same pixels -- a camera whose range reaches past it
    is refused at load time by the sensor plugin, so this is a backstop."""
    frame = np.array([[1000, 20000], [DEFAULT_DEPTH_MAX_M * 1000, 0]], np.uint16)
    decoded = rvl.decompress(encode_depth(frame, fmt="rvl")[20:], frame.size)
    assert decoded.tolist() == [1000, 0, int(DEFAULT_DEPTH_MAX_M * 1000), 0]


def test_compressed_depth_refuses_a_frame_that_is_not_16_bit_depth():
    with pytest.raises(TypeError, match="16-bit"):
        encode_depth(np.zeros((4, 4), np.float32))
    with pytest.raises(TypeError, match="16-bit"):
        encode_depth(np.zeros((4, 4, 3), np.uint8))


def test_an_unknown_depth_codec_names_the_ones_that_exist():
    with pytest.raises(ValueError, match="expected one of"):
        encode_depth(depth_frame(), fmt="zstd")
