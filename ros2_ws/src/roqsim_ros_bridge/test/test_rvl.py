"""rvl: round-trip, the wire bytes, and the shapes that break a reimplementation.

No ROS imports -- the codec has none either, which is the point of splitting it out: these run in a
plain venv.

The golden vector below was produced by the reference C++ encoder
(``compressed_depth_image_transport::RvlCodec::CompressRVL``, compiled from the upstream source
against this ROS distro's header). Round-tripping alone would only prove this module is
self-consistent; the golden is what proves it is *compatible* -- a wrong nibble or word order still
decodes to something when both ends share the mistake.
"""

from __future__ import annotations

import numpy as np
import pytest

from roqsim_ros_bridge import rvl

#: A 4x4 frame with a leading zero run, interior runs, rising and falling deltas, and a trailing zero
#: (so the last pair has no values, and the last word is padded).
GOLDEN_PIXELS = np.array(
    [[0, 0, 1000, 1001], [1003, 0, 0, 2000], [2001, 1999, 0, 0], [0, 500, 501, 0]], np.uint16
)
GOLDEN_BYTES = b"$\xf3\x8a##\xf3\xa9#!\xe5\xde2\x00\x00\x00\x00"


def test_the_bytes_are_the_reference_encoder_s():
    assert rvl.compress(GOLDEN_PIXELS) == GOLDEN_BYTES
    assert np.array_equal(
        rvl.decompress(GOLDEN_BYTES, GOLDEN_PIXELS.size), GOLDEN_PIXELS.reshape(-1)
    )


@pytest.mark.parametrize(
    "pixels",
    [
        pytest.param(np.zeros(64, np.uint16), id="all no-return"),
        pytest.param(np.arange(1, 65, dtype=np.uint16), id="all returns"),
        pytest.param(np.array([5, 7, 0, 0, 9, 0], np.uint16), id="opens with a return"),
        pytest.param(np.array([0, 0, 5, 7, 0, 9], np.uint16), id="closes with a return"),
        pytest.param(np.array([1234], np.uint16), id="one pixel"),
        pytest.param(np.array([0, 65535, 1, 65535, 0, 32768], np.uint16), id="extreme deltas"),
        pytest.param(np.full(1000, 65535, np.uint16), id="saturated"),
    ],
)
def test_round_trip_is_exact(pixels):
    """RVL is lossless, so this is equality, not tolerance."""
    assert np.array_equal(rvl.decompress(rvl.compress(pixels), pixels.size), pixels)


def test_round_trip_over_random_shapes_and_densities():
    """Run structure is where an off-by-one hides: the length, the density and the first/last pixel
    all change which runs are empty."""
    rng = np.random.default_rng(4)
    for _ in range(200):
        n = int(rng.integers(1, 400))
        pixels = ((rng.random(n) < rng.random()) * rng.integers(1, 65536, n)).astype(np.uint16)
        assert np.array_equal(rvl.decompress(rvl.compress(pixels), n), pixels)


def test_output_is_whole_words_within_the_buffer_the_reference_allocates():
    """Nibbles go into 32-bit words, so a partial word is padded rather than truncated -- the output
    is always a multiple of four bytes.

    The bound is the reference *encoder*'s own buffer, `3 * numPixels + 12`. Its header claims
    `1.5 * numPixels + 4`, which does not hold for uncorrelated 16-bit values: a zigzagged delta of
    that size takes six nibbles, i.e. three bytes for one pixel. Random data is the worst case here
    and real depth is nowhere near it, but a buffer sized from the header's number would overflow."""
    rng = np.random.default_rng(5)
    for n in (1, 7, 8, 9, 1000):
        pixels = rng.integers(1, 65536, n).astype(np.uint16)
        data = rvl.compress(pixels)
        assert len(data) % 4 == 0
        assert len(data) <= 3 * n + 12


def test_a_smooth_surface_takes_the_small_delta_path():
    """Not a performance claim -- a codec that coded whole values instead of deltas would still
    round-trip, and only the size would show it."""
    y, x = np.mgrid[0:120, 0:160]
    frame = (1000 + 800 * x / 160).astype(np.uint16)  # a 5 mm step per pixel, a fifth no-return
    frame[y > 96] = 0
    assert len(rvl.compress(frame)) < frame.nbytes / 2


def test_only_16_bit_depth_is_accepted():
    with pytest.raises(TypeError, match="16-bit"):
        rvl.compress(np.zeros((4, 4), np.float32))
