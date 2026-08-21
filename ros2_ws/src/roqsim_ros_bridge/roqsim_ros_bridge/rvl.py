"""RVL: the lossless 16-bit depth codec ``image_transport`` carries as ``compressedDepth``.

Wilson's Rapid Value Lossless coding (Microsoft Research, 2017, *Fast Lossless Depth Image
Compression*), as ``compressed_depth_image_transport``'s ``RvlCodec`` implements it -- which is the
definition that matters here, since the point of this module is that a stream from a roqsim run and
one from a real RealSense driver are the same bytes. The reference is ``rvl_codec.cpp`` /
``rvl_codec.hpp`` in ros-perception/image_transport_plugins (BSD-3); this is an independent
vectorised implementation of that algorithm, not a translation of its loops.

Deliberately free of ROS imports and of the ``compressedDepth`` framing (the header, the dimensions):
this module is the codec, :mod:`roqsim_ros_bridge.image_codec` puts it on the wire, and the registry
owns the ``format`` string. It runs in a plain venv, which is where its tests run.

**The format, in the three places a reimplementation goes wrong silently.**

*Symbols.* The image is walked as alternating runs: a count of zeros ("no return" pixels), a count of
nonzeros, then one value per nonzero pixel -- its difference from the previous nonzero pixel, zigzagged
(``(delta << 1) ^ (delta >> 31)``) so a small negative step stays a small number. The first pair may
have a zero count of 0, and the last pair a nonzero count of 0.

*Nibbles.* Each symbol is a variable-length sequence of nibbles, 3 payload bits each, low bits first,
with bit ``0x8`` set on every nibble but the last: 7 costs one nibble, 8 costs two.

**What it costs, measured.** 35 ms for a 1280x720 frame and 7.7 ms for 848x480 (a RealSense's own
depth profile), both compressing about 5:1 -- 370 kB against 1.84 MB raw. That is an order above the
JPEG colour path's 2.9 ms, because there is no C library for RVL: this is numpy doing per-nibble work
that the reference does in a tight C loop, and the encode runs on the physics thread. A camera pair
publishing 1280x720 depth at 5 Hz therefore spends about a third of a second per simulated second in
here, so the endpoint is `lazy` (nothing subscribed, nothing encoded) and a world that records the
compressed stream at full resolution should expect to pay for it.

*Words.* Nibbles are packed **most-significant first** into 32-bit words (``word <<= 4; word |=
nibble``) and each full word is written in native byte order -- so on a little-endian machine the first
nibble of a word lands in the high nibble of that word's *last* byte. A trailing partial word is
shifted up so its written nibbles keep the high end, and is emitted whole: output is always a multiple
of four bytes. Get the order wrong and the stream still decodes to *something*, which is why the tests
compare bytes against the reference encoder rather than only round-tripping.
"""

from __future__ import annotations

import numpy as np

#: A symbol needs one nibble more than the number of these thresholds its value reaches.
_NIBBLE_THRESHOLDS = (1 << (3 * np.arange(1, 12))).astype(np.uint32)


def compress(pixels: np.ndarray) -> bytes:
    """Compress a uint16 depth image (0 = no return) to an RVL stream.

    Bit-for-bit what ``RvlCodec::CompressRVL`` produces for the same pixels, including the
    zero-padded trailing word.
    """
    if pixels.dtype != np.uint16:
        raise TypeError(f"RVL codes 16-bit depth; got dtype {pixels.dtype}")
    return _pack_words(_symbols(np.ascontiguousarray(pixels).reshape(-1)))


def decompress(data: bytes, num_pixels: int) -> np.ndarray:
    """Decode an RVL stream back into ``num_pixels`` uint16 values.

    The inverse of :func:`compress` -- RVL is lossless, so this returns the original pixels exactly.
    Used by the tests, and to read a depth stream recorded from hardware.
    """
    values = _unpack_symbols(data)
    out = np.zeros(num_pixels, dtype=np.uint16)
    cursor = pos = previous = 0
    while pos < num_pixels:
        zeros, nonzeros = int(values[cursor]), int(values[cursor + 1])
        cursor += 2
        pos += zeros  # `out` is zero there already
        if nonzeros:
            zigzag = values[cursor : cursor + nonzeros].astype(np.int64)
            cursor += nonzeros
            deltas = (zigzag >> 1) ^ -(zigzag & 1)
            run = previous + np.cumsum(deltas)
            out[pos : pos + nonzeros] = run
            previous, pos = int(run[-1]), pos + nonzeros
    return out


def _symbols(flat: np.ndarray) -> np.ndarray:
    """The symbol stream for one image: [zeros, nonzeros, zigzag deltas...] per run pair."""
    seen = flat != 0
    lengths = np.diff(np.concatenate(([0], np.flatnonzero(np.diff(seen)) + 1, [flat.size])))
    # The walk always reads a zero run before a nonzero one, so an image opening with a return opens
    # with an empty zero run, and one ending in no-returns closes with an empty nonzero run.
    if flat.size and seen[0]:
        lengths = np.concatenate(([0], lengths))
    if lengths.size % 2:
        lengths = np.concatenate((lengths, [0]))
    zeros, nonzeros = lengths[0::2], lengths[1::2]

    values = flat[seen].astype(np.int64)
    deltas = values - np.concatenate(([0], values[:-1]))
    zigzag = ((deltas << 1) ^ (deltas >> 63)).astype(np.uint64)

    pairs = zeros.size
    symbols = np.empty(2 * pairs + values.size, dtype=np.uint64)
    # A pair's header sits after every value the pairs before it emitted.
    header = 2 * np.arange(pairs) + np.concatenate(([0], np.cumsum(nonzeros)[:-1]))
    symbols[header] = zeros
    symbols[header + 1] = nonzeros
    is_header = np.zeros(symbols.size, dtype=bool)
    is_header[header] = True
    is_header[header + 1] = True
    symbols[~is_header] = zigzag  # in pixel order, which is the order the runs emit them in
    return symbols


def _pack_words(symbols: np.ndarray) -> bytes:
    """Symbols -> variable-length nibbles -> nibble-first 32-bit words -> native bytes.

    Flat passes over the *nibble* stream rather than a (symbols x max-nibbles) matrix: depth deltas
    are millimetres, so nearly every symbol is one or two nibbles and such a matrix is mostly
    padding -- it measured four times slower than this.
    """
    counts = (np.searchsorted(_NIBBLE_THRESHOLDS, symbols, side="right") + 1).astype(np.int32)
    total = int(counts.sum())
    if not total:
        return b""
    owner = np.repeat(np.arange(counts.size, dtype=np.int32), counts)  # the symbol a nibble is of
    starts = np.cumsum(counts, dtype=np.int64) - counts
    place = (np.arange(total, dtype=np.int64) - starts[owner]).astype(np.uint32)
    nibbles = (symbols[owner] >> (3 * place)) & np.uint32(7)
    nibbles += np.uint32(8) * (place != counts[owner].astype(np.uint32) - np.uint32(1))

    # Zero-padding to a whole word IS the reference's `word << 4 * (8 - nibblesWritten)`. Two
    # nibbles make a byte, and a word's four bytes go out reversed: on a little-endian machine the
    # first nibble of a word belongs in the high nibble of that word's last byte.
    padded = np.zeros(total + (-total % 8), dtype=np.uint8)
    padded[:total] = nibbles
    return ((padded[0::2] << 4) | padded[1::2]).reshape(-1, 4)[:, ::-1].tobytes()


def _unpack_symbols(data: bytes) -> np.ndarray:
    """Native words -> nibbles -> symbol values (the inverse of :func:`_pack_words`)."""
    ordered = np.frombuffer(data, dtype=np.uint8).reshape(-1, 4)[:, ::-1].reshape(-1)
    nibbles = np.empty(ordered.size * 2, dtype=np.uint8)
    nibbles[0::2] = ordered >> 4
    nibbles[1::2] = ordered & 0x0F
    ends = (nibbles & 0x8) == 0
    # A symbol runs up to and including the first nibble without the continuation bit, and a
    # symbol's nibbles hold disjoint bit ranges -- so summing a group is the same as or-ing it.
    # Trailing padding decodes as extra zero symbols, which the caller never reads.
    boundaries = np.flatnonzero(ends)
    if not boundaries.size:
        return np.zeros(0, dtype=np.uint32)
    starts = np.concatenate(([0], boundaries[:-1] + 1))
    lengths = np.diff(np.concatenate((starts, [nibbles.size])))
    place = (np.arange(nibbles.size, dtype=np.int64) - np.repeat(starts, lengths)).astype(np.uint32)
    return np.add.reduceat((nibbles & 0x7).astype(np.uint32) << (3 * place), starts)
