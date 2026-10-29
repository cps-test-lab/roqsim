"""The ``EXT_meshopt_compression`` bitstreams, written with numpy.

``roqsim export gltf --meshopt`` compresses every vertex attribute and index buffer of the file with
the codecs of the glTF extension ``EXT_meshopt_compression``, which three.js, Babylon.js and
gltfpack decode. This module writes the three things that takes, and nothing that reads them back:

* :func:`encode_vertex_buffer` -- the ``ATTRIBUTES`` mode (vertex codec, version 0): each byte of a
  vertex is stored as the zigzag delta from the same byte of the vertex before, sixteen vertices to
  a group, each group at the smallest of 0, 2, 4 or 8 bits per delta;
* :func:`encode_index_buffer` -- the ``TRIANGLES`` mode (index codec, version 1): a triangle that
  shares an edge with one of the last sixteen is one code byte naming the edge and its third vertex;
* :func:`exponential_filter` -- the ``EXPONENTIAL`` filter, which stores each float as a 24-bit
  mantissa and an exponent shared by a whole attribute component, so the low bytes that carry
  nothing but float noise are gone before the vertex codec sees them.

:func:`fetch_order` puts a mesh's vertices in the order its triangles first use them, which is what
both codecs are built for: a new vertex is then always the next one, and neighbouring vertices sit
next to each other.

The bitstreams are specified in the extension's README (``KhronosGroup/glTF``,
``extensions/2.0/Vendor/EXT_meshopt_compression``); the tests decode them with a port of the
reference decoder.
"""

from __future__ import annotations

import numpy as np

#: The first byte of a version-0 vertex stream and of a version-1 index stream.
VERTEX_HEADER = 0xA0
INDEX_HEADER = 0xE1

#: The aux table of the index codec. Only entry 0 is used: three vertices that are all new.
_CODEAUX = bytes(16)


def _zigzag8(delta: np.ndarray) -> np.ndarray:
    d = delta.astype(np.int8).astype(np.int16)
    return (((d << 1) ^ (d >> 7)) & 0xFF).astype(np.uint8)


def _encode_groups(groups: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Encode rows of 16 zigzagged byte deltas: ``(mode, bytes, length)`` per row.

    ``mode`` is the 2-bit header value (0, 2, 4 or 8 bits per delta); ``bytes`` holds each row's
    encoding left-aligned in 32 columns, ``length`` how many of them count.
    """
    rows = groups.shape[0]
    zero = ~groups.any(axis=1)
    esc2 = groups >= 3
    esc4 = groups >= 15
    cost = np.stack([4 + esc2.sum(1), 8 + esc4.sum(1), np.full(rows, 16)], axis=1)
    mode = np.where(zero, 0, 1 + np.argmin(cost, axis=1))
    out = np.zeros((rows, 32), np.uint8)
    length = np.zeros(rows, np.int64)

    def place(sel, packed, escape):
        width = packed.shape[1]
        out[sel, :width] = packed
        rank = np.cumsum(escape, axis=1) - 1
        r, c = np.nonzero(escape)
        out[np.flatnonzero(sel)[r], width + rank[r, c]] = groups[sel][r, c]
        length[sel] = width + escape.sum(1)

    for m, bits, sentinel in ((1, 2, 3), (2, 4, 15)):
        sel = mode == m
        if not sel.any():
            continue
        g = np.minimum(groups[sel], sentinel).astype(np.uint16)
        per = 8 // bits
        shifts = (8 - bits) - bits * np.arange(per)  # the first value in the high bits
        packed = (g.reshape(-1, 16 // per, per) << shifts).sum(axis=2).astype(np.uint8)
        place(sel, packed, groups[sel] >= sentinel)
    sel = mode == 3
    out[sel, :16] = groups[sel]
    length[sel] = 16
    return mode, out, length


def encode_vertex_buffer(data: np.ndarray) -> bytes:
    """``data`` (``(count, stride)`` uint8, ``stride`` a multiple of 4 up to 256) as an
    ``ATTRIBUTES`` stream."""
    data = np.ascontiguousarray(data, np.uint8)
    count, stride = data.shape
    if stride % 4 or not 0 < stride <= 256:
        raise ValueError(f"a vertex stride of {stride} bytes is not a multiple of 4 up to 256")
    block = min((8192 // stride) & ~15, 256)
    baseline = data[0].copy() if count else np.zeros(stride, np.uint8)
    previous = np.vstack([baseline[None, :], data[:-1]]) if count else data
    deltas = _zigzag8(data.astype(np.int16) - previous.astype(np.int16))

    out = [bytes([VERTEX_HEADER])]
    for base in range(0, count, block):
        chunk = deltas[base : base + block]
        n = len(chunk)
        ngroups = (n + 15) // 16
        padded = np.zeros((ngroups * 16, stride), np.uint8)
        padded[:n] = chunk
        # (stride, groups, 16): one row of 16 deltas per byte of the vertex and group of vertices.
        groups = padded.T.reshape(stride * ngroups, 16)
        mode, enc, length = _encode_groups(groups)
        keep = np.arange(32)[None, :] < length[:, None]
        mode = mode.reshape(stride, ngroups)
        flat = enc[keep]
        ends = np.cumsum(length.reshape(stride, ngroups).sum(axis=1))
        start = 0
        header = np.zeros((stride, (ngroups + 3) // 4), np.uint8)
        for g in range(ngroups):
            header[:, g >> 2] |= (mode[:, g] << ((g & 3) * 2)).astype(np.uint8)
        for k in range(stride):
            out.append(header[k].tobytes())
            out.append(flat[start : ends[k]].tobytes())
            start = ends[k]
    # The tail: the baseline the first deltas are taken from, padded to at least 32 bytes.
    out.append(bytes(max(0, 32 - stride)))
    out.append(baseline.tobytes())
    return b"".join(out)


def _leb128(value: int, out: bytearray) -> None:
    while value >= 0x80:
        out.append((value & 0x7F) | 0x80)
        value >>= 7
    out.append(value)


def _zigzag(delta: int) -> int:
    return delta << 1 if delta >= 0 else ((-delta) << 1) - 1


class _IndexWriter:
    """The index codec's state, as its decoder keeps it, and the stream written so far."""

    def __init__(self) -> None:
        self.codes = bytearray()
        self.data = bytearray()
        self.edges: list[tuple[int, int]] = [(0, 0)] * 16  # newest first
        self.vertices: list[int] = [0] * 16  # newest first
        self.next = 0  # the next new vertex
        self.last = 0  # the last vertex given explicitly

    def push_vertex(self, v: int) -> None:
        self.vertices.insert(0, v)
        del self.vertices[16:]

    def push_edge(self, a: int, b: int) -> None:
        self.edges.insert(0, (a, b))
        del self.edges[16:]

    def explicit(self, v: int, out: bytearray) -> None:
        _leb128(_zigzag(v - self.last), out)
        self.last = v

    def third(self, c: int) -> int:
        """The code of a triangle's third vertex after a shared edge; pushes it when new."""
        if c == self.next:
            self.next += 1
            self.push_vertex(c)
            return 0
        if c in self.vertices[1:13]:
            return self.vertices.index(c, 1, 13)
        if c in (self.last - 1, self.last + 1):
            code = 0x0D if c == self.last - 1 else 0x0E
            self.last = c
        else:
            code = 0x0F
            self.explicit(c, self.data)
        self.push_vertex(c)
        return code

    def other(self, v: int, fifo: list[int], out: bytearray, *, allow_next: bool = True) -> int:
        """The 4-bit code of a second or third vertex of a triangle with no shared edge."""
        if allow_next and v == self.next:
            self.next += 1
            return 0
        if v in fifo:
            return 1 + fifo.index(v)
        self.explicit(v, out)
        return 0x0F


def encode_index_buffer(faces: np.ndarray) -> bytes:
    """Triangles ``faces`` (``(m, 3)``) as a ``TRIANGLES`` stream (index codec version 1).

    Each triangle is written in one of its three rotations, so its winding is kept. Vertices are
    best numbered in first-use order (:func:`fetch_order`), which makes most new vertices the
    ``next`` one.
    """
    w = _IndexWriter()
    for a, b, c in np.asarray(faces, np.int64).reshape(-1, 3).tolist():
        found = -1
        for x, y, z in ((a, b, c), (b, c, a), (c, a, b)):
            try:
                found = w.edges.index((x, y), 0, 15)
            except ValueError:
                continue
            a, b, c = x, y, z
            break
        if found >= 0:
            w.codes.append((found << 4) | w.third(c))
            w.push_edge(c, b)
            w.push_edge(a, c)
            continue

        # No shared edge: three vertices of their own. A first vertex that is the next new one
        # costs no data, so rotate one there when the triangle has it.
        for x, y, z in ((a, b, c), (b, c, a), (c, a, b)):
            if x == w.next:
                a, b, c = x, y, z
                break
        if (a, b, c) == (w.next, w.next + 1, w.next + 2):
            w.codes.append(0xF0)  # aux entry 0: all three are new
            w.next += 3
            for v in (a, b, c):
                w.push_vertex(v)
        else:
            fifo = w.vertices[:14]  # as the decoder reads it, before this triangle pushes
            head, tail = bytearray(), bytearray()
            if a == w.next:
                lead = 0x0E
                w.next += 1
            else:
                lead = 0x0F
                w.explicit(a, head)
            z = w.other(b, fifo, tail)
            # z = w = 0 would read as "restart numbering at 0"; spell c out instead.
            wc = w.other(c, fifo, tail, allow_next=not (z == 0 and lead == 0x0F))
            w.codes.append(0xF0 | lead)
            w.data.append((z << 4) | wc)
            w.data.extend(head)
            w.data.extend(tail)
            w.push_vertex(a)
            if z in (0, 0x0F):
                w.push_vertex(b)
            if wc in (0, 0x0F):
                w.push_vertex(c)
        w.push_edge(b, a)
        w.push_edge(c, b)
        w.push_edge(a, c)
    return bytes([INDEX_HEADER]) + bytes(w.codes) + bytes(w.data) + _CODEAUX


def fetch_order(faces: np.ndarray, nvert: int) -> tuple[np.ndarray, np.ndarray]:
    """Number vertices in the order the faces first use them: ``(order, faces)``.

    ``order[k]`` is the old index of new vertex ``k``; a vertex no face uses is dropped.
    """
    flat = np.asarray(faces, np.int64).ravel()
    _, first = np.unique(flat, return_index=True)
    order = flat[np.sort(first)]
    remap = np.full(nvert, -1, np.int64)
    remap[order] = np.arange(len(order))
    return order, remap[flat].reshape(-1, 3)


def exponential_filter(values: np.ndarray, step_exponent) -> tuple[np.ndarray, np.ndarray]:
    """Floats as the ``EXPONENTIAL`` filter stores them: ``(int32 words, decoded floats)``.

    Each component ``k`` is a multiple of ``2 ** step_exponent[k]`` (or of the one exponent given
    for all), rounded to the nearest; the exponent is raised where a value would not fit the
    24-bit mantissa. The decoded floats are what a reader gets back, for the accessor's bounds.
    """
    v = np.asarray(values, np.float64)
    v = v.reshape(len(v), -1)
    exp = np.broadcast_to(np.asarray(step_exponent, np.int64), (v.shape[1],)).copy()
    peak = np.abs(v).max(axis=0) if len(v) else np.zeros(v.shape[1])
    with np.errstate(divide="ignore"):
        need = np.where(peak > 0, np.ceil(np.log2(np.maximum(peak, 1e-300) / (2**23 - 1))), -127)
    exp = np.clip(np.maximum(exp, need.astype(np.int64)), -100, 100)
    mantissa = np.rint(v / np.exp2(exp)).astype(np.int64)
    words = ((exp[None, :] & 0xFF) << 24) | (mantissa & 0xFFFFFF)
    decoded = (mantissa * np.exp2(exp)).astype(np.float32)
    return words.astype(np.uint32).view(np.int32), decoded
