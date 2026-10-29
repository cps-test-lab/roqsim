"""A decoder of the ``EXT_meshopt_compression`` bitstreams roqsim writes, for the tests.

A plain port of the extension's reference decoder (``meshopt_decoder_reference.js`` in the
meshoptimizer distribution), written to be read rather than to be fast: the tests decode what
:mod:`roqsim.meshopt` encodes and compare it with what went in, so a loader of the file gets back
exactly the data the exporter meant.
"""

from __future__ import annotations

import numpy as np


def _dezig(v: int) -> int:
    return ~(v >> 1) if v & 1 else v >> 1


def decode_vertex_buffer(source: bytes, count: int, stride: int) -> np.ndarray:
    assert source[0] == 0xA0, hex(source[0])
    block = min((0x2000 // stride) & ~0x0F, 0x100)
    tail = max(stride, 32)
    temp = bytearray(source[len(source) - stride :])
    target = np.zeros((count, stride), np.uint8)
    off = 1
    for base in range(0, count, block):
        n = min(count - base, block)
        ngroups = (n + 15) // 16
        nheader = (ngroups + 3) // 4
        deltas = bytearray(block * stride + 16)
        for byte in range(stride):
            dbase = byte * n
            hdr = off
            off += nheader
            for g in range(ngroups):
                mode = (source[hdr + (g >> 2)] >> ((g & 3) << 1)) & 3
                bits = (0, 2, 4, 8)[mode]
                at = dbase + (g << 4)
                if bits == 0:
                    deltas[at : at + 16] = bytes(16)
                elif bits == 2:
                    sb = off
                    off += 4
                    for m in range(16):
                        d = (source[sb + (m >> 2)] >> (6 - ((m & 3) << 1))) & 3
                        if d == 3:
                            d = source[off]
                            off += 1
                        deltas[at + m] = d
                elif bits == 4:
                    sb = off
                    off += 8
                    for m in range(16):
                        d = (source[sb + (m >> 1)] >> (4 - ((m & 1) << 2))) & 0x0F
                        if d == 0x0F:
                            d = source[off]
                            off += 1
                        deltas[at + m] = d
                else:
                    deltas[at : at + 16] = source[off : off + 16]
                    off += 16
        for e in range(n):
            for byte in range(stride):
                temp[byte] = (temp[byte] + _dezig(deltas[byte * n + e])) & 0xFF
            target[base + e] = np.frombuffer(bytes(temp), np.uint8)
    assert off == len(source) - tail, (off, len(source), tail)
    return target


def decode_index_buffer(source: bytes, count: int) -> np.ndarray:
    assert source[0] == 0xE1 and count % 3 == 0
    ntri = count // 3
    code = 1
    data = 1 + ntri
    aux = len(source) - 16
    out = []
    nxt = last = 0
    edges = [0] * 32
    verts = [0] * 16

    def push(fifo, v):
        fifo.insert(0, v)
        del fifo[len(fifo) - 1]

    def leb():
        nonlocal data
        n = shift = 0
        while True:
            b = source[data]
            data += 1
            n |= (b & 0x7F) << shift
            shift += 7
            if b < 0x80:
                return n

    def index(v):
        nonlocal last
        last += _dezig(v)
        return last

    for _ in range(ntri):
        c0 = source[code]
        code += 1
        b0, b1 = c0 >> 4, c0 & 0x0F
        if b0 < 0x0F:
            a, b = edges[b0 * 2], edges[b0 * 2 + 1]
            if b1 == 0:
                c = nxt
                nxt += 1
                push(verts, c)
            elif b1 < 0x0D:
                c = verts[b1]
            elif b1 == 0x0D:
                last -= 1
                c = last
                push(verts, c)
            elif b1 == 0x0E:
                last += 1
                c = last
                push(verts, c)
            else:
                c = index(leb())
                push(verts, c)
            for v in (b, c, c, a):
                push(edges, v)
        else:
            if b1 < 0x0E:
                e = source[aux + b1]
                z, w = e >> 4, e & 0x0F
                a = nxt
                nxt += 1
                if z == 0:
                    b = nxt
                    nxt += 1
                else:
                    b = verts[z - 1]
                if w == 0:
                    c = nxt
                    nxt += 1
                else:
                    c = verts[w - 1]
                push(verts, a)
                if z == 0:
                    push(verts, b)
                if w == 0:
                    push(verts, c)
            else:
                e = source[data]
                data += 1
                if e == 0:
                    nxt = 0
                z, w = e >> 4, e & 0x0F
                if b1 == 0x0E:
                    a = nxt
                    nxt += 1
                else:
                    a = index(leb())
                if z == 0:
                    b = nxt
                    nxt += 1
                elif z == 0x0F:
                    b = index(leb())
                else:
                    b = verts[z - 1]
                if w == 0:
                    c = nxt
                    nxt += 1
                elif w == 0x0F:
                    c = index(leb())
                else:
                    c = verts[w - 1]
                push(verts, a)
                if z in (0, 0x0F):
                    push(verts, b)
                if w in (0, 0x0F):
                    push(verts, c)
            for v in (a, b, b, c, c, a):
                push(edges, v)
        out.append((a, b, c))
    assert data <= aux, (data, aux)
    return np.asarray(out, np.int64).reshape(-1, 3)


def unfilter_exponential(words: np.ndarray) -> np.ndarray:
    w = np.asarray(words).view(np.int32).astype(np.int64)
    exp = w >> 24
    mantissa = ((w & 0xFFFFFF) ^ 0x800000) - 0x800000
    return (mantissa * np.exp2(exp)).astype(np.float32)
