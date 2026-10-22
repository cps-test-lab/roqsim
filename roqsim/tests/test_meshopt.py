"""roqsim.meshopt: the EXT_meshopt_compression bitstreams decode to what went in.

Decoded with a port of the extension's reference decoder (``meshopt_decode``).
"""

from __future__ import annotations

import meshopt_decode
import numpy as np
import pytest

from roqsim import meshopt, tessellate


@pytest.mark.parametrize(
    "count, stride", [(1, 4), (5, 12), (16, 8), (17, 16), (300, 12), (4000, 4)]
)
def test_a_vertex_buffer_decodes_to_its_bytes(count, stride):
    rng = np.random.default_rng(count * stride)
    noise = rng.integers(0, 256, (count, stride)).astype(np.uint8)
    noise[::3] = 0  # groups of zero deltas
    smooth = np.cumsum(rng.integers(0, 3, (count, stride)), axis=0).astype(np.uint8)
    for data in (noise, smooth):
        encoded = meshopt.encode_vertex_buffer(data)
        np.testing.assert_array_equal(
            meshopt_decode.decode_vertex_buffer(encoded, count, stride), data
        )


def test_a_vertex_stride_the_codec_cannot_take_is_refused():
    with pytest.raises(ValueError, match="multiple of 4"):
        meshopt.encode_vertex_buffer(np.zeros((3, 6), np.uint8))


def _same_triangles(decoded, faces):
    """Equal triangle by triangle, each in one of its rotations, so its winding is kept."""
    assert len(decoded) == len(faces)
    for got, (a, b, c) in zip(decoded.tolist(), faces.tolist(), strict=True):
        assert tuple(got) in {(a, b, c), (b, c, a), (c, a, b)}


@pytest.mark.parametrize("seed", range(12))
def test_an_index_buffer_decodes_to_its_triangles(seed):
    rng = np.random.default_rng(seed)
    nvert = int(rng.integers(3, 400))
    faces = rng.integers(0, nvert, (int(rng.integers(1, 600)), 3))
    faces = faces[
        (faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 0] != faces[:, 2])
    ]
    if seed % 2:
        _order, faces = meshopt.fetch_order(faces, nvert)
    _same_triangles(
        meshopt_decode.decode_index_buffer(meshopt.encode_index_buffer(faces), faces.size), faces
    )


def test_a_connected_mesh_costs_about_a_byte_and_a_half_a_triangle():
    verts, faces = tessellate.rings([(1.0, z) for z in np.linspace(0, 1, 9)], 32)
    _order, faces = meshopt.fetch_order(faces, len(verts))
    encoded = meshopt.encode_index_buffer(faces)
    _same_triangles(meshopt_decode.decode_index_buffer(encoded, faces.size), faces)
    assert len(encoded) < 2 * len(faces)


def test_fetch_order_numbers_vertices_by_first_use_and_drops_unused_ones():
    faces = np.array([[5, 2, 7], [7, 2, 9]])
    order, renumbered = meshopt.fetch_order(faces, 12)
    np.testing.assert_array_equal(order, [5, 2, 7, 9])
    np.testing.assert_array_equal(renumbered, [[0, 1, 2], [2, 1, 3]])


def test_the_exponential_filter_rounds_to_its_step_and_widens_it_for_large_values():
    values = np.array([[0.1, -3.0], [1e7, 2.5]])
    words, decoded = meshopt.exponential_filter(values, -12)
    np.testing.assert_array_equal(meshopt_decode.unfilter_exponential(words), decoded)
    assert abs(decoded[0, 1] - values[0, 1]) <= 2.0**-13
    assert abs(decoded[0, 0] - values[0, 0]) > 0  # column 0 holds 1e7: its step had to grow
    assert abs(decoded[1, 0] - 1e7) / 1e7 < 2.0**-22
