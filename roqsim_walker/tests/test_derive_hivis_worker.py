"""`roqsim walker derive-hivis-worker`'s parts that run without its source: the subdivision, the
credits, and its refusal of a mesh that is not the source it was written for."""

from __future__ import annotations

import numpy as np
import pytest

from roqsim_walker.cli import derive_hivis_worker as derive


def _octahedron():
    P = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], float)
    F = np.array(
        [[0, 2, 4], [2, 1, 4], [1, 3, 4], [3, 0, 4], [2, 0, 5], [1, 2, 5], [3, 1, 5], [0, 3, 5]]
    )
    return P, F


def test_loop_subdivision_rounds_a_closed_mesh_and_carries_uv_and_weights():
    P, F = _octahedron()
    Fuv = np.random.default_rng(0).random((len(F), 3, 2))
    W = np.eye(6)
    region = np.array(["a"] * 4 + ["b"] * 4, dtype=object)
    P2, F2, Fuv2, region2, W2 = derive.loop_subdivide(P, F, Fuv, region, W)
    assert len(F2) == 4 * len(F) and len(P2) == len(P) + 12  # one vertex per edge
    r = np.linalg.norm(P2, axis=1)
    assert r.max() < 1.0 and r.min() > 0.5  # pulled in towards a sphere, not collapsed
    np.testing.assert_allclose(W2.sum(axis=1), 1.0)
    # each child's corners lie in its parent's UV triangle: the corners, or their midpoints
    child = Fuv2[3 * len(F) :]  # the middle triangles
    parent = Fuv
    np.testing.assert_allclose(child[:, 0], 0.5 * (parent[:, 0] + parent[:, 1]))
    assert list(region2[: len(F)]) == list(region)


def test_a_boundary_stays_on_its_boundary_curve():
    """An open sheet's border is subdivided along itself, not pulled into the sheet."""
    P = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], float)
    F = np.array([[0, 1, 2], [0, 2, 3]])
    P2, *_ = derive.loop_subdivide(
        P, F, np.zeros((2, 3, 2)), np.array(["a", "a"], object), np.eye(4)
    )
    np.testing.assert_allclose(P2[:, 2], 0.0)
    assert P2[:, :2].min() >= 0.0 and P2[:, :2].max() <= 1.0


def test_the_credits_name_the_source_and_the_changes():
    source = (
        "OpScrubsWalk\n\nSource:  https://example.org/a\nLicence: CC-BY 4.0\nAuthor:  Someone\n"
    )
    text = derive.credits_text(source)
    assert "Source:  https://example.org/a" in text and "Licence: CC-BY 4.0" in text
    assert "derive-hivis-worker" in text and "CARLA" in text
    with pytest.raises(SystemExit):
        derive.credits_text("no attribution here\n")


def test_a_mesh_that_is_not_the_source_is_refused():
    P, F = _octahedron()
    uv = np.tile([[0.5, 0.5]], (len(P), 1))
    with pytest.raises(SystemExit, match="not the source"):
        derive.islands(P, F, uv)
