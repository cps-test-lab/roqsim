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


def _body(skeleton, rng, per_joint=60):
    """Points round every bone of ``skeleton``'s T-pose bind, each skinned to the bone's joint, with
    the lowest on z = 0: a stand-in for a mesh, in the frame a blueprint's mesh is in."""
    from roqsim_walker import skin
    from roqsim_walker.humanoid import CHILDREN, JOINT_NAMES, to_skeleton

    bind = skin._bind_bones(tpose=True, skeleton=to_skeleton(skeleton))
    pts, w = [], []
    for j, name in enumerate(JOINT_NAMES):
        a = bind[name][0]
        b = bind[CHILDREN[name][0]][0] if CHILDREN[name] else a + [0.05, 0.0, -0.02]
        t = rng.random((per_joint, 1))
        p = a + t * (b - a) + rng.normal(scale=0.03, size=(per_joint, 3))
        wj = np.zeros((per_joint, len(JOINT_NAMES)))
        wj[:, j] = 1.0
        pts.append(p)
        w.append(wj)
    P, W = np.vstack(pts), np.vstack(w)
    return P - [0.0, 0.0, P[:, 2].min()], W


def test_measure_takes_every_shape_value_from_the_derived_mesh():
    """Each value the blueprint states of its mesh's shape is the derived mesh's, against the
    bones the skin is bound to at runtime; the bones are the source's, lowered with the mesh."""
    from roqsim_walker import skin
    from roqsim_walker.cli import import_actor
    from roqsim_walker.humanoid import to_skeleton

    skeleton = {
        "root_height": 0.95,
        "height": 9.9,
        "foot_tip": [9.9, 9.9, 9.9],
        "offsets": {"toe_l": [0.11, 0.0, -0.1], "toe_r": [0.11, 0.0, -0.1]},
    }
    lowered = 0.0021
    P, W = _body(dict(skeleton, root_height=0.95 - lowered), np.random.default_rng(3))
    got = derive.measure(P, W, skeleton, tpose=True, lowered=lowered)
    sk = got["skeleton"]
    assert sk["root_height"] == pytest.approx(0.95 - lowered, abs=1e-4)
    assert sk["height"] == pytest.approx(P[:, 2].max(), abs=1e-4)
    assert sk["offsets"] == skeleton["offsets"]  # the source rig's bones, kept
    bind = {n: pq[0] for n, pq in skin._bind_bones(tpose=True, skeleton=to_skeleton(sk)).items()}
    sole, tip = import_actor.measure_sole(P, W, bind)
    assert sk["foot_tip"] == tip
    assert got["collision"] == import_actor.measure_collision(P, W, bind)
    for s in ("l", "r"):
        assert got["sole"][s]["heel"] == sole[s]["heel"]
        assert got["sole"][s]["toe"] == sole[s]["toe"]
        assert got["sole"][s]["hull"] == import_actor.measure_hull(P, W)[s]
    # the heel point is where the ankle's part of the mesh is lowest: measured from the lowered bones
    from roqsim_walker.humanoid import JOINT_NAMES

    heel_z = bind["ankle_l"][2] + got["sole"]["l"]["heel"][2]
    ankle_part = P[W.argmax(axis=1) == JOINT_NAMES.index("ankle_l")]
    assert heel_z == pytest.approx(ankle_part[:, 2].min(), abs=1e-4)
