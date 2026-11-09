"""``import_actor.measure``: what a blueprint's ``walker.json`` says of its mesh, measured against
the bones MuJoCo poses the skin with (``skin._bind_bones``), not the source rig's."""

from __future__ import annotations

import numpy as np
import pytest

from roqsim_walker import skin
from roqsim_walker.cli import import_actor
from roqsim_walker.humanoid import CHILDREN, JOINT_NAMES, to_skeleton

SKELETON = {
    "root_height": 0.93,
    "offsets": {"toe_l": [0.11, 0.0, -0.1], "toe_r": [0.11, 0.0, -0.1], "knee_l": [0, 0, -0.42]},
}
RADIUS = 0.05


def _tube_body(skeleton, n=48):
    """A tube of radius ``RADIUS`` round every bone of ``skeleton``'s T-pose bind -- the bones the
    skin is bound to at runtime -- each ring skinned to the bone's joint, the lowest point on z=0."""
    bind = skin._bind_bones(tpose=True, skeleton=to_skeleton(skeleton))
    pts, w = [], []
    for j, name in enumerate(JOINT_NAMES):
        if not CHILDREN[name]:
            continue
        a, b = bind[name][0], bind[CHILDREN[name][0]][0]
        axis = (b - a) / np.linalg.norm(b - a)
        u = np.cross(axis, [0.3, 0.5, 0.8])
        u /= np.linalg.norm(u)
        v = np.cross(axis, u)
        for t in np.linspace(0.1, 0.9, 9):
            ang = np.linspace(0, 2 * np.pi, n, endpoint=False)[:, None]
            pts.append(a + t * (b - a) + RADIUS * (np.cos(ang) * u + np.sin(ang) * v))
            wj = np.zeros((n, len(JOINT_NAMES)))
            wj[:, j] = 1.0
            w.append(wj)
    for s in ("l", "r"):  # a box of a toe under each toe joint
        toe = bind[f"toe_{s}"][0]
        box = toe + np.random.default_rng(0).uniform(
            [-0.02, -0.04, -0.03], [0.06, 0.04, 0.0], (60, 3)
        )
        wj = np.zeros((60, len(JOINT_NAMES)))
        wj[:, JOINT_NAMES.index(f"toe_{s}")] = 1.0
        pts.append(box)
        w.append(wj)
    P, W = np.vstack(pts), np.vstack(w)
    shift = P[:, 2].min()
    sk = dict(skeleton, root_height=skeleton["root_height"] - shift)
    return P - [0.0, 0.0, shift], W, sk


def test_capsules_and_soles_are_measured_against_the_runtime_bones():
    P, W, sk = _tube_body(SKELETON)
    got = import_actor.measure(P, W, sk, tpose=True, flip=False)
    for name in ("pelvis", "spine", "hip_l", "knee_l", "ankle_l", "shoulder_r", "elbow_r"):
        assert got["collision"][name] == pytest.approx(RADIUS, abs=2e-4), name
    bind = {n: pq[0] for n, pq in skin._bind_bones(True, to_skeleton(sk)).items()}
    dom = W.argmax(axis=1)
    for s in ("l", "r"):
        heel = bind[f"ankle_{s}"] + got["sole"][s]["heel"]
        toe = bind[f"toe_{s}"] + got["sole"][s]["toe"]
        assert heel[2] == pytest.approx(
            P[dom == JOINT_NAMES.index(f"ankle_{s}"), 2].min(), abs=1e-4
        )
        assert toe[2] == pytest.approx(0.0, abs=1e-4)
        assert set(got["sole"][s]["hull"]) == {"joints", "at", "w"}
    assert got["height"] == pytest.approx(P[:, 2].max(), abs=1e-4)


def test_a_mesh_facing_minus_x_is_measured_as_the_skin_turns_it():
    P, W, sk = _tube_body(SKELETON)
    turned = P * [-1.0, -1.0, 1.0]
    assert import_actor.measure(turned, W, sk, True, flip=True) == import_actor.measure(
        P, W, sk, True, flip=False
    )
