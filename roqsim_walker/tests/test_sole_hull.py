"""A shoe's hull: the points the foot grounding is taken on, measured by the blueprint tools
(``import_actor.measure_hull``) and skinned at runtime (``animation._lowest_sole``)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from roqsim_walker import animation
from roqsim_walker.cli import import_actor
from roqsim_walker.humanoid import DEFAULT_SKELETON, JOINT_NAMES, forward_kinematics

IDENT = {n: np.array([1.0, 0.0, 0.0, 0.0]) for n in JOINT_NAMES}
COL = {n: i for i, n in enumerate(JOINT_NAMES)}


def _rest():
    return forward_kinematics([0.0, 0.0, DEFAULT_SKELETON.root_height], 0.0, IDENT)


def _shoes(rng, n=400):
    """Two shoes of points round the rest pose's ankles and toes, each skinned to its nearer
    joint (a few to the knee besides), with soles on z = 0."""
    rest = _rest()
    pts, w = [], []
    for s in ("l", "r"):
        ankle, toe = rest[f"ankle_{s}"][0], rest[f"toe_{s}"][0]
        lo = np.array([ankle[0] - 0.06, ankle[1] - 0.05, 0.0])
        hi = np.array([toe[0] + 0.06, ankle[1] + 0.05, ankle[2] + 0.02])
        p = lo + rng.random((n, 3)) * (hi - lo)
        wi = np.zeros((n, len(JOINT_NAMES)))
        near_toe = np.linalg.norm(p - toe, axis=1) < np.linalg.norm(p - ankle, axis=1)
        wi[near_toe, COL[f"toe_{s}"]] = 1.0
        wi[~near_toe, COL[f"ankle_{s}"]] = 1.0
        wi[:10, COL[f"ankle_{s}"]] = 0.7  # a band of the ankle followed partly by the knee
        wi[:10, COL[f"knee_{s}"]] = 0.3
        wi[:10, COL[f"toe_{s}"]] = 0.0
        pts.append(p)
        w.append(wi)
    return np.vstack(pts), np.vstack(w)


def _skinned(poses, rest, verts, w):
    """Every point skinned as MuJoCo skins a mesh: its joints' moves since the bind, blended."""
    out = np.zeros_like(verts)
    for j, name in enumerate(JOINT_NAMES):
        if not w[:, j].any():
            continue
        p0, q0 = rest[name]
        p, q = poses[name]
        local = animation._rotate_rows(animation._conj(q0), verts - p0)
        out += w[:, j : j + 1] * (p + animation._rotate_rows(q, local))
    return out


def _foot_pose(rng):
    """A rest pose with the ankles and toes turned a random way, up to 70 degrees."""
    rot = dict(IDENT)
    for name in ("knee_l", "ankle_l", "toe_l", "knee_r", "ankle_r", "toe_r"):
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        half = 0.5 * rng.uniform(-1.2, 1.2)
        rot[name] = np.array([np.cos(half), *(np.sin(half) * axis)])
    return forward_kinematics([0.0, 0.0, DEFAULT_SKELETON.root_height], 0.3, rot)


def test_the_hull_grounds_on_the_lowest_point_of_the_skinned_shoe():
    rng = np.random.default_rng(1)
    verts, w = _shoes(rng)
    hull = import_actor.measure_hull(verts, w)
    assert hull["l"]["joints"] == ["knee_l", "ankle_l", "toe_l"]
    assert len(hull["l"]["at"]) < len(verts) / 2  # the corners, not the whole shoe
    sole = {s: {"heel": [0, 0, 0], "toe": [0, 0, 0], "hull": hull[s]} for s in ("l", "r")}
    st = SimpleNamespace(
        sole=sole, sole_hull=animation._sole_hull(sole, DEFAULT_SKELETON), skeleton=None
    )
    rest = _rest()
    for _ in range(20):
        poses = _foot_pose(rng)
        want = _skinned(poses, rest, verts, w)[:, 2].min()
        assert animation._lowest_sole(poses, st) == pytest.approx(want, abs=2e-4)


def test_without_a_hull_the_heel_and_toe_ground_as_before():
    sole = {s: {"heel": [0.0, 0.0, -0.08], "toe": [0.05, 0.0, -0.02]} for s in ("l", "r")}
    assert animation._sole_hull(sole, DEFAULT_SKELETON) is None
    assert animation._sole_hull(None, DEFAULT_SKELETON) is None
    st = SimpleNamespace(sole=sole, sole_hull=None, skeleton=DEFAULT_SKELETON)
    rest = _rest()
    want = min(rest["ankle_l"][0][2] - 0.08, rest["toe_l"][0][2] - 0.02)
    assert animation._lowest_sole(rest, st) == pytest.approx(want)


def test_a_malformed_hull_is_refused():
    hull = {"joints": ["ankle_l"], "at": [[0.0, 0.0, 0.0]], "w": [[0.5, 0.5]]}
    with pytest.raises(ValueError, match="hull"):
        animation._sole_hull({"l": {"hull": hull}, "r": {"hull": hull}}, DEFAULT_SKELETON)


def test_a_shoe_skinned_off_its_leg_is_refused():
    verts, w = _shoes(np.random.default_rng(2))
    w[:, COL["wrist_l"]] += 0.5
    with pytest.raises(SystemExit, match="off its leg"):
        import_actor.measure_hull(verts, w / w.sum(axis=1, keepdims=True))


def _rocker_shoes(n=600):
    """Two shoes whose soles curve up to the heel and the toe (a rocker sole), so the lowest point
    of each joint's part lies inside the sole, at neither end: the ankle's 3 cm in front of the
    heel, the toe's at the toe part's rear edge."""
    rest = _rest()
    rng = np.random.default_rng(4)
    pts, w = [], []
    for s in ("l", "r"):
        ankle, toe = rest[f"ankle_{s}"][0], rest[f"toe_{s}"][0]
        heel_x, tip_x, split_x = ankle[0] - 0.06, toe[0] + 0.06, toe[0] - 0.02
        low_x = heel_x + 0.03
        x = rng.uniform(heel_x, tip_x, n)
        y = ankle[1] + rng.uniform(-0.04, 0.04, n)
        bottom = 4.0 * (x - low_x) ** 2  # 4 cm up at 10 cm from the lowest point
        z = bottom + np.where(rng.random(n) < 0.5, 0.0, rng.uniform(0.0, 0.08, n))
        wi = np.zeros((n, len(JOINT_NAMES)))
        wi[x < split_x, COL[f"ankle_{s}"]] = 1.0
        wi[x >= split_x, COL[f"toe_{s}"]] = 1.0
        pts.append(np.stack([x, y, z], axis=1))
        w.append(wi)
    verts = np.vstack(pts)
    return verts - [0.0, 0.0, verts[:, 2].min()], np.vstack(w), rest


def test_the_sole_points_are_the_lowest_of_each_joints_part_of_the_shoe():
    """Without a hull, a walker stands on its heel and toe points: they are the lowest points of
    the vertices the ankle and the toe carry, not the ends of the shoe's lowest centimetre."""
    verts, w, rest = _rocker_shoes()
    bind = {n: pq[0] for n, pq in rest.items()}
    sole, tip = import_actor.measure_sole(verts, w, bind)
    dom = w.argmax(axis=1)
    for s in ("l", "r"):
        ankle_part = verts[dom == COL[f"ankle_{s}"]]
        toe_part = verts[dom == COL[f"toe_{s}"]]
        heel = bind[f"ankle_{s}"] + sole[s]["heel"]
        toe = bind[f"toe_{s}"] + sole[s]["toe"]
        assert heel[2] == pytest.approx(ankle_part[:, 2].min(), abs=1e-4)
        assert toe[2] == pytest.approx(toe_part[:, 2].min(), abs=1e-4)
        assert heel[0] > ankle_part[:, 0].min() + 0.02  # inside the sole, not at the heel
    # standing, the grounding finds the shoes' lowest point, which is on the floor
    st = SimpleNamespace(sole=sole, sole_hull=None, skeleton=DEFAULT_SKELETON)
    assert animation._lowest_sole(rest, st) == pytest.approx(0.0, abs=1e-4)
