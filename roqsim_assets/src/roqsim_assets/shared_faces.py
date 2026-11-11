"""Two drawn parts showing a face in one place: the flicker a picture taken from one angle hides.

Where two geoms' outward faces lie in one plane, face the same way and overlap, or two round parts
share one curved surface, the renderer's depth test cannot tell which is in front: the overlap
flickers between the two as the camera moves (z-fighting). A still render often looks right, so this
is measured on the compiled model instead.

:func:`shared_faces` compares every pair of drawn geoms (:func:`roqsim.rendering.drawn_by_default`:
groups 0-2 and a window, not fully transparent) whose bounds touch:

* **flat faces**: a box's six sides, a cylinder's two ends, a plane, and a mesh's triangles, whose
  outward side is the one their winding gives (counter-clockwise seen from outside, as MuJoCo and
  every exporter here write them). Two faces conflict where they lie in one plane, face the same way
  and share an area. Faces pressed together back to back (a part resting on another) look away from
  each other and are hidden, and a face pointing down at ``z = 0`` is on the floor, out of sight;
* **round surfaces**: cylinders or capsules of one radius on one axis over an overlapping length,
  spheres of one radius on one centre, a sphere on the centre of a round capsule end of its radius,
  and two capsules of one radius on different axes whose round ends share one centre (an elbow):
  where those centres coincide the round ends are one surface, and where they lie apart two spheres
  of one radius meet in a circle, which shows no area.

A collision-only proxy (group 3) or a part kept out of every picture (group 4, alpha 0) may lie
where it likes. The remedy for a pair found is to inset the touching part by about 1 mm, so its face
sits just inside the other; a render cannot show the difference.

A pair whose bounds touch and that no rule above decides is not passed: an ellipsoid against a
sphere, capsule or ellipsoid, and any heightfield or SDF geom. :func:`shared_faces` raises
:class:`UncomparableGeoms` naming each such pair, after comparing everything else.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations

import mujoco
import numpy as np

from roqsim.rendering import drawn_by_default

#: m: the plane distance, in-plane overlap and radius difference below which two faces count as one.
#: A model's numbers are written to about six decimals, and a mesh's vertices are stored in single
#: precision (about 1e-7 of its extent): a rotated mesh's face then strays from the plane of a part
#: placed against it by up to a few 1e-6 m. Fixed rather than scaled to the model, since the larger
#: share of that error is the written decimals, which do not grow with the model; still a hundredth
#: of the 1 mm inset that remedies a pair.
TOL = 1e-5
#: The sides a cylinder's flat ends are drawn with here: enough that a cap inside another face is
#: seen as inside it.
CAP_SIDES = 24
#: m: the half extent an infinite plane (``size="0 0 ..."``) is compared over.
INFINITE_PLANE = 1e4

_G = mujoco.mjtGeom
#: Types whose surface is curved in every direction: no flat face, and no rule against each other
#: beyond the ones :func:`_round_hit` knows.
_CURVED = {_G.mjGEOM_SPHERE, _G.mjGEOM_CAPSULE, _G.mjGEOM_ELLIPSOID}
_FLAT = {_G.mjGEOM_PLANE, _G.mjGEOM_BOX, _G.mjGEOM_CYLINDER, _G.mjGEOM_MESH}


@dataclass(frozen=True)
class SharedFace:
    """Two drawn geoms showing a face in one place.

    ``kind`` is ``"plane"`` (flat faces in one plane), ``"cylinder"`` (one curved side) or
    ``"sphere"`` (one round surface: two spheres, or a round capsule end against a sphere or
    another's). ``at`` is the centre of the shared area in the world frame (a sphere's: its centre),
    ``direction`` the outward normal both faces show (plane), the common axis (cylinder) or ``None``
    (sphere), and ``area`` the shared area in m².
    """

    first: str
    second: str
    kind: str
    at: tuple[float, float, float]
    direction: tuple[float, float, float] | None
    area: float

    def describe(self) -> str:
        """One line naming the pair and where they touch."""
        at = ", ".join(f"{c:+.3f}" for c in self.at)
        where = {
            "plane": "faces in one plane",
            "cylinder": "one curved side",
            "sphere": "one sphere",
        }[self.kind]
        facing = ""
        if self.direction is not None:
            word = "facing" if self.kind == "plane" else "along"
            facing = f" {word} ({', '.join(f'{c:+.2f}' for c in self.direction)})"
        return (
            f"{self.first} / {self.second}: {where} at ({at}) m{facing}, {self.area * 1e4:.1f} cm²"
        )

    def as_dict(self) -> dict:
        return {
            "first": self.first,
            "second": self.second,
            "kind": self.kind,
            "at": [round(c, 6) for c in self.at],
            "direction": None if self.direction is None else [round(c, 6) for c in self.direction],
            "area": round(self.area, 9),
        }


class UncomparableGeoms(ValueError):
    """Pairs of drawn geoms whose bounds touch and that :func:`shared_faces` cannot decide.

    ``pairs`` holds ``(first, second, reason)`` for each; ``found`` the pairs that were decided and
    do show a face in one place, so a caller reporting both loses neither.
    """

    def __init__(self, pairs: list[tuple[str, str, str]], found: list[SharedFace]):
        self.pairs, self.found = pairs, found
        listed = "; ".join(f"{a} / {b}: {why}" for a, b, why in pairs)
        super().__init__(f"{len(pairs)} pair(s) of drawn geoms not compared: {listed}")


def _name(model: mujoco.MjModel, g: int) -> str:
    return model.geom(g).name or f"geom {g}"


def _faces(model: mujoco.MjModel, data: mujoco.MjData, g: int) -> tuple[np.ndarray, np.ndarray]:
    """A geom's flat faces in the world as triangles (n, 3, 3) with their outward normals (n, 3)."""
    R, c, h = data.geom_xmat[g].reshape(3, 3), data.geom_xpos[g], model.geom_size[g]
    t = _G(int(model.geom_type[g]))
    if t == _G.mjGEOM_MESH:
        k = model.geom_dataid[g]
        start, count = model.mesh_vertadr[k], model.mesh_vertnum[k]
        v = model.mesh_vert[start : start + count]
        f = model.mesh_face[model.mesh_faceadr[k] : model.mesh_faceadr[k] + model.mesh_facenum[k]]
        tri = (c + v.astype(float) @ R.T)[f]
        n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])  # a mesh's faces wind outward
    elif t in (_G.mjGEOM_BOX, _G.mjGEOM_PLANE):
        if t == _G.mjGEOM_PLANE:
            hx, hy = (s if s > 0 else INFINITE_PLANE for s in h[:2])
            v = np.array([[sx * hx, sy * hy, 0.0] for sx in (-1, 1) for sy in (-1, 1)])
            f = np.array([[0, 2, 3], [0, 3, 1]])  # counter-clockwise seen from +z, its drawn side
        else:
            v = np.array(
                [
                    [sx * h[0], sy * h[1], sz * h[2]]
                    for sx in (-1, 1)
                    for sy in (-1, 1)
                    for sz in (-1, 1)
                ]
            )
            f = np.array(
                [[0, 2, 3], [0, 3, 1], [4, 5, 7], [4, 7, 6], [0, 1, 5], [0, 5, 4],
                 [2, 6, 7], [2, 7, 3], [0, 4, 6], [0, 6, 2], [1, 3, 7], [1, 7, 5]]
            )  # fmt: skip
        tri = (c + v @ R.T)[f]
        n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        if t == _G.mjGEOM_BOX:  # a box is convex: its faces look away from its centre
            n *= np.sign(np.einsum("ij,ij->i", n, tri[:, 0] - c))[:, None]
    elif t == _G.mjGEOM_CYLINDER:
        a = np.linspace(0, 2 * math.pi, CAP_SIDES, endpoint=False)
        ring = np.c_[h[0] * np.cos(a), h[0] * np.sin(a)]
        tri, n = [], []
        for s in (-1, 1):
            p = c + np.c_[ring, np.full(CAP_SIDES, s * h[1])] @ R.T
            mid = c + s * h[1] * R[:, 2]
            tri += [(mid, p[i], p[(i + 1) % CAP_SIDES]) for i in range(CAP_SIDES)]
            n += [s * R[:, 2]] * CAP_SIDES
        tri, n = np.array(tri), np.array(n)
    else:  # curved all over: no flat face
        return np.zeros((0, 3, 3)), np.zeros((0, 3))
    length = np.linalg.norm(n, axis=1)
    keep = length > 1e-12  # a degenerate face shows nothing
    return tri[keep], n[keep] / length[keep, None]


def _overlap_2d(a: np.ndarray, b: np.ndarray) -> bool:
    """Whether two triangles in one plane share an area wider than TOL (separating axes)."""
    for tri in (a, b):
        for i in range(3):
            e = tri[(i + 1) % 3] - tri[i]
            if np.linalg.norm(e) <= TOL:
                continue
            ax = np.array([-e[1], e[0]]) / np.linalg.norm(e)
            pa, pb = a @ ax, b @ ax
            if min(pa.max(), pb.max()) - max(pa.min(), pb.min()) <= TOL:
                return False
    return True


def _ccw(p: np.ndarray) -> np.ndarray:
    return p if _area(p) >= 0 else p[::-1]


def _area(p: np.ndarray) -> float:
    x, y = p[:, 0], p[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _centroid(p: np.ndarray) -> np.ndarray:
    """A polygon's centroid; its vertices' mean where it has no area."""
    a = _area(p)
    if abs(a) <= TOL * TOL:
        return p.mean(0)
    q = np.roll(p, -1, axis=0)
    cross = p[:, 0] * q[:, 1] - q[:, 0] * p[:, 1]
    return ((p + q) * cross[:, None]).sum(0) / (6 * a)


def _clip(subject: np.ndarray, clip: np.ndarray) -> np.ndarray:
    """The part of convex polygon ``subject`` inside triangle ``clip`` (Sutherland-Hodgman, 2D)."""
    out = _ccw(subject)
    clip = _ccw(clip)
    for i in range(len(clip)):
        if not len(out):
            break
        a, b = clip[i], clip[(i + 1) % len(clip)]
        e = b - a

        def side(p, a=a, e=e):
            return e[0] * (p[1] - a[1]) - e[1] * (p[0] - a[0])

        kept = []
        for j in range(len(out)):
            p, q = out[j], out[(j + 1) % len(out)]
            sp, sq = side(p), side(q)
            if sp >= 0:
                kept.append(p)
            if (sp >= 0) != (sq >= 0):
                kept.append(p + (q - p) * (sp / (sp - sq)))
        out = np.array(kept)
    return out


def _within(faces, lo: np.ndarray, hi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The faces whose bounds reach into the box ``lo``..``hi``: only they can meet the other
    geom's, and keeping the rest out keeps the pairwise comparison of two large meshes small."""
    tri, n = faces
    keep = np.all(tri.min(1) <= hi + TOL, axis=1) & np.all(tri.max(1) >= lo - TOL, axis=1)
    return tri[keep], n[keep]


#: Faces of one geom compared against all of the other's at once: bounds the memory two large
#: meshes take to a few rows of their pairwise comparison.
_CHUNK = 1024


def _candidates(na, da, shown, nb, db):
    """The face pairs (i, j) that face the same way in one plane, the first face shown."""
    for start in range(0, len(na), _CHUNK):
        rows = slice(start, start + _CHUNK)
        same = (na[rows] @ nb.T > 1 - 1e-9) & shown[rows, None]
        near = np.abs(da[rows, None] - db[None, :]) <= TOL
        for i, j in zip(*np.nonzero(same & near), strict=True):
            yield start + i, j


def _flat_hit(fa, fb) -> tuple[np.ndarray, np.ndarray, float] | None:
    """Where two geoms show an outward face each in one plane, the same way, over a shared area:
    (centre, normal, area) of the largest such plane, or None."""
    (ta, na), (tb, nb) = fa, fb
    if not len(ta) or not len(tb):
        return None
    shown = ~((na[:, 2] < -1 + 1e-9) & (np.abs(ta[:, :, 2]).max(1) <= TOL))  # down on the floor
    da, db = np.einsum("ij,ij->i", na, ta[:, 0]), np.einsum("ij,ij->i", nb, tb[:, 0])
    planes: dict[tuple, list] = {}
    for i, j in _candidates(na, da, shown, nb, db):
        a, b, n = ta[i], tb[j], na[i]
        u = (a[1] - a[0]) / np.linalg.norm(a[1] - a[0])
        basis = np.c_[u, np.cross(n, u)]
        pa, pb = (a - a[0]) @ basis, (b - a[0]) @ basis
        if not _overlap_2d(pa, pb):
            continue
        piece = _clip(pa, pb)
        area = abs(_area(piece)) if len(piece) >= 3 else 0.0
        centre = a[0] + basis @ (_centroid(piece) if len(piece) else pa.mean(0))
        key = (*np.round(n, 6), round(float(da[i]), 6))
        planes.setdefault(key, []).append((centre, n, area))
    if not planes:
        return None
    best = max(planes.values(), key=lambda pieces: sum(p[2] for p in pieces))
    total = sum(p[2] for p in best)
    if total > 0:
        centre = sum(p[0] * p[2] for p in best) / total
    else:
        centre = np.mean([p[0] for p in best], axis=0)
    return centre, best[0][1], total


def _caps(
    model: mujoco.MjModel, data: mujoco.MjData, g: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    """A sphere's or capsule's round surfaces as (centre, outward axis); a sphere's axis is zero,
    as it is round all over."""
    if int(model.geom_type[g]) == _G.mjGEOM_SPHERE:
        return [(data.geom_xpos[g], np.zeros(3))]
    axis = data.geom_xmat[g].reshape(3, 3)[:, 2]
    half = axis * model.geom_size[g][1]
    return [(data.geom_xpos[g] + half, axis), (data.geom_xpos[g] - half, -axis)]


def _cap_hit(model, data, g, h) -> tuple[np.ndarray, float] | None:
    """Where a sphere and a capsule, or two capsules, of one radius have a round surface on one
    centre: (that centre, the area both show), or None.

    A sphere covers all directions and a capsule's round end the half around its outward axis, so a
    sphere on a capsule end shares that half (2 pi r²); two capsule ends whose outward axes are
    ``theta`` apart share the lune where both halves meet, 2 (pi - theta) r², which vanishes for two
    capsules end to end on one axis: a lune no wider than TOL, (pi - theta) r, is none, as rounding
    leaves theta a hair short of pi in a straight chain off the coordinate axes (with its joints
    written to about six decimals; coarser, the lune is reported). Ends whose
    centres lie apart, being spheres of one radius, meet in a circle at most.
    """
    r = model.geom_size[g][0]
    if abs(r - model.geom_size[h][0]) > TOL:
        return None
    best = None
    for pg, ag in _caps(model, data, g):
        for ph, ah in _caps(model, data, h):
            if np.linalg.norm(pg - ph) > TOL:
                continue
            if not ag.any() or not ah.any():
                lune = math.pi
            else:
                lune = math.pi - math.acos(float(np.clip(ag @ ah, -1.0, 1.0)))
            area = 2 * lune * r * r
            if lune * r > TOL and (best is None or area > best[1]):
                best = (pg.copy(), area)
    return best


def _round_hit(model, data, g, h) -> tuple[str, np.ndarray, np.ndarray | None, float] | None:
    """Whether two round geoms share a curved surface: cylinders or capsules on one axis with one
    radius and overlapping lengths, spheres on one centre with one radius, or a round capsule end
    on the centre of a sphere or of another capsule's end of its radius (:func:`_cap_hit`)."""
    tg, th = _G(int(model.geom_type[g])), _G(int(model.geom_type[h]))
    sg, sh = model.geom_size[g], model.geom_size[h]
    cg, ch = data.geom_xpos[g], data.geom_xpos[h]
    if tg == th == _G.mjGEOM_SPHERE:
        if abs(sg[0] - sh[0]) <= TOL and np.linalg.norm(cg - ch) <= TOL:
            return "sphere", cg.copy(), None, 4 * math.pi * sg[0] ** 2
        return None
    if {tg, th} <= {_G.mjGEOM_SPHERE, _G.mjGEOM_CAPSULE}:
        side = _side_hit(model, data, g, h) if tg == th else None
        if side is not None:
            return side
        cap = _cap_hit(model, data, g, h)
        return None if cap is None else ("sphere", cap[0], None, cap[1])
    return _side_hit(model, data, g, h)


def _side_hit(model, data, g, h) -> tuple[str, np.ndarray, np.ndarray, float] | None:
    """Cylinders or capsules of one radius on one axis over an overlapping length."""
    tg, th = _G(int(model.geom_type[g])), _G(int(model.geom_type[h]))
    sg, sh = model.geom_size[g], model.geom_size[h]
    cg, ch = data.geom_xpos[g], data.geom_xpos[h]
    side = (_G.mjGEOM_CYLINDER, _G.mjGEOM_CAPSULE)
    if tg not in side or th not in side or abs(sg[0] - sh[0]) > TOL:
        return None
    ag = data.geom_xmat[g].reshape(3, 3)[:, 2]
    ah = data.geom_xmat[h].reshape(3, 3)[:, 2]
    off = ch - cg
    if abs(abs(ag @ ah) - 1) > 1e-9 or np.linalg.norm(off - (off @ ag) * ag) > TOL:
        return None
    s = off @ ag
    lo, hi = max(-sg[1], s - sh[1]), min(sg[1], s + sh[1])
    if hi - lo <= TOL:
        return None
    return "cylinder", cg + (lo + hi) / 2 * ag, ag.copy(), 2 * math.pi * sg[0] * (hi - lo)


def _type_name(t: _G) -> str:
    return t.name.removeprefix("mjGEOM_").lower()


def _uncomparable(model: mujoco.MjModel, g: int, h: int) -> str | None:
    """Why no rule here decides geoms ``g`` and ``h``, or None when one does."""
    tg, th = _G(int(model.geom_type[g])), _G(int(model.geom_type[h]))
    for t in (tg, th):
        if t not in _FLAT and t not in _CURVED:
            return f"geom type {_type_name(t)} is not compared"
    if tg not in _CURVED or th not in _CURVED or _G.mjGEOM_ELLIPSOID not in (tg, th):
        return None
    return f"{' and '.join(sorted(map(_type_name, (tg, th))))}: an ellipsoid is not compared"


def _bounds(model, data, g) -> tuple[np.ndarray, np.ndarray]:
    """A geom's box in the world, round its own bounding box (a plane's: its rectangle)."""
    R, c = data.geom_xmat[g].reshape(3, 3), data.geom_xpos[g]
    if int(model.geom_type[g]) == _G.mjGEOM_PLANE:
        h = [s if s > 0 else INFINITE_PLANE for s in model.geom_size[g][:2]]
        mid, half = c, np.abs(R) @ np.array([h[0], h[1], 0.0])
    else:
        box = model.geom_aabb[g]
        mid, half = c + R @ box[:3], np.abs(R) @ box[3:]
    return mid - half, mid + half


def shared_faces(model: mujoco.MjModel, data: mujoco.MjData | None = None) -> list[SharedFace]:
    """Every pair of drawn geoms that show a face in one place, with where they touch.

    ``data`` holds the pose to check, already forwarded (``mj_forward`` or ``mj_kinematics``); with
    none, the model's own ``qpos0``. Raises :class:`UncomparableGeoms` after comparing every pair it
    can, when a pair whose bounds touch is one no rule here decides (the module docstring).
    """
    if data is None:
        data = mujoco.MjData(model)
        mujoco.mj_kinematics(model, data)
    shown = [g for g in range(model.ngeom) if drawn_by_default(model, g)]
    box = {g: _bounds(model, data, g) for g in shown}
    flat: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    found: list[SharedFace] = []
    skipped: list[tuple[str, str, str]] = []
    for g, h in combinations(shown, 2):
        if np.any(box[g][0] > box[h][1] + TOL) or np.any(box[h][0] > box[g][1] + TOL):
            continue
        why = _uncomparable(model, g, h)
        if why is not None:
            skipped.append((_name(model, g), _name(model, h), why))
            continue
        hit = _round_hit(model, data, g, h)
        if hit is not None:
            kind, at, axis, area = hit
            direction = None if axis is None else tuple(float(x) for x in axis)
            found.append(
                SharedFace(_name(model, g), _name(model, h), kind,
                           tuple(float(x) for x in at), direction, float(area))
            )  # fmt: skip
            continue
        for k in (g, h):
            if k not in flat:
                flat[k] = _faces(model, data, k)
        lo, hi = np.maximum(box[g][0], box[h][0]), np.minimum(box[g][1], box[h][1])
        hit = _flat_hit(_within(flat[g], lo, hi), _within(flat[h], lo, hi))
        if hit is not None:
            at, n, area = hit
            found.append(
                SharedFace(_name(model, g), _name(model, h), "plane",
                           tuple(float(x) for x in at), tuple(float(x) for x in n), float(area))
            )  # fmt: skip
    if skipped:
        raise UncomparableGeoms(skipped, found)
    return found
