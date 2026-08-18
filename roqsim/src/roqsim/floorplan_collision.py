"""Exact convex wall colliders for a floorplan, parsed from its json-ld source.

A floorplan mesh (``<env>/3d-mesh/<name>.stl``) is concave, and MuJoCo collides a mesh only by its
*convex hull* -- useless for a building interior, where the hull is a solid block filling every room
(see :mod:`roqsim_mobile.plugins.floorplan`). But the mesh's generator (scenery_builder /
Floorplan-DSL) also emits a structured json-ld description (``<env>/json-ld/``) in which every wall
and column is an exact convex polyhedron. :func:`wall_colliders` reads that and returns one convex
vertex set per collider, ready to inject as a MuJoCo mesh geom.

Lives in the core because it has two consumers on different branches of the package graph and is a
source-format reader belonging to neither: :mod:`roqsim_mobile.plugins.floorplan` builds these colliders
at run time, and ``roqsim scenes jsonld-to-scene`` bakes them into a ``scene.json``. Putting it in either
sibling would make the other depend on a package it has no other use for.

Geometry lives in a tree of frames related by ``pose-A-wrt-B`` coordinates and corner ``position``
coordinates. We solve every frame's world pose by treating each pose as a *bidirectional* edge and
walking out from ``world-frame`` (a simple parent-walk fails: a room frame is pinned only through an
anchor child wall). Then:

* each **wall** polyhedron -> an oriented box; **door** openings that overlap it are subtracted along
  the wall's length so the robot can drive through doorways;
* **window** openings are ignored (the sill is above the floor, so the wall still blocks a ground
  robot), and **columns** are emitted as-is.

Returns world-frame vertices only; MuJoCo builds the convex hull at compile time.

Ported from our earlier in-house nav prototype's ``floorplan_collision.py``.
"""

from __future__ import annotations

import json
import logging
import os
from collections import defaultdict, deque

import numpy as np

logger = logging.getLogger("roqsim.floorplan_collision")

_EPS = 1e-6
_MIN_SEG = 1e-3  # drop wall slivers shorter than this (m) after door cuts


def wall_colliders(mesh_path: str) -> list[np.ndarray]:
    """Convex collider vertex sets (each an ``(n, 3)`` world-frame array) for the floorplan whose mesh
    is ``mesh_path``.

    The json-ld is looked up next to the mesh at ``<env>/json-ld/``. Returns ``[]`` (with a warning)
    when that data is missing or unreadable, so the caller degrades to visual-only walls.
    """
    jdir = _json_ld_dir(mesh_path)
    if jdir is None:
        logger.warning(
            "no json-ld found next to %s; using visual-only walls", os.path.basename(mesh_path)
        )
        return []
    try:
        return _parse(_load_graph(jdir))
    except Exception as exc:  # noqa: BLE001 - never block the build
        logger.warning("could not parse json-ld in %s (%s); using visual-only walls", jdir, exc)
        return []


def _json_ld_dir(mesh_path):
    """``<env>/json-ld`` for a mesh at ``<env>/3d-mesh/<name>.stl``, if it holds any ``.json``."""
    env_root = os.path.dirname(os.path.dirname(os.path.abspath(mesh_path)))
    jdir = os.path.join(env_root, "json-ld")
    if os.path.isdir(jdir) and any(f.endswith(".json") for f in os.listdir(jdir)):
        return jdir
    return None


def _load_graph(jdir):
    """Merge the ``@graph`` of the json-ld in ``jdir``.

    Generators differ in how they split the data: some emit ``polyhedron.json`` + ``coordinate.json``,
    others a single combined document (e.g. scenery_builder's ``floorplan.fpm.json``). Either way the
    union of the graphs holds the polyhedra and their coordinate frames.
    """
    names = sorted(f for f in os.listdir(jdir) if f.endswith(".json"))
    graph = []
    for name in names:
        with open(os.path.join(jdir, name)) as fh:
            graph.extend(json.load(fh).get("@graph", []))
    return graph


def _parse(graph):
    polys = [e for e in graph if "Polyhedron" in _types(e)]
    world = _solve_frames(graph)
    corners = _corner_table(graph)

    def corner_world(cid):
        frame, x, y, z = corners[cid]
        return (world[frame] @ np.array([x, y, z, 1.0]))[:3]

    def poly_world(poly):
        return np.array([corner_world(c) for c in poly["points"] if c in corners])

    def kind(pid):
        return next((k for k in ("door", "window", "entry", "column") if k in pid), "wall")

    doors = [p for p in polys if kind(p["@id"]) in ("door", "entry")]
    out = []
    for p in polys:
        k = kind(p["@id"])
        if k in ("door", "entry", "window"):
            continue
        if k == "column":
            out.append(poly_world(p))  # convex as-is
            continue
        out.extend(_wall_boxes(p, world, corner_world, poly_world, doors))
    return _dedupe(out)


def _dedupe(parts):
    """Drop geometrically identical colliders.

    A wall shared by two spaces is emitted once per space (e.g. ``corridor-wall-1`` and
    ``room_251-wall-0`` are the same wall), which would otherwise inject two coincident collider geoms
    and so double the contacts on that wall.
    """
    out, seen = [], set()
    for part in parts:
        key = tuple(np.round(np.sort(np.asarray(part, dtype=float), axis=0), 6).ravel())
        if key in seen:
            continue
        seen.add(key)
        out.append(part)
    if len(out) != len(parts):
        logger.info(
            "dropped %d duplicate wall colliders (walls shared by two spaces)",
            len(parts) - len(out),
        )
    return out


def _split_curie(identifier: str) -> tuple[str, str]:
    """``"rooms:pose-a-wrt-b"`` -> ``("rooms:", "pose-a-wrt-b")``; unprefixed ids -> ``("", id)``.

    Identifiers may be plain (our earlier in-house nav prototype's fixtures) or CURIEs with a generator-chosen prefix
    (scenery_builder emits ``rooms:``). Frames are keyed by their *full* id, because ``as-seen-by``
    and the polyhedron ``points`` reference them that way; only the ``pose-``/``position-`` tokens
    need the prefix stripped before parsing.
    """
    prefix, sep, local = identifier.partition(":")
    return (prefix + sep, local) if sep else ("", identifier)


def _referenced_frame(identifier: str, token: str) -> str:
    """Full frame/position id named by ``<prefix>:<token><name>-wrt-<other>``."""
    prefix, local = _split_curie(identifier)
    return prefix + local[len(token) :].split("-wrt-")[0]


def _solve_frames(graph):
    """Map every frame id -> its 4x4 world transform.

    Each ``pose-A-wrt-B`` is an undirected edge (A in B, and its inverse); BFS from the world frame.
    """
    edges = defaultdict(list)
    for e in graph:
        if "PoseCoordinate" not in _types(e):
            continue
        a = _referenced_frame(e["of-pose"], "pose-")
        b = e["as-seen-by"]
        m = _mat(e.get("x"), e.get("y"), e.get("z"), e.get("alpha"))
        edges[b].append((a, m))
        edges[a].append((b, np.linalg.inv(m)))

    root = next((f for f in edges if _split_curie(f)[1] == "world-frame"), None)
    if root is None:
        raise ValueError("no 'world-frame' in the json-ld coordinate graph")

    world = {root: np.eye(4)}
    q = deque([root])
    while q:
        f = q.popleft()
        for nb, m in edges[f]:
            if nb not in world:
                world[nb] = world[f] @ m
                q.append(nb)
    return world


def _corner_table(graph):
    out = {}
    for e in graph:
        if "PositionCoordinate" not in _types(e):
            continue
        cid = _referenced_frame(e["of-position"], "position-")
        out[cid] = (e["as-seen-by"], _f(e.get("x")), _f(e.get("y")), _f(e.get("z")))
    return out


def _wall_boxes(poly, world, corner_world, poly_world, doors):
    """One convex box per length-segment of a wall after subtracting any door openings that overlap it.

    The wall's own frame makes its corners axis-aligned, so the cut is a 1-D interval subtraction
    along local x.
    """
    frame = poly["@id"][: -len("-polyhedron")] + "-frame"
    if frame not in world:
        return [poly_world(poly)]  # no frame -> emit as convex hull
    wt = world[frame]
    inv = np.linalg.inv(wt)

    def to_local(pts):
        h = np.c_[pts, np.ones(len(pts))]
        return (h @ inv.T)[:, :3]

    lp = to_local(poly_world(poly))
    lo, hi = lp.min(0), lp.max(0)  # local AABB of the wall

    spans = []
    for d in doors:
        dl = to_local(poly_world(d))
        dlo, dhi = dl.min(0), dl.max(0)
        # overlap across thickness (y) and height (z), and within the wall length
        if (
            dlo[1] < hi[1] - _EPS
            and lo[1] < dhi[1] - _EPS
            and dlo[2] < hi[2] - _EPS
            and lo[2] < dhi[2] - _EPS
            and dhi[0] > lo[0] + _EPS
            and dlo[0] < hi[0] - _EPS
        ):
            spans.append((max(dlo[0], lo[0]), min(dhi[0], hi[0])))

    segs = [(lo[0], hi[0])]
    for s0, s1 in sorted(spans):
        nxt = []
        for a, b in segs:
            if s1 <= a or s0 >= b:
                nxt.append((a, b))
                continue
            if a < s0:
                nxt.append((a, s0))
            if s1 < b:
                nxt.append((s1, b))
        segs = nxt

    boxes = []
    for a, b in segs:
        if b - a <= _MIN_SEG:
            continue
        verts = np.array(
            [[x, y, z] for x in (a, b) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]
        )
        boxes.append((np.c_[verts, np.ones(8)] @ wt.T)[:, :3])  # back to world
    return boxes


def _types(e):
    t = e.get("@type")
    return t if isinstance(t, list) else [t]


def _f(v):
    return float(v) if v is not None else 0.0


def _mat(x, y, z, yaw):
    c, s = np.cos(_f(yaw)), np.sin(_f(yaw))
    return np.array([[c, -s, 0, _f(x)], [s, c, 0, _f(y)], [0, 0, 1, _f(z)], [0, 0, 0, 1.0]])
