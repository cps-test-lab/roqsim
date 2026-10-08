"""Triangle meshes of MuJoCo's primitive geoms, in the geom's own frame.

Every exporter that has to spell a primitive as triangles -- ``roqsim export mesh`` and
``roqsim export gltf`` -- takes it from here, so a capsule is drawn one way. Each function returns
``(vertices, faces)``: ``vertices`` an ``(n, 3)`` float array, ``faces`` an ``(m, 3)`` int array wound
counter-clockwise seen from outside, so every normal points away from the solid.

Poles and seams are emitted as duplicated vertices; a caller that needs a closed solid merges them
(``export_mesh``'s weld pass, which also drops the degenerate triangles that leaves). That keeps each
generator a readable ring-and-cap construction instead of carrying its own special cases.
"""

from __future__ import annotations

import numpy as np


def rings(rows, segments: int) -> tuple[np.ndarray, np.ndarray]:
    """Triangulate a stack of ``(radius, z)`` rings into (vertices, faces), closing in longitude."""
    lon = np.linspace(0.0, 2.0 * np.pi, segments, endpoint=False)
    verts = np.concatenate(
        [
            np.stack([r * np.cos(lon), r * np.sin(lon), np.full(segments, z)], axis=1)
            for r, z in rows
        ]
    )
    faces = []
    for i in range(len(rows) - 1):
        for j in range(segments):
            k = (j + 1) % segments
            a, b = i * segments + j, i * segments + k
            c, d = (i + 1) * segments + j, (i + 1) * segments + k
            # Wound so the normal points AWAY from the axis: with rows ordered bottom-to-top and
            # longitude increasing, (a,c,d) faces inward. An inverted primitive renders as a hole and
            # CAD reads it as a void, and nothing about the file looks wrong -- test_closed_geoms
            # _wind_outward is the guard.
            faces += [[a, d, c], [a, b, d]]
    return verts, np.asarray(faces, dtype=int)


def cylinder(radius: float, half_length: float, segments: int) -> tuple[np.ndarray, np.ndarray]:
    """A closed cylinder about local z (MuJoCo's convention), as (vertices, faces)."""
    rows = [(0.0, -half_length), (radius, -half_length), (radius, half_length), (0.0, half_length)]
    return rings(rows, segments)


def box(size) -> tuple[np.ndarray, np.ndarray]:
    """An axis-aligned box of half-extents ``size``, as (vertices, faces)."""
    sx, sy, sz = (float(v) for v in size)
    verts = np.array(
        [[x, y, z] for x in (-sx, sx) for y in (-sy, sy) for z in (-sz, sz)], dtype=float
    )
    faces = np.array(
        [
            [0, 1, 3],
            [0, 3, 2],
            [4, 7, 5],
            [4, 6, 7],
            [0, 4, 5],
            [0, 5, 1],
            [2, 3, 7],
            [2, 7, 6],
            [0, 2, 6],
            [0, 6, 4],
            [1, 5, 7],
            [1, 7, 3],
        ],
        dtype=int,
    )
    return verts, faces


def sphere(radius: float, segments: int) -> tuple[np.ndarray, np.ndarray]:
    lat = np.linspace(-np.pi / 2, np.pi / 2, max(3, segments // 2 + 1))
    rows = [(radius * np.cos(a), radius * np.sin(a)) for a in lat]
    return rings(rows, segments)


def capsule(radius: float, half_length: float, segments: int) -> tuple[np.ndarray, np.ndarray]:
    """A cylinder about local z closed by two hemispheres -- MuJoCo's capsule."""
    n = max(2, segments // 4)
    lower = np.linspace(-np.pi / 2, 0.0, n + 1)
    upper = np.linspace(0.0, np.pi / 2, n + 1)
    rows = [(radius * np.cos(a), -half_length + radius * np.sin(a)) for a in lower]
    rows += [(radius * np.cos(a), half_length + radius * np.sin(a)) for a in upper]
    return rings(rows, segments)


def ellipsoid(size, segments: int) -> tuple[np.ndarray, np.ndarray]:
    verts, faces = sphere(1.0, segments)
    return verts * np.asarray(size, dtype=float)[:3], faces


# --------------------------------------------------------------------------------------------------
# Textured surfaces. A drawing of a primitive needs what a closed solid must not have: a seam where
# the texture coordinate wraps, so the round shapes are built here as parametric patches with
# duplicated seam vertices, each vertex carrying its normal and MuJoCo's own texture coordinate.
# --------------------------------------------------------------------------------------------------


def _grid(na: int, nb: int, fn):
    """A patch over ``(a, b)`` in ``[0, 1]^2`` on an ``na`` x ``nb`` grid: (verts, normals, uv, faces).

    ``fn(a, b)`` returns ``(positions, normals, uv)`` for arrays ``a``, ``b``.
    """
    a, b = np.meshgrid(np.linspace(0.0, 1.0, na + 1), np.linspace(0.0, 1.0, nb + 1), indexing="ij")
    pos, nrm, uv = fn(a.ravel(), b.ravel())
    idx = np.arange((na + 1) * (nb + 1)).reshape(na + 1, nb + 1)
    q00, q10 = idx[:-1, :-1].ravel(), idx[1:, :-1].ravel()
    q01, q11 = idx[:-1, 1:].ravel(), idx[1:, 1:].ravel()
    faces = np.concatenate([np.stack([q00, q10, q11], 1), np.stack([q00, q11, q01], 1)])
    return pos, nrm, uv, faces


def _merge(*patches):
    verts, normals, uv, faces, base = [], [], [], [], 0
    for v, n, t, f in patches:
        verts.append(v)
        normals.append(n)
        uv.append(t)
        faces.append(f + base)
        base += len(v)
    return np.concatenate(verts), np.concatenate(normals), np.concatenate(uv), np.concatenate(faces)


def _orient(verts, normals, uv, faces):
    """Drop degenerate triangles (a pole, a disk's centre) and wind the rest along their normals."""
    tri = verts[faces]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    keep = np.linalg.norm(cross, axis=1) > 1e-12
    faces, cross = faces[keep], cross[keep]
    flip = np.einsum("ij,ij->i", cross, normals[faces].sum(axis=1)) < 0
    faces[flip] = faces[flip][:, ::-1]
    return verts, normals, uv, faces


def _unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-30)


def _sphere_patch(radii, slices: int, stacks: int):
    """MuJoCo's sphere: ``u = az / 2pi``, ``v = 1/2 - el / pi`` (the top pole at ``v = 0``)."""
    radii = np.asarray(radii, float)

    def fn(a, b):
        az, el = 2.0 * np.pi * a, np.pi * (0.5 - b)
        unit = np.stack([np.cos(az) * np.cos(el), np.sin(az) * np.cos(el), np.sin(el)], 1)
        return unit * radii, _unit(unit / radii), np.stack([a, b], 1)

    return _grid(slices, stacks, fn)


def _side_patch(radius: float, half: float, slices: int, stacks: int):
    """MuJoCo's open cylinder: ``u = az / 2pi``, ``v = (1 - h) / 2``, ``h`` from -1 to 1 along z."""

    def fn(a, b):
        az = 2.0 * np.pi * a
        ring = np.stack([np.cos(az), np.sin(az), np.zeros_like(a)], 1)
        pos = ring * radius
        pos[:, 2] = half * (1.0 - 2.0 * b)
        return pos, ring, np.stack([a, b], 1)

    return _grid(slices, stacks, fn)


def _disk_patch(radius: float, z: float, sign: int, slices: int, rings: int):
    """MuJoCo's disk: ``(u, v) = 1/2 + (x, y) / 2`` over the unit disk, facing ``sign`` z."""

    def fn(a, b):
        az = 2.0 * np.pi * a
        xy = np.stack([np.cos(az) * b, np.sin(az) * b], 1)
        pos = np.c_[xy * radius, np.full_like(a, z)]
        return pos, np.tile([0.0, 0.0, float(sign)], (len(a), 1)), 0.5 + 0.5 * xy

    return _grid(slices, rings, fn)


def _cap_patch(radius: float, half: float, sign: int, slices: int, stacks: int):
    """MuJoCo's capsule end: a hemisphere whose ``v`` runs 1 -> 0 from equator to pole on top, and
    0 -> 1 from equator to pole underneath."""

    def fn(a, b):
        az = 2.0 * np.pi * a
        el = sign * 0.5 * np.pi * b  # b: equator (0) to pole (1)
        unit = np.stack([np.cos(az) * np.cos(el), np.sin(az) * np.cos(el), np.sin(el)], 1)
        pos = unit * radius
        pos[:, 2] += sign * half
        v = 1.0 - b if sign > 0 else b
        return pos, unit, np.stack([a, v], 1)

    return _grid(slices, stacks, fn)


#: A box face: (axis it faces along, sign, the two in-face axes MuJoCo's u and v run along).
_BOX_FACES = [(2, 1, 0, 1), (2, -1, 0, 1), (0, 1, 1, 2), (0, -1, 1, 2), (1, -1, 0, 2), (1, 1, 0, 2)]


def _box_faces(size):
    """MuJoCo's box: each face mapped from its two other axes, ``u = (a + 1) / 2``,
    ``v = 1 - (b + 1) / 2`` in the unit box."""
    size = np.asarray(size[:3], float)
    patches = []
    for axis, sign, ua, va in _BOX_FACES:

        def fn(a, b, axis=axis, sign=sign, ua=ua, va=va):
            unit = np.zeros((len(a), 3))
            unit[:, axis] = sign
            unit[:, ua] = 2.0 * a - 1.0
            unit[:, va] = 2.0 * b - 1.0
            normal = np.zeros((len(a), 3))
            normal[:, axis] = sign
            return unit * size, normal, np.stack([a, 1.0 - b], 1)

        patches.append(_grid(1, 1, fn))
    return _merge(*patches)


def _plane_quad(size, extent: float):
    """MuJoCo's plane: a finite side spans ``u`` (``v``) 0 -> 1 (1 -> 0) over its size; an infinite
    side has ``u = x / 2`` (``v = -y / 2``), and is drawn here ``extent`` either way."""
    sx = float(size[0]) if size[0] > 0 else extent
    sy = float(size[1]) if size[1] > 0 else extent

    def fn(a, b):
        x, y = sx * (2.0 * a - 1.0), sy * (2.0 * b - 1.0)
        u = (x + sx) / (2.0 * sx) if size[0] > 0 else 0.5 * x
        v = 1.0 - (y + sy) / (2.0 * sy) if size[1] > 0 else -0.5 * y
        pos = np.stack([x, y, np.zeros_like(a)], 1)
        return pos, np.tile([0.0, 0.0, 1.0], (len(a), 1)), np.stack([u, v], 1)

    return _grid(1, 1, fn)


def textured_surface(gtype: str, size, segments: int, extent: float = 1.0):
    """A primitive as MuJoCo draws it: ``(verts, normals, uv, faces)`` in the geom's frame.

    Positions and normals are exact for the shape; ``uv`` is the texture coordinate MuJoCo's
    renderer gives the same point, before ``texrepeat`` and ``texuniform`` scale it (the
    parametrisation of MuJoCo 3.14's ``src/render/classic/render_context.c``: ``sphere``,
    ``halfSphere``, ``cylinder``, ``disk``, the box and ``makePlane``). ``segments`` is the count
    around a round shape; half as many run from pole to pole. An infinite plane side is ``extent``
    long either way of the origin.
    """
    slices = max(3, int(segments))
    stacks = max(2, slices // 2)
    if gtype == "plane":
        surface = _plane_quad(size, extent)
    elif gtype == "box":
        surface = _box_faces(size)
    elif gtype in ("sphere", "ellipsoid"):
        radii = [size[0]] * 3 if gtype == "sphere" else size[:3]
        surface = _sphere_patch(radii, slices, stacks)
    elif gtype == "cylinder":
        r, h = float(size[0]), float(size[1])
        surface = _merge(
            _side_patch(r, h, slices, max(1, stacks // 4)),
            _disk_patch(r, h, 1, slices, max(1, stacks // 4)),
            _disk_patch(r, -h, -1, slices, max(1, stacks // 4)),
        )
    elif gtype == "capsule":
        r, h = float(size[0]), float(size[1])
        surface = _merge(
            _side_patch(r, h, slices, max(1, stacks // 4)),
            _cap_patch(r, h, 1, slices, max(1, stacks // 2)),
            _cap_patch(r, h, -1, slices, max(1, stacks // 2)),
        )
    else:
        raise ValueError(f"no surface for a {gtype} geom")
    return _orient(*surface)
