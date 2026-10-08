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
