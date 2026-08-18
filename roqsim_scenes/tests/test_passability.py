"""The check a picture cannot make: did the convex hulls wall off a way through?

MuJoCo collides a mesh by its convex hull, so a wall with a doorway cut out of it is a solid wall --
while every renderer and ``mj_ray`` still show the doorway open. secorolab shipped that way and 87 %
of the building was unreachable, through four campaigns, because nothing anyone looked at could show
it. These tests pin the shapes that must and must not trip the check.
"""

from __future__ import annotations

import numpy as np

from roqsim_scenes.passability import closed_passages


def _box(lo, hi):
    """A closed axis-aligned box as ``(verts, faces)``."""
    x0, y0, z0 = lo
    x1, y1, z1 = hi
    verts = np.array([[x, y, z] for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)], float)
    quads = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    faces = np.array([t for a, b, c, d in quads for t in ((a, b, c), (a, c, d))])
    return verts, faces


def _room_with_gap(gap=0.9, t=0.1, h=2.2, size=4.0):
    """Four walls around a room, the north wall split into two boxes with *gap* between them.

    The honest way to model a doorway: the opening is the absence of geometry, so no hull can close
    it. This is what ``jsonld-to-scene`` produces.
    """
    s, g = size, gap / 2
    return [
        _box((-s, s - t, 0), (-g, s, h)),  # north wall, west of the gap
        _box((g, s - t, 0), (s, s, h)),  # north wall, east of the gap
        _box((-s, -s, 0), (s, -s + t, h)),
        _box((-s, -s, 0), (-s + t, s, h)),
        _box((s - t, -s, 0), (s, s, h)),
    ]


def _wall_with_hole(gap=0.9, t=0.1, h=2.2, size=4.0):
    """The same room, but the north wall is ONE part with the doorway as a hole through it.

    Geometrically identical to :func:`_room_with_gap` -- and its convex hull is the solid wall, so the
    doorway exists for every renderer and for no robot. This is what baking a fused floorplan mesh
    produces.
    """
    s, g = size, gap / 2
    west = _box((-s, s - t, 0), (-g, s, h))
    east = _box((g, s - t, 0), (s, s, h))
    merged = (
        np.vstack([west[0], east[0]]),
        np.vstack([west[1], east[1] + len(west[0])]),
    )
    return [merged] + _room_with_gap(gap, t, h, size)[2:]


def test_doorway_as_absent_geometry_is_not_flagged():
    assert closed_passages(_room_with_gap()) == []


def test_doorway_as_a_hole_in_one_part_is_flagged():
    sealed = closed_passages(_wall_with_hole())
    assert sealed, "a hull that fills the only doorway must be refused"
    # The room is what got sealed off, so the report must point INTO it, not at the wall.
    (cx, cy) = sealed[0].centre
    assert abs(cx) < 1.0 and abs(cy) < 2.0, f"expected the room's interior, got ({cx:.2f}, {cy:.2f})"
    assert sealed[0].area > 40.0, f"an 8x8 m room, got {sealed[0].area:.1f} m^2"


def test_a_gap_narrower_than_the_probe_is_not_a_passage():
    """A 4 cm gap is a modelling seam, not a doorway.

    The check must not fire on one, or every chamfer and mesh seam in the corpus becomes an import
    failure -- which is how a loud check gets silenced and stops protecting anything.
    """
    assert closed_passages(_wall_with_hole(gap=0.04)) == []


def test_a_solid_wall_with_no_doorway_is_not_flagged():
    """Sealing a room DELIBERATELY is not the failure; inventing the seal is.

    A room with no opening at all is sealed in the triangles too, so there is no way through for a
    hull to close and nothing to report.
    """
    s, t, h = 4.0, 0.1, 2.2
    walls = [
        _box((-s, s - t, 0), (s, s, h)),
        _box((-s, -s, 0), (s, -s + t, h)),
        _box((-s, -s, 0), (-s + t, s, h)),
        _box((s - t, -s, 0), (s, s, h)),
    ]
    assert closed_passages(walls) == []


def test_a_zero_thickness_wall_is_someone_else_s_failure():
    """A surface with no volume is not this check's to catch, and pretending otherwise would hide it.

    Its 2D hull is a line, so it seals nothing here -- and it seals nothing in MuJoCo either, which
    refuses to hull it at all ("coplanar vertices, cannot compute convex hull") when the scene is
    baked. That refusal is the one the secorolab port actually hit, and read as a substrate quirk to
    work around with a thickening script instead of as a wrong-route signal.

    Pinned as a test because the tempting "fix" is to make this check fire on zero-thickness parts
    too. It must not: the part never becomes a collider, so no passage is closed, and a check that
    claimed otherwise would be reporting a defect that is not there while the real one -- a *thick*
    wall with a hole in it -- looks identical in its output.
    """
    s, h = 4.0, 2.2
    quad = (
        np.array([[-s, s, 0], [s, s, 0], [s, s, h], [-s, s, h]], float),
        np.array([[0, 1, 2], [0, 2, 3]]),
    )
    room = _room_with_gap()[2:]  # three solid walls, north side open
    assert closed_passages(room) == [], "an open-north room is reachable and must pass"
    assert closed_passages(room + [quad]) == []
