"""Regression tests for the SDF import path: the two ways a scene silently sinks its robot.

Both bugs below were found in the warehouse port, both produced the *same* user-visible symptom ("the
robot sinks into the floor"), and neither is visible in the MJCF -- the model loads, compiles and
steps. They only show up when something stands on the floor. Hence tests.

The third group covers a shape neither graph cut can decompose -- a closed *ring* of walls, whose hull
is the filled room -- and the footprint cut that does decompose it exactly.
"""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from roqsim_scenes import scene_mesh_io as mio
from roqsim_scenes.cli import scene_to_mjcf


def _hollow_box(inner=5.0, t=0.2, h=3.0):
    """Four walls enclosing a room: one connected component, hull = the whole room."""
    verts, faces = [], []

    def box(lo, hi):
        base = len(verts)
        x0, y0, z0 = lo
        x1, y1, z1 = hi
        verts.extend([[x, y, z] for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)])
        for a, b, c in [
            (0, 1, 3),
            (0, 3, 2),
            (4, 6, 7),
            (4, 7, 5),
            (0, 4, 5),
            (0, 5, 1),
            (2, 3, 7),
            (2, 7, 6),
            (0, 2, 6),
            (0, 6, 4),
            (1, 5, 7),
            (1, 7, 3),
        ]:
            faces.append([base + a, base + b, base + c])

    box((-inner, -inner, 0), (-inner + t, inner, h))
    box((inner - t, -inner, 0), (inner, inner, h))
    box((-inner, -inner, 0), (inner, -inner + t, h))
    box((-inner, inner - t, 0), (inner, inner, h))
    return np.array(verts, float), np.array(faces, np.int64)


def test_connected_components_alone_leave_a_room_swallowing_hull():
    """The premise of the convex split: components cannot separate welded walls."""
    v, f = _hollow_box()
    parts = mio.split_components(v, f)
    assert len(parts) == 1  # the four walls are one component
    ext = parts[0][0].max(axis=0) - parts[0][0].min(axis=0)
    assert ext[0] > 9 and ext[1] > 9  # its hull spans the whole room


def test_split_convex_parts_recovers_the_four_walls():
    v, f = _hollow_box()
    parts = mio.split_convex_parts(v, f)
    assert len(parts) == 4, "a room's shell must decompose into its walls, not stay one brick"
    for pv, _, _ in parts:
        ext = pv.max(axis=0) - pv.min(axis=0)
        # Every piece is a thin slab -- no piece may span the room and swallow its interior.
        assert min(ext[0], ext[1]) < 1.0, f"piece is not a wall slab: extent {ext}"


def test_split_convex_parts_leaves_convex_geometry_whole():
    """A convex mesh has no reflex edges: it must survive as ONE piece, not shatter per triangle."""
    v, f = mio.box("1 2 3")
    assert len(mio.split_convex_parts(v, f)) == 1
    v, f = mio.cylinder(0.5, 1.0)
    assert len(mio.split_convex_parts(v, f)) == 1


def test_convex_split_carries_uvs_alongside_vertices():
    v, f = _hollow_box()
    uv = np.random.default_rng(0).random((len(v), 2))
    for pv, _, puv in mio.split_convex_parts(v, f, uv=uv):
        assert puv is not None and len(puv) == len(pv)


def _manifest(ground_z=None, bounds_min=(-10.0, -10.0, -0.1)):
    m = {
        "name": "t",
        "objects": [],
        "bounds_min": list(bounds_min),
        "bounds_max": [10.0, 10.0, 5.0],
    }
    if ground_z is not None:
        m["ground_z"] = ground_z
    return m


def _floor_z(manifest, config):
    spec = mujoco.MjSpec()
    scene_to_mjcf._add_ground_plane(spec, manifest, [0.0, 0.0, 0.0], config)
    return spec.worldbody.geoms[0].pos[2]


def test_ground_plane_follows_the_source_world_not_the_scene_floor():
    """warehouse.sdf states its ground at z=0 and drops the building to -0.1, so the scene's LOWEST
    geometry is an outdoor apron. Taking the bounds as the ground puts the floor 10 cm under the one
    the robot drives on."""
    assert _floor_z(_manifest(ground_z=0.0), {}) == 0.0


def test_ground_plane_falls_back_to_bounds_when_the_source_states_nothing():
    assert _floor_z(_manifest(ground_z=None), {}) == -0.1


def test_scene_yaml_ground_z_overrides_the_source():
    assert _floor_z(_manifest(ground_z=0.0), {"ground_z": 1.25}) == 1.25


def _wall_ring(outer=10.0, t=0.2, h=3.0):
    """A closed loop of walls as ONE solid extruded ring -- the shape neither graph cut decomposes.

    Not the same mesh as ``_hollow_box``: there the four walls are four closed boxes that happen to
    touch, so reflex edges at the joins separate them. Here the ring is a single closed surface running
    all the way round, capped top and bottom, which is what a floorplan exporter and a Gazebo building
    model both produce (``turtlebot3_world``'s wall is exactly this, with six sides instead of four).
    """
    o, i = outer / 2, outer / 2 - t
    ring = [(-o, -o, -i, -i), (o, -o, i, -i), (o, o, i, i), (-o, o, -i, i)]
    verts = []
    for ox, oy, ix, iy in ring:
        verts += [[ox, oy, 0.0], [ox, oy, h], [ix, iy, 0.0], [ix, iy, h]]
    faces = []
    n = len(ring)
    for k in range(n):
        ob, ot, ib, it = (4 * k + j for j in range(4))
        nb, nt, jb, jt = (4 * ((k + 1) % n) + j for j in range(4))
        faces += [
            [ob, nb, nt],
            [ob, nt, ot],  # outer wall face
            [jb, ib, it],
            [jb, it, jt],  # inner wall face, normal into the room
            [ob, ib, jb],
            [ob, jb, nb],  # bottom cap
            [ot, nt, jt],
            [ot, jt, it],  # top cap
        ]
    verts = np.array(verts, float)
    faces = np.array(faces, np.int64)
    tri = verts[faces]
    if np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum() < 0:
        faces = faces[:, ::-1]  # outward normals, which is what the reflex test reads
    return verts, faces


def test_a_wall_ring_defeats_both_graph_cuts():
    """The premise of the footprint cut: a ring is one component AND one convex piece."""
    v, f = _wall_ring()
    assert len(mio.split_components(v, f)) == 1
    assert len(mio.split_convex_parts(v, f)) == 1, "a ring has no reflex edge that separates it"


def test_the_footprint_cut_recovers_the_four_walls_of_a_ring():
    v, f = _wall_ring()
    parts = mio.split_extruded_shell(v, f)
    assert parts is not None and len(parts) == 4, "a rectangular ring is four wall prisms"
    for pv, _, _ in parts:
        ext = pv.max(axis=0) - pv.min(axis=0)
        assert min(ext[0], ext[1]) < 1.0, f"piece is not a wall slab: extent {ext}"


def test_the_footprint_cut_keeps_every_cubic_metre_of_the_wall():
    """Exact, not approximate: the pieces are the ring, so nothing is shaved off or invented."""
    v, f = _wall_ring(outer=10.0, t=0.2, h=3.0)
    parts = mio.split_extruded_shell(v, f)

    def volume(pv, pf):
        t = pv[pf]
        return abs(np.einsum("ij,ij->i", t[:, 0], np.cross(t[:, 1], t[:, 2])).sum() / 6)

    ring = (10.0**2 - 9.6**2) * 3.0
    assert sum(volume(pv, pf) for pv, pf, _ in parts) == pytest.approx(ring, rel=1e-9)


def test_no_piece_of_a_cut_ring_reaches_into_the_room():
    """What the whole cut is for: the interior must stay empty of collision geometry."""
    from scipy.spatial import ConvexHull

    v, f = _wall_ring()
    parts = mio.split_extruded_shell(v, f)
    hulls = [ConvexHull(pv) for pv, _, _ in parts]
    for x in np.linspace(-4.5, 4.5, 19):
        for y in np.linspace(-4.5, 4.5, 19):
            p = np.array([x, y, 1.5])
            assert not any(
                np.all(h.equations[:, :3] @ p + h.equations[:, 3] <= 1e-9) for h in hulls
            )


def test_the_footprint_cut_declines_what_is_not_an_extrusion():
    """It must abstain rather than approximate: the caller's refusal message is the better answer."""
    assert mio.split_extruded_shell(*mio.sphere(1.0)) is None, "a sphere has no footprint"
    v, f = _wall_ring()
    tilted = v.copy()
    tilted[:, 2] += 0.3 * tilted[:, 0]  # a sloped top: no longer two horizontal planes
    assert mio.split_extruded_shell(tilted, f) is None


_Y_UP_DAE = """<?xml version="1.0"?>
<COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema" version="1.4.1">
 <asset><unit name="metre" meter="1"/><up_axis>Y_UP</up_axis></asset>
 <library_geometries>
  <geometry id="g"><mesh>
   <source id="pos"><float_array id="pa" count="9">0 0 0 1 0 0 0 2 0</float_array>
    <technique_common><accessor source="#pa" count="3" stride="3">
     <param name="X" type="float"/><param name="Y" type="float"/><param name="Z" type="float"/>
    </accessor></technique_common></source>
   <vertices id="v"><input semantic="POSITION" source="#pos"/></vertices>
   <triangles count="1"><input semantic="VERTEX" source="#v" offset="0"/><p>0 1 2</p></triangles>
  </mesh></geometry>
 </library_geometries>
 <library_visual_scenes><visual_scene id="s"><node><instance_geometry url="#g"/></node></visual_scene>
 </library_visual_scenes>
 <scene><instance_visual_scene url="#s"/></scene>
</COLLADA>
"""


def test_a_y_up_collada_file_is_rotated_to_z_up(tmp_path):
    """The default, and it must stay the default: the spec is on the declaration's side."""
    dae = tmp_path / "y_up.dae"
    dae.write_text(_Y_UP_DAE)
    verts = mio.read_collada(dae)[0].verts
    assert verts[2].tolist() == [0.0, 0.0, 2.0], "the authored +Y vertex has to come back as +Z"


def test_a_files_up_axis_can_be_overridden_per_file(tmp_path):
    """For an asset whose declaration contradicts its own data.

    ``turtlebot3_world``'s wall declares Y_UP over Z-up geometry: honouring it stands the room's fence
    up on edge, 6.6 m tall and 1.27 m thick, which neither the world it is used in nor the occupancy
    grid published beside it agrees with. The override is per file and the importer announces it,
    because it overrules what the asset says about itself.
    """
    dae = tmp_path / "mislabelled.dae"
    dae.write_text(_Y_UP_DAE)
    verts = mio.read_collada(dae, ignore_up_axis=True)[0].verts
    assert verts[2].tolist() == [0.0, 2.0, 0.0], "coordinates have to come back as authored"
    assert mio.read_mesh(dae, ignore_up_axis=True)[0].verts[2].tolist() == [0.0, 2.0, 0.0]
