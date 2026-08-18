"""Regression tests for the SDF import path: the two ways a scene silently sinks its robot.

Both bugs below were found in the warehouse port, both produced the *same* user-visible symptom ("the
robot sinks into the floor"), and neither is visible in the MJCF -- the model loads, compiles and
steps. They only show up when something stands on the floor. Hence tests.
"""

from __future__ import annotations

import mujoco
import numpy as np

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
