# Copyright (C) 2026 Frederik Pasch
# SPDX-License-Identifier: Apache-2.0
"""The GPU backend's hits against the CPU cast: the same geom on every ray but the known kinds.

For a 2D lidar, a VLP-16 and a Mid-360 (:mod:`_worlds`), in the default room and in the depot, with
the obstacle present and made absent through :mod:`roqsim.presence`, each lidar's own rays are cast
by the backend it opened (``mjwarp``) and by :func:`roqsim.raycast.cast` on the same state. Every
ray they disagree on is put in one of the kinds the backend's docstring names, and each kind is
bounded:

- ``tie``: both hit, other geoms, at the same distance (within :data:`TIE_M`) -- two visible
  geoms meeting the ray at one point, such as a sensor's visual mesh and its collision cylinder;
- ``edge``: a ray whose CPU answer itself changes when its origin or direction moves by the
  float32 error of the device copy (:data:`NUDGE_RAD`) -- a ray grazing an edge;
- ``thin_mesh``: one side passes a mesh the other hits, reading what lies behind it or nothing --
  another geom, or another of the same mesh's triangles;
- ``beyond_cutoff``: the CPU reports a hit beyond the cutoff on a geom the GPU passed nearer
  (:func:`roqsim.raycast.cull_beyond_cutoff`);
- anything else is ``unexplained``, and there must be none.

Every other ray names the same geom on both, at distances within :data:`DIST_TOL_M`.
"""

from __future__ import annotations

import _worlds
import mujoco
import numpy as np
import pytest

from roqsim import raycast
from roqsim.presence import set_present

#: Distance agreement on a ray both backends give the same geom: float32 on the device against
#: float64, over a world tens of metres across.
DIST_TOL_M = 5e-5
#: Two hits this close are one point of the scene, whichever geom is named.
TIE_M = 1e-4
#: How far the probe for an edge moves a ray: well above float32's error on a direction, and
#: under a millimetre at the lidars' ranges.
NUDGE_RAD = 2e-5

#: Per lidar, the most rays of each kind a sweep may disagree on. Measured on these worlds with
#: :func:`classify`: the 2D lidar and the VLP-16 none; the Mid-360 (20,160 rays) one tie (the VLP-16's
#: visual mesh and its collision cylinder), two grazing the obstacle's edges and two passing the
#: depot's ceiling fans, each with a margin of a few rays.
BOUNDS = {
    "sick_tim571": {"tie": 1, "edge": 1, "thin_mesh": 0, "beyond_cutoff": 1},
    "velodyne_vlp16": {"tie": 1, "edge": 1, "thin_mesh": 0, "beyond_cutoff": 1},
    "mid360": {"tie": 3, "edge": 5, "thin_mesh": 4, "beyond_cutoff": 3},
}


def _rays(engine, plugin):
    d = engine.ctx.data
    origin = np.array(d.site_xpos[plugin._site_id])
    dirs = plugin._local_dirs @ np.array(d.site_xmat[plugin._site_id]).reshape(3, 3).T
    return origin, dirs


def _cpu(engine, plugin, origin, dirs):
    m, d = engine.ctx.model, engine.ctx.data
    return raycast.cast(
        m, d, origin, dirs, cutoff=plugin.range_max, bodyexclude=plugin._bodyexclude
    )


def _robust(engine, plugin, origin, dirs, i) -> bool:
    """Whether ray *i*'s CPU answer survives a nudge of its direction and its origin."""
    ref = _cpu(engine, plugin, origin, dirs[i : i + 1]).geomid[0]
    a, b = _basis(dirs[i])
    for e in (a, -a, b, -b):
        nudged = dirs[i] + NUDGE_RAD * e
        nudged /= np.linalg.norm(nudged)
        if _cpu(engine, plugin, origin, nudged[None]).geomid[0] != ref:
            return False
        if _cpu(engine, plugin, origin + 1e-5 * e, dirs[i : i + 1]).geomid[0] != ref:
            return False
    return True


def _basis(v):
    a = np.cross(v, [0.0, 0.0, 1.0] if abs(v[2]) < 0.9 else [1.0, 0.0, 0.0])
    a /= np.linalg.norm(a)
    return a, np.cross(v, a)


def classify(engine, plugin, cpu, gpu, origin, dirs) -> dict:
    """Every ray the two disagree on, by kind; plus the distance error on the rays they agree on.

    A ray disagrees when the geoms differ, or when the distances differ by more than
    :data:`DIST_TOL_M` on the same geom (a ray entering a mesh at another of its triangles).
    """
    m = engine.ctx.model
    kinds = {"tie": [], "edge": [], "thin_mesh": [], "beyond_cutoff": [], "unexplained": []}
    both = (cpu.geomid >= 0) & (gpu.geomid >= 0)
    far = both & (np.abs(cpu.dist - gpu.dist) > DIST_TOL_M)
    disagree = (cpu.geomid != gpu.geomid) | far
    agree_hit = ~disagree & both
    err = np.abs(cpu.dist - gpu.dist)[agree_hit]
    mesh = mujoco.mjtGeom.mjGEOM_MESH
    for i in np.nonzero(disagree)[0]:
        c, g = int(cpu.geomid[i]), int(gpu.geomid[i])
        dc, dg = float(cpu.dist[i]), float(gpu.dist[i])
        if c >= 0 and g >= 0 and c != g and abs(dc - dg) <= TIE_M:
            kind = "tie"
        elif not _robust(engine, plugin, origin, dirs, i):
            kind = "edge"
        elif (c >= 0 and m.geom_type[c] == mesh and (g < 0 or dg > dc)) or (
            g >= 0 and m.geom_type[g] == mesh and (c < 0 or dc > dg)
        ):
            kind = "thin_mesh"
        elif c >= 0 and dc > plugin.range_max and g < 0:
            kind = "beyond_cutoff"
        else:
            kind = "unexplained"
        kinds[kind].append({"ray": int(i), "cpu": [_name(m, c), dc], "gpu": [_name(m, g), dg]})
    return {
        "kinds": kinds,
        "same_geom": float((cpu.geomid == gpu.geomid).mean()),
        "dist_err_max_m": float(err.max()) if err.size else 0.0,
        "rays": int(len(dirs)),
    }


def _name(m, g):
    if g < 0:
        return None
    return mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or f"geom{g}"


def _device_model(plugin) -> str:
    return {"lidar2d": "sick_tim571", "vlp16": "velodyne_vlp16", "mid360": "mid360"}[
        plugin.address.split(".")[0]
    ]


@pytest.fixture(scope="module", params=["room", "depot"])
def world(request, gpu):
    path = _worlds.depot_world() if request.param == "depot" else None
    engine = _worlds.build(_worlds.lidar_world(path, "mjwarp"))
    for _ in range(25):
        engine.step()
    yield request.param, engine
    engine.shutdown()


@pytest.mark.parametrize("present", [True, False], ids=["present", "absent"])
def test_the_gpu_names_the_cpus_geom_on_every_ray_but_the_known_kinds(world, present):
    name, engine = world
    obstacle = engine.ctx.entities.get("obstacle")
    set_present(engine.ctx, obstacle, present)
    mujoco.mj_forward(engine.ctx.model, engine.ctx.data)
    obstacle_geoms = set(np.nonzero(engine.ctx.model.geom_bodyid == obstacle_body(engine))[0])
    try:
        for plugin in _worlds.lidar_plugins(engine):
            origin, dirs = _rays(engine, plugin)
            cpu = _cpu(engine, plugin, origin, dirs)
            gpu = plugin._raycaster.cast(
                engine.ctx.model,
                engine.ctx.data,
                origin,
                dirs,
                cutoff=plugin.range_max,
                bodyexclude=plugin._bodyexclude,
            )
            got = classify(engine, plugin, cpu, gpu, origin, dirs)
            where = f"{name}, obstacle {'present' if present else 'absent'}, {plugin.address}"
            assert got["dist_err_max_m"] <= DIST_TOL_M, where
            assert not got["kinds"]["unexplained"], (where, got["kinds"]["unexplained"])
            for kind, bound in BOUNDS[_device_model(plugin)].items():
                assert len(got["kinds"][kind]) <= bound, (where, kind, got["kinds"][kind])
            on_obstacle = np.isin(gpu.geomid, list(obstacle_geoms))
            if present:
                # The obstacle stands in every scan: the lidars see it, on both backends.
                assert on_obstacle.any(), where
            else:
                # Absent, nothing reads it, and the rays that would have read the wall behind it.
                assert not on_obstacle.any(), where
                assert not np.isin(cpu.geomid, list(obstacle_geoms)).any(), where
    finally:
        set_present(engine.ctx, obstacle, True)
        mujoco.mj_forward(engine.ctx.model, engine.ctx.data)


def obstacle_body(engine) -> int:
    return mujoco.mj_name2id(engine.ctx.model, mujoco.mjtObj.mjOBJ_BODY, "obs_obstacle")


def test_the_mask_the_excluded_body_and_flg_static_hold_on_the_gpu(world):
    _name_, engine = world
    m, d = engine.ctx.model, engine.ctx.data
    plugin = _worlds.lidar_plugins(engine)[-1]  # the Mid-360: rays every way
    origin, dirs = _rays(engine, plugin)
    backend = plugin._raycaster
    for kwargs in (
        {"geomgroup": None},
        {"geomgroup": np.array([1, 0, 0, 0, 0, 0], np.uint8)},
        {"flg_static": False},
        {"bodyexclude": obstacle_body(engine)},
    ):
        cpu = raycast.cast(m, d, origin, dirs, cutoff=plugin.range_max, **kwargs)
        gpu = backend.cast(m, d, origin, dirs, cutoff=plugin.range_max, **kwargs)
        same = (cpu.geomid == gpu.geomid).mean()
        assert same >= 0.999, (kwargs, same)
        if "bodyexclude" in kwargs:
            assert not np.isin(
                gpu.geomid, np.nonzero(m.geom_bodyid == kwargs["bodyexclude"])[0]
            ).any()
        if kwargs.get("flg_static") is False:
            assert (
                not (gpu.geomid >= 0).any()
                or (m.body_weldid[m.geom_bodyid[gpu.geomid[gpu.geomid >= 0]]] != 0).all()
            )


def test_cast_many_is_cast_from_each_origin(world):
    _name_, engine = world
    m, d = engine.ctx.model, engine.ctx.data
    plugin = _worlds.lidar_plugins(engine)[0]
    origin, dirs = _rays(engine, plugin)
    origins = origin + np.array([[0.0, 0.0, 0.0], [0.3, 0.1, 0.0], [-0.2, 0.4, 0.2]])
    many = plugin._raycaster.cast_many(m, d, origins, dirs, cutoff=plugin.range_max, normals=True)
    assert many.dist.shape == (3, len(dirs)) and many.normal.shape == (3, len(dirs), 3)
    for k, o in enumerate(origins):
        one = plugin._raycaster.cast(m, d, o, dirs, cutoff=plugin.range_max)
        assert many.geomid[k].tobytes() == one.geomid.tobytes()
        assert many.dist[k].tobytes() == one.dist.tobytes()
    ref = raycast.cast_many(m, d, origins, dirs, cutoff=plugin.range_max, normals=True)
    assert (ref.geomid == many.geomid).mean() >= 0.995
    hit = (ref.geomid == many.geomid) & (ref.geomid >= 0)
    # Normals: unit, and the CPU's own on the rays both agree on (a box's face, a plane's up).
    assert np.allclose(np.linalg.norm(many.normal[hit], axis=-1), 1.0, atol=1e-4)
    assert np.mean(np.einsum("ij,ij->i", ref.normal[hit], many.normal[hit]) > 0.999) >= 0.99


def test_the_same_state_twice_gives_the_same_bytes(world):
    _name_, engine = world
    m, d = engine.ctx.model, engine.ctx.data
    plugin = _worlds.lidar_plugins(engine)[-1]
    origin, dirs = _rays(engine, plugin)
    backend = plugin._raycaster
    first = backend.cast(
        m, d, origin, dirs, cutoff=plugin.range_max, bodyexclude=plugin._bodyexclude
    )
    first = (first.dist.copy(), first.geomid.copy())
    # Upload and refit again from nothing, as a fresh tick would.
    backend._last.clear()
    again = backend.cast(
        m, d, origin, dirs, cutoff=plugin.range_max, bodyexclude=plugin._bodyexclude
    )
    assert again.dist.tobytes() == first[0].tobytes()
    assert again.geomid.tobytes() == first[1].tobytes()


def test_every_lidar_shares_one_device_scene_refitted_once_per_state(world):
    _name_, engine = world
    lidars = _worlds.lidar_plugins(engine)
    backends = {id(p._raycaster) for p in lidars}
    assert len(backends) == 1
    backend = lidars[0]._raycaster
    before = backend.refits
    # A state with every geom pose changed, as a moving robot's would be: the first cast uploads and
    # refits, the other lidars find it current.
    d = engine.ctx.data
    saved = (d.geom_xpos.copy(), d.geom_xmat.copy())
    d.geom_xpos[:] += 1e-3
    try:
        for p in lidars:
            p.capture(engine.ctx)
        assert backend.refits == before + 1
    finally:
        d.geom_xpos[:], d.geom_xmat[:] = saved
