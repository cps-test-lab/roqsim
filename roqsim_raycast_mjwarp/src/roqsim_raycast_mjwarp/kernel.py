# Copyright (C) 2026 Frederik Pasch
# SPDX-License-Identifier: Apache-2.0
"""The cast kernel: MuJoCo Warp's BVH ray search, with a mesh hit from behind counted as MuJoCo does.

``mjw.rays`` with a render context culls a mesh triangle the ray meets from behind
(``ray._ray_geom_mesh_bvh`` passes ``cull_backfaces=True``, stating that ``mj_ray`` does the same).
MuJoCo's own ``mj_ray`` does not: from inside a closed mesh it reports the face the ray leaves by,
and through a one-sided mesh it reports the triangle from either side. A lidar whose origin is
inside a mesh -- a robot that drove into a visual-only pillar -- then reads the pillar on the CPU
and the room beyond it on the GPU. This kernel is MuJoCo Warp 4.0.1's ``_ray_bvh`` with meshes
searched without culling; everything else -- the BVH traversal, ``_ray_eliminate``, the primitive and
height-field intersections -- is MuJoCo Warp's own function, called unchanged. It reaches into
MuJoCo Warp's private module, so the ``[mjwarp]`` extra pins its version and the parity test guards
it.

Importing this module imports Warp; :mod:`roqsim_raycast_mjwarp.backend` imports it only when a
backend opens.
"""

from __future__ import annotations

import warp as wp
from mujoco_warp._src.ray import _ray_eliminate, ray_geom, ray_mesh_with_bvh
from mujoco_warp._src.types import MJ_MAXVAL, GeomType, vec6


@wp.func
def _geom_hit(
    body_weldid: wp.array[int],
    geom_type: wp.array[int],
    geom_bodyid: wp.array[int],
    geom_dataid: wp.array2d[int],
    geom_matid: wp.array2d[int],
    geom_group: wp.array[int],
    geom_size: wp.array2d[wp.vec3],
    geom_rgba: wp.array2d[wp.vec4],
    mat_rgba: wp.array2d[wp.vec4],
    geom_xpos_in: wp.array2d[wp.vec3],
    geom_xmat_in: wp.array2d[wp.mat33],
    worldid: int,
    pnt: wp.vec3,
    vec: wp.vec3,
    geomgroup: vec6,
    flg_static: bool,
    bodyexclude: int,
    geomid: int,
    mesh_bvh_id: wp.array[wp.uint64],
    hfield_bvh_id: wp.array[wp.uint64],
    min_dist: float,
) -> tuple[float, wp.vec3]:
    if not _ray_eliminate(
        body_weldid,
        geom_bodyid,
        geom_matid[worldid % geom_matid.shape[0]],
        geom_group,
        geom_rgba[worldid % geom_rgba.shape[0]],
        mat_rgba[worldid % mat_rgba.shape[0]],
        geomid,
        geomgroup,
        flg_static,
        bodyexclude,
    ):
        pos = geom_xpos_in[worldid, geomid]
        mat = geom_xmat_in[worldid, geomid]
        gtype = geom_type[geomid]
        if gtype == GeomType.MESH or gtype == GeomType.HFIELD:
            # A mesh is searched as mj_ray searches it, hit from either side; a height field keeps
            # MuJoCo Warp's rule.
            is_mesh = gtype == GeomType.MESH
            bvh_ids = mesh_bvh_id if is_mesh else hfield_bvh_id
            t, n, u, v, f, mid = ray_mesh_with_bvh(
                bvh_ids,
                geom_dataid[worldid % geom_dataid.shape[0], geomid],
                pos,
                mat,
                pnt,
                vec,
                min_dist,
                not is_mesh,
            )
            if t >= 0.0 and t < min_dist:
                return t, n
        else:
            return ray_geom(
                pos, mat, geom_size[worldid % geom_size.shape[0], geomid], pnt, vec, gtype
            )
    return -1.0, wp.vec3()


@wp.kernel
def cast_bvh(
    ngeom: int,
    body_weldid: wp.array[int],
    geom_type: wp.array[int],
    geom_bodyid: wp.array[int],
    geom_dataid: wp.array2d[int],
    geom_matid: wp.array2d[int],
    geom_group: wp.array[int],
    geom_size: wp.array2d[wp.vec3],
    geom_rgba: wp.array2d[wp.vec4],
    mat_rgba: wp.array2d[wp.vec4],
    geom_xpos_in: wp.array2d[wp.vec3],
    geom_xmat_in: wp.array2d[wp.mat33],
    pnt: wp.array2d[wp.vec3],
    vec: wp.array2d[wp.vec3],
    geomgroup: vec6,
    flg_static: bool,
    bodyexclude: wp.array[int],
    bvh_id: wp.uint64,
    group_root: wp.array[int],
    enabled_geom_ids: wp.array[int],
    mesh_bvh_id: wp.array[wp.uint64],
    hfield_bvh_id: wp.array[wp.uint64],
    dist_out: wp.array2d[float],
    geomid_out: wp.array2d[int],
    normal_out: wp.array2d[wp.vec3],
):
    worldid, rayid = wp.tid()
    ray_origin = pnt[worldid % pnt.shape[0], rayid]
    ray_dir = vec[worldid % vec.shape[0], rayid]
    body_exclude = bodyexclude[rayid]
    min_dist = float(MJ_MAXVAL)
    min_geomid = int(-1)  # noqa: UP018 -- Warp: a dynamic variable, not a constant
    min_normal = wp.vec3()
    query = wp.bvh_query_ray(bvh_id, ray_origin, ray_dir, group_root[worldid])
    bounds_nr = int(0)  # noqa: UP018 -- Warp: a dynamic variable, not a constant
    while wp.bvh_query_next(query, bounds_nr, min_dist):
        geomid = enabled_geom_ids[bounds_nr - (worldid * ngeom)]
        dist, normal = _geom_hit(
            body_weldid,
            geom_type,
            geom_bodyid,
            geom_dataid,
            geom_matid,
            geom_group,
            geom_size,
            geom_rgba,
            mat_rgba,
            geom_xpos_in,
            geom_xmat_in,
            worldid,
            ray_origin,
            ray_dir,
            geomgroup,
            flg_static,
            body_exclude,
            geomid,
            mesh_bvh_id,
            hfield_bvh_id,
            min_dist,
        )
        if dist >= 0.0 and dist < min_dist:
            min_dist = dist
            min_geomid = geomid
            min_normal = normal
    if min_dist >= MJ_MAXVAL:
        dist_out[worldid, rayid] = -1.0
    else:
        dist_out[worldid, rayid] = min_dist
    geomid_out[worldid, rayid] = min_geomid
    normal_out[worldid, rayid] = min_normal


def launch(m, d, rc, pnt, vec, geomgroup, flg_static, bodyexclude, dist, geomid, normal) -> None:
    """``mjw.rays(..., rc)``'s launch, through :func:`cast_bvh`."""
    wp.launch(
        cast_bvh,
        dim=(d.nworld, pnt.shape[1]),
        inputs=[
            rc.bvh_ngeom,
            m.body_weldid,
            m.geom_type,
            m.geom_bodyid,
            m.geom_dataid,
            m.geom_matid,
            m.geom_group,
            m.geom_size,
            m.geom_rgba,
            m.mat_rgba,
            d.geom_xpos,
            d.geom_xmat,
            pnt,
            vec,
            geomgroup,
            flg_static,
            bodyexclude,
            rc.bvh_id,
            rc.group_root,
            rc.enabled_geom_ids,
            rc.mesh_bvh_id,
            rc.hfield_bvh_id,
        ],
        outputs=[dist, geomid, normal],
    )
