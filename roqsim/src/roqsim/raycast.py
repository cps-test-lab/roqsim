"""The one batched raycaster: every ``mj_multiRay`` call in the tree goes through here.

**The visibility mask is the default, not an argument you remember.** ``mj_multiRay``'s ``geomgroup``
of ``None`` means *every* group, including :data:`roqsim.presence.ABSENT_GEOM_GROUP` -- so a caller
that omits the mask sees entities that have been made absent, and nothing downstream can tell. Here
the mask is what you get unless you write
``geomgroup=None`` on purpose, which turns "forgot the mask" from the default into a visible choice.

Threading this is unsafe, and that is measured rather than assumed
-----------------------------------------------------------------

Rays are independent and ``mj_multiRay`` releases the GIL, so splitting one batch across threads
looks free -- measured 1.5x at 360 rays and 3.1x at 20160, bit-identical output. It is not free, and
the reason is not in the ray maths:

**``mj_multiRay`` allocates from ``mjData``'s stack** -- 96 bytes on a fresh ``mjData``, whatever the
ray count. That stack is a single bump pointer in the shared ``mjData``, so concurrent callers race on
it. Hammering one ``mjData`` with 8 threads x 40000 single-ray casts, a sampler thread observed
``pstack`` peak at **2112** bytes (eight-plus overlapping 96-byte frames) and, on return,
``pstack`` was **1632 rather than 0** -- MuJoCo documents that a function restores ``pstack`` on
return, and concurrency breaks that invariant. The leak is monotonic, so a long trial (1800 s at
10 Hz is ~18k casts) walks toward stack exhaustion; the failure mode is a core dump, not a wrong
number.

Results were *correct* in every one of those 320000 casts, which is exactly what makes this worth
writing down: the race corrupts an allocator invariant rather than the output, so it survives any
amount of parity testing and then fails as a crash much later, in a campaign. MuJoCo's own guidance
is one ``mjData`` per thread; the ray functions are not exempt.

So this module is single-threaded on purpose. If 3D-lidar throughput ever becomes the binding
constraint, the viable route is a pool of per-worker ``mjData`` clones with the geom poses copied in
before each cast -- which needs the set of ``mjData`` fields the ray path reads to be pinned down and
kept pinned across MuJoCo upgrades. That is a real project, not a flag.

**No GPU backend yet.** MJWarp's ``mjw.rays`` can express this module's cast: it takes
``geomgroup``, ``flg_static`` and an excluded body per ray, and applies them inside the nearest-hit
search, so an excluded geom does not hide what is behind it and an absent obstacle reads as the wall
behind it. A backend on it needs a device copy of the scene (``mjw.put_model``, which refuses a
model MJWarp cannot simulate), the geom poses uploaded every tick, a BVH (without one every ray
tests every mesh), and a Warp build that matches the host's CUDA driver. Its cost is a fixed
hand-over per sweep, so it pays for 3D lidars and mesh-heavy worlds, not for a 2D sweep. See
``docs/architecture.rst``.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from .presence import visible_geomgroup_mask

#: Built once: the mask is a constant of the group layout, and rebuilding it per cast would allocate
#: on every tick of every sensor.
VISIBLE_GROUPS = visible_geomgroup_mask()

#: Sentinel for "caller said nothing", so that an explicit ``geomgroup=None`` ("every group,
#: including absent entities") stays expressible and stays distinguishable from the default.
_UNSET = object()


@dataclass
class RayHits:
    """Per-ray nearest-hit results, in the order the directions were given.

    ``dist`` is the Euclidean hit distance with ``-1`` for a miss -- raw ``mj_multiRay`` semantics,
    deliberately not reinterpreted here. ``cutoff`` is a culling *hint* to MuJoCo and not a clamp, so
    a hit beyond it can still be reported; a caller that needs a range window applies it itself (the
    lidar base does).
    """

    dist: np.ndarray  # (nray,) or (npts, nray) float64, -1 = miss
    geomid: np.ndarray  # same shape, int32, -1 = miss
    normal: np.ndarray | None = None  # (..., nray, 3) float64 when requested

    @property
    def nray(self) -> int:
        """Rays per origin -- the last axis, so this reads the same for :func:`cast` and
        :func:`cast_many`."""
        return self.dist.shape[-1]


def buffers(nray: int, *, normals: bool = False) -> RayHits:
    """Allocate reusable output buffers for ``nray`` rays.

    Callers that cast every tick allocate once in ``configure`` and pass the result as ``out=``, so
    the per-cast path does not allocate.
    """
    return RayHits(
        dist=np.full(nray, -1.0, dtype=np.float64),
        geomid=np.full(nray, -1, dtype=np.int32),
        normal=np.zeros((nray, 3), dtype=np.float64) if normals else None,
    )


def _resolve_mask(geomgroup):
    return VISIBLE_GROUPS if geomgroup is _UNSET else geomgroup


def _plane_hits(origins, cutoff, model, data, hits) -> None:
    """Drop, in place, the far plane hits that a compact geom in the same place would not give.

    MuJoCo culls a geom whose bounding sphere lies farther from the ray origin than ``cutoff``, but
    it bounds no plane, so it intersects every plane whatever the cutoff: a floor a metre below a
    15 cm cliff sensor is reported at a metre. The cast keeps a plane to the rule a geom whose
    bounding sphere sits at the plane's position follows. A plane hit nearer than ``cutoff`` stands
    wherever the plane's position is, which is what makes a floor centred at the world origin
    visible to a short-range probe two metres out. A hit beyond ``cutoff`` stands only while that
    position is within ``cutoff`` of the origin, as it would for such a geom, since ``cutoff`` is a
    hint and not a clamp. ``origins`` is ``(npts, 3)`` against hits shaped ``(npts, nray)``.
    """
    plane = model.geom_type == mujoco.mjtGeom.mjGEOM_PLANE
    if not plane.any():
        return
    dist, geomid = hits.dist.reshape(len(origins), -1), hits.geomid.reshape(len(origins), -1)
    far = (geomid >= 0) & (dist > cutoff)
    if not far.any():
        return
    far[far] = plane[geomid[far]]
    rows, cols = np.nonzero(far)
    away = np.linalg.norm(data.geom_xpos[geomid[rows, cols]] - origins[rows], axis=1) > cutoff
    rows, cols = rows[away], cols[away]
    dist[rows, cols] = -1.0
    geomid[rows, cols] = -1
    if hits.normal is not None:
        hits.normal.reshape(len(origins), -1, 3)[rows, cols] = 0.0


def cast(
    model,
    data,
    origin,
    dirs,
    *,
    cutoff: float,
    geomgroup=_UNSET,
    flg_static: bool = True,
    bodyexclude: int = -1,
    out: RayHits | None = None,
) -> RayHits:
    """Cast ``dirs`` from ``origin`` and return the nearest hit per ray.

    ``dirs`` is ``(nray, 3)`` or a flat ``(3 * nray,)`` of unit world-frame directions. ``geomgroup``
    defaults to :data:`VISIBLE_GROUPS`; pass ``None`` to include absent entities, or an explicit
    6-element ``uint8`` mask. ``out`` reuses buffers from :func:`buffers` (and decides whether
    normals are computed, since MuJoCo fills them only when given somewhere to write).
    """
    d = np.ascontiguousarray(dirs, dtype=np.float64).reshape(-1)
    if d.size % 3:
        raise ValueError(f"raycast: dirs has {d.size} values, not a multiple of 3")
    nray = d.size // 3
    hits = out if out is not None else buffers(nray)
    if hits.dist.ndim != 1 or hits.nray != nray:
        raise ValueError(f"raycast: out sized for {hits.nray} rays, given {nray} directions")
    o = np.ascontiguousarray(origin, dtype=np.float64).reshape(3)
    mujoco.mj_multiRay(
        model,
        data,
        o,
        d,
        _resolve_mask(geomgroup),
        flg_static,
        bodyexclude,
        hits.geomid,
        hits.dist,
        # MuJoCo wants the normals flat; the buffer is (nray, 3) and C-contiguous, so this is a view.
        None if hits.normal is None else hits.normal.reshape(-1),
        nray,
        cutoff,
    )
    _plane_hits(o.reshape(1, 3), cutoff, model, data, hits)
    return hits


def cast_many(
    model,
    data,
    origins,
    dirs,
    *,
    cutoff: float,
    geomgroup=_UNSET,
    flg_static: bool = True,
    bodyexclude: int = -1,
    normals: bool = False,
) -> RayHits:
    """The same ``dirs`` cast from *each* of ``origins``; results shaped ``(len(origins), nray)``.

    ``mj_multiRay`` takes one origin, so a probe that asks the same question at many places -- the
    coverage sampler's six axis rays per grid point -- is irreducibly one call per origin and cannot
    be collapsed into a single batch.

    What this still buys over a hand-written loop is the allocation discipline and a caller that can
    then classify every point at once: the flat views are computed before the loop, so each iteration
    passes slices rather than building a fresh array. Measured on a 26569-point grid that is ~1.5x
    faster than a per-point loop with per-point classification -- from the vectorised
    numpy, not from concurrency, which this module refuses (see the module docstring).
    """
    o = np.ascontiguousarray(origins, dtype=np.float64).reshape(-1, 3)
    d = np.ascontiguousarray(dirs, dtype=np.float64).reshape(-1)
    if d.size % 3:
        raise ValueError(f"raycast: dirs has {d.size} values, not a multiple of 3")
    npts, nray = len(o), d.size // 3
    mask = _resolve_mask(geomgroup)
    hits = RayHits(
        dist=np.full((npts, nray), -1.0, dtype=np.float64),
        geomid=np.full((npts, nray), -1, dtype=np.int32),
        normal=np.zeros((npts, nray, 3), dtype=np.float64) if normals else None,
    )
    # Flat views, once: slicing these per origin costs a view rather than a reshape + allocation.
    dist_f, geom_f = hits.dist.reshape(-1), hits.geomid.reshape(-1)
    nrm_f = None if hits.normal is None else hits.normal.reshape(-1)
    for i in range(npts):
        a = i * nray
        mujoco.mj_multiRay(
            model,
            data,
            o[i],
            d,
            mask,
            flg_static,
            bodyexclude,
            geom_f[a : a + nray],
            dist_f[a : a + nray],
            None if nrm_f is None else nrm_f[3 * a : 3 * (a + nray)],
            nray,
            cutoff,
        )
    _plane_hits(o, cutoff, model, data, hits)
    return hits
