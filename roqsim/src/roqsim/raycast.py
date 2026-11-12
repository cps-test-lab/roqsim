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


def cull_beyond_cutoff(origins, cutoff, model, data, hits) -> None:
    """Drop, in place, the hits beyond ``cutoff`` that ``mj_multiRay`` would not have reported.

    For a backend whose search does not cull: ``mj_multiRay`` skips a geom whose bounding sphere
    (``geom_rbound`` about ``geom_xpos``) lies wholly farther than ``cutoff`` from the ray's origin,
    so a hit beyond ``cutoff`` on such a geom is one the CPU never reports. A plane's ``rbound`` is 0,
    so for planes this is the far-plane rule of :func:`_plane_hits`. ``origins`` is ``(npts, 3)``
    against hits shaped ``(npts, nray)``.

    It cannot give what the CPU reports *instead*: where the nearest hit is on a culled geom, the CPU
    may still report a farther hit on a geom it did not cull, and this leaves a miss. Both lie beyond
    ``cutoff``, which every lidar drops (:mod:`roqsim_sensors.plugins.lidar_common`).
    """
    o = np.asarray(origins, dtype=np.float64).reshape(-1, 3)
    dist, geomid = hits.dist.reshape(len(o), -1), hits.geomid.reshape(len(o), -1)
    far = (geomid >= 0) & (dist > cutoff)
    if not far.any():
        return
    rows, cols = np.nonzero(far)
    g = geomid[rows, cols]
    away = np.linalg.norm(data.geom_xpos[g] - o[rows], axis=1) - model.geom_rbound[g] > cutoff
    rows, cols = rows[away], cols[away]
    dist[rows, cols] = -1.0
    geomid[rows, cols] = -1
    if hits.normal is not None:
        hits.normal.reshape(len(o), -1, 3)[rows, cols] = 0.0


# -- backends --------------------------------------------------------------------------------------

#: The backend a world casts through when it names none: :func:`cast` and :func:`cast_many`.
DEFAULT_RAYCASTER = "cpu"


@dataclass
class Setting:
    """One key of a backend's settings block (``sim.raycasters.<name>``): its default and meaning."""

    default: object
    meaning: str


class Raycaster:
    """Base of a ray-cast backend: the same cast as :func:`cast` and :func:`cast_many`, elsewhere.

    One instance serves every sensor of a world that chose it (:class:`RaycastService`), so it
    holds whatever it built from the model once -- a device copy of the scene, an acceleration
    structure -- and brings it up to date with the state it is handed on each call. It is opened
    after the model compiled, and called only on the physics thread, with that thread's ``model``
    and ``data``: it reads them, and never writes them.

    Every backend honours what :func:`cast` honours: the ``geomgroup`` mask (the visible groups by
    default), ``flg_static``, the excluded body, a geom whose resolved alpha is 0, and the rule for
    hits beyond ``cutoff`` (:func:`cull_beyond_cutoff`). A backend that cannot run where it was
    chosen raises from :meth:`open`, naming the reason; the service turns that into a load failure
    naming the sensor, and nothing casts on another backend instead.
    """

    #: Its settings block, key by key.
    SETTINGS: dict[str, Setting] = {}

    def __init__(self, settings=None) -> None:
        self.settings = {k: s.default for k, s in self.SETTINGS.items()}
        self.settings.update(dict(settings or {}))

    @classmethod
    def validate_config(cls, settings) -> list[str]:
        """Errors in a settings block: a key it does not declare is one."""
        unknown = sorted(set(settings or {}) - set(cls.SETTINGS))
        if unknown:
            known = ", ".join(sorted(cls.SETTINGS)) or "(none)"
            return [f"unknown key(s) {', '.join(unknown)}; valid: {known}"]
        return []

    def open(self, model) -> None:
        """Build what this backend needs from the compiled ``model``; raise if it cannot run."""

    def cast(
        self,
        model,
        data,
        origin,
        dirs,
        *,
        cutoff,
        geomgroup=_UNSET,
        flg_static=True,
        bodyexclude=-1,
        out=None,
    ) -> RayHits:
        raise NotImplementedError

    def cast_many(
        self,
        model,
        data,
        origins,
        dirs,
        *,
        cutoff,
        geomgroup=_UNSET,
        flg_static=True,
        bodyexclude=-1,
        normals=False,
    ) -> RayHits:
        raise NotImplementedError

    def close(self) -> None:
        """Release what :meth:`open` built."""


class CpuRaycaster(Raycaster):
    """MuJoCo's ``mj_multiRay`` on the physics thread: :func:`cast` and :func:`cast_many`, unchanged."""

    def cast(
        self,
        model,
        data,
        origin,
        dirs,
        *,
        cutoff,
        geomgroup=_UNSET,
        flg_static=True,
        bodyexclude=-1,
        out=None,
    ) -> RayHits:
        return cast(
            model,
            data,
            origin,
            dirs,
            cutoff=cutoff,
            geomgroup=geomgroup,
            flg_static=flg_static,
            bodyexclude=bodyexclude,
            out=out,
        )

    def cast_many(
        self,
        model,
        data,
        origins,
        dirs,
        *,
        cutoff,
        geomgroup=_UNSET,
        flg_static=True,
        bodyexclude=-1,
        normals=False,
    ) -> RayHits:
        return cast_many(
            model,
            data,
            origins,
            dirs,
            cutoff=cutoff,
            geomgroup=geomgroup,
            flg_static=flg_static,
            bodyexclude=bodyexclude,
            normals=normals,
        )


class RaycastService:
    """The ray-cast backends of one world: each opened once, on first use, and shared.

    ``sim.raycaster`` names the backend a sensor casts through when it names none of its own, and
    ``sim.raycasters.<name>`` holds a backend's settings, as ``sim.renderer`` and ``sim.renderers``
    do for cameras (``docs/rendering.rst``). Every sensor that chose a backend gets the same
    instance, so a device scene exists once per world and is brought up to date once per state,
    however many lidars cast in it.
    """

    def __init__(self, model, *, raycaster: str | None = None, raycasters=None, base_dir=None):
        self.model = model
        self.default = raycaster or DEFAULT_RAYCASTER
        self.settings = {str(k): dict(v or {}) for k, v in (raycasters or {}).items()}
        self.base_dir = base_dir
        self.open_backends: dict[str, Raycaster] = {}

    @classmethod
    def of(cls, ctx) -> RaycastService:
        """*ctx*'s service, made from its world's ``sim`` block on first use."""
        service = getattr(ctx, "raycast", None)
        if service is None or service.model is not ctx.model:
            sim = (ctx.config or {}).get("sim") or {}
            service = cls(
                ctx.model,
                raycaster=sim.get("raycaster"),
                raycasters=sim.get("raycasters"),
                base_dir=getattr(ctx, "base_dir", None),
            )
            ctx.raycast = service
        return service

    def get(self, name: str | None, who: str) -> Raycaster:
        """The open backend *name* (the world's default when ``None``), for the sensor *who*.

        A backend that cannot open fails here, naming *who*, the backend and the reason.
        """
        from .plugin import PluginError  # noqa: PLC0415 -- plugin imports nothing of ours
        from .registry import resolve_raycaster  # noqa: PLC0415

        name = name or self.default
        if name in self.open_backends:
            return self.open_backends[name]
        if name == DEFAULT_RAYCASTER and not self.settings.get(name):
            backend = CpuRaycaster()
        else:
            try:
                cls = resolve_raycaster(name, self.base_dir)
            except PluginError as err:
                raise PluginError(f"{who}: {err}") from err
            settings = self.settings.get(name, {})
            errors = cls.validate_config(settings)
            if errors:
                raise PluginError(
                    f"{who}: raycaster {name!r}: "
                    + "; ".join(f"sim.raycasters.{name}: {e}" for e in errors)
                )
            backend = cls(settings)
            try:
                backend.open(self.model)
            except Exception as err:
                backend.close()
                raise PluginError(f"{who}: raycaster {name!r} cannot run here: {err}") from err
        self.open_backends[name] = backend
        return backend

    def close(self) -> None:
        for backend in self.open_backends.values():
            backend.close()
        self.open_backends.clear()
