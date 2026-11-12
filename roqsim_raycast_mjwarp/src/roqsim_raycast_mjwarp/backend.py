# Copyright (C) 2026 Frederik Pasch
# SPDX-License-Identifier: Apache-2.0
"""The ``mjwarp`` ray-cast backend: roqsim's cast on a CUDA GPU through MuJoCo Warp's ``mjw.rays``.

A world chooses it with ``sim.raycaster: mjwarp``, or a single lidar with ``raycaster: mjwarp``
(``docs/rendering.rst``, "Choosing a ray caster"). It stands behind :mod:`roqsim.raycast`, so the
lidar plugins and every other caller of a :class:`~roqsim.raycast.Raycaster` take it unchanged.

What it holds, once per world
-----------------------------

:meth:`MjwarpRaycaster.open` runs when the first sensor that chose it is configured, after the model
compiled, and builds the device scene once: ``mjw.put_model`` (which refuses a model MuJoCo Warp
cannot simulate, and the world then fails to load naming the sensor and the reason), a device
``Data`` that only ever receives geom poses, and a render context with no camera whose scene BVH
spans every geom group the model uses. Nothing is recompiled afterwards.

What it does per cast
---------------------

1. **Brings the device scene up to the state it is handed.** The geom poses (``geom_xpos``,
   ``geom_xmat``, as float32) are compared with those last uploaded; when they differ they are
   uploaded and the BVH is refitted. Every lidar casting in the same tick therefore finds the
   poses current and refits nothing: one upload and one refit per tick, however many lidars share
   the scene. The model fields presence writes at run time (``geom_group``, ``geom_rgba``,
   ``mat_rgba``) are compared and uploaded the same way, so an entity made absent
   (:mod:`roqsim.presence`) is absent on the device from the next cast.
2. **Casts every ray in one launch** over MuJoCo Warp's BVH (:mod:`roqsim_raycast_mjwarp.kernel`,
   ``mjw.rays``' search with a mesh hit from either side, as ``mj_ray`` hits it): the mask,
   ``flg_static`` and the excluded body (per ray) are applied inside the nearest-hit search, as
   ``mj_multiRay`` applies them, and a geom whose resolved alpha is 0 is skipped as there.
   :meth:`cast_many` is the same launch with a ray per (origin, direction).
3. **Reads distances and geom ids back** into page-locked host memory and applies MuJoCo's culling
   beyond ``cutoff``, which ``mjw.rays`` does not do (:func:`roqsim.raycast.cull_beyond_cutoff`; for
   a plane it is roqsim's far-plane rule).

Where it differs from the CPU
-----------------------------

The search runs in float32 against MuJoCo's float64, so distances agree to micrometres, and a ray
can name another geom than the CPU where the answer is not robust to that: a ray grazing an edge, or
a tie between two visible geoms at the same distance. Beyond ``cutoff``,
where the CPU's nearest geom was culled, the CPU may report a farther hit where this reports none;
the lidars drop every hit beyond their range either way. The parity test bounds each kind
(``tests/test_parity.py``).

It fails loudly and never falls back: no MuJoCo Warp installed, no CUDA device, a Warp build the
driver cannot run, or a model ``put_model`` refuses -- :meth:`open` raises with the reason, and the
world does not load.
"""

from __future__ import annotations

import threading

import numpy as np

from roqsim import raycast
from roqsim.raycast import Raycaster, RayHits, Setting

#: What to install when MuJoCo Warp is missing.
INSTALL_HINT = "pip install 'roqsim_raycast_mjwarp[mjwarp]'"


class MjwarpRaycaster(Raycaster):
    """roqsim's ray cast on a CUDA GPU through MuJoCo Warp's ``mjw.rays``, with a BVH."""

    SETTINGS = {
        "device": Setting("cuda:0", "the CUDA device the scene is built on (a Warp device name)"),
    }

    def __init__(self, settings=None) -> None:
        super().__init__(settings)
        self._lock = threading.Lock()
        self._wp = None
        self._mjw = None
        self._kernel = None
        self._device = None
        self._M = None
        self._D = None
        self._rc = None
        self._last: dict[str, np.ndarray] = {}
        self._buffers: dict[int, dict] = {}
        #: Uploads of the geom poses with a BVH refit, and launches: what a bench and a test read.
        self.refits = 0
        self.casts = 0

    @classmethod
    def validate_config(cls, settings) -> list[str]:
        errors = super().validate_config(settings)
        device = (settings or {}).get("device", "cuda:0")
        if not isinstance(device, str) or not device.startswith("cuda"):
            errors.append(f"'device' must name a CUDA device such as 'cuda:0', got {device!r}")
        return errors

    # -- open / close -------------------------------------------------------------------------

    def open(self, model) -> None:
        wp, mjw = _import_warp()
        try:
            wp.init()
        except Exception as err:  # noqa: BLE001 -- Warp raises what its native loader raises
            raise RuntimeError(f"Warp {wp.__version__} cannot initialise: {err}") from err
        if not wp.get_cuda_device_count():
            raise RuntimeError(_no_cuda_reason(wp))
        device = self.settings["device"]
        try:
            self._device = wp.get_device(device)
        except Exception as err:  # noqa: BLE001
            raise RuntimeError(f"no CUDA device {device!r}: {err}") from err
        self._wp, self._mjw = wp, mjw
        from . import kernel  # noqa: PLC0415 -- imports Warp, so only once a backend opens

        self._kernel = kernel
        import mujoco  # noqa: PLC0415

        with wp.ScopedDevice(self._device):
            try:
                self._M = mjw.put_model(model)
            except Exception as err:  # noqa: BLE001 -- put_model raises NotImplementedError & co.
                raise RuntimeError(
                    f"MuJoCo Warp refuses this model (mjw.put_model: {type(err).__name__}: {err})"
                ) from err
            data = mujoco.MjData(model)
            mujoco.mj_kinematics(model, data)
            # The device Data is only ever handed geom poses; its contact and constraint buffers
            # are sized to nothing, since nothing steps it.
            self._D = mjw.put_data(model, data, nconmax=1, njmax=1)
            groups = list(range(max(6, int(model.geom_group.max(initial=0)) + 1)))
            self._rc = mjw.create_render_context(
                model,
                nworld=1,
                cam_active=[],
                render_rgb=False,
                render_depth=False,
                use_textures=False,
                enabled_geom_groups=groups,
            )
        self._last = {}
        self._sync(model, data)

    def close(self) -> None:
        self._buffers.clear()
        self._rc = self._D = self._M = None
        self._last = {}

    # -- the cast -----------------------------------------------------------------------------

    def cast(
        self,
        model,
        data,
        origin,
        dirs,
        *,
        cutoff,
        geomgroup=raycast._UNSET,
        flg_static=True,
        bodyexclude=-1,
        out=None,
    ) -> RayHits:
        d = np.ascontiguousarray(dirs, dtype=np.float64).reshape(-1)
        if d.size % 3:
            raise ValueError(f"raycast: dirs has {d.size} values, not a multiple of 3")
        nray = d.size // 3
        hits = out if out is not None else raycast.buffers(nray)
        if hits.dist.ndim != 1 or hits.nray != nray:
            raise ValueError(f"raycast: out sized for {hits.nray} rays, given {nray} directions")
        o = np.ascontiguousarray(origin, dtype=np.float64).reshape(1, 3)
        self._launch(
            model, data, o, d.reshape(nray, 3), cutoff, geomgroup, flg_static, bodyexclude, hits
        )
        return hits

    def cast_many(
        self,
        model,
        data,
        origins,
        dirs,
        *,
        cutoff,
        geomgroup=raycast._UNSET,
        flg_static=True,
        bodyexclude=-1,
        normals=False,
    ) -> RayHits:
        o = np.ascontiguousarray(origins, dtype=np.float64).reshape(-1, 3)
        d = np.ascontiguousarray(dirs, dtype=np.float64).reshape(-1)
        if d.size % 3:
            raise ValueError(f"raycast: dirs has {d.size} values, not a multiple of 3")
        npts, nray = len(o), d.size // 3
        hits = RayHits(
            dist=np.full((npts, nray), -1.0, dtype=np.float64),
            geomid=np.full((npts, nray), -1, dtype=np.int32),
            normal=np.zeros((npts, nray, 3), dtype=np.float64) if normals else None,
        )
        self._launch(
            model, data, o, d.reshape(nray, 3), cutoff, geomgroup, flg_static, bodyexclude, hits
        )
        return hits

    def _launch(self, model, data, origins, dirs, cutoff, geomgroup, flg_static, bodyexclude, hits):
        """One launch over every (origin, direction); ``hits`` is filled in place."""
        wp = self._wp
        if self._M is None:
            raise RuntimeError("raycaster 'mjwarp' is not open")
        npts, nray = len(origins), len(dirs)
        total = npts * nray
        with self._lock, wp.ScopedDevice(self._device):
            self._sync(model, data)
            buf = self._buffers_for(total)
            pnt = buf["pnt_h"].numpy().reshape(npts, nray, 3)
            pnt[:] = origins[:, None, :]
            vec = buf["vec_h"].numpy().reshape(npts, nray, 3)
            vec[:] = dirs[None, :, :]
            wp.copy(buf["pnt"], buf["pnt_h"])
            wp.copy(buf["vec"], buf["vec_h"])
            if buf["bodyexclude_value"] != bodyexclude:
                buf["bodyexclude"].fill_(int(bodyexclude))
                buf["bodyexclude_value"] = bodyexclude
            self._kernel.launch(
                self._M,
                self._D,
                self._rc,
                buf["pnt"],
                buf["vec"],
                _vec6(geomgroup),
                bool(flg_static),
                buf["bodyexclude"],
                buf["dist"],
                buf["geomid"],
                buf["normal"],
            )
            wp.copy(buf["dist_h"], buf["dist"])
            wp.copy(buf["geomid_h"], buf["geomid"])
            if hits.normal is not None:
                wp.copy(buf["normal_h"], buf["normal"])
            wp.synchronize_device(self._device)
            hits.dist.reshape(-1)[:] = buf["dist_h"].numpy().reshape(-1)
            hits.geomid.reshape(-1)[:] = buf["geomid_h"].numpy().reshape(-1)
            if hits.normal is not None:
                nrm = hits.normal.reshape(-1, 3)
                nrm[:] = buf["normal_h"].numpy().reshape(-1, 3)
                nrm[hits.geomid.reshape(-1) < 0] = 0.0
            self.casts += 1
        raycast.cull_beyond_cutoff(origins, float(cutoff), model, data, hits)

    def _sync(self, model, data) -> None:
        """Upload what changed since the last cast: geom poses (then refit), presence fields."""
        mjw = self._mjw
        xpos = np.asarray(data.geom_xpos, dtype=np.float32)
        xmat = np.asarray(data.geom_xmat, dtype=np.float32)
        if not (_same(self._last.get("xpos"), xpos) and _same(self._last.get("xmat"), xmat)):
            self._D.geom_xpos.assign(xpos.reshape(1, -1, 3))
            self._D.geom_xmat.assign(xmat.reshape(1, -1, 3, 3))
            mjw.refit_bvh(self._M, self._D, self._rc)
            self._last["xpos"], self._last["xmat"] = xpos, xmat
            self.refits += 1
        group = np.asarray(model.geom_group, dtype=np.int32)
        if not _same(self._last.get("group"), group):
            self._M.geom_group.assign(group)
            self._last["group"] = group.copy()
        for key, field in (("geom_rgba", model.geom_rgba), ("mat_rgba", model.mat_rgba)):
            if field.size == 0:
                continue
            rgba = np.asarray(field, dtype=np.float32)
            if not _same(self._last.get(key), rgba):
                target = getattr(self._M, key)
                target.assign(rgba.reshape(target.shape[0], -1, 4))
                self._last[key] = rgba.copy()

    def _buffers_for(self, total: int) -> dict:
        buf = self._buffers.get(total)
        if buf is None:
            wp = self._wp
            dev = self._device
            buf = {
                "pnt": wp.empty((1, total), dtype=wp.vec3, device=dev),
                "vec": wp.empty((1, total), dtype=wp.vec3, device=dev),
                "dist": wp.empty((1, total), dtype=float, device=dev),
                "geomid": wp.empty((1, total), dtype=int, device=dev),
                "normal": wp.empty((1, total), dtype=wp.vec3, device=dev),
                "bodyexclude": wp.full(total, -1, dtype=int, device=dev),
                "bodyexclude_value": -1,
                "pnt_h": wp.empty((1, total), dtype=wp.vec3, device="cpu", pinned=True),
                "vec_h": wp.empty((1, total), dtype=wp.vec3, device="cpu", pinned=True),
                "dist_h": wp.empty((1, total), dtype=float, device="cpu", pinned=True),
                "geomid_h": wp.empty((1, total), dtype=int, device="cpu", pinned=True),
                "normal_h": wp.empty((1, total), dtype=wp.vec3, device="cpu", pinned=True),
            }
            self._buffers[total] = buf
        return buf


def _same(a, b) -> bool:
    return a is not None and a.shape == b.shape and np.array_equal(a, b)


def _vec6(geomgroup):
    from mujoco_warp._src.types import vec6  # noqa: PLC0415 -- MuJoCo Warp's own mask type

    mask = raycast.VISIBLE_GROUPS if geomgroup is raycast._UNSET else geomgroup
    if mask is None:
        return vec6(-1, -1, -1, -1, -1, -1)
    return vec6(*[int(v) for v in np.asarray(mask).reshape(6)])


def _import_warp():
    try:
        import warp as wp  # noqa: PLC0415

        # Before MuJoCo Warp, whose import initialises Warp: a world's log is not Warp's banner.
        wp.config.log_level = wp.LOG_WARNING
        import mujoco_warp as mjw  # noqa: PLC0415
    except ImportError as err:
        raise RuntimeError(f"MuJoCo Warp is not installed ({err}); {INSTALL_HINT}") from err
    return wp, mjw


def _no_cuda_reason(wp) -> str:
    """Why Warp sees no CUDA device, in the terms a fix needs: its build and the driver."""

    def ask(fn):
        try:
            return getattr(wp, fn)()
        except Exception:  # noqa: BLE001 -- a host without a driver answers by raising
            return None

    def ver(v):
        return ".".join(str(x) for x in v) if isinstance(v, (tuple, list)) else str(v)

    toolkit, driver = ask("get_cuda_toolkit_version"), ask("get_cuda_driver_version")
    built = f"Warp {wp.__version__}, built for CUDA {ver(toolkit)}"
    if not driver:
        return f"no CUDA device: {built}, finds no CUDA device or driver on this host"
    if toolkit and tuple(driver)[:1] < tuple(toolkit)[:1]:
        return (
            f"no CUDA device: {built}, cannot run on this host's driver, which supports CUDA "
            f"{ver(driver)}; install the Warp build that matches the driver (Warp publishes a "
            f"CUDA 12 build of each release on its release page)"
        )
    return f"no CUDA device: {built}, sees none beside a driver for CUDA {ver(driver)}"
