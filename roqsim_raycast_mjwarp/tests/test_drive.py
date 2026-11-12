# Copyright (C) 2026 Frederik Pasch
# SPDX-License-Identifier: Apache-2.0
"""A Jackal driving through the depot with a VLP-16 and a Mid-360: the GPU run's clouds are the CPU's.

The same world twice, stepped in lockstep with the Mid-360's range noise on: once with every lidar
on the CPU, once as a user writes it, with no raycaster named, so ``auto`` puts the Mid-360 on the
GPU and the planar VLP-16 scan on the CPU. The noise is drawn after the cast, per ray, from
``ctx.rng_for`` on either backend, so a point differs only where its own ray hit differently. Each
frame is compared point by point, and each point that differs is a ray
:func:`test_parity.classify` explains.
"""

from __future__ import annotations

import _worlds
import numpy as np
import pytest
from test_parity import classify

#: Seconds of driving: 160 frames of each lidar at 10 Hz, long enough for the circle to carry the
#: Mid-360 through one of the depot's visual-only pillars, where a mesh is seen from inside.
TRIAL_S = 16.0
#: The most points of one Mid-360 frame (20,160 rays) that may differ from the CPU's.
MAX_POINTS_PER_FRAME = 6


@pytest.fixture(scope="module")
def runs(gpu):
    cpu = _worlds.build(_worlds.drive_world(_worlds.depot_world(), "cpu"))
    # The world as a user writes it: no raycaster named, so auto chooses per lidar.
    gpu_run = _worlds.build(_worlds.drive_world(_worlds.depot_world(), None))
    yield cpu, gpu_run
    cpu.shutdown()
    gpu_run.shutdown()


def test_auto_puts_the_3d_lidar_on_the_gpu_and_the_planar_scan_on_the_cpu(runs):
    from roqsim import raycast
    from roqsim_raycast_mjwarp.backend import MjwarpRaycaster

    _cpu, gpu = runs
    velodyne, mid360 = _worlds.lidar_plugins(gpu)
    assert isinstance(velodyne._raycaster, raycast.CpuRaycaster)  # 898 rays
    assert isinstance(mid360._raycaster, MjwarpRaycaster)  # 20,160 rays
    chosen = {c["source"]: c["details"] for c in gpu.ctx.conditions if c["code"] == "raycaster"}
    assert chosen[velodyne.address]["backend"] == "cpu"
    assert "fewer than the" in chosen[velodyne.address]["reason"]
    assert chosen[mid360.address]["backend"] == "mjwarp"


def test_the_gpu_drive_differs_from_the_cpus_only_on_the_rays_parity_bounds(runs):
    """Point by point, with the noise on: a point differs only where its ray hit differently."""
    cpu, gpu = runs
    lc, lg = _worlds.lidar_plugins(cpu), _worlds.lidar_plugins(gpu)
    backend = lg[1]._raycaster
    refits0 = backend.refits
    last = [None] * len(lc)
    differing = {p.address: [] for p in lc}
    ticks = 0
    for _ in range(int(TRIAL_S / cpu.ctx.dt)):
        cpu.step()
        gpu.step()
        cast = False
        for k, (a, b) in enumerate(zip(lc, lg, strict=True)):
            if a.latest is None or a.latest is last[k]:
                continue
            cast = cast or k == 1
            last[k] = a.latest
            pa, pb = _payload(a.latest), _payload(b.latest)
            assert pa.shape == pb.shape, a.address
            bad = np.nonzero(np.abs(pa - pb).reshape(len(pa), -1).max(axis=1) > 1e-4)[0]
            differing[a.address].append(len(bad))
            if len(bad):
                got = classify(gpu, b, a._hits, b._hits, *_rays(gpu, b))
                assert not got["kinds"]["unexplained"], (a.address, got["kinds"])
                explained = {r["ray"] for rays in got["kinds"].values() for r in rays}
                # Noise is drawn per ray, so no other point may move.
                assert set(bad.tolist()) <= explained, (a.address, bad, explained)
        ticks += cast
    # The robot drove the same path: a lidar changes nothing it reads.
    assert np.array_equal(cpu.ctx.data.qpos, gpu.ctx.data.qpos)
    assert ticks >= 150
    # One upload and refit per tick in which the GPU's lidar cast.
    assert backend.refits - refits0 == ticks
    velodyne, mid360 = (p.address for p in lc)
    assert max(differing[velodyne]) == 0
    assert max(differing[mid360]) <= MAX_POINTS_PER_FRAME, differing[mid360]


def _payload(value) -> np.ndarray:
    points = getattr(value, "points", None)
    return np.asarray(points if points is not None else value.ranges, dtype=np.float64)


def _rays(engine, plugin):
    d = engine.ctx.data
    origin = np.array(d.site_xpos[plugin._site_id])
    dirs = plugin._local_dirs @ np.array(d.site_xmat[plugin._site_id]).reshape(3, 3).T
    return origin, dirs
