# Copyright (C) 2026 Frederik Pasch
# SPDX-License-Identifier: Apache-2.0
"""Where the backend cannot run, the world does not load, and says which lidar, which backend, why.

These run on any host: Warp is replaced by a stand-in that reports what a host without the GPU, the
driver or the model support would.
"""

from __future__ import annotations

import types

import _worlds
import pytest

from roqsim.plugin import PluginError
from roqsim_raycast_mjwarp import backend


def _fake_warp(cuda_devices: int, put_model=None):
    wp = types.SimpleNamespace(
        __version__="1.19.0",
        init=lambda: None,
        get_cuda_device_count=lambda: cuda_devices,
        get_device=lambda name: name,
        ScopedDevice=lambda dev: _Null(),
        get_cuda_toolkit_version=lambda: (13, 4),
        get_cuda_driver_version=lambda: (12, 2),
    )

    def refuse(model):
        raise NotImplementedError("flex and height field in one model")

    mjw = types.SimpleNamespace(put_model=put_model or refuse)
    return wp, mjw


class _Null:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _load(world: dict):
    return _worlds.build(world)


ONE_LIDAR = (_worlds.LIDARS[2],)


def test_no_mujoco_warp_installed_fails_naming_the_lidar_backend_and_install(monkeypatch):
    def missing():
        raise RuntimeError(
            f"MuJoCo Warp is not installed (No module named 'warp'); {backend.INSTALL_HINT}"
        )

    monkeypatch.setattr(backend, "_import_warp", missing)
    with pytest.raises(
        PluginError,
        match=r"mid360\.livox_mid360: raycaster 'mjwarp' cannot run here: MuJoCo Warp is not installed.*roqsim_raycast_mjwarp\[mjwarp\]",
    ):
        _load(_worlds.lidar_world(None, "mjwarp", ONE_LIDAR))


def test_no_cuda_device_fails_naming_warps_build_and_the_driver(monkeypatch):
    monkeypatch.setattr(backend, "_import_warp", lambda: _fake_warp(0))
    with pytest.raises(
        PluginError,
        match=r"raycaster 'mjwarp' cannot run here: no CUDA device: Warp 1\.19\.0, built for CUDA 13\.4, cannot run on this host's driver, which supports CUDA 12\.2",
    ):
        _load(_worlds.lidar_world(None, "mjwarp", ONE_LIDAR))


def test_a_model_put_model_refuses_fails_with_its_reason(monkeypatch):
    monkeypatch.setattr(backend, "_import_warp", lambda: _fake_warp(1))
    with pytest.raises(
        PluginError,
        match=r"MuJoCo Warp refuses this model \(mjw\.put_model: NotImplementedError: flex and height field",
    ):
        _load(_worlds.lidar_world(None, "mjwarp", ONE_LIDAR))


def test_the_world_level_choice_reaches_every_lidar(monkeypatch):
    monkeypatch.setattr(backend, "_import_warp", lambda: _fake_warp(0))
    doc = _worlds.lidar_world(None, None, ONE_LIDAR)
    doc["sim"]["raycaster"] = "mjwarp"
    with pytest.raises(PluginError, match="raycaster 'mjwarp' cannot run here"):
        _load(doc)


def test_settings_are_checked_at_load():
    from roqsim.config import validate_raycast

    validate_raycast({"raycaster": "mjwarp", "raycasters": {"mjwarp": {"device": "cuda:1"}}}, None)
    with pytest.raises(
        PluginError, match="sim.raycasters.mjwarp: 'device' must name a CUDA device"
    ):
        validate_raycast({"raycasters": {"mjwarp": {"device": "cpu"}}}, None)
    with pytest.raises(PluginError, match="sim.raycasters.mjwarp: unknown key"):
        validate_raycast({"raycasters": {"mjwarp": {"bvh": False}}}, None)
