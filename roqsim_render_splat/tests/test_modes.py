# Copyright (C) 2026 Frederik Pasch
# SPDX-License-Identifier: Apache-2.0
"""The splat renderer behind a running world, in ``sync`` and in ``async``.

A picture is a function of the state and the seed: two ``sync`` runs agree byte for byte, and every
frame ``async`` publishes is the one ``sync`` drew for the same stamp. The world is the example
room with two RealSense cameras and a crate that falls through their view, so the moving body is
drawn into the room by depth in every frame. Needs a CUDA GPU.
"""

from __future__ import annotations

import math
import shutil
import time
from pathlib import Path

import mujoco

import roqsim_render_splat
from roqsim.config import load_config_from_dict
from roqsim.engine import Engine
from roqsim.plugin import Plugin

EXAMPLE = Path(roqsim_render_splat.WORLDS_DIR) / "splat_room_example"


class _Crate(Plugin):
    """A crate that falls onto the floor in the cameras' view; the world draws without MSAA."""

    def build(self, spec: mujoco.MjSpec, ctx) -> None:
        spec.visual.quality.offsamples = 0
        body = spec.worldbody.add_body(name="crate", pos=[0.0, 0.0, 1.2])
        body.add_freejoint()
        body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.2, 0.2, 0.2], rgba=[0.2, 0.8, 0.3, 1])


def _cameras(n: int) -> list[dict]:
    out = []
    for i in range(n):
        a = 2 * math.pi * i / n + 0.3
        x, y = 1.2 * math.cos(a), 1.0 * math.sin(a)
        out.append(
            {
                "spawn_sensor": {
                    "model": "realsense_d415",
                    "prefix": f"cam{i}_",
                    "namespace": f"cam{i}",
                    "pose": {
                        "position": {"x": x, "y": y, "z": 0.5},
                        "orientation": {"yaw": math.atan2(-y, -x)},
                    },
                },
                "name": f"cam{i}",
                "components": [{"realsense_d415": {"on_demand": True, "rate_hz": 50}}],
            }
        )
    return out


def _engine(tmp_path: Path, mode: str) -> Engine:
    for name in ("room.ply", "collision.xml"):
        shutil.copy(EXAMPLE / name, tmp_path / name)
    world = {
        "sim": {
            "timestep": 0.002,
            "world": "collision.xml",
            "renderer": "splat",
            "render": {"mode": mode},
            "renderers": {
                "splat": {
                    "splat": {
                        "file": "room.ply",
                        "pose": {
                            "position": {"x": 0.0, "y": 0.0, "z": 1.0},
                            "orientation": {"roll": -1.5708, "pitch": 0.0, "yaw": 0.35},
                        },
                        "scale": 0.5,
                    }
                }
            },
        },
        "components": [{f"{__name__}:_Crate": {}}, *_cameras(2)],
    }
    engine = Engine(load_config_from_dict(world, base_dir=tmp_path))
    engine.ctx.seed = 1
    engine.setup()
    engine.reset()
    return engine


def _images(engine):
    return [ep for ep in engine.ctx.interface.all() if ep.name == "image"]


def _run(tmp_path: Path, mode: str, ticks: int = 60) -> dict:
    engine = _engine(tmp_path, mode)
    for ep in _images(engine):
        ep.claim()
    out: dict = {}
    deadline = time.monotonic() + 60.0
    try:
        for i in range(ticks * 50):
            if i >= ticks and (mode == "sync" or len(out) >= 4 or time.monotonic() > deadline):
                break
            engine.step()
            if mode == "async":
                time.sleep(0.004)
            for ep in _images(engine):
                image = ep.read()
                if image is not None:
                    out[(ep.producer, round(image.stamp, 9))] = image.data.tobytes()
    finally:
        engine.shutdown()
    return out


def test_sync_is_deterministic_and_an_async_frame_is_the_sync_frame_of_its_stamp(gpu, tmp_path):
    first = _run(tmp_path, "sync")
    second = _run(tmp_path, "sync")
    assert first and first == second
    assert len(set(first.values())) > 2  # the crate falls through the frames
    late = _run(tmp_path, "async")
    assert late
    assert set(late) <= set(first)
    assert all(first[key] == pixels for key, pixels in late.items())
