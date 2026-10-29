# Copyright (C) 2026 Frederik Pasch
# SPDX-License-Identifier: Apache-2.0
"""More than one worker: a renderer's cameras dealt to several instances, with no camera knowing.

``sim.render.workers: N`` gives each renderer N instances -- N contexts, and in ``async`` mode N
threads -- and deals its cameras to them in turn. GL-free: the renderer records which instance drew
which camera.
"""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from roqsim.config import load_config_from_dict
from roqsim.engine import Engine
from roqsim.plugin import Plugin, PluginError
from roqsim.render_api import Frame, Renderer, RendererCapabilities

#: (instance id, thread name, camera id) for every frame drawn, and each instance's cameras.
DRAWN: list[tuple[int, str, str]] = []
OPENED: dict[int, list[str]] = {}


class Recorder(Renderer):
    capabilities = RendererCapabilities(images=frozenset({"rgb"}), batched=True)

    def open(self, scene):
        OPENED[id(self)] = [c.id for c in scene.cameras]

    def render(self, snapshot, requests):
        thread = threading.current_thread().name
        out = []
        for r in requests:
            DRAWN.append((id(self), thread, r.id))
            out.append(
                Frame(
                    r.id,
                    snapshot.sim_time,
                    snapshot.step,
                    rgb=np.zeros((r.camera.height, r.camera.width, 3), np.uint8),
                )
            )
        return out


class Cams(Plugin):
    def build(self, spec, ctx):
        for i in range(4):
            spec.worldbody.add_camera(name=f"c{i}", pos=[1, i, 0.2])

    def configure(self, ctx):
        self.views = [
            ctx.render.view(f"cam{i}", camera=f"c{i}", width=4, height=3, wants=lambda: ("rgb",))
            for i in range(4)
        ]


def _engine(mode: str, workers: int) -> Engine:
    DRAWN.clear()
    OPENED.clear()
    cfg = load_config_from_dict(
        {
            "sim": {
                "renderer": f"{__name__}:Recorder",
                "render": {"mode": mode, "workers": workers},
            },
            "components": [{f"{__name__}:Cams": {}}],
        }
    )
    engine = Engine(cfg)
    engine.ctx.seed = 1
    engine.setup()
    engine.reset()
    return engine


@pytest.mark.parametrize("mode", ["sync", "async"])
def test_two_workers_share_four_cameras(mode):
    engine = _engine(mode, 2)
    try:
        for _ in range(3):
            engine.step()
            if mode == "async":
                time.sleep(0.02)
        deadline = time.monotonic() + 30
        while any(v.frame is None for v in engine.plugins[0].views):
            assert time.monotonic() < deadline
            time.sleep(0.01)
    finally:
        engine.shutdown()
    assert sorted(OPENED.values()) == [["cam0", "cam2"], ["cam1", "cam3"]]
    by_instance: dict[int, set[str]] = {}
    for instance, _, cam in DRAWN:
        by_instance.setdefault(instance, set()).add(cam)
    assert sorted(map(sorted, by_instance.values())) == [["cam0", "cam2"], ["cam1", "cam3"]]
    threads = {thread for _, thread, _ in DRAWN}
    if mode == "async":
        assert threads == {f"roqsim-render-{__name__}:Recorder-{i}" for i in (0, 1)}
    else:
        assert threads == {threading.main_thread().name}


def test_one_worker_is_the_default():
    engine = _engine("sync", 1)
    try:
        engine.step()
    finally:
        engine.shutdown()
    assert len(OPENED) == 1


def test_workers_is_a_whole_number_of_at_least_one():
    for bad in (0, -1, 1.5, True):
        with pytest.raises(PluginError, match="sim.render.workers"):
            load_config_from_dict({"sim": {"render": {"workers": bad}}, "components": []})
