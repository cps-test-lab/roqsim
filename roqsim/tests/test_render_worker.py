# Copyright (C) 2026 Frederik Pasch
# SPDX-License-Identifier: Apache-2.0
"""The render service's two modes: ``sync`` waits each tick for its frames, ``async`` renders on a
worker thread from a snapshot while physics steps on, newest snapshot first, never queueing.

GL-free: the renderers here are stand-ins that record what they are handed and how long they take.
"""

from __future__ import annotations

import threading
import time

import mujoco
import numpy as np
import pytest

from roqsim.config import load_config_from_dict
from roqsim.engine import Engine
from roqsim.plugin import Plugin, PluginError
from roqsim.render_api import Frame, RenderError, Renderer, RendererCapabilities

#: What the stand-ins saw, by renderer class: (thread, snapshot step, id of the data they read).
SEEN: dict[str, list] = {}


class Slow(Renderer):
    """Takes ``delay`` per batch; each frame's pixel is its snapshot's step."""

    capabilities = RendererCapabilities(images=frozenset({"rgb"}), deterministic=True)
    delay = 0.03

    def render(self, snapshot, requests):
        SEEN.setdefault(type(self).__name__, []).append(
            (threading.current_thread().name, snapshot.step, id(snapshot.data))
        )
        time.sleep(self.delay)
        return [
            Frame(
                r.id,
                snapshot.sim_time,
                snapshot.step,
                rgb=np.full((r.camera.height, r.camera.width, 3), snapshot.step % 256, np.uint8),
            )
            for r in requests
        ]


class Quick(Slow):
    delay = 0.0


class Boom(Slow):
    """Fails at its third batch."""

    delay = 0.0

    def render(self, snapshot, requests):
        if len(SEEN.get("Boom", [])) >= 2:
            raise RuntimeError("the GPU went away")
        return super().render(snapshot, requests)


class Cam(Plugin):
    """One model camera, due every tick, as a camera plugin registers its view."""

    def build(self, spec, ctx):
        spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.1, 0, 0])
        spec.worldbody.add_camera(name="c", pos=[1, 0, 0.2])

    def configure(self, ctx):
        self.view = ctx.render.view("cam", camera="c", width=4, height=3, wants=lambda: ("rgb",))


def _settle(service, ticks: int, timeout: float = 30.0) -> None:
    """Wait until the worker has drawn or dropped every one of *ticks* batches."""
    deadline = time.monotonic() + timeout
    while service.stats["batches"] + service.stats["dropped"] < ticks:
        assert time.monotonic() < deadline, f"the worker settled {service.stats} of {ticks}"
        time.sleep(0.01)


def _engine(renderer: str, mode: str) -> Engine:
    SEEN.clear()
    cfg = load_config_from_dict(
        {
            "sim": {
                "timestep": 0.002,
                "renderer": f"{__name__}:{renderer}",
                "render": {"mode": mode},
            },
            "components": [{f"{__name__}:Cam": {}}],
        }
    )
    engine = Engine(cfg)
    engine.ctx.seed = 1
    engine.setup()
    engine.reset()
    return engine


def test_sync_waits_for_this_ticks_frame():
    engine = _engine("Slow", "sync")
    try:
        for _ in range(3):
            engine.step()
            frame = engine.plugins[0].view.frame
            assert frame.sim_time == engine.ctx.sim_time  # the tick's own, waited for
    finally:
        engine.shutdown()
    assert {thread for thread, _, _ in SEEN["Slow"]} == {threading.main_thread().name}


def test_async_steps_on_renders_the_newest_snapshot_and_drops_the_stale():
    engine = _engine("Slow", "async")
    try:
        t0 = time.perf_counter()
        for _ in range(40):
            engine.step()
        stepping = time.perf_counter() - t0
        service = engine.ctx.render
        _settle(service, 40)
        frame = engine.plugins[0].view.frame
    finally:
        engine.shutdown()
    seen = SEEN["Slow"]
    # Physics did not wait: 40 ticks took far less than 40 renders.
    assert stepping < 40 * Slow.delay / 2
    # Off the physics thread, in order, and most snapshots never drawn: newest wins, nothing queues.
    assert all(thread.startswith("roqsim-render-") for thread, _, _ in seen)
    steps = [step for _, step, _ in seen]
    assert steps == sorted(steps) and len(steps) < 40
    assert service.stats["dropped"] > 0
    assert service.stats["dropped"] + len(steps) == 40
    # The last frame is the last snapshot's, stamped with its own time.
    assert frame.step == 40 and frame.sim_time == pytest.approx(40 * 0.002)


def test_an_async_frame_is_stamped_with_its_snapshot_not_the_tick_that_reads_it():
    engine = _engine("Slow", "async")
    try:
        for _ in range(30):
            engine.step()
            frame = engine.plugins[0].view.frame
            if frame is not None:
                assert frame.sim_time <= engine.ctx.sim_time
                assert int(frame.rgb[0, 0, 0]) == frame.step % 256
    finally:
        engine.shutdown()


@pytest.mark.parametrize("mode", ["sync", "async"])
def test_a_renderer_never_reads_the_live_data(mode):
    engine = _engine("Quick", mode)
    try:
        for _ in range(5):
            engine.step()
        _settle(engine.ctx.render, 5)
        live = id(engine.ctx.data)
    finally:
        engine.shutdown()
    assert SEEN["Quick"] and all(data_id != live for _, _, data_id in SEEN["Quick"])


def test_a_worker_that_died_fails_the_next_tick_loudly():
    engine = _engine("Boom", "async")
    try:
        with pytest.raises(RenderError, match="the GPU went away"):
            for _ in range(50):
                engine.step()
                time.sleep(0.005)
    finally:
        engine.shutdown()


def test_lockstep_cannot_render_async():
    with pytest.raises(PluginError, match="sim.sync.enabled"):
        load_config_from_dict(
            {"sim": {"render": {"mode": "async"}, "sync": {"enabled": True}}, "components": []}
        )
