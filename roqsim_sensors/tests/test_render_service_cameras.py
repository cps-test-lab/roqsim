# Copyright (C) 2026 Frederik Pasch
# SPDX-License-Identifier: Apache-2.0
"""A world's cameras draw through one render service: what that costs, and that it changes no frame.

What scales is tested here with the cameras themselves: one GL context for any number of cameras,
nothing drawn for a camera nobody reads, every frame stamped with the
sim time of the state it was drawn from, and a Spot body camera's three streams as they were drawn
before the service, pixel for pixel.
"""

from __future__ import annotations

import gc
import math

import mujoco
import numpy as np
import pytest

from roqsim.config import load_config_from_dict
from roqsim.engine import Engine
from roqsim.plugin import Plugin
from roqsim.rendering import FrameRenderer
from roqsim_sensors.plugins.camera_common import encode_color


def _ring(n: int, *, on_demand: bool = True, sim: dict | None = None) -> list[dict]:
    out = []
    for i in range(n):
        a = 2 * math.pi * i / n
        x, y = 2.5 * math.cos(a), 2.5 * math.sin(a)
        out.append(
            {
                "spawn_sensor": {
                    "model": "realsense_d415",
                    "prefix": f"cam{i}_",
                    "namespace": f"cam{i}",
                    "pose": {
                        "position": {"x": x, "y": y, "z": 0.6},
                        "orientation": {"yaw": math.atan2(-y, -x)},
                    },
                },
                "name": f"cam{i}",
                "components": [{"realsense_d415": {"on_demand": on_demand, "rate_hz": 50}}],
            }
        )
    return out


class _Props(Plugin):
    """Something in the middle of the room for the cameras to see, a see-through pane included.

    The world draws without MSAA: under MuJoCo's default 4x MSAA a frame is not bit-reproducible on
    every GPU, before the render service as after it (see ``test_render_service_pixels.py``), and
    these tests compare frames exactly.
    """

    def build(self, spec: mujoco.MjSpec, ctx) -> None:
        spec.visual.quality.offsamples = 0
        spec.worldbody.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[0, 0, 0.3],
            size=[0.3, 0.3, 0.3],
            rgba=[0.9, 0.3, 0.2, 1],
        )
        spec.worldbody.add_geom(
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            pos=[0.5, 0.5, 0.4],
            size=[0.2, 0, 0],
            rgba=[0.2, 0.8, 0.3, 1],
        )
        spec.worldbody.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[-0.6, 0, 0.5],
            size=[0.02, 0.5, 0.5],
            rgba=[0.6, 0.8, 1, 0.3],
        )


def _props() -> list[dict]:
    return [{f"{__name__}:_Props": {}}]


def _engine(components: list[dict], sim: dict | None = None) -> Engine:
    cfg = load_config_from_dict(
        {"sim": {"timestep": 0.002, **(sim or {})}, "components": components}
    )
    engine = Engine(cfg)
    engine.ctx.seed = 1
    try:
        engine.setup()
    except Exception as err:  # noqa: BLE001 -- the GL backend check raises its own type
        if "GL" in str(err):
            pytest.skip(f"no usable offscreen GL here: {err}")
        raise
    engine.reset()
    return engine


def _images(engine: Engine, which=None) -> list:
    return [
        ep
        for ep in engine.ctx.interface.all()
        if ep.name == "image" and (which is None or ep.producer.split(".")[0] in which)
    ]


def _gl_contexts() -> int:
    from mujoco.rendering.classic import gl_context

    gc.collect()
    cls = gl_context.GLContext
    return sum(1 for o in gc.get_objects() if cls is not None and isinstance(o, cls))


def test_one_gl_context_for_eight_cameras():
    before = _gl_contexts()
    engine = _engine([*_props(), *_ring(8)])
    for ep in _images(engine):
        ep.claim()
    for _ in range(50):
        engine.step()
    service = engine.ctx.render
    renderer = service.renderer()
    assert len(service.views) == 8
    assert _gl_contexts() - before == 1
    ticks = renderer.stats["batches"]
    assert ticks == 5  # 50 Hz over 0.1 s of 2 ms steps
    assert renderer.stats["draws"] == 8 * ticks  # every due camera, in one context
    engine.shutdown()
    assert _gl_contexts() - before == 0


def test_a_camera_nobody_reads_is_in_no_batch_and_opens_no_context():
    engine = _engine([*_props(), *_ring(3)])
    for _ in range(20):
        engine.step()
    assert engine.ctx.render.renderer() is None  # no batch ever came
    (wanted,) = _images(engine, {"cam1"})
    wanted.claim()
    for _ in range(20):
        engine.step()
    renderer = engine.ctx.render.renderer()
    assert renderer.stats["draws"] == renderer.stats["batches"] == 2  # cam1 only, twice
    views = {v.id.split("/")[0]: v for v in engine.ctx.render.views}
    assert views["cam1.realsense_d415"].frame is not None
    assert views["cam0.realsense_d415"].frame is None and views["cam2.realsense_d415"].frame is None
    engine.shutdown()


def test_every_frame_carries_its_snapshots_sim_time():
    engine = _engine([*_props(), *_ring(2)])
    for ep in _images(engine):
        ep.claim()
    stamps = []
    for _ in range(30):
        engine.step()
        for view in engine.ctx.render.views:
            frame = view.frame
            if frame is not None and frame.sim_time == engine.ctx.sim_time:
                stamps.append(frame.sim_time)
    assert stamps  # synchronous: a due tick's frame is drawn from that tick's state
    for ep in _images(engine):
        image = ep.read()
        assert image.stamp == engine.ctx.render.views[0].frame.sim_time
    engine.shutdown()


def _spot_station():
    return {
        "spawn_sensor": {
            "model": "spot_body_camera",
            "device_name": "frontleft",
            "pose": {"position": {"x": -1.2, "z": 0.5}},
        },
        "name": "station",
        "components": [{"spot_body_camera": {"registered_depth": True}}],
    }


def test_a_spot_station_draws_what_its_two_renderers_drew_before():
    """Fisheye colour (mono8), depth through the depth camera, registered depth through the colour
    one -- each equal to a renderer of its own drawing the same state with the same planes."""
    engine = _engine([*_props(), _spot_station()])
    eps = {ep.name: ep for ep in engine.ctx.interface.all() if ep.producer.startswith("station")}
    for name in ("image", "depth", "registered_depth"):
        eps[name].claim()
    for _ in range(80):  # two frames at 15 Hz: compare the second (see test_render_service_pixels)
        engine.step()
    from roqsim_sensors.plugins.spot_body_camera import SpotBodyCameraPlugin

    plugin = next(p for p in engine.plugins if isinstance(p, SpotBodyCameraPlugin))
    m, d = engine.ctx.model, engine.ctx.data
    stamp = eps["image"].read().stamp
    assert stamp == plugin._views["color"].frame.sim_time
    # The state the second frame was drawn from: step back to it.
    replay = _engine([*_props(), _spot_station()])
    while replay.ctx.sim_time < stamp - 1e-9:
        replay.step()
    near, far = plugin._planes
    fb = (
        max(int(m.vis.global_.offwidth), plugin._intr.width, plugin._depth_intr.width),
        max(int(m.vis.global_.offheight), plugin._intr.height, plugin._depth_intr.height),
    )
    got_rgb = eps["image"].read().data
    got_depth = plugin._depth.copy()
    got_registered = plugin._registered.copy()
    old_colour = _Before(replay.ctx.model, plugin._intr, plugin._cam_id, near, far, fb)
    old_depth = _Before(replay.ctx.model, plugin._depth_intr, plugin._depth_cam_id, near, far, fb)
    rd = replay.ctx.data
    try:
        np.testing.assert_array_equal(got_rgb, encode_color(old_colour.rgb(rd), "mono8"))
        np.testing.assert_array_equal(got_depth, plugin._clipped(old_depth.depth(rd))[0])
        np.testing.assert_array_equal(got_registered, plugin._clipped(old_colour.depth(rd))[0])
    finally:
        old_colour.close()
        old_depth.close()
        replay.shutdown()
        engine.shutdown()


class _Before:
    """A renderer of its own per camera, as every camera had before the service, in the framebuffer
    the service draws in, each pass drawn twice. It is a FrameRenderer, which
    ``roqsim/tests/test_render_service_pixels.py`` holds to the path before the service pixel for
    pixel; what this adds is that a camera's views drawn in one batch change nothing."""

    def __init__(self, model, intr, cam_id, near, far, fb):
        import copy

        self.model = copy.copy(model)
        self.model.vis.global_.offwidth, self.model.vis.global_.offheight = fb
        self.fr = FrameRenderer(
            self.model, intr.width, intr.height, camera=cam_id, near=near, far=far
        )

    def rgb(self, data):
        self.fr.render(data)
        return self.fr.render(data)

    def depth(self, data):
        self.fr.render_depth(data)
        return self.fr.render_depth(data)

    def close(self):
        self.fr.close()


def _run(ticks: int = 60) -> dict[float, bytes]:
    """Every published colour frame of two noisy cameras, by its stamp, over *ticks* steps."""
    ring = _ring(2)
    for entry in ring:
        entry["components"][0]["realsense_d415"]["noise"] = {"gaussian_sigma": 4.0}
    engine = _engine([*_props(), *ring])
    for ep in _images(engine):
        ep.claim()
    out: dict[float, bytes] = {}
    try:
        for _ in range(ticks):
            engine.step()
            for ep in _images(engine):
                image = ep.read()
                if image is not None:
                    out[(ep.producer, round(image.stamp, 9))] = image.data.tobytes()
    finally:
        engine.shutdown()
    return out


def test_a_run_is_reproduced_byte_for_byte():
    """A picture is a function of the state and the seed: two runs agree, noise included."""
    first, second = _run(), _run()
    assert first and first == second
