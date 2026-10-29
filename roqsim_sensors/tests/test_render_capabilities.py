# Copyright (C) 2026 Frederik Pasch
# SPDX-License-Identifier: Apache-2.0
"""A camera draws through the renderer it names, and one that cannot give what it asks for is
refused when the world loads -- naming the camera, the renderer, what was asked and what it gives --
never drawn by MuJoCo's renderer instead."""

from __future__ import annotations

import numpy as np
import pytest

from roqsim.config import load_config_from_dict
from roqsim.engine import Engine
from roqsim.plugin import PluginError
from roqsim.render_api import Frame, Renderer, RendererCapabilities

#: The colour a Flat renderer fills every view with.
FLAT = (12, 34, 56)


class Flat(Renderer):
    """Colour only: a solid fill per view, stamped with the snapshot."""

    capabilities = RendererCapabilities(
        images=frozenset({"rgb"}),
        camera_models=frozenset({"pinhole", "free"}),
        deterministic=True,
    )

    def render(self, snapshot, requests):
        out = []
        for r in requests:
            rgb = np.empty((r.camera.height, r.camera.width, 3), np.uint8)
            rgb[:] = FLAT
            out.append(Frame(r.id, snapshot.sim_time, snapshot.step, rgb=rgb))
        return out


FLAT_REF = f"{__name__}:Flat"
D435 = "roqsim_sensors.plugins.realsense_d435:RealsenseD435Plugin"


def _camera(name: str, x: float, **config) -> dict:
    return {
        "spawn_sensor": {
            "model": "realsense_d435",
            "prefix": f"{name}_",
            "namespace": name,
            "pose": {"position": {"x": x, "z": 0.5}, "orientation": {"yaw": 3.14159}},
        },
        "name": name,
        "components": [{"realsense_d435": config}],
    }


def _engine(components, sim=None) -> Engine:
    engine = Engine(load_config_from_dict({"sim": dict(sim or {}), "components": components}))
    engine.ctx.seed = 1
    return engine


def test_a_renderer_without_depth_is_refused_for_a_depth_camera_at_load():
    engine = _engine([_camera("cam", 1.5, depth=True, renderer=FLAT_REF)])
    with pytest.raises(PluginError) as err:
        engine.setup()
    text = str(err.value)
    assert "cam.realsense_d435" in text
    assert FLAT_REF in text
    assert "image 'depth'" in text
    assert "renders images rgb" in text


def test_a_world_choosing_a_renderer_is_refused_the_same_way():
    engine = _engine([_camera("cam", 1.5, depth=True)], sim={"renderer": FLAT_REF})
    with pytest.raises(PluginError, match="image 'depth'"):
        engine.setup()


def test_an_unknown_renderer_on_a_camera_is_said_at_load():
    with pytest.raises(PluginError, match="unknown renderer 'nope_xyz'"):
        _engine([_camera("cam", 1.5, renderer="nope_xyz")]).setup()


def test_two_cameras_draw_through_their_own_renderers():
    engine = _engine([_camera("flat", 1.5, renderer=FLAT_REF), _camera("real", -1.5)])
    engine.setup()
    engine.reset()
    images = {
        ep.producer.split(".")[0]: ep for ep in engine.ctx.interface.all() if ep.name == "image"
    }
    for ep in images.values():
        ep.claim()
    for _ in range(20):
        engine.step()
    flat = images["flat"].read().data
    real = images["real"].read().data
    assert (flat == np.array(FLAT, np.uint8)).all()
    assert not (real == np.array(FLAT, np.uint8)).all() and real.max() > 0
    assert engine.ctx.render.renderer(FLAT_REF) is not None
    assert engine.ctx.render.renderer("mujoco") is not None
    engine.shutdown()


def test_roqsim_render_draws_through_the_renderer_the_flag_names(tmp_path):
    """`--renderer` is `--set sim.renderer=...` as a flag, and `roqsim render` honours it."""
    import yaml
    from PIL import Image

    from roqsim import render

    world = tmp_path / "w.yaml"
    world.write_text(yaml.safe_dump({"sim": {}, "components": [_camera("cam", 1.5)]}))
    out = tmp_path / "shot.png"
    assert render.main(
        [str(world), "--out", str(out), "--size", "64x48", "--renderer", FLAT_REF]
    ) in (
        0,
        None,
    )
    pixels = np.asarray(Image.open(out).convert("RGB"))
    assert (pixels == np.array(FLAT, np.uint8)).all()
