# Copyright (C) 2026 Frederik Pasch
# SPDX-License-Identifier: Apache-2.0
"""A renderer in another process stands behind the render service unchanged.

``render_stub_server.py`` serves roqsim's render socket from a process of its own and answers with
synthetic images; the world names the ``socket`` renderer, which starts it, sends it the scene
once and each tick's batch, and fails the tick loudly once it is gone.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest

from roqsim.config import load_config_from_dict
from roqsim.engine import Engine
from roqsim.plugin import Plugin, PluginError
from roqsim.presence import set_present
from roqsim.render_api import RenderError

STUB = Path(__file__).with_name("render_stub_server.py")


class Cams(Plugin):
    """Two model cameras over a box, each registered with the render service as a view due every
    tick, the way a camera plugin registers one."""

    def build(self, spec, ctx):
        spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.2, 0.2, 0.2])
        spec.worldbody.add_camera(name="front", pos=[1, 0, 0.3], xyaxes=[0, 1, 0, 0, 0, 1])
        spec.worldbody.add_camera(name="top", pos=[0, 0, 2])

    def configure(self, ctx):
        images = tuple(self.config.get("images", ("rgb", "depth")))
        self.views = [
            ctx.render.view(
                f"stub/{name}",
                camera=name,
                width=w,
                height=h,
                images=images,
                wants=lambda i=images: i,
                geomgroup=self.config.get("geomgroup"),
            )
            for name, (w, h) in (("front", (32, 24)), ("top", (16, 16)))
        ]


def _engine(tmp_path, images=("rgb", "depth"), settings=None, geomgroup=None) -> Engine:
    address = f"unix:{tmp_path / 'render.sock'}"
    socket_settings = {
        "address": address,
        "command": [sys.executable, str(STUB), "{address}", str(tmp_path / "report.json")],
        "images": ["rgb", "depth"],
        "connect_timeout_s": 60,
        "timeout_s": 30,
        **(settings or {}),
    }
    cfg = load_config_from_dict(
        {
            "sim": {
                "timestep": 0.002,
                "renderer": "socket",
                "renderers": {"socket": socket_settings},
            },
            "components": [{f"{__name__}:Cams": {"images": list(images), "geomgroup": geomgroup}}],
        }
    )
    engine = Engine(cfg)
    engine.ctx.seed = 1
    return engine


def test_a_world_draws_through_a_renderer_in_another_process(tmp_path):
    engine = _engine(tmp_path)
    engine.setup()
    engine.reset()
    try:
        for _ in range(3):
            engine.step()
        renderer = engine.ctx.render.renderer()
        assert renderer.capabilities_for(renderer.settings).in_process is False
        assert renderer._proc is not None and renderer._proc.poll() is None  # its own process
        front, top = (v.frame for v in engine.plugins[0].views)
        step = round(engine.ctx.sim_time / engine.ctx.dt)
        for frame, (w, h), view_id in (
            (front, (32, 24), "stub/front"),
            (top, (16, 16), "stub/top"),
        ):
            assert frame.sim_time == pytest.approx(engine.ctx.sim_time) and frame.step == step
            assert frame.rgb.shape == (h, w, 3) and frame.depth.shape == (h, w)
            assert (frame.rgb[..., 0] == step % 256).all()
            assert (frame.rgb[..., 1] == len(view_id)).all()
            assert (frame.rgb[0, :, 2] == np.arange(w)).all()
            assert frame.depth[0, 0] == pytest.approx(1.0 + 0.001 * step)
    finally:
        engine.shutdown()
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["cameras"] == ["stub/front", "stub/top"]
    assert report["bodies"] == engine.ctx.model.nbody
    assert report["glb_magic"] == "glTF" and report["glb_bytes"] > 0  # the scene, once, as GLB
    # Each tick carries the poses and the presence the server draws.
    assert report["render_blobs"] == [
        "model/flex_group",
        "model/flex_rgba",
        "model/geom_group",
        "model/geom_rgba",
        "xpos",
        "xquat",
    ]


def test_a_renderer_process_that_died_fails_the_next_tick(tmp_path):
    engine = _engine(tmp_path)
    engine.setup()
    engine.reset()
    try:
        engine.step()
        proc = engine.ctx.render.renderer()._proc
        proc.kill()
        proc.wait()
        with pytest.raises(RenderError, match="failed during 'render'"):
            engine.step()
    finally:
        engine.shutdown()


def test_an_image_the_server_does_not_declare_is_refused_at_load(tmp_path):
    engine = _engine(tmp_path, images=("rgb", "segmentation"))
    with pytest.raises(PluginError, match="image 'segmentation'"):
        engine.setup()


def test_a_server_nobody_listens_at_is_said(tmp_path):
    engine = _engine(tmp_path, settings={"command": None, "connect_timeout_s": 0.3})
    engine.setup()
    engine.reset()
    try:
        with pytest.raises(RenderError, match="no renderer accepted a connection"):
            engine.step()
    finally:
        engine.shutdown()


def test_a_server_that_never_listens_is_stopped_with_the_failed_open(tmp_path):
    pid_file = tmp_path / "server.pid"
    never_listens = (
        f"import os, time; open({str(pid_file)!r}, 'w').write(str(os.getpid())); time.sleep(60)"
    )
    engine = _engine(
        tmp_path,
        settings={"command": [sys.executable, "-c", never_listens], "connect_timeout_s": 2},
    )
    engine.setup()
    engine.reset()
    try:
        with pytest.raises(RenderError, match="no renderer accepted a connection"):
            engine.step()
    finally:
        engine.shutdown()
    pid = int(pid_file.read_text())
    with pytest.raises(ProcessLookupError):  # terminated and reaped, not left running
        os.kill(pid, 0)


#: A soft block, spawned as the entity ``prop``.
SOFT_BLOCK = """<mujoco><worldbody><body name="soft_block">
  <flexcomp name="soft" type="grid" count="3 3 3" spacing=".02 .02 .02" dim="3" radius=".002"
            mass="0.1" rgba="1 0 0 1"><edge equality="false"/>
    <elasticity young="5e4" poisson="0.3" damping="0.002"/></flexcomp>
</body></worldbody></mujoco>"""


def test_an_absent_flex_is_not_drawn_by_a_renderer_in_another_process(tmp_path):
    block = tmp_path / "soft_block.xml"
    block.write_text(SOFT_BLOCK, encoding="utf-8")
    address = f"unix:{tmp_path / 'render.sock'}"
    cfg = load_config_from_dict(
        {
            "sim": {
                "timestep": 0.002,
                "renderer": "socket",
                "renderers": {
                    "socket": {
                        "address": address,
                        "command": [
                            sys.executable,
                            str(STUB),
                            "{address}",
                            str(tmp_path / "r.json"),
                        ],
                        "images": ["rgb", "depth"],
                    }
                },
            },
            "components": [
                {"spawn_model": {"model": str(block), "prefix": "p_"}, "name": "prop"},
                {f"{__name__}:Cams": {}},
            ],
        },
        base_dir=tmp_path,
    )
    engine = Engine(cfg)
    engine.ctx.seed = 1
    engine.setup()
    engine.reset()
    try:
        engine.step()
        set_present(engine.ctx, engine.ctx.entities.get("prop"), False)
        engine.step()
    finally:
        engine.shutdown()
    report = json.loads((tmp_path / "r.json").read_text())
    assert report["flexes_drawn"] == [1, 0]


def test_a_camera_drawing_some_geom_groups_is_refused_by_a_server_that_draws_every_group(tmp_path):
    engine = _engine(tmp_path, settings={}, geomgroup=[2])
    with pytest.raises(PluginError, match="only some geom groups"):
        engine.setup()


def test_a_server_that_declares_geom_groups_is_sent_the_cameras_groups(tmp_path):
    engine = _engine(tmp_path, settings={"geom_groups": True}, geomgroup=[2])
    engine.setup()
    engine.reset()
    try:
        engine.step()
    finally:
        engine.shutdown()
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["geomgroups"] == [[2], [2]]


def test_its_settings_are_validated_at_load():
    with pytest.raises(PluginError, match="'address' is required"):
        load_config_from_dict(
            {"sim": {"renderer": "socket", "renderers": {"socket": {}}}, "components": []}
        )
