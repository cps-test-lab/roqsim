# Copyright (C) 2026 Frederik Pasch
# SPDX-License-Identifier: Apache-2.0
"""The service holds every renderer to the frames it was asked for, and says whose failure is whose.

A frame stamped with another time, or an image of another size or type, fails the tick naming the
renderer and the camera, rather than reaching a camera plugin. ``roqsim renderers describe`` prints
what a renderer gives as its settings make it. Only MuJoCo's renderer needs GL: another renderer
opens on a host whose GL cannot render, and its own failure to open is told as its own.
"""

from __future__ import annotations

import re

import mujoco
import numpy as np
import pytest

from roqsim import render, renderers, rendering
from roqsim.render_api import Frame, Renderer, RendererCapabilities, RenderError, Setting
from roqsim.rendering import FrameRenderer, GLBackendError, RenderService

XML = '<mujoco><worldbody><camera name="c" pos="1 0 0" xyaxes="0 1 0 0 0 1"/></worldbody></mujoco>'


class Liar(Renderer):
    """Answers each request with a frame that is wrong in the way its ``lie`` setting names."""

    capabilities = RendererCapabilities(images=frozenset({"rgb", "depth"}), in_process=True)
    SETTINGS = {"lie": Setting(None, "what to get wrong: stamp, step, id, shape, dtype, or none")}

    def render(self, snapshot, requests):
        lie = self.settings["lie"]
        frames = []
        for r in requests:
            h, w = r.camera.height, r.camera.width
            frame = Frame(
                "someone-else" if lie == "id" else r.id,
                snapshot.sim_time + (1.0 if lie == "stamp" else 0.0),
                snapshot.step + (1 if lie == "step" else 0),
            )
            if "rgb" in r.images:
                shape = (h + 1, w, 3) if lie == "shape" else (h, w, 3)
                frame.rgb = np.zeros(shape, np.float32 if lie == "dtype" else np.uint8)
            if "depth" in r.images:
                frame.depth = np.ones((h, w), np.float32)
            frames.append(frame)
        return frames


class Widens(Renderer):
    """Gives depth only when its settings say so, as the socket renderer's capabilities do."""

    capabilities = RendererCapabilities(images=frozenset({"rgb"}))
    SETTINGS = {"depth": Setting(True, "whether it renders depth")}

    @classmethod
    def capabilities_for(cls, settings):
        images = {"rgb", "depth"} if {"depth": True, **dict(settings)}["depth"] else {"rgb"}
        return RendererCapabilities(images=frozenset(images))


def _draw(lie, images=("rgb",)):
    model = mujoco.MjModel.from_xml_string(XML)
    data = mujoco.MjData(model)
    name = f"{__name__}:Liar"
    service = RenderService(model, renderer=name, renderers={name: {"lie": lie}})
    view = service.view("cam", camera="c", width=4, height=3, images=images)
    try:
        return service.render_now([(view, images)], data)
    finally:
        service.close()


def test_a_truthful_frame_passes():
    (frame,) = _draw(None, ("rgb", "depth"))
    assert frame.rgb.shape == (3, 4, 3) and frame.depth.shape == (3, 4)


@pytest.mark.parametrize(
    ("lie", "said"),
    [
        ("stamp", "stamped t=1.0"),
        ("step", "step 1, the snapshot"),
        ("id", "camera 'someone-else'"),
        ("shape", "rgb (4, 4, 3) uint8, expected (3, 4, 3) uint8"),
        ("dtype", "rgb (3, 4, 3) float32, expected (3, 4, 3) uint8"),
    ],
)
def test_a_frame_that_is_not_the_one_asked_for_fails_naming_renderer_and_camera(lie, said):
    with pytest.raises(RenderError, match=re.escape(said)) as err:
        _draw(lie)
    assert "Liar" in str(err.value) and "'cam'" in str(err.value)


def test_describe_prints_what_the_settings_give():
    described = renderers.describe_renderer(f"{__name__}:Widens")
    assert described["capabilities"]["images"] == ["depth", "rgb"]


def test_a_renderer_without_gl_opens_where_gl_cannot_render(monkeypatch):
    def no_gl():
        raise GLBackendError("MUJOCO_GL names a backend that cannot render here")

    monkeypatch.setattr(rendering, "check_gl_backend", no_gl)
    model = mujoco.MjModel.from_xml_string(XML)
    data = mujoco.MjData(model)
    frame = FrameRenderer(model, 4, 3, camera="c", renderer=f"{__name__}:Liar")
    try:
        assert frame.render(data).shape == (3, 4, 3)
    finally:
        frame.close()
    with pytest.raises(GLBackendError):  # MuJoCo's own renderer still checks the backend
        FrameRenderer(model, 4, 3, camera="c")


def test_roqsim_render_tells_a_renderer_that_cannot_open_by_its_own_failure(tmp_path):
    model = mujoco.MjModel.from_xml_string(XML)
    settings = {"address": f"unix:{tmp_path / 'nobody.sock'}", "connect_timeout_s": 0.2}
    with pytest.raises(render.RenderError) as err:
        render.open_frame_renderer(model, 4, 3, camera="c", renderer="socket", settings=settings)
    assert not isinstance(err.value, render.RenderGLError)
    assert "renderer 'socket' could not open" in str(err.value)
    assert "no renderer accepted a connection" in str(err.value)
    assert "MUJOCO_GL" not in str(err.value)
