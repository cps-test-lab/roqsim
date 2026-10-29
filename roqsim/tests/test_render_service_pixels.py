# Copyright (C) 2026 Frederik Pasch
# SPDX-License-Identifier: Apache-2.0
"""A frame drawn through the render service is the frame roqsim drew before it, pixel for pixel.

"Before" is the path every camera took until the service: a ``mujoco.Renderer`` of its own, its
``update_scene``, the renderer's clip planes set on its scene, see-through geoms out of the shadow
pass, and its depth read and linearised with those planes. It is kept here, as :class:`Before`,
and drawn on the same machine in the same process, so the comparison holds on any GPU: what it
asserts is that the service changed nothing, not what a GPU draws.

Under MuJoCo's 4x MSAA a frame is not bit-reproducible on every GPU, before the service as after
it: on some GPUs a first frame differs from later ones, the frame depends on the size of the
framebuffer it is drawn in, and a run may differ from another in a handful of values by one level.
Without MSAA it is exact. So the exact comparisons draw the model without MSAA (``offsamples="0"``), and one test
keeps MuJoCo's default and holds the service to that variation: one level, on a tiny fraction of
the pixels. Both sides also draw every pass twice and compare the second, and :class:`Before`
draws in the framebuffer the service draws in -- before the service a renderer's was the model's
declared size or the camera's, whichever was larger.
"""

from __future__ import annotations

import copy

import mujoco
import numpy as np
import pytest

from roqsim.rendering import (
    FrameRenderer,
    RenderService,
    clear_geoms_cast_no_shadow,
    has_clear_geoms,
    hold_gc,
    linear_depth,
    set_clip_planes,
)

#: A lit room with a textured floor, a see-through pane, an orthographic and a perspective camera,
#: and a body that moves, so the scene build, the shadows and the planes all matter.
XML = """
<mujoco>
  <visual>
    <global offwidth="320" offheight="240"/><map znear="0.05"/><quality offsamples="{samples}"/>
  </visual>
  <asset>
    <texture name="grid" type="2d" builtin="checker" width="64" height="64" rgb1=".2 .3 .4"
             rgb2=".8 .8 .7"/>
    <material name="grid" texture="grid" texrepeat="4 4"/>
  </asset>
  <worldbody>
    <light pos="0 0 4" dir="0 0 -1" castshadow="true"/>
    <light pos="2 2 3" dir="-1 -1 -1"/>
    <geom type="plane" size="4 4 .1" material="grid"/>
    <geom name="pane" type="box" pos="0.6 0 0.5" size=".02 .5 .5" rgba=".6 .8 1 .3"/>
    <body name="crate" pos="0 0 .3">
      <freejoint/>
      <geom type="box" size=".2 .2 .2" rgba=".9 .3 .2 1"/>
      <geom type="sphere" pos="0 0 .3" size=".1" rgba=".2 .9 .3 1"/>
    </body>
    <camera name="persp" pos="2 0 1" xyaxes="0 1 0 -0.4 0 1" fovy="55"/>
    <camera name="side" pos="0 2.2 .8" xyaxes="-1 0 0 0 -0.3 1" fovy="40"/>
    <camera name="ortho" pos="0 0 3" xyaxes="1 0 0 0 1 0" projection="orthographic" fovy="3"/>
  </worldbody>
</mujoco>
"""


@pytest.fixture
def world():
    model = mujoco.MjModel.from_xml_string(XML.replace("{samples}", "0"))
    data = mujoco.MjData(model)
    data.qpos[:3] = [0.1, -0.2, 0.35]
    mujoco.mj_forward(model, data)
    return model, data


class Before:
    """The render path before the service: a ``mujoco.Renderer`` per camera."""

    def __init__(self, model, width, height, camera, near=None, far=None, fb=None):
        # Its own model copy, sized as the old renderer sized the shared one.
        self.model = copy.copy(model)
        fbw, fbh = fb or (width, height)
        self.model.vis.global_.offwidth = max(fbw, int(model.vis.global_.offwidth))
        self.model.vis.global_.offheight = max(fbh, int(model.vis.global_.offheight))
        self.r = mujoco.Renderer(self.model, height, width)
        self.camera, self.near, self.far = camera, near, far
        self.vopt = mujoco.MjvOption()
        self.vopt.geomgroup[4] = 0

    def _update(self, data):
        self.r.update_scene(data, self.camera, scene_option=self.vopt)
        set_clip_planes(self.r.scene, self.near, self.far)

    def rgb(self, data):
        # Each pass drawn twice and the second kept: a renderer's first frame may differ from its
        # later ones (see the module docstring), and a camera renders more than one frame.
        self._rgb(data)
        return self._rgb(data)

    def _rgb(self, data):
        self._update(data)
        if has_clear_geoms(self.model):
            clear_geoms_cast_no_shadow(self.r.scene)
        with hold_gc():
            return self.r.render()

    def depth(self, data):
        self._depth(data)
        return self._depth(data)

    def _depth(self, data):
        r = self.r
        self._update(data)
        scene = r.scene
        cam = scene.camera[0]
        near, far, ortho = float(cam.frustum_near), float(cam.frustum_far), bool(cam.orthographic)
        flags = scene.flags.copy()
        scene.flags[mujoco.mjtRndFlag.mjRND_SEGMENT] = True
        scene.flags[mujoco.mjtRndFlag.mjRND_IDCOLOR] = True
        buffer = np.empty((r.height, r.width), dtype=np.float32)
        try:
            r._gl_context.make_current()
            with hold_gc():
                mujoco.mjr_render(r._rect, scene, r._mjr_context)
                mujoco.mjr_readPixels(None, buffer, r._rect, r._mjr_context)
        finally:
            np.copyto(scene.flags, flags)
        depth = linear_depth(buffer, near, far, ortho)
        depth[buffer == 0.0] = np.inf
        return np.ascontiguousarray(np.flipud(depth))

    def segmentation(self, data):
        self._segmentation(data)
        return self._segmentation(data)

    def _segmentation(self, data):
        self._update(data)
        self.r.enable_segmentation_rendering()
        try:
            with hold_gc():
                return self.r.render()
        finally:
            self.r.disable_segmentation_rendering()

    def close(self):
        # Under its own context: mujoco.Renderer.close frees the GL context first, and the deletes
        # of the render context would land in whichever other renderer's context is current.
        r = self.r
        r._gl_context.make_current()
        r._mjr_context.free()
        r._mjr_context = None
        r.close()


def _cam(model, name):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name)


@pytest.mark.parametrize("camera", ["persp", "ortho", "free"])
def test_a_frame_renderer_draws_what_a_renderer_drew_before(world, camera):
    model, data = world
    cam = mujoco.MjvCamera() if camera == "free" else _cam(model, camera)
    if camera == "free":
        mujoco.mjv_defaultFreeCamera(model, cam)
    planes = {"near": 0.02, "far": 9.0} if camera == "persp" else {}
    new = FrameRenderer(model, 160, 120, camera=cam, **planes)
    old = Before(model, 160, 120, cam, **planes)
    try:
        for _ in range(2):
            rgb, depth, seg = (
                new.render(data),
                new.render_depth(data),
                new.render_segmentation(data),
            )
        np.testing.assert_array_equal(rgb, old.rgb(data))
        np.testing.assert_array_equal(depth, old.depth(data))
        np.testing.assert_array_equal(seg, old.segmentation(data))
    finally:
        new.close()
        old.close()


def test_a_batch_of_cameras_draws_what_each_drew_alone(world):
    """Three cameras of different sizes in one context, each as a renderer of its own drew it."""
    model, data = world
    service = RenderService(model)
    sizes = {"persp": (200, 150), "side": (96, 64), "ortho": (128, 128)}
    views = {
        name: service.view(name, camera=name, width=w, height=h, near=0.02, images=("rgb", "depth"))
        for name, (w, h) in sizes.items()
    }
    for _ in range(2):
        frames = service.render_now([(v, ("rgb", "depth")) for v in views.values()], data)
    renderer = service.renderer()
    assert renderer.stats["scene_builds"] == 6  # each camera its own, in two batches of three
    # The framebuffer the service draws in: the largest camera, or the model's declared size.
    fb = (max(320, 200), max(240, 150))
    try:
        for (name, (w, h)), frame in zip(sizes.items(), frames, strict=True):
            old = Before(model, w, h, _cam(model, name), near=0.02, fb=fb)
            np.testing.assert_array_equal(frame.rgb, old.rgb(data), err_msg=name)
            np.testing.assert_array_equal(frame.depth, old.depth(data), err_msg=name)
            old.close()
    finally:
        service.close()


def test_the_renderer_never_writes_the_shared_model(world):
    model, data = world
    before = (int(model.vis.global_.offwidth), int(model.vis.global_.offheight))
    fr = FrameRenderer(model, 640, 480, camera="persp")
    fr.render(data)
    fr.close()
    assert (int(model.vis.global_.offwidth), int(model.vis.global_.offheight)) == before


def test_under_msaa_the_service_stays_within_the_gpus_own_variation():
    """MuJoCo's default 4x MSAA: one level, on at most a thousandth of the values, for every pass."""
    model = mujoco.MjModel.from_xml_string(XML.replace("{samples}", "4"))
    data = mujoco.MjData(model)
    data.qpos[:3] = [0.1, -0.2, 0.35]
    mujoco.mj_forward(model, data)
    new = FrameRenderer(model, 160, 120, camera="persp", near=0.02)
    old = Before(model, 160, 120, _cam(model, "persp"), near=0.02)
    try:
        for _ in range(2):
            rgb = new.render(data)
        reference = old.rgb(data)
        diff = np.abs(rgb.astype(int) - reference.astype(int))
        assert diff.max() <= 1
        assert (diff > 0).mean() <= 1e-3
    finally:
        new.close()
        old.close()
