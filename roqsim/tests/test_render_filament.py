# Copyright (C) 2026 Frederik Pasch
# SPDX-License-Identifier: Apache-2.0
"""MuJoCo's Filament renderer behind the render service (:mod:`roqsim.render_filament`).

Drawn through Vulkan, on whatever device this machine's Vulkan selects (lavapipe on a machine
without a GPU). Its labels and its depth are held to MuJoCo's classic renderer and to geometry;
its colour is not compared pixel by pixel (it is a different renderer), only checked to show the
scene. Skipped, with Filament's reason, where no Vulkan context opens at all.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import mujoco
import numpy as np
import pytest

from roqsim import render_filament
from roqsim.registry import resolve_renderer
from roqsim.render_api import RenderError
from roqsim.render_filament import FilamentRenderer, headlight, linear_depth
from roqsim.rendering import FrameRenderer, RenderService


def _vulkan() -> str | None:
    try:
        render_filament.probe()
    except RenderError as err:
        return str(err)
    return None


_NO_VULKAN = _vulkan()
needs_vulkan = pytest.mark.skipif(_NO_VULKAN is not None, reason=f"{_NO_VULKAN}")

#: A textured floor, a body that moves, a site, and cameras of every projection the service
#: knows: a pinhole, an orthographic one, one with an off-centre principal point and non-square
#: pixels.
XML = """
<mujoco>
  <visual><global offwidth="320" offheight="240"/></visual>
  <asset>
    <texture name="grid" type="2d" builtin="checker" width="64" height="64" rgb1=".2 .3 .4"
             rgb2=".8 .8 .7"/>
    <material name="grid" texture="grid" texrepeat="4 4"/>
  </asset>
  <worldbody>
    <light pos="0 0 4" dir="0 0 -1" castshadow="true"/>
    <geom name="floor" type="plane" size="4 4 .1" material="grid"/>
    <body name="crate" pos="0 0 .3">
      <freejoint/>
      <geom name="box" type="box" size=".2 .2 .2" rgba=".9 .3 .2 1"/>
      <geom name="ball" type="sphere" pos="0 0 .3" size=".1" rgba=".2 .9 .3 1"/>
    </body>
    <geom name="post" type="cylinder" pos=".7 .3 .3" size=".08 .3" rgba=".9 .9 .2 1" group="1"/>
    <site name="marker" pos="-.6 0 .3" size=".08" rgba="0 0 1 1"/>
    <camera name="persp" pos="0 -2 1.2" xyaxes="1 0 0 0 .5 .87"/>
    <camera name="ortho" pos="0 -2 1.2" xyaxes="1 0 0 0 .5 .87" projection="orthographic"
            fovy="2"/>
    <camera name="offset" pos="0 -2 1.2" xyaxes="1 0 0 0 .5 .87" sensorsize=".004 .003"
            resolution="320 200" focalpixel="250 300" principalpixel="30 -20"/>
  </worldbody>
</mujoco>
"""


@pytest.fixture
def scene():
    model = mujoco.MjModel.from_xml_string(XML)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def _draw(model, data, camera, renderer, image="rgb", **kw):
    frame = FrameRenderer(model, 320, 200, camera=camera, renderer=renderer, **kw)
    try:
        if image == "rgb":
            return frame.render(data)
        if image == "depth":
            return frame.render_depth(data)
        return frame.render_segmentation(data)
    finally:
        frame.close()


# -- what it declares --------------------------------------------------------------------------


def test_it_is_registered_and_declares_what_it_gives():
    cls = resolve_renderer("filament")
    assert cls is FilamentRenderer
    caps = cls.capabilities
    assert caps.images == {"rgb", "depth", "segmentation"}
    assert caps.camera_models == {"pinhole", "orthographic", "free"}
    assert caps.geom_groups and caps.principal_point and caps.non_square_pixels
    assert not caps.decorations and not caps.moving_only


def test_a_frame_is_a_function_of_the_state_only_without_post_processing():
    assert not FilamentRenderer.capabilities_for({}).deterministic
    assert not FilamentRenderer.capabilities_for({"post_processing": True}).deterministic
    assert FilamentRenderer.capabilities_for({"post_processing": False}).deterministic


def test_settings_are_booleans_and_known_keys():
    assert FilamentRenderer.validate_config({"shadows": False, "software": True}) == []
    assert "must be srgb or model" in FilamentRenderer.validate_config({"colour_textures": "x"})[0]
    errors = FilamentRenderer.validate_config({"shadows": "yes", "shadow": True})
    assert any("'shadows' must be true or false" in e for e in errors)
    assert any("unknown key 'shadow'" in e and "did you mean 'shadows'" in e for e in errors)


def test_a_camera_asking_for_decorations_is_refused_at_load(scene):
    model, _ = scene
    service = RenderService(model, renderer="filament")
    service.view("c", camera="persp", width=32, height=24, decorated=True)
    with pytest.raises(Exception, match="decorations"):
        service.check()


# -- the rules it applies, without a GPU ----------------------------------------------------------


def test_depth_is_linearised_from_reversed_z():
    near, far = 0.5, 3.0
    z = np.array([0.6, 1.7, 2.9], np.float64)
    d = near * (far - z) / (z * (far - near))
    out = linear_depth(np.append(d, 0.0).astype(np.float32), near, far)
    np.testing.assert_allclose(out[:3], z, rtol=1e-5)
    assert np.isinf(out[3])
    ortho = linear_depth(np.array([(far - 1.2) / (far - near)], np.float32), near, far, True)
    np.testing.assert_allclose(ortho, [1.2], rtol=1e-5)


def _lit(extra="", light='<light pos="0 0 3"/>', visual=""):
    return mujoco.MjModel.from_xml_string(
        f"<mujoco><visual>{visual}</visual><custom>{extra}</custom>"
        f"<worldbody>{light}<geom size='.1'/></worldbody></mujoco>"
    )


def test_the_headlight_follows_mujoco_s_rule():
    on, colour, lux = headlight(_lit())
    assert on and lux == render_filament.FALLBACK_HEADLIGHT_LUX
    assert colour == pytest.approx((0.4, 0.4, 0.4))  # visual/headlight/diffuse
    numeric = '<numeric name="filament.fallback.head_light_intensity" data="1234"/>'
    assert headlight(_lit(numeric))[2] == 1234.0
    # A photometric light switches the whole scene to physical lighting: no headlight.
    assert not headlight(_lit(light='<light pos="0 0 3" intensity="500"/>'))[0]
    assert not headlight(_lit(visual='<headlight active="0"/>'))[0]


def test_a_cpu_rasteriser_is_refused_where_a_gpu_was_given(monkeypatch):
    from roqsim import gl

    monkeypatch.setattr(gl, "gpu_visible", lambda: True)
    with pytest.raises(RenderError, match="CPU rasteriser"):
        render_filament.check_device("llvmpipe (LLVM 19.1.7, 256 bits)", allow_software=False)
    render_filament.check_device("llvmpipe (LLVM 19.1.7, 256 bits)", allow_software=True)
    render_filament.check_device("NVIDIA GeForce RTX 4090", allow_software=False)
    monkeypatch.setattr(gl, "gpu_visible", lambda: False)
    render_filament.check_device("llvmpipe (LLVM 19.1.7, 256 bits)", allow_software=False)


# -- what it draws -------------------------------------------------------------------------------


@needs_vulkan
@pytest.mark.parametrize("camera", ["persp", "ortho", "offset"])
def test_labels_agree_with_mujoco_s_renderer(scene, camera):
    model, data = scene
    classic = _draw(model, data, camera, "mujoco", "segmentation")
    filament = _draw(model, data, camera, "filament", "segmentation")
    assert filament.shape == classic.shape and filament.dtype == np.int32
    agree = (filament == classic).all(-1).mean()
    assert agree > 0.998, f"{camera}: labels agree on {agree:.4f} of the pixels"
    site = int(mujoco.mjtObj.mjOBJ_SITE)
    assert ((filament[..., 0] == 0) & (filament[..., 1] == site)).any(), "the site is labelled"


@needs_vulkan
def test_a_batch_of_cameras_and_images_is_drawn_as_each_alone(scene):
    """Several cameras and every image in one batch: each frame is the one drawn by itself."""
    model, data = scene
    service = RenderService(
        model, renderer="filament", renderers={"filament": {"post_processing": False}}
    )
    images = ("rgb", "depth", "segmentation")
    views = [
        service.view(c, camera=c, width=160, height=100, near=0.1, far=20, images=images)
        for c in ("persp", "ortho", "offset")
    ]
    try:
        batch = service.render_now([(v, images) for v in views], data)
        for view, frame in zip(views, batch, strict=True):
            (alone,) = service.render_now([(view, images)], data)
            assert np.array_equal(frame.rgb, alone.rgb), view.id
            assert np.array_equal(frame.segmentation, alone.segmentation), view.id
            np.testing.assert_array_equal(frame.depth, alone.depth)
        assert service.renderer("filament").stats["submits"] == 18  # one call per image
    finally:
        service.close()


@needs_vulkan
def test_depth_is_metres_along_the_view_axis():
    """A wall square to the camera reads its distance; the floor reads the pinhole's geometry."""
    model = mujoco.MjModel.from_xml_string(
        """<mujoco><worldbody><light pos="0 0 4" dir="0 0 -1"/>
        <geom type="box" pos="0 2 0" size="3 .1 3"/><geom type="plane" pos="0 0 -1" size="5 5 .1"/>
        <camera name="c" pos="0 -1 0" xyaxes="1 0 0 0 0 1" fovy="60"/></worldbody></mujoco>"""
    )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    depth = _draw(model, data, "c", "filament", "depth", near=0.1, far=20)
    assert depth[60, 160] == pytest.approx(2.9, abs=1e-3)
    fy = 100 / np.tan(np.radians(30))  # 200 rows, a vertical field of view of 60 degrees
    for row in (170, 185, 195):
        # The floor 1 m below, seen through the centre of pixel row `row`.
        assert depth[row, 160] == pytest.approx(fy / (row + 0.5 - 100), rel=2e-3)


@needs_vulkan
def test_nothing_drawn_is_infinitely_far(scene):
    model, data = scene
    depth = _draw(model, data, "persp", "filament", "depth", near=0.1, far=20)
    classic = _draw(model, data, "persp", "mujoco", "depth", near=0.1, far=20)
    assert np.isinf(depth[0]).all()
    assert (np.isinf(depth) == np.isinf(classic)).mean() > 0.999


@needs_vulkan
def test_colour_shows_the_scene(scene):
    model, data = scene
    rgb = _draw(model, data, "persp", "filament")
    assert rgb.shape == (200, 320, 3) and rgb.dtype == np.uint8
    seg = _draw(model, data, "persp", "filament", "segmentation")
    box = seg[..., 0] == mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "box")
    red = rgb[box].astype(int)
    assert (red[:, 0] > red[:, 1] + 40).mean() > 0.9, "the red box is drawn red"
    # Black where the model sets no colour, but for the dithering's level or two.
    assert rgb[0].max() <= 2, "the background is black where the model sets no colour"


def _texture_and_rgba(colorspace="auto", settings=None):
    """A colour as a flat texture and as an rgba, side by side; the two patches as Filament draws them."""
    colour = "0.55 0.65 0.15"
    model = mujoco.MjModel.from_xml_string(
        f"""<mujoco><asset>
        <texture name="t" type="2d" builtin="flat" rgb1="{colour}" width="16" height="16"
                 colorspace="{colorspace}"/>
        <material name="textured" texture="t"/><material name="plain" rgba="{colour} 1"/></asset>
        <worldbody><light pos="0 0 3" dir="0 0 -1"/>
        <geom type="box" pos="-.5 0 0" size=".4 .4 .01" material="textured"/>
        <geom type="box" pos=".5 0 0" size=".4 .4 .01" material="plain"/>
        <camera name="c" pos="0 0 2"/></worldbody></mujoco>"""
    )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    frame = FrameRenderer(model, 200, 100, camera="c", renderer="filament", settings=settings)
    try:
        rgb = frame.render(data).astype(int)
    finally:
        frame.close()
    assert model.tex_colorspace[0] == {"auto": 0, "linear": 1, "sRGB": 2}[colorspace], "not written"
    return rgb[40:60, 40:60].mean((0, 1)), rgb[40:60, 140:160].mean((0, 1))


@needs_vulkan
def test_a_colour_texture_is_drawn_as_the_same_colour_given_as_rgba():
    textured, plain = _texture_and_rgba()
    assert np.abs(textured - plain).max() < 4
    # MuJoCo's own reading of `auto` (linear) draws it lighter and paler, and so does `linear`.
    for linear in (
        _texture_and_rgba(settings={"colour_textures": "model"})[0],
        _texture_and_rgba("linear")[0],
    ):
        assert linear[2] > plain[2] + 30, "the blue of a linear lime is lifted"


@needs_vulkan
def test_without_post_processing_a_frame_repeats_exactly(scene):
    model, data = scene
    frame = FrameRenderer(
        model, 160, 100, camera="persp", renderer="filament", settings={"post_processing": False}
    )
    try:
        first = frame.render(data)
        assert all(np.array_equal(first, frame.render(data)) for _ in range(3))
    finally:
        frame.close()


@needs_vulkan
def test_a_camera_draws_only_the_groups_it_names(scene):
    model, data = scene
    post = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "post")
    every = _draw(model, data, "persp", "filament", "segmentation", geomgroup=(0, 1))
    some = _draw(model, data, "persp", "filament", "segmentation", geomgroup=(0,))
    assert (every[..., 0] == post).any()
    assert not (some[..., 0] == post).any()


@needs_vulkan
def test_what_roqsim_hides_at_run_time_is_not_drawn(scene):
    """Presence moves an absent geom to its own group and zeroes its alpha, at run time."""
    from roqsim.presence import ABSENT_GEOM_GROUP

    model, data = scene
    box = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "box")
    frame = FrameRenderer(model, 160, 100, camera="persp", renderer="filament")
    try:
        assert (frame.render_segmentation(data)[..., 0] == box).any()
        model.geom_group[box] = ABSENT_GEOM_GROUP
        assert not (frame.render_segmentation(data)[..., 0] == box).any()
        model.geom_group[box] = 0
        model.geom_rgba[box, 3] = 0.0
        assert not (frame.render_segmentation(data)[..., 0] == box).any()
        model.geom_rgba[box, 3] = 1.0
        assert (frame.render_segmentation(data)[..., 0] == box).any(), "drawn again"
    finally:
        frame.close()


@needs_vulkan
def test_two_textured_worlds_draw_at_once(scene):
    """One Filament context holds one textured model at a time; each renderer borrows its own."""
    model, data = scene
    other = mujoco.MjModel.from_xml_string(XML.replace('rgb1=".2 .3 .4"', 'rgb1=".9 .1 .1"'))
    other_data = mujoco.MjData(other)
    mujoco.mj_forward(other, other_data)
    a = FrameRenderer(model, 160, 100, camera="persp", renderer="filament")
    b = FrameRenderer(other, 160, 100, camera="persp", renderer="filament")
    try:
        a.render(data)
        b.render(other_data)
        a.render(data)
    finally:
        a.close()
        b.close()


def _child(code: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )


_CHILD_SCENE = """
    import roqsim, mujoco, threading
    from roqsim.rendering import FrameRenderer
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><light pos="0 0 3"/><geom size=".1"/></worldbody></mujoco>')
    data = mujoco.MjData(model)
    def draw(renderer):
        frame = FrameRenderer(model, 64, 48, renderer=renderer)
        frame.render(data)
        frame.close()
"""


@needs_vulkan
def test_mujoco_s_renderer_still_opens_after_a_filament_one_closed():
    """Destroying a Filament context ends the EGL display the classic renderer shares; none is."""
    run = _child(
        _CHILD_SCENE
        + """
    for renderer in ("mujoco", "filament", "mujoco", "filament", "mujoco"):
        draw(renderer)
    print("drawn")
    """
    )
    assert run.returncode == 0 and "drawn" in run.stdout, run.stderr[-2000:]


@needs_vulkan
def test_closed_or_collected_off_its_thread_it_does_not_end_the_process():
    """Filament aborts when its objects are destroyed off their thread; they are kept instead."""
    run = _child(
        _CHILD_SCENE
        + """
    import gc
    box = {}
    def make():
        box["frame"] = FrameRenderer(model, 64, 48, renderer="filament")
        box["frame"].render(data)
    worker = threading.Thread(target=make); worker.start(); worker.join()
    box["frame"].close()
    box.clear(); gc.collect()
    worker = threading.Thread(target=make); worker.start(); worker.join()
    box.clear(); gc.collect()
    print("survived")
    """
    )
    assert run.returncode == 0 and "survived" in run.stdout, run.stderr[-2000:]


def test_a_machine_without_vulkan_is_named_not_aborted():
    """Filament ends the process when no Vulkan driver loads; the probe says so in a child."""
    run = _child(
        """
    import os
    os.environ["VK_ICD_FILENAMES"] = "/nonexistent/icd.json"
    os.environ["VK_DRIVER_FILES"] = "/nonexistent/icd.json"
    from roqsim import render_filament
    from roqsim.render_api import RenderError
    try:
        render_filament.probe()
    except RenderError as err:
        print("refused:", str(err).splitlines()[0])
    else:
        print("opened")
    """
    )
    assert run.returncode == 0, run.stderr[-2000:]
    assert "refused: filament: Filament could not open a headless Vulkan context" in run.stdout
