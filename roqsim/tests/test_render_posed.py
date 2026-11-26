# Copyright (C) 2026 Frederik Pasch
# SPDX-License-Identifier: Apache-2.0
"""A posed camera: drawn at the pose and intrinsics its request carries, with no camera of the model.

A renderer declares whether it draws one (``RendererCapabilities.posed``); one that does not is
refused, naming what it lacks, and nothing falls back to another view.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np
import pytest

from roqsim.plugin import PluginError
from roqsim.presence import ABSENT_GEOM_GROUP, WINDOW_GEOM_GROUP
from roqsim.render_api import IMAGES, CameraSpec, RendererCapabilities, RenderError
from roqsim.rendering import RenderService, camera_intrinsics, posed_gl_camera


def _posed(**kw) -> CameraSpec:
    return CameraSpec(id="v", width=64, height=48, intrinsics=(50.0, 50.0, 31.5, 23.5), **kw)


def test_a_camera_with_intrinsics_and_no_model_camera_is_posed():
    assert _posed().posed
    # A model camera, or one without intrinsics, is not.
    assert not _posed(mujoco_camera=0).posed
    assert not CameraSpec(id="v", width=4, height=4).posed


def test_a_renderer_that_does_not_draw_a_posed_camera_says_so():
    assert _posed().missing_from(RendererCapabilities()) == ["a camera at a given pose"]
    assert _posed().missing_from(RendererCapabilities(posed=True)) == []


def test_the_capability_is_listed():
    assert RendererCapabilities().as_dict()["posed"] is False
    assert RendererCapabilities(posed=True).as_dict()["posed"] is True


# -- MuJoCo's renderer draws a posed camera ------------------------------------------------------

W, H = 160, 120

#: A floor, three coloured bodies at different depths, a light, and a model camera with a sensor, a
#: focal length per axis and an off-centre principal point. No MSAA: a frame is exact.
XML = """
<mujoco>
  <visual><quality offsamples="0"/><global offwidth="200" offheight="200"/></visual>
  <worldbody>
    <light pos="0 0 3" castshadow="false"/>
    <geom name="floor" type="plane" size="5 5 .1" rgba=".5 .6 .7 1"/>
    <geom name="box" type="box" pos="1 .3 .3" size=".2 .3 .3" rgba="1 0 0 1"/>
    <geom name="ball" type="sphere" pos="1.5 -.4 .5" size=".2" rgba="0 1 0 1"/>
    <geom name="bar" type="box" pos="2 0 1.0" size=".1 .5 .1" rgba="0 0 1 1"/>
    <camera name="cam" pos="-1 0 .6" xyaxes="0 -1 0 .2 0 1" {sensor}/>
  </worldbody>
</mujoco>
"""

#: The model camera's sensor: centred, off-centre, and with non-square pixels.
SENSORS = {
    "centred": 'fovy="50"',
    "off_centre": 'sensorsize="0.0048 0.0036" resolution="160 120" focal="0.004 0.004" '
    'principal="0.0003 -0.0002"',
    "non_square": 'sensorsize="0.0048 0.0036" resolution="160 120" focal="0.004 0.0042" '
    'principal="-0.0001 0.0002"',
}


def _world(sensor: str):
    model = mujoco.MjModel.from_xml_string(XML.format(sensor=SENSORS[sensor]))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def _cam_pose(model, data, cam: int = 0):
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, data.cam_xmat[cam])
    return np.array(data.cam_xpos[cam]), quat


def _draw(service, views, data):
    return service.render_now([(v, IMAGES) for v in views], data)


@pytest.mark.parametrize("sensor", sorted(SENSORS))
def test_a_posed_view_is_the_picture_of_the_model_camera_at_its_pose(sensor):
    """Colour, depth and segmentation, pixel for pixel, also when two sizes share one batch."""
    model, data = _world(sensor)
    service = RenderService(model)
    try:
        pos, quat = _cam_pose(model, data)
        ref = service.view("ref", camera=0, width=W, height=H, images=IMAGES)
        posed = service.view_at(
            "posed",
            width=W,
            height=H,
            intrinsics=camera_intrinsics(model, 0, W, H, float(model.cam_fovy[0])),
            pos=pos,
            quat=quat,
            images=IMAGES,
        )
        small_ref = service.view("small_ref", camera=0, width=W // 2, height=H // 2, images=IMAGES)
        small = service.view_at(
            "small",
            width=W // 2,
            height=H // 2,
            intrinsics=camera_intrinsics(model, 0, W // 2, H // 2, float(model.cam_fovy[0])),
            pos=pos,
            quat=quat,
            images=IMAGES,
        )
        a, b, c, d = _draw(service, [ref, posed, small_ref, small], data)
        for kind in IMAGES:
            assert np.array_equal(a.image(kind), b.image(kind)), kind
            assert np.array_equal(c.image(kind), d.image(kind)), kind
        assert (np.asarray(a.segmentation)[..., 0] >= 0).any()
    finally:
        service.close()


def _posed_pair(model, data, intr_a, intr_b, quat_b=None, size=(W, H)):
    """Two posed views of the model camera's pose, the second with other intrinsics or rotation."""
    service = RenderService(model)
    pos, quat = _cam_pose(model, data)
    w, h = size
    a = service.view_at(
        "a", width=w, height=h, intrinsics=intr_a, pos=pos, quat=quat, images=IMAGES
    )
    b = service.view_at(
        "b",
        width=w,
        height=h,
        intrinsics=intr_b,
        pos=pos,
        quat=quat if quat_b is None else quat_b,
        images=IMAGES,
    )
    try:
        return _draw(service, [a, b], data)
    finally:
        service.close()


def test_an_off_centre_principal_point_shifts_the_picture_by_exactly_that_many_pixels():
    model, data = _world("centred")
    fx, fy, cx, cy = camera_intrinsics(model, 0, W, H, float(model.cam_fovy[0]))
    a, b = _posed_pair(model, data, (fx, fy, cx, cy), (fx, fy, cx + 7, cy - 4))
    # The optical axis 7 px right and 4 px up: the picture moves with it. Colour and labels to the
    # bit; depth to float32's rounding of another frustum's interpolation.
    for kind in ("rgb", "segmentation"):
        assert np.array_equal(b.image(kind)[0 : H - 4, 7:W], a.image(kind)[4:H, 0 : W - 7]), kind
    np.testing.assert_allclose(b.depth[0 : H - 4, 7:W], a.depth[4:H, 0 : W - 7], rtol=1e-5)


def test_a_rolled_view_is_the_unrolled_one_rotated():
    """A roll of 90 degrees about the view axis, square pixels and image, the principal point at
    the centre: the picture is the unrolled one turned by a quarter."""
    model, data = _world("centred")
    n = 96
    f = 80.0
    centre = (n - 1) / 2.0
    _pos, quat = _cam_pose(model, data)
    roll = np.zeros(4)
    mujoco.mju_axisAngle2Quat(roll, [0.0, 0.0, 1.0], math.pi / 2)  # about the camera's own z
    rolled = np.zeros(4)
    mujoco.mju_mulQuat(rolled, quat, roll)
    a, b = _posed_pair(model, data, (f, f, centre, centre), (f, f, centre, centre), rolled, (n, n))
    # Rolling the camera by +90 degrees about its view axis turns the picture by -90 degrees. The
    # labels to the bit, depth to float32's rounding; colour to one level at a few pixels, where
    # MuJoCo's lighting interpolates a triangle's colours along other edges.
    assert np.array_equal(b.segmentation, np.rot90(a.segmentation, k=-1))
    np.testing.assert_allclose(b.depth, np.rot90(a.depth, k=-1), rtol=1e-5)
    colour = np.abs(b.rgb.astype(int) - np.rot90(a.rgb, k=-1).astype(int))
    assert colour.max() <= 1 and (colour.max(axis=2) > 0).mean() < 0.001


def test_depth_is_metric_along_the_view_axis_at_every_pixel():
    """Off-centre too: depth times the length of the pixel's ray (z = 1) is the ray cast's distance.

    Checked where the picture shows a flat face (the floor, the boxes): a sphere is drawn tessellated,
    and its drawn surface is not the one a ray meets."""
    model, data = _world("off_centre")
    intr = camera_intrinsics(model, 0, W, H, float(model.cam_fovy[0]))
    service = RenderService(model)
    pos, quat = _cam_pose(model, data)
    view = service.view_at(
        "d",
        width=W,
        height=H,
        intrinsics=intr,
        pos=pos,
        quat=quat,
        near=0.05,
        images=("depth", "segmentation"),
    )
    try:
        (frame,) = service.render_now([(view, ("depth", "segmentation"))], data)
    finally:
        service.close()
    fx, fy, cx, cy = intr
    mat = np.asarray(data.cam_xmat[0]).reshape(3, 3)
    geomid = np.zeros(1, np.int32)
    flat = {mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n) for n in ("floor", "box", "bar")}
    checked = 0
    for v in range(3, H, 13):
        for u in range(2, W, 17):
            ray_cv = np.array([(u - cx) / fx, (v - cy) / fy, 1.0])  # x right, y down, z forward
            ray = mat @ (ray_cv * np.array([1.0, -1.0, -1.0]))
            dist = mujoco.mj_ray(model, data, pos, ray / np.linalg.norm(ray), None, 1, -1, geomid)
            depth = float(frame.depth[v, u])
            if int(frame.segmentation[v, u, 0]) not in flat and dist >= 0:
                continue
            if dist < 0:
                assert not np.isfinite(depth)
                continue
            assert depth * np.linalg.norm(ray_cv) == pytest.approx(dist, abs=1e-3), (u, v)
            checked += 1
    assert checked > 50


def test_a_posed_view_draws_the_window_group_and_leaves_an_absent_entity_out():
    """A world drawn with baked lighting keeps its static geometry in the window group: drawn. A
    geom in the absent group, between it and the camera: not drawn."""
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><visual><headlight ambient="1 1 1"/><quality offsamples="0"/></visual><worldbody>'
        f'<geom name="baked" type="box" size="0.5 0.5 0.5" pos="0 3 0.9" rgba="1 0 0 1" '
        f'group="{WINDOW_GEOM_GROUP}"/>'
        f'<geom name="absent" type="box" size="2 0.1 2" pos="0 1.5 0.9" rgba="0 0 1 1" '
        f'group="{ABSENT_GEOM_GROUP}"/>'
        "</worldbody></mujoco>"
    )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    # Looking along +y from the origin at 0.9 m: MuJoCo's camera frame with -z along +y, +y up.
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, np.array([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]]).reshape(-1))
    service = RenderService(model)
    view = service.view_at(
        "v",
        width=64,
        height=48,
        intrinsics=(50.0, 50.0, 31.5, 23.5),
        pos=[0, 0, 0.9],
        quat=quat,
        images=IMAGES,
    )
    try:
        (frame,) = service.render_now([(view, IMAGES)], data)
    finally:
        service.close()
    ids = frame.segmentation[..., 0]
    assert (ids == 0).mean() > 0.05  # the baked copy
    assert not (ids == 1).any()  # the absent wall, nowhere
    red = (frame.rgb[..., 0] > 150) & (frame.rgb[..., 2] < 80)
    assert red.mean() > 0.05


def test_the_headlight_follows_a_posed_view():
    """MuJoCo lights a scene's headlight at its camera: at the posed view, not at the free camera
    the scene was built through."""
    model, data = _world("centred")
    service = RenderService(model)
    pos, quat = _cam_pose(model, data)
    view = service.view_at(
        "v", width=W, height=H, intrinsics=(100.0, 100.0, 79.5, 59.5), pos=pos, quat=quat
    )
    try:
        service.render_now([(view, ("rgb",))], data)
        scene = service.renderer().scene
        head = [scene.lights[i] for i in range(scene.nlight) if scene.lights[i].headlight]
        assert head and np.allclose(head[0].pos, pos, atol=1e-5)
    finally:
        service.close()


def test_posed_gl_camera_counts_the_principal_point_from_the_pixel_centre():
    """The frustum's edges are the image's outer edges: half a pixel beyond the outer centres."""
    cam = mujoco.MjvGLCamera()
    posed_gl_camera(cam, [0, 0, 0], [1, 0, 0, 0], (100.0, 100.0, 9.5, 4.5), 20, 10, 1.0, 10.0)
    left = cam.frustum_center - cam.frustum_width
    right = cam.frustum_center + cam.frustum_width
    assert left == pytest.approx(-0.1) and right == pytest.approx(0.1)
    assert cam.frustum_top == pytest.approx(0.05) and cam.frustum_bottom == pytest.approx(-0.05)


# -- refusals ------------------------------------------------------------------------------------


def test_a_renderer_that_does_not_draw_a_posed_camera_is_refused_and_never_opened(tmp_path):
    path = tmp_path / "noposed.py"
    path.write_text(
        "from roqsim.render_api import IMAGES, Renderer, RendererCapabilities\n"
        "OPENED = []\n"
        "class NoPose(Renderer):\n"
        "    capabilities = RendererCapabilities(images=frozenset(IMAGES), principal_point=True,\n"
        "                                        non_square_pixels=True)\n"
        "    def open(self, scene):\n"
        "        OPENED.append(1)\n"
    )
    model, data = _world("centred")
    ref = f"{path}:NoPose"
    service = RenderService(model, renderer=ref)
    pos, quat = _cam_pose(model, data)
    view = service.view_at(
        "v", width=W, height=H, intrinsics=(90.0, 90.0, 79.5, 59.5), pos=pos, quat=quat
    )
    with pytest.raises(PluginError, match="a camera at a given pose"):
        service.check()
    with pytest.raises(RenderError, match="a camera at a given pose"):
        service.render_now([(view, ("rgb",))], data)
    assert service.renderer(ref) is None
    service.close()


def test_a_far_plane_inside_the_near_one_is_refused():
    model, data = _world("centred")
    service = RenderService(model)
    pos, quat = _cam_pose(model, data)
    kw = {"width": W, "height": H, "intrinsics": (90.0, 90.0, 79.5, 59.5), "pos": pos, "quat": quat}
    with pytest.raises(ValueError, match="not beyond the near one"):
        service.view_at("v", near=2.0, far=1.0, **kw)
    with pytest.raises(ValueError, match="> 0"):
        service.view_at("v", near=0.0, **kw)
    # A far plane given alone, inside the model's own near plane: refused when drawn.
    view = service.view_at("w", far=1e-6, **kw)
    with pytest.raises(RenderError, match="not beyond the near one"):
        service.render_now([(view, ("rgb",))], data)
    service.close()
