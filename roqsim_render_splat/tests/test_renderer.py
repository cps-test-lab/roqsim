# Copyright (C) 2026 Frederik Pasch
# SPDX-License-Identifier: Apache-2.0
"""The splat renderer draws a capture where its pose puts it, and the moving bodies into it by depth.

Every test but the last two needs a CUDA GPU (the ``gpu`` fixture skips, saying so, where there is
none). The splats are written by the test: a few Gaussians at known places, so where they must land
in a picture is computed, not looked at.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np
import pytest

from roqsim.plugin import PluginError
from roqsim.pose import rpy_to_quat
from roqsim.render_api import RenderError
from roqsim.rendering import RenderService
from roqsim_render_splat.ply import Splat, write_splat
from roqsim_render_splat.renderer import SplatRenderer, splat_placement
from roqsim_render_splat.synthetic import SH_C0

#: Where the test captures sit: tipped, turned, offset and scaled, as a real capture is.
PLACEMENT = {
    "pose": {
        "position": {"x": 0.2, "y": -0.1, "z": 0.5},
        "orientation": {"roll": -1.5708, "pitch": 0.0, "yaw": 0.3},
    },
    "scale": 0.5,
}

#: A camera at 1 m looking along +x, a lens camera with its principal point off centre and
#: non-square pixels, and a free box that moves.
XML = """
<mujoco>
  <visual><global offwidth="128" offheight="96"/><quality offsamples="0"/></visual>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1" directional="true"/>
    <geom name="floor" type="plane" size="5 5 .1" rgba="1 1 1 1"/>
    <geom name="static_post" type="box" pos="2 0.8 1" size=".1 .1 .5" rgba="1 1 0 1"/>
    <camera name="cam" pos="0 0 1" xyaxes="0 -1 0 0 0 1" fovy="60"/>
    <camera name="lens" pos="0 0 1" xyaxes="0 -1 0 0 0 1" sensorsize="0.0064 0.0048"
            focal="0.004 0.005" principal="0.0004 -0.0002" resolution="64 48"/>
    <body name="crate" pos="1.5 0 1">
      <freejoint/>
      <geom type="box" size=".2 .2 .2" rgba="0 1 0 1"/>
    </body>
  </worldbody>
</mujoco>
"""


def _capture_points(world_points: np.ndarray) -> np.ndarray:
    rot, trans, scale = splat_placement(PLACEMENT)
    return (rot.T @ (world_points - trans).T).T / scale


def _write(tmp_path, world_points, colours, sigma_m=0.02, name="room.ply") -> str:
    """A splat of round, opaque Gaussians at *world_points* (m) with *colours* (0-1)."""
    n = len(world_points)
    scale = PLACEMENT["scale"]
    splat = Splat(
        means=_capture_points(np.asarray(world_points, float)).astype(np.float32),
        quats=np.tile([1.0, 0.0, 0.0, 0.0], (n, 1)).astype(np.float32),
        log_scales=np.full((n, 3), math.log(sigma_m / scale), dtype=np.float32),
        opacity_logits=np.full(n, 8.0, dtype=np.float32),
        sh=((np.asarray(colours, float) - 0.5) / SH_C0).astype(np.float32)[:, None, :],
        sh_degree=0,
    )
    write_splat(tmp_path / name, splat)
    return name


def _wall(x: float, colour, half: float = 3.0, step: float = 0.05):
    """A dense square of Gaussians in the plane at *x*, facing the camera."""
    ys, zs = np.meshgrid(np.arange(-half, half, step), np.arange(1 - half, 1 + half, step))
    pts = np.stack([np.full(ys.size, x), ys.ravel(), zs.ravel()], axis=1)
    return pts, np.tile(colour, (len(pts), 1))


def _service(tmp_path, file, cameras=("cam",), images=("rgb", "depth"), **settings):
    model = mujoco.MjModel.from_xml_string(XML)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    block = {"splat": {"file": file, **PLACEMENT}, **settings}
    service = RenderService(model, renderer="splat", renderers={"splat": block}, base_dir=tmp_path)
    views = [service.view(c, camera=c, width=128, height=96, images=images) for c in cameras]
    service.check()
    return service, views, model, data


def _draw(service, views, data, images=("rgb", "depth")):
    return service.render_now([(v, images) for v in views], data)


def _expected_pixel(model, data, camera: str, point, spec) -> tuple[float, float, float]:
    """Where *point* projects in *camera* (column, row from the top-left pixel's centre) and its
    depth along the view axis."""
    cam = model.camera(camera).id
    pos, mat = data.cam_xpos[cam], data.cam_xmat[cam].reshape(3, 3)
    d = np.asarray(point) - pos
    right, up, back = mat[:, 0], mat[:, 1], mat[:, 2]
    x, y, z = d @ right, -(d @ up), -(d @ back)
    fx, fy, cx, cy = spec.intrinsics
    return cx + fx * x / z, cy + fy * y / z, z


def _centroid(rgb: np.ndarray) -> tuple[float, float]:
    weight = rgb.astype(float).sum(axis=2)
    rows, cols = np.indices(weight.shape)
    total = weight.sum()
    return float((cols * weight).sum() / total), float((rows * weight).sum() / total)


def _move_crate(model, data, x: float) -> None:
    data.qpos[:3] = [x, 0.0, 1.0]
    mujoco.mj_forward(model, data)


@pytest.mark.parametrize("camera", ["cam", "lens"])
def test_a_gaussian_lands_where_the_pose_and_the_camera_put_it(gpu, tmp_path, camera):
    point = np.array([[3.0, 0.4, 1.3]])
    file = _write(tmp_path, point, [[1.0, 1.0, 1.0]], sigma_m=0.01)
    service, (view,), model, data = _service(tmp_path, file, cameras=(camera,))
    _move_crate(model, data, -5.0)  # out of view, behind the camera
    try:
        (frame,) = _draw(service, [view], data)
    finally:
        service.close()
    col, row, z = _expected_pixel(model, data, camera, point[0], view.spec)
    got_col, got_row = _centroid(frame.rgb)
    assert got_col == pytest.approx(col, abs=0.25) and got_row == pytest.approx(row, abs=0.25)
    # Depth along the view axis, in metres: the scale of the capture is applied.
    assert frame.depth[int(round(row)), int(round(col))] == pytest.approx(z, abs=0.01)
    assert np.isinf(frame.depth[0, 0])  # nothing there: no return


def test_a_moving_body_in_front_hides_the_room_and_behind_it_is_hidden(gpu, tmp_path):
    pts, colours = _wall(3.0, [1.0, 0.0, 0.0])
    service, (view,), model, data = _service(tmp_path, _write(tmp_path, pts, colours, sigma_m=0.04))
    try:
        _move_crate(model, data, 1.5)  # in front of the wall: its face is at 1.3 m
        (front,) = _draw(service, [view], data)
        _move_crate(model, data, 4.0)  # behind the wall
        (behind,) = _draw(service, [view], data)
    finally:
        service.close()
    centre = front.rgb[48, 64]
    assert centre[1] > 150 and centre[1] > centre[0] + 80, centre  # the green crate, lit
    assert front.depth[48, 64] == pytest.approx(1.3, abs=0.01)
    assert behind.rgb[48, 64][0] > 200 and behind.rgb[48, 64][1] < 30  # the red wall
    assert behind.depth[48, 64] == pytest.approx(3.0, abs=0.03)
    # The static geometry (the floor, the yellow post) is the world's collision, not drawn.
    yellow = (front.rgb[..., 0] > 150) & (front.rgb[..., 1] > 150) & (front.rgb[..., 2] < 80)
    assert not yellow.any()


def test_a_frame_is_a_function_of_the_state(gpu, tmp_path):
    pts, colours = _wall(3.0, [0.2, 0.5, 0.9])
    rng = np.random.default_rng(0)
    colours = np.clip(colours + rng.normal(0, 0.2, colours.shape), 0, 1)
    file = _write(tmp_path, pts + rng.normal(0, 0.05, pts.shape), colours, sigma_m=0.05)
    frames = []
    for _ in range(2):
        service, (view,), model, data = _service(tmp_path, file)
        _move_crate(model, data, 1.5)
        try:
            frames.append(_draw(service, [view], data)[0])
            frames.append(_draw(service, [view], data)[0])
        finally:
            service.close()
    for other in frames[1:]:
        np.testing.assert_array_equal(other.rgb, frames[0].rgb)
        np.testing.assert_array_equal(other.depth, frames[0].depth)


def test_a_batch_draws_what_each_camera_drew_alone(gpu, tmp_path):
    pts, colours = _wall(3.0, [0.9, 0.6, 0.1])
    file = _write(tmp_path, pts, colours, sigma_m=0.04)
    service, views, model, data = _service(tmp_path, file, cameras=("cam", "lens"))
    try:
        both = _draw(service, views, data)
        alone = [_draw(service, [v], data)[0] for v in views]
    finally:
        service.close()
    for a, b in zip(both, alone, strict=True):
        np.testing.assert_array_equal(a.rgb, b.rgb)
        np.testing.assert_array_equal(a.depth, b.depth)


def test_a_camera_asking_for_segmentation_is_refused_at_load(tmp_path):
    model = mujoco.MjModel.from_xml_string(XML)
    service = RenderService(
        model,
        renderer="splat",
        renderers={"splat": {"splat": {"file": "room.ply"}}},
        base_dir=tmp_path,
    )
    service.view("cam", camera="cam", width=8, height=8, images=("rgb", "segmentation"))
    with pytest.raises(
        PluginError, match="camera 'cam' asks renderer 'splat' for image 'segmentation'"
    ):
        service.check()


def test_without_a_gpu_it_refuses_to_open_and_names_why(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    renderer = SplatRenderer({"splat": {"file": "room.ply"}})
    model = mujoco.MjModel.from_xml_string(XML)
    from roqsim.render_api import SceneDescription
    from roqsim.rendering import moving_bodies

    scene = SceneDescription(model, moving_bodies(model), [], {}, base_dir=tmp_path)
    with pytest.raises(RenderError, match=r"draws on a CUDA GPU \('cuda:0'\), and torch sees none"):
        renderer.open(scene)


def test_the_placement_reads_a_pose_as_any_world_pose_is_read():
    rot, trans, scale = splat_placement(PLACEMENT)
    q = rpy_to_quat(-1.5708, 0.0, 0.3)
    m = np.zeros(9)
    mujoco.mju_quat2Mat(m, np.asarray(q))
    np.testing.assert_allclose(rot, m.reshape(3, 3), atol=1e-9)
    assert trans.tolist() == [0.2, -0.1, 0.5] and scale == 0.5


def test_roqsim_render_draws_the_example_world_as_installed(gpu, tmp_path):
    """The README's last command: the shipped example, by reference, through the splat renderer."""
    from PIL import Image

    from roqsim.render import main

    out = tmp_path / "shot.png"
    assert main(["roqsim_render_splat:splat_room_example", "--out", str(out)]) in (0, None)
    pixels = np.asarray(Image.open(out).convert("RGB"))
    assert pixels.shape == (540, 960, 3)
    # The room (its checkered floor, its walls) fills the picture: nothing is background black.
    assert (pixels.sum(axis=2) == 0).mean() < 0.01


def test_a_batch_split_into_passes_draws_the_same_frames(gpu, tmp_path):
    """``max_pass_mib`` bounds the GPU memory of a pass; the pictures do not change."""
    pts, colours = _wall(3.0, [0.3, 0.7, 0.4])
    file = _write(tmp_path, pts, colours, sigma_m=0.04)
    one, views, model, data = _service(tmp_path, file, cameras=("cam", "lens"))
    many, views2, _, _ = _service(tmp_path, file, cameras=("cam", "lens"), max_pass_mib=1e-6)
    try:
        a = _draw(one, views, data)
        b = _draw(many, views2, data)
        assert many.renderer("splat").stats["passes"] == 2  # one camera a pass
        assert one.renderer("splat").stats["passes"] == 1
    finally:
        one.close()
        many.close()
    for x, y in zip(a, b, strict=True):
        np.testing.assert_array_equal(x.rgb, y.rgb)
        np.testing.assert_array_equal(x.depth, y.depth)


def test_faint_gaussians_are_dropped_at_load_when_asked(gpu, tmp_path):
    pts, colours = _wall(3.0, [0.3, 0.7, 0.4])
    file = _write(tmp_path, pts, colours, sigma_m=0.04)  # every one 0.9997 opaque
    service, views, _, data = _service(tmp_path, file, min_opacity=0.5)
    try:
        _draw(service, views, data)
        assert service.renderer("splat").stats["gaussians"] == len(pts)
    finally:
        service.close()
    service, views, _, data = _service(tmp_path, file, min_opacity=0.9999)
    try:
        with pytest.raises(
            RenderError, match=r"min_opacity 0.9999 drops every one of the capture's"
        ):
            _draw(service, views, data)
    finally:
        service.close()
