# Copyright (C) 2026 Frederik Pasch
# SPDX-License-Identifier: Apache-2.0
"""Two renderers, one projection: a point of known position lands on the same pixel in MuJoCo's
picture of a posed camera and in the splat renderer's, to the sub-pixel.

The splat renderer draws on a CUDA GPU through gsplat; where it cannot open, its package's own tests
skip. What is checked here runs anywhere the package imports, which needs neither: the projection
the splat renderer hands gsplat (its world-to-camera matrix and its 3x3 intrinsics, gsplat counting
pixels from the image's corner) against the frustum MuJoCo's renderer draws a posed camera through,
and against the picture MuJoCo draws. Not covered: gsplat's own rasterisation of a Gaussian (its
screen-space dilation and its tile sort), the compositing of moving bodies, and the CUDA path; the
splat package's GPU tests check those where a GPU is (``roqsim_render_splat/tests``).
"""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from roqsim.render_api import CameraRequest, CameraSpec
from roqsim.rendering import RenderService, posed_gl_camera

splat = pytest.importorskip(
    "roqsim_render_splat.renderer", reason="the splat renderer lives in roqsim_render_splat"
)

W, H = 200, 150

#: Cameras of the test: centred, off-centre with non-square pixels, and rolled.
CAMERAS = {
    "centred": ((160.0, 160.0, 99.5, 74.5), 0.0),
    "off_centre": ((170.0, 182.0, 88.25, 81.75), 0.0),
    "rolled": ((165.0, 165.0, 104.0, 70.0), 0.6),
}

#: Points in front of the camera (world frame), spread over the picture and in depth.
POINTS = np.array([[2.0, 0.0, 1.0], [2.5, 0.6, 1.4], [3.0, -0.9, 0.7], [1.6, 0.3, 0.8]])


def _quat(roll: float) -> np.ndarray:
    """A camera at the origin's height 1 m looking along +x, +z up, rolled by *roll* about its axis
    (MuJoCo's camera frame: -z along the view, +y up)."""
    base = np.zeros(4)
    mujoco.mju_mat2Quat(base, np.array([[0, 0, -1.0], [-1.0, 0, 0], [0, 1.0, 0]]).reshape(-1))
    turn = np.zeros(4)
    mujoco.mju_axisAngle2Quat(turn, [0.0, 0.0, 1.0], roll)
    out = np.zeros(4)
    mujoco.mju_mulQuat(out, base, turn)
    return out


POS = np.array([0.0, 0.0, 1.0])


def _request(name: str) -> CameraRequest:
    intrinsics, roll = CAMERAS[name]
    spec = CameraSpec(id=name, width=W, height=H, intrinsics=intrinsics)
    return CameraRequest(spec, frozenset({"rgb"}), POS, _quat(roll), fovy=0.0)


def _splat_pixels(request: CameraRequest) -> np.ndarray:
    """Pixels, from the top-left pixel's centre, of POINTS through what the splat renderer gives
    gsplat (corner coordinates, hence the half pixel)."""
    placement = (np.eye(3), np.zeros(3), 1.0)
    view = splat.view_matrix(request.pos, request.quat, placement)
    k = splat.intrinsics_matrix(request)
    cam = (view[:3, :3] @ POINTS.T).T + view[:3, 3]
    uv = (k @ cam.T).T
    return uv[:, :2] / uv[:, 2:] - 0.5


def _gl_pixels(request: CameraRequest) -> np.ndarray:
    """Pixels of POINTS through the frustum MuJoCo's renderer draws the posed camera with:
    ``glFrustum`` and the viewport, OpenGL rasterising at pixel centres."""
    gl = mujoco.MjvGLCamera()
    spec = request.camera
    posed_gl_camera(gl, request.pos, request.quat, spec.intrinsics, W, H, 0.05, 50.0)
    fwd, up = np.array(gl.forward, float), np.array(gl.up, float)
    right = np.cross(fwd, up)
    n = float(gl.frustum_near)
    left = gl.frustum_center - gl.frustum_width
    rgt = gl.frustum_center + gl.frustum_width
    bottom, top = float(gl.frustum_bottom), float(gl.frustum_top)
    out = []
    for p in POINTS:
        d = p - np.array(gl.pos, float)
        xe, ye, ze = d @ right, d @ up, -(d @ fwd)
        xn = (2 * n / (rgt - left) * xe + (rgt + left) / (rgt - left) * ze) / -ze
        yn = (2 * n / (top - bottom) * ye + (top + bottom) / (top - bottom) * ze) / -ze
        out.append(((xn + 1) / 2 * W - 0.5, (1 - yn) / 2 * H - 0.5))
    return np.array(out)


def _pinhole_pixels(request: CameraRequest) -> np.ndarray:
    """The pinhole's own answer: the camera frame from the pose, then ``fx x/z + cx``."""
    fx, fy, cx, cy = request.camera.intrinsics
    mat = np.zeros(9)
    mujoco.mju_quat2Mat(mat, request.quat)
    mat = mat.reshape(3, 3)
    local = (POINTS - request.pos) @ mat  # MuJoCo's camera frame: x right, y up, -z forward
    x, y, z = local[:, 0], -local[:, 1], -local[:, 2]
    return np.stack([fx * x / z + cx, fy * y / z + cy], axis=1)


@pytest.mark.parametrize("name", sorted(CAMERAS))
def test_the_splat_renderer_and_mujoco_s_project_a_point_to_the_same_pixel(name):
    request = _request(name)
    want = _pinhole_pixels(request)
    np.testing.assert_allclose(_splat_pixels(request), want, atol=1e-6)
    np.testing.assert_allclose(_gl_pixels(request), want, atol=1e-4)  # float32 GL camera


@pytest.mark.parametrize("name", sorted(CAMERAS))
def test_mujoco_draws_a_point_where_the_splat_renderer_projects_it(name):
    """A small sphere at each point, drawn by MuJoCo through the posed camera at four times the size
    (the intrinsics scaled about the image's corner, as a smaller picture of the same camera): the
    centroid of its pixels, brought back to the picture's size, is the splat renderer's pixel to a
    tenth of a pixel. The sphere is two pixels of the picture across, so the perspective's offset of
    its outline from its centre stays a few thousandths of a pixel."""
    request = _request(name)
    fx, fy, cx, cy = request.camera.intrinsics
    s = 4
    big = (fx * s, fy * s, (cx + 0.5) * s - 0.5, (cy + 0.5) * s - 0.5)
    want = _splat_pixels(request)
    for point, (u, v) in zip(POINTS, want, strict=True):
        z = float(np.linalg.norm(point - POS))
        radius = 1.0 * z / fx  # a pixel of the picture
        model = mujoco.MjModel.from_xml_string(
            '<mujoco><visual><quality offsamples="0"/>'
            f'<global offwidth="{W * s}" offheight="{H * s}"/></visual>'
            f'<worldbody><geom type="sphere" size="{radius}" pos="{point[0]} {point[1]} '
            f'{point[2]}"/></worldbody></mujoco>'
        )
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        service = RenderService(model)
        view = service.view_at(
            name,
            width=W * s,
            height=H * s,
            intrinsics=big,
            pos=request.pos,
            quat=request.quat,
            near=0.05,
            far=50.0,
            images=("segmentation",),
        )
        try:
            (frame,) = service.render_now([(view, ("segmentation",))], data)
        finally:
            service.close()
        rows, cols = np.nonzero(frame.segmentation[..., 0] == 0)
        assert len(rows) > 20, f"{name}: the sphere at {point} is not in the picture"
        got_u = (float(cols.mean()) + 0.5) / s - 0.5
        got_v = (float(rows.mean()) + 0.5) / s - 0.5
        assert got_u == pytest.approx(u, abs=0.1), (name, point)
        assert got_v == pytest.approx(v, abs=0.1), (name, point)
