"""Clip planes per camera: each camera renders with its own, and the model keeps its own.

MuJoCo derives every camera's near plane from the model (``visual/map znear`` times the extent), so
in a large world it stands beyond what a camera close to an object needs. A camera's ``clip_near``
moves the plane in its own renderer's scene alone; the viewer, an export and every other camera keep
the model's.
"""

from __future__ import annotations

import mujoco
import numpy as np
import pytest
from roqsim_sensors.plugins.camera_common import CameraPlugin
from roqsim_sensors.plugins.depth_camera import DepthCameraPlugin

from roqsim.config import load_config_from_dict
from roqsim.context import SimContext
from roqsim.engine import Engine
from roqsim.plugin import Plugin
from roqsim.rendering import FrameRenderer
from roqsim.scene_source import declared_cameras

D415 = "roqsim_sensors.plugins.realsense_d415:RealsenseD415Plugin"
OAKD = "roqsim_sensors.plugins.oakd_camera:OakDCameraPlugin"

#: How far the red plate's face stands in front of the cameras, in metres.
PLATE = 0.15


class _BigRoom(Plugin):
    """A red plate 0.15 m in front of two cameras, a blue wall 3 m behind it, in a world 40 m
    across, where the model's own near plane (znear 0.01 of the extent) stands about 0.2 m out."""

    def build(self, spec: mujoco.MjSpec, ctx: SimContext) -> None:
        spec.worldbody.add_geom(
            type=mujoco.mjtGeom.mjGEOM_PLANE, size=[20, 20, 0.1], rgba=[0.3, 0.3, 0.3, 1]
        )
        spec.worldbody.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[-PLATE - 0.01, 0, 0.5],
            size=[0.01, 0.03, 0.03],
            rgba=[1, 0, 0, 1],
        )
        spec.worldbody.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX, pos=[-3, 0, 2], size=[0.1, 10, 2], rgba=[0, 0, 1, 1]
        )
        for name, y in (("cam", 0.0), ("cam2", 0.02)):
            spec.worldbody.add_camera(
                name=name, pos=[0, y, 0.5], xyaxes=[0, 1, 0, 0, 0, 1], fovy=45, resolution=[64, 48]
            )


class _SmallRoom(Plugin):
    """The plate and the camera, a wall 1 m out, in a world 2 m across: the near plane is 2 cm out."""

    def build(self, spec: mujoco.MjSpec, ctx: SimContext) -> None:
        spec.stat.extent = 2.0
        spec.worldbody.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[-PLATE - 0.01, 0, 0.5],
            size=[0.01, 0.03, 0.03],
            rgba=[1, 0, 0, 1],
        )
        spec.worldbody.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX, pos=[-1.1, 0, 0.5], size=[0.1, 1, 1], rgba=[0, 0, 1, 1]
        )
        spec.worldbody.add_camera(
            name="cam", pos=[0, 0, 0.5], xyaxes=[0, 1, 0, 0, 0, 1], fovy=45, resolution=[64, 48]
        )


def _engine(*cameras: tuple[str, str, dict], scene: str = "_BigRoom") -> Engine:
    """An engine on *scene* with ``(plugin ref, name, config)`` cameras, one step taken."""
    components = [{f"{__name__}:{scene}": {}}]
    components += [{ref: config, "name": name} for ref, name, config in cameras]
    engine = Engine(load_config_from_dict({"sim": {}, "components": components}))
    engine.ctx.seed = 1
    engine.setup()
    engine.reset()
    engine.step()
    return engine


def _plugin(engine: Engine, name: str) -> CameraPlugin:
    return next(p for p in engine.plugins if isinstance(p, CameraPlugin) and p.name == name)


def _sees_the_plate(frame: np.ndarray) -> bool:
    r, g, b = frame[24, 32].astype(int)
    return r > 150 and g < 60 and b < 60


def _model_planes(m: mujoco.MjModel) -> tuple[float, float]:
    return float(m.vis.map.znear), float(m.vis.map.zfar)


def test_the_models_own_plane_stands_beyond_the_plate():
    """The premise of every test here: without a camera's own plane, the plate is cut away."""
    engine = _engine()
    m = engine.ctx.model
    assert m.vis.map.znear * m.stat.extent > 0.18
    frames = FrameRenderer(m, 64, 48, camera=m.camera("cam").id)
    assert not _sees_the_plate(frames.render(engine.ctx.data))
    frames.close()
    engine.shutdown()


def test_each_colour_camera_renders_with_its_own_near_plane_and_the_model_keeps_its_own():
    engine = _engine()
    before = _model_planes(engine.ctx.model)
    engine.shutdown()
    engine = _engine(
        (D415, "near", {"camera": "cam", "clip_near": 0.05}),
        (D415, "nearer", {"camera": "cam2", "clip_near": 0.1}),
        (D415, "beyond", {"camera": "cam", "clip_near": 0.18}),
    )
    m = engine.ctx.model
    assert _sees_the_plate(_plugin(engine, "near")._color_frame)
    assert _sees_the_plate(_plugin(engine, "nearer")._color_frame)
    assert not _sees_the_plate(_plugin(engine, "beyond")._color_frame)
    # Nothing a camera did reached the model: the viewer and an export render with its planes.
    assert _model_planes(m) == before
    frames = FrameRenderer(m, 64, 48, camera=m.camera("cam").id)
    assert not _sees_the_plate(frames.render(engine.ctx.data))
    frames.close()
    engine.shutdown()


def test_the_model_is_not_written_during_a_render():
    """The planes live in the renderer's scene: the model reads its own while a frame is drawn."""
    engine = _engine()
    m = engine.ctx.model
    before = _model_planes(m)
    frames = FrameRenderer(m, 64, 48, camera=m.camera("cam").id, near=0.05, far=10.0)
    during = []

    def look(scene):
        during.append(_model_planes(m))
        assert scene.camera[0].frustum_near == pytest.approx(0.05)
        assert scene.camera[0].frustum_far == pytest.approx(10.0)

    assert _sees_the_plate(frames.render(engine.ctx.data, decorate=look))
    frames.render_depth(engine.ctx.data)
    assert during == [before] and _model_planes(m) == before
    frames.close()
    engine.shutdown()


def test_a_moved_near_plane_keeps_the_field_of_view():
    """The frustum's sides scale with its near plane, so the picture is the same one, nearer cut.

    Both planes here cut the plate and nothing else, so the two frames must agree.
    """
    engine = _engine()
    m, d = engine.ctx.model, engine.ctx.data
    cam = m.camera("cam").id
    own = FrameRenderer(m, 64, 48, camera=cam)
    moved = FrameRenderer(m, 64, 48, camera=cam, near=1.0)
    a, b = own.render(d).astype(int), moved.render(d).astype(int)
    # A zoomed or shifted view would move the wall's edge and the floor's horizon in every row; the
    # rows above the floor's grazing near edge (the bottom two) agree to a level.
    assert np.abs(a - b)[:40].max() <= 2
    own.close()
    moved.close()
    engine.shutdown()


@pytest.mark.parametrize(("near", "far"), [(None, None), (0.05, 50.0), (0.5, 4.0), (1.0, 3.5)])
def test_depth_is_in_metres_with_any_planes(near, far):
    """Linearised with the planes the frame was drawn with, depth reads the wall's true distance."""
    engine = _engine()
    m, d = engine.ctx.model, engine.ctx.data
    frames = FrameRenderer(m, 64, 48, camera=m.camera("cam").id, near=near, far=far)
    depth = frames.render_depth(d)
    # The top rows look over the plate at the wall's face, 2.9 m away along the view axis.
    assert depth[0, 32] == pytest.approx(2.9, abs=2e-3)
    if near is not None and near < PLATE:
        assert depth[24, 32] == pytest.approx(PLATE, abs=1e-3)
    frames.close()
    engine.shutdown()


def test_nothing_drawn_reads_infinite_depth():
    engine = _engine()
    m, d = engine.ctx.model, engine.ctx.data
    frames = FrameRenderer(m, 64, 48, camera=m.camera("cam").id, near=0.05, far=2.0)
    assert np.isinf(frames.render_depth(d)[0, 32])  # the wall lies beyond this far plane
    frames.close()
    engine.shutdown()


def test_two_depth_cameras_read_the_plate_from_their_own_near_planes():
    engine = _engine(
        (OAKD, "near", {"camera": "cam", "depth_camera": "cam", "clip_near": 0.1}),
        (OAKD, "beyond", {"camera": "cam2", "depth_camera": "cam2", "clip_near": 0.18}),
    )
    before = _model_planes(engine.ctx.model)
    near, beyond = _plugin(engine, "near"), _plugin(engine, "beyond")
    assert near._depth[24, 32] == pytest.approx(PLATE, abs=1e-3)
    assert _sees_the_plate(near._color_frame)
    assert not _sees_the_plate(beyond._color_frame)
    assert beyond._depth[24, 32] != pytest.approx(PLATE, abs=0.01)
    assert _model_planes(engine.ctx.model) == before
    engine.shutdown()


def test_a_surface_nearer_than_the_working_range_occludes_and_reads_no_return():
    """The device's planes are the model's widened, never narrowed: a plane at clip_near would cut
    the plate away and let the wall behind it read as a range."""
    engine = _engine(
        (OAKD, "far", {"camera": "cam", "depth_camera": "cam", "clip_near": 0.3}),
        scene="_SmallRoom",
    )
    m = engine.ctx.model
    assert m.vis.map.znear * m.stat.extent < PLATE
    plugin = _plugin(engine, "far")
    assert _sees_the_plate(plugin._color_frame)
    assert np.isinf(plugin._depth[24, 32])
    engine.shutdown()


def test_a_depth_cameras_colour_stream_sees_past_its_working_range():
    engine = _engine((OAKD, "short", {"camera": "cam", "depth_camera": "cam", "clip_far": 1.0}))
    plugin = _plugin(engine, "short")
    r, g, b = plugin._color_frame[0, 32].astype(int)
    assert b > 150 and r < 60  # the wall, 2.9 m out
    assert np.isinf(plugin._depth[0, 32])  # and no depth return from it
    engine.shutdown()


def test_the_scene_reports_each_cameras_own_planes():
    engine = _engine(
        (D415, "own", {"camera": "cam", "clip_near": 0.05, "clip_far": 9.0}),
        (D415, "model", {"camera": "cam2"}),
        (OAKD, "depth", {"camera": "cam", "depth_camera": "cam", "clip_near": 0.1}),
    )
    m = engine.ctx.model
    extent = float(m.stat.extent)
    planes = {c.sensor.split(".")[-1]: (c.znear, c.zfar) for c in declared_cameras(engine)}
    assert planes["own"] == (0.05, 9.0)
    assert planes["model"] == pytest.approx((m.vis.map.znear * extent, m.vis.map.zfar * extent))
    assert planes["depth"] == pytest.approx((0.1, m.vis.map.zfar * extent))
    engine.shutdown()


@pytest.mark.parametrize(
    ("config", "key"),
    [
        ({"clip_near": 0}, "'clip_near'"),
        ({"clip_near": -0.1}, "'clip_near'"),
        ({"clip_far": 0}, "'clip_far'"),
        ({"clip_near": 2.0, "clip_far": 1.0}, "'clip_far'"),
    ],
)
def test_an_invalid_colour_plane_is_a_config_error(config, key):
    from roqsim_sensors.plugins.realsense_d415 import RealsenseD415Plugin

    errors = RealsenseD415Plugin({}).validate_config(config)
    assert any(key in e for e in errors), errors


def test_a_depth_cameras_near_range_must_be_a_distance():
    errors = DepthCameraPlugin({}).validate_config({"clip_near": 0})
    assert any("'clip_near'" in e for e in errors), errors


def test_a_near_plane_beyond_the_models_far_plane_is_refused_at_configure():
    with pytest.raises(Exception, match="clip planes"):
        _engine((D415, "lost", {"camera": "cam", "clip_near": 1e4}))


@pytest.mark.parametrize("plane", [0.0, -1.0, float("nan"), float("inf")])
def test_a_renderer_refuses_a_plane_that_is_no_distance(plane):
    engine = _engine()
    frames = FrameRenderer(engine.ctx.model, 8, 8)
    with pytest.raises(ValueError):
        frames.near = plane
    with pytest.raises(ValueError):
        frames.far = plane
    frames.close()
    engine.shutdown()
