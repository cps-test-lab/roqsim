"""What a camera draws and what it senses: render planes per camera, the sensing range after them.

MuJoCo derives every camera's near plane from the model (``visual/map znear`` times the extent), so
in a large world it stands far beyond what a camera close to an object needs. A sensor camera draws
from its ``render_near`` instead, in its own renderer's scene; the model, the viewer and an export
keep the model's planes. A depth camera's ``clip_near``/``clip_far`` only mask the rendered depth.
"""

from __future__ import annotations

import mujoco
import numpy as np
import pytest
from roqsim_sensors.plugins.camera_common import DEFAULT_RENDER_NEAR, CameraPlugin

from roqsim.config import load_config_from_dict
from roqsim.context import SimContext
from roqsim.engine import Engine
from roqsim.plugin import Plugin, PluginError
from roqsim.rendering import FrameRenderer
from roqsim.scene_source import declared_cameras

D415 = "roqsim_sensors.plugins.realsense_d415:RealsenseD415Plugin"
OAKD = "roqsim_sensors.plugins.oakd_camera:OakDCameraPlugin"

#: How far the red plate's face stands in front of the cameras, in metres.
PLATE = 0.05
#: How far the blue wall's face stands in front of the cameras, in metres.
WALL = 2.9
#: The world's extent: the model's own near plane (znear 0.01 of it) stands 0.5 m out.
EXTENT = 50.0


class _BigRoom(Plugin):
    """A red plate 5 cm in front of two cameras and a blue wall behind it, in a world whose
    extent puts the model's own near plane 0.5 m out."""

    def build(self, spec: mujoco.MjSpec, ctx: SimContext) -> None:
        # The engine's own floor is the ground; a second plane here would be coplanar with it and
        # z-fight, and every face below stays clear of the floor for the same reason.
        spec.stat.extent = EXTENT
        spec.worldbody.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[-PLATE - 0.005, 0, 0.5],
            size=[0.005, 0.01, 0.01],
            rgba=[1, 0, 0, 1],
        )
        spec.worldbody.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[-WALL - 0.1, 0, 2.05],
            size=[0.1, 4, 2],
            rgba=[0, 0, 1, 1],
        )
        for name, y in (("cam", 0.0), ("cam2", 0.001)):
            spec.worldbody.add_camera(
                name=name, pos=[0, y, 0.5], xyaxes=[0, 1, 0, 0, 0, 1], fovy=45, resolution=[64, 48]
            )


def _engine(*cameras: tuple[str, str, dict]) -> Engine:
    """An engine on :class:`_BigRoom` with ``(plugin ref, name, config)`` cameras, stepped once."""
    components = [{f"{__name__}:_BigRoom": {}}]
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
    return r > 100 and g < 40 and b < 40


def _sees_the_wall(frame: np.ndarray, row: int = 2) -> bool:
    r, g, b = frame[row, 32].astype(int)
    return b > 150 and r < 60


def _model_planes(m: mujoco.MjModel) -> tuple[float, float]:
    return float(m.vis.map.znear), float(m.vis.map.zfar)


def _model_far(m: mujoco.MjModel) -> float:
    return float(m.vis.map.zfar) * float(m.stat.extent)


def test_the_models_own_near_plane_cuts_the_plate_away():
    """The premise: drawn with the model's planes, as the viewer and an export are, the plate is
    gone."""
    engine = _engine()
    m = engine.ctx.model
    assert m.vis.map.znear * m.stat.extent == pytest.approx(0.5)
    frames = FrameRenderer(m, 64, 48, camera=m.camera("cam").id)
    frame = frames.render(engine.ctx.data)
    assert not _sees_the_plate(frame) and _sees_the_wall(frame, 24)
    frames.close()
    engine.shutdown()


def test_a_colour_camera_sees_from_its_render_near_and_the_model_keeps_its_planes():
    engine = _engine()
    before = _model_planes(engine.ctx.model)
    engine.shutdown()
    engine = _engine(
        (D415, "default", {"camera": "cam"}),
        (D415, "near", {"camera": "cam2", "render_near": 0.03}),
        (D415, "beyond", {"camera": "cam", "render_near": 0.1}),
    )
    m = engine.ctx.model
    assert _sees_the_plate(_plugin(engine, "default")._color_frame)
    assert _sees_the_plate(_plugin(engine, "near")._color_frame)
    assert not _sees_the_plate(_plugin(engine, "beyond")._color_frame)
    # Nothing a camera did reached the model: the viewer and an export draw with its planes.
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
    frames = FrameRenderer(m, 64, 48, camera=m.camera("cam").id, near=0.01, far=10.0)
    during = []

    def look(scene):
        during.append(_model_planes(m))
        assert scene.camera[0].frustum_near == pytest.approx(0.01)
        assert scene.camera[0].frustum_far == pytest.approx(10.0)

    assert _sees_the_plate(frames.render(engine.ctx.data, decorate=look))
    frames.render_depth(engine.ctx.data)
    assert during == [before] and _model_planes(m) == before
    frames.close()
    engine.shutdown()


def test_a_moved_near_plane_keeps_the_field_of_view():
    """The frustum's sides scale with its near plane, so the view's angles are the model's and the
    plate covers the same pixels whichever near plane draws it."""
    engine = _engine()
    m, d = engine.ctx.model, engine.ctx.data
    cam = m.camera("cam").id

    def angles(near):
        frames = FrameRenderer(m, 64, 48, camera=cam, near=near)
        frame = frames.render(d)
        mask = (frame[..., 0] > 100) & (frame[..., 2] < 40)
        sides = [
            (
                c.frustum_top / c.frustum_near,
                c.frustum_bottom / c.frustum_near,
                c.frustum_center / c.frustum_near,
                c.frustum_width / c.frustum_near,
            )
            for c in frames.scene.camera
        ]
        frames.close()
        return np.array(sides), mask

    own, _ = angles(None)
    for near in (0.01, 0.04, 1.0):
        np.testing.assert_allclose(angles(near)[0], own, rtol=1e-5)
    near_mask, nearer_mask = angles(0.04)[1], angles(0.01)[1]
    assert near_mask.any() and np.array_equal(near_mask, nearer_mask)
    engine.shutdown()


def test_a_depth_camera_marks_a_surface_nearer_than_its_range_no_return_and_it_still_occludes():
    """Drawn from render_near, sensed from clip_near: the plate is in the picture and hides the
    wall, and its pixels read "no return" rather than the wall's range behind it."""
    engine = _engine(
        (OAKD, "d435like", {"camera": "cam", "depth_camera": "cam", "clip_near": 0.28})
    )
    plugin = _plugin(engine, "d435like")
    assert _sees_the_plate(plugin._color_frame)
    assert np.isinf(plugin._depth[24, 32])
    assert plugin._depth[2, 32] == pytest.approx(WALL, abs=1e-4)  # above the plate: the wall
    engine.shutdown()


def test_a_depth_cameras_colour_stream_sees_past_its_sensing_range():
    engine = _engine((OAKD, "short", {"camera": "cam", "depth_camera": "cam", "clip_far": 1.0}))
    plugin = _plugin(engine, "short")
    assert _sees_the_wall(plugin._color_frame)
    assert np.isinf(plugin._depth[2, 32])  # and no depth return from it
    engine.shutdown()


def _wall_at(distance: float) -> tuple[mujoco.MjModel, mujoco.MjData]:
    """A wall *distance* m in front of camera ``cam``, in a world of :data:`EXTENT`."""
    model = mujoco.MjModel.from_xml_string(
        f"""<mujoco>
          <statistic extent="{EXTENT}"/>
          <worldbody>
            <geom type="box" pos="{-distance - 0.5} 0 0.5" size="0.5 100 100"/>
            <camera name="cam" pos="0 0 0.5" xyaxes="0 1 0 0 0 1" fovy="45" resolution="64 48"/>
          </worldbody>
        </mujoco>"""
    )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


#: The bound on the error of depth drawn from the default render_near to the model's far plane
#: (2.5 km here), per range. Measured over walls at 40 distances from each range to 0.37 % beyond
#: it (EGL): at worst 1.2e-7 m at 1 m, 9.5e-7 m at 10 m and 7.6e-6 m at 50 m, one float32 step of
#: the distance. The bound is twice that step.
DEPTH_TOLERANCE = {1.0: 2.4e-7, 10.0: 1.9e-6, 50.0: 1.53e-5}


@pytest.mark.parametrize("base", sorted(DEPTH_TOLERANCE))
@pytest.mark.parametrize("offset", [0.0, 0.0011, 0.0023, 0.0037])
def test_depth_from_the_default_render_near_is_metres_at_any_range(base, offset):
    distance = base * (1.0 + offset)
    m, d = _wall_at(distance)
    frames = FrameRenderer(
        m, 64, 48, camera=m.camera("cam").id, near=DEFAULT_RENDER_NEAR, far=_model_far(m)
    )
    assert np.abs(frames.render_depth(d) - distance).max() <= DEPTH_TOLERANCE[base]
    frames.close()


@pytest.mark.parametrize(("near", "far"), [(None, None), (0.01, 50.0), (0.5, 4.0), (1.0, 3.5)])
def test_depth_is_in_metres_with_any_planes(near, far):
    """Linearised with the planes the frame was drawn with, depth reads the wall's true distance."""
    engine = _engine()
    m, d = engine.ctx.model, engine.ctx.data
    frames = FrameRenderer(m, 64, 48, camera=m.camera("cam").id, near=near, far=far)
    depth = frames.render_depth(d)
    assert depth[2, 32] == pytest.approx(WALL, abs=1e-4)
    if near is not None and near < PLATE:
        assert depth[24, 32] == pytest.approx(PLATE, abs=1e-5)
    frames.close()
    engine.shutdown()


def test_nothing_drawn_reads_infinite_depth():
    engine = _engine()
    m, d = engine.ctx.model, engine.ctx.data
    frames = FrameRenderer(m, 64, 48, camera=m.camera("cam").id, near=1.0, far=2.0)
    assert np.isinf(frames.render_depth(d)[2, 32])  # the wall lies beyond this far plane
    frames.close()
    engine.shutdown()


def test_the_scene_reports_each_cameras_render_planes():
    engine = _engine(
        (D415, "own", {"camera": "cam", "render_near": 0.02}),
        (D415, "default", {"camera": "cam2"}),
        (OAKD, "depth", {"camera": "cam", "depth_camera": "cam", "clip_near": 0.1}),
    )
    far = _model_far(engine.ctx.model)
    planes = {c.sensor.split(".")[-1]: (c.znear, c.zfar) for c in declared_cameras(engine)}
    assert planes["own"] == pytest.approx((0.02, far))
    assert planes["default"] == pytest.approx((DEFAULT_RENDER_NEAR, far))
    assert planes["depth"] == pytest.approx((DEFAULT_RENDER_NEAR, far))  # not its sensing range
    engine.shutdown()


@pytest.mark.parametrize("render_near", [0, -0.1, float("nan"), float("inf")])
def test_a_render_near_that_is_no_distance_is_a_config_error(render_near):
    from roqsim_sensors.plugins.realsense_d415 import RealsenseD415Plugin

    errors = RealsenseD415Plugin({}).validate_config({"render_near": render_near})
    assert any("'render_near'" in e for e in errors), errors


def test_a_depth_camera_must_render_from_no_further_than_it_senses():
    from roqsim_sensors.plugins.oakd_camera import OakDCameraPlugin

    errors = OakDCameraPlugin({}).validate_config({"render_near": 0.5, "clip_near": 0.3})
    assert any("'clip_near'" in e and "'render_near'" in e for e in errors), errors
    assert OakDCameraPlugin({}).validate_config({"render_near": 0.3, "clip_near": 0.3}) == []


def test_a_render_near_beyond_the_models_far_plane_is_a_config_error_at_configure():
    with pytest.raises(PluginError, match=r"lost.*'render_near' \(10000 m\) must be nearer"):
        _engine((D415, "lost", {"camera": "cam", "render_near": 1e4}))


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
