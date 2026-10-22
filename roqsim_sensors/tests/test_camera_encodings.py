"""The colour stream's wire encoding, depth registered into the colour image, and the depth stamp."""

from __future__ import annotations

import mujoco
import numpy as np
import pytest
from roqsim_sensors.plugins.camera_common import LUMA_BT601, encode_color

from roqsim.config import PluginError, load_config_from_dict
from roqsim.context import SimContext
from roqsim.engine import Engine
from roqsim.plugin import Plugin
from roqsim.types import Image

D435 = "roqsim_sensors.plugins.realsense_d435:RealsenseD435Plugin"
OAKD = "roqsim_sensors.plugins.oakd_camera:OakDCameraPlugin"


class _TwoCameraScene(Plugin):
    """A red box 2 m away along -x (where both cameras look), seen by a colour camera ``cam`` and a wider, coarser depth camera
    ``cam_depth`` 5 cm to its left -- a device with its own depth optics."""

    def build(self, spec: mujoco.MjSpec, ctx: SimContext) -> None:
        spec.worldbody.add_geom(
            type=mujoco.mjtGeom.mjGEOM_PLANE, size=[5, 5, 0.1], rgba=[0.3, 0.3, 0.3, 1]
        )
        spec.worldbody.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[-2, 0, 0.5],
            size=[0.3, 0.3, 0.3],
            rgba=[0.8, 0.1, 0.1, 1],
        )
        spec.worldbody.add_camera(
            name="cam", pos=[0, 0, 0.5], xyaxes=[0, 1, 0, 0, 0, 1], fovy=45, resolution=[64, 48]
        )
        spec.worldbody.add_camera(
            name="cam_depth",
            pos=[0, 0.05, 0.5],
            xyaxes=[0, 1, 0, 0, 0, 1],
            fovy=60,
            resolution=[32, 24],
        )


def _engine(plugin_ref: str = D435, **config) -> Engine:
    cfg = load_config_from_dict(
        {
            "sim": {},
            "components": [
                {f"{__name__}:_TwoCameraScene": {}},
                {plugin_ref: {"camera": "cam", **config}},
            ],
        }
    )
    engine = Engine(cfg)
    engine.ctx.seed = 1
    engine.setup()
    engine.reset()
    engine.step()
    return engine


def _endpoint(engine: Engine, name: str):
    return next((e for e in engine.ctx.interface.all() if e.name == name), None)


def _topic(ep) -> str:
    return ep.topic or ep.backend["ros2"].get("topic") or ep.name


# -- colour encoding --------------------------------------------------------------------------


def test_rgb8_is_the_default_and_unchanged():
    engine = _engine(OAKD, depth_camera="cam")
    frame = _endpoint(engine, "image").read()
    assert frame.encoding == "rgb8" and frame.data.shape == (48, 64, 3)
    # The red box: red dominates the centre of an RGB frame.
    r, g, b = frame.data[24, 32]
    assert r > g and r > b
    engine.shutdown()


def test_bgr8_is_the_channel_swap_of_the_render():
    rgb = _endpoint(_engine(OAKD, depth_camera="cam"), "image").read().data
    engine = _engine(OAKD, depth_camera="cam", color_encoding="bgr8")
    frame = _endpoint(engine, "image").read()
    assert frame.encoding == "bgr8"
    np.testing.assert_array_equal(frame.data, rgb[..., ::-1])
    b, g, r = frame.data[24, 32]
    assert r > g and r > b
    engine.shutdown()


def test_mono8_is_the_bt601_luma_of_the_render():
    rgb = _endpoint(_engine(OAKD, depth_camera="cam"), "image").read().data
    engine = _engine(OAKD, depth_camera="cam", color_encoding="mono8")
    frame = _endpoint(engine, "image").read()
    assert (
        frame.encoding == "mono8" and frame.data.shape == (48, 64) and frame.data.dtype == np.uint8
    )
    want = np.rint(rgb.astype(np.float32) @ LUMA_BT601).astype(np.uint8)
    np.testing.assert_array_equal(frame.data, want)
    # The compressed companion reads the same pixels.
    assert _endpoint(engine, "image_compressed").read().data is frame.data
    engine.shutdown()


def test_mono8_matches_opencv_to_a_grey_level():
    cv2 = pytest.importorskip("cv2")
    rgb = np.random.default_rng(0).integers(0, 256, (16, 16, 3), dtype=np.uint8)
    diff = encode_color(rgb, "mono8").astype(int) - cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    assert np.abs(diff).max() <= 1


def test_noise_on_a_mono8_stream_is_in_grey_levels():
    clean = _endpoint(_engine(OAKD, depth_camera="cam", color_encoding="mono8"), "image").read()
    noisy = _endpoint(
        _engine(OAKD, depth_camera="cam", color_encoding="mono8", noise={"gaussian_sigma": 5.0}),
        "image",
    ).read()
    assert noisy.data.shape == clean.data.shape
    assert 2.0 < np.std(noisy.data.astype(float) - clean.data.astype(float)) < 8.0


def test_an_unknown_colour_encoding_is_refused():
    with pytest.raises(PluginError, match="color_encoding"):
        _engine(OAKD, depth_camera="cam", color_encoding="yuv422")


# -- registered depth -------------------------------------------------------------------------


def test_registered_depth_is_off_by_default():
    engine = _engine(depth=True, depth_camera="cam_depth")
    assert _endpoint(engine, "registered_depth") is None
    assert _endpoint(engine, "registered_depth_camera_info") is None
    engine.shutdown()


def test_registered_depth_has_the_colour_size_intrinsics_and_frame():
    engine = _engine(depth=True, depth_camera="cam_depth", registered_depth=True)
    depth = _endpoint(engine, "depth").read()
    registered = _endpoint(engine, "registered_depth")
    frame = registered.read()
    assert depth.data.shape == (24, 32)
    assert frame.data.shape == (48, 64) and frame.encoding == depth.encoding
    assert (
        registered.backend["ros2"]["frame_id"]
        == _endpoint(engine, "image").backend["ros2"]["frame_id"]
    )
    info = _endpoint(engine, "registered_depth_camera_info")
    assert info.read() == _endpoint(engine, "camera_info").read()
    # realsense-ros's aligned-depth topics, the info beside its image.
    assert _topic(registered) == "camera/aligned_depth_to_color/image_raw"
    assert _topic(info) == "camera/aligned_depth_to_color/camera_info"
    engine.shutdown()


def test_registered_depth_is_the_depth_of_each_colour_pixel():
    """The box face is 1.7 m from both cameras (it faces them squarely), so the registered pixel at
    the colour image's centre and the depth camera's own centre read the same range."""
    engine = _engine(depth=True, depth_camera="cam_depth", registered_depth=True)
    registered = _endpoint(engine, "registered_depth").read().data
    depth = _endpoint(engine, "depth").read().data
    assert registered[24, 32] == pytest.approx(1.7, abs=1e-3)
    assert depth[12, 16] == pytest.approx(1.7, abs=1e-3)
    engine.shutdown()


def test_registered_depth_follows_the_depth_encoding():
    engine = _engine(
        depth=True, depth_camera="cam_depth", registered_depth=True, depth_encoding="16UC1"
    )
    frame = _endpoint(engine, "registered_depth").read()
    assert frame.encoding == "16UC1" and frame.data.dtype == np.uint16
    assert frame.data[24, 32] == 1700
    engine.shutdown()


def test_registered_depth_topic_can_be_renamed_and_its_info_follows():
    engine = _engine(
        depth=True,
        depth_camera="cam_depth",
        registered_depth=True,
        topics={"registered_depth": "aligned/image"},
    )
    assert _endpoint(engine, "registered_depth").topic == "aligned/image"
    engine.shutdown()


def test_a_registered_depth_subscriber_alone_justifies_the_render():
    engine = _engine(depth=True, depth_camera="cam_depth", registered_depth=True)
    for ep in engine.ctx.interface.all():
        ep.has_subscribers = lambda: False
    registered = _endpoint(engine, "registered_depth")
    registered.has_subscribers = lambda: True
    plugin = next(p for p in engine.plugins if hasattr(p, "registered_depth"))
    plugin.on_reset(engine.ctx)
    assert plugin._due(engine.ctx)
    engine.shutdown()


def test_registered_depth_through_the_colour_camera_is_refused():
    with pytest.raises(PluginError, match="registered_depth"):
        _engine(OAKD, depth_camera="cam", registered_depth=True)


# -- the depth frame's stamp ------------------------------------------------------------------


def test_a_depth_frame_carries_the_sim_time_it_was_captured_at():
    engine = _engine(depth=True, depth_camera="cam_depth", registered_depth=True)
    for name in ("image", "depth", "registered_depth"):
        frame = _endpoint(engine, name).read()
        assert isinstance(frame, Image)
        assert frame.stamp == pytest.approx(engine.ctx.sim_time), name
    engine.shutdown()


# -- the near clip plane ----------------------------------------------------------------------


class _BigRoom(Plugin):
    """A box 0.15 m in front of a depth camera, in a world 40 m across, where MuJoCo's default near
    plane (znear 0.01 of the extent) stands about 0.2 m out."""

    def build(self, spec: mujoco.MjSpec, ctx: SimContext) -> None:
        spec.worldbody.add_geom(
            type=mujoco.mjtGeom.mjGEOM_PLANE, size=[20, 20, 0.1], rgba=[0.3, 0.3, 0.3, 1]
        )
        spec.worldbody.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX, pos=[15, 15, 1], size=[1, 1, 1], rgba=[1, 1, 1, 1]
        )
        spec.worldbody.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX, pos=[-0.2, 0, 0.5], size=[0.05, 0.3, 0.3]
        )
        spec.worldbody.add_camera(
            name="cam", pos=[0, 0, 0.5], xyaxes=[0, 1, 0, 0, 0, 1], fovy=45, resolution=[64, 48]
        )


def test_a_depth_camera_sees_from_its_own_clip_near_in_a_large_world():
    cfg = load_config_from_dict(
        {
            "sim": {},
            "components": [
                {f"{__name__}:_BigRoom": {}},
                {OAKD: {"camera": "cam", "depth_camera": "cam", "clip_near": 0.1}},
            ],
        }
    )
    engine = Engine(cfg)
    engine.ctx.seed = 1
    engine.setup()
    m = engine.ctx.model
    assert m.vis.map.znear * m.stat.extent == pytest.approx(0.1)
    engine.reset()
    engine.step()
    depth = _endpoint(engine, "depth").read().data
    assert depth[24, 32] == pytest.approx(0.15, abs=1e-3)
    engine.shutdown()


def test_the_near_plane_is_never_pushed_out():
    engine = _engine(OAKD, depth_camera="cam", clip_near=5.0, clip_far=10.0)
    m = engine.ctx.model
    assert m.vis.map.znear * m.stat.extent < 5.0
    engine.shutdown()


def test_a_second_camera_at_the_same_range_leaves_the_plane_where_the_first_put_it(caplog):
    import logging

    cfg = load_config_from_dict(
        {
            "sim": {},
            "components": [
                {f"{__name__}:_BigRoom": {}},
                {OAKD: {"camera": "cam", "depth_camera": "cam", "clip_near": 0.1}},
                {OAKD: {"camera": "cam", "depth_camera": "cam", "clip_near": 0.1}, "name": "two"},
            ],
        }
    )
    engine = Engine(cfg)
    engine.ctx.seed = 1
    with caplog.at_level(logging.INFO, logger="roqsim_sensors.plugins.depth_camera"):
        engine.setup()
    assert len([r for r in caplog.records if "lowered" in r.getMessage()]) == 1
    engine.shutdown()
