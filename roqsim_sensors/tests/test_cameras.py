"""Camera plugin checks: intrinsics math, capture shapes/dtypes, and subscriber-gated rendering."""

from __future__ import annotations

import math

import mujoco
import numpy as np
import pytest
from roqsim_sensors.plugins.camera_common import intrinsics_from_model
from roqsim_sensors.plugins.realsense_d435 import RealsenseD435Plugin

from roqsim.config import load_config_from_dict
from roqsim.context import SimContext
from roqsim.engine import Engine
from roqsim.plugin import Plugin


class _CameraScene(Plugin):
    """A lit box in front of a single MuJoCo camera named ``cam``."""

    def build(self, spec: mujoco.MjSpec, ctx: SimContext) -> None:
        spec.worldbody.add_geom(
            type=mujoco.mjtGeom.mjGEOM_PLANE, size=[5, 5, 0.1], rgba=[0.3, 0.3, 0.3, 1]
        )
        spec.worldbody.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[2, 0, 0.5],
            size=[0.3, 0.3, 0.3],
            rgba=[0.8, 0.1, 0.1, 1],
        )
        spec.worldbody.add_camera(
            name="cam", pos=[0, 0, 0.5], xyaxes=[0, 1, 0, 0, 0, 1], fovy=45, resolution=[64, 48]
        )


def _world(plugin_ref: str, **config):
    cfg = {
        "sim": {},
        "plugins": [
            {f"{__name__}:_CameraScene": {}},
            {plugin_ref: {"camera": "cam", **config}},
        ],
    }
    return load_config_from_dict(cfg)


def _endpoint(engine: Engine, name: str):
    return next((e for e in engine.ctx.interface.all() if e.name == name), None)


# -- intrinsics -----------------------------------------------------------------------------


def test_intrinsics_from_model_matches_pinhole_formula():
    xml = """<mujoco><worldbody>
        <geom type="plane" size="5 5 0.1"/>
        <camera name="cam" fovy="60" resolution="64 48"/>
    </worldbody></mujoco>"""
    m = mujoco.MjModel.from_xml_string(xml)
    intr = intrinsics_from_model(m, 0)
    expected_f = 48 / (2.0 * math.tan(math.radians(60) / 2.0))
    assert intr.width == 64 and intr.height == 48
    assert math.isclose(intr.fx, expected_f) and intr.fx == intr.fy
    assert intr.cx == 32.0 and intr.cy == 24.0


def test_intrinsics_from_model_falls_back_when_resolution_unset():
    xml = """<mujoco><worldbody>
        <geom type="plane" size="5 5 0.1"/>
        <camera name="cam"/>
    </worldbody></mujoco>"""
    m = mujoco.MjModel.from_xml_string(xml)
    intr = intrinsics_from_model(m, 0, default_width=320, default_height=240)
    assert intr.width == 320 and intr.height == 240


# -- capture (real mujoco.Renderer, needs a GL backend e.g. MUJOCO_GL=egl) ------------------


def test_oakd_camera_captures_rgb_and_depth():
    engine = Engine(_world("roqsim_sensors.plugins.oakd_camera:OakDCameraPlugin"))
    engine.setup()
    engine.reset()
    engine.step()
    rgb = _endpoint(engine, "image").read()
    depth = _endpoint(engine, "depth").read()
    info = _endpoint(engine, "camera_info").read()
    assert rgb.shape == (48, 64, 3) and rgb.dtype == np.uint8
    assert rgb.max() > 0  # not a black frame -- the box/plane actually rendered
    assert depth.shape == (48, 64) and depth.dtype == np.float32
    assert np.isfinite(depth).any()  # the plane is within clip range
    assert info.width == 64 and info.height == 48 and info.fx == info.fy


def test_zivid_captures_rgb_and_depth_with_working_range_defaults():
    engine = Engine(_world("roqsim_sensors.plugins.zivid:ZividPlugin"))
    engine.setup()
    engine.reset()
    engine.step()
    rgb = _endpoint(engine, "image").read()
    depth = _endpoint(engine, "depth").read()
    info = _endpoint(engine, "camera_info").read()
    assert rgb.shape == (48, 64, 3) and rgb.dtype == np.uint8 and rgb.max() > 0
    assert depth.shape == (48, 64) and depth.dtype == np.float32
    # The scene box sits 2 m in front of the camera -- inside the XL250's 1.3-5 m working range.
    assert np.isfinite(depth).any()
    assert info.width == 64 and info.height == 48


def test_zivid_applies_datasheet_working_range_and_depth_topic():
    from roqsim_sensors.plugins.zivid import ZividPlugin

    p = ZividPlugin({"camera": "cam"})
    assert (p.clip_near, p.clip_far) == (1.3, 5.0)  # XL250 recommended-to-extended working range
    assert ZividPlugin({"camera": "cam", "clip_near": 0.5}).clip_near == 0.5  # world can override


D435 = "roqsim_sensors.plugins.realsense_d435:RealsenseD435Plugin"


def _stepped(plugin_ref: str, **config):
    engine = Engine(_world(plugin_ref, **config))
    engine.setup()
    engine.reset()
    engine.step()
    return engine


def test_realsense_d435_publishes_colour_only_by_default():
    engine = _stepped(D435)
    rgb = _endpoint(engine, "image").read()
    assert rgb.shape == (48, 64, 3) and rgb.dtype == np.uint8 and rgb.max() > 0
    assert _endpoint(engine, "camera_info") is not None
    # Depth and the cloud are opt-in: a 640x480 cloud is up to 307k points per capture, so a world
    # that did not ask for one must not pay for it.
    assert _endpoint(engine, "depth") is None
    assert _endpoint(engine, "points") is None


def test_realsense_d435_depth_is_opt_in_and_uses_realsense_topics():
    engine = _stepped(D435, depth=True)
    ep = _endpoint(engine, "depth")
    depth = ep.read()
    assert depth.shape == (48, 64) and depth.dtype == np.float32
    assert np.isfinite(depth).any()  # the box at 2 m is inside the D435's 0.28-3.0 m range
    ros = ep.backend["ros2"]
    assert ros["topic"] == "camera/depth/image_rect_raw"
    assert ros["frame_id"] == "camera_depth_optical_frame"
    assert ros["encoding"] == "32FC1"
    assert _endpoint(engine, "points") is None  # depth alone does not imply the cloud


def test_realsense_d435_working_range_defaults_to_the_datasheet():
    from roqsim_sensors.plugins.realsense_d435 import RealsenseD435Plugin as D

    p = D({"camera": "cam"})
    # Not the OAK-D's 0.3-100 m: clip_far is what decides whether the wall behind the subject lands
    # in an occupancy map as an obstacle.
    assert (p.clip_near, p.clip_far) == (0.28, 3.0)
    assert D({"camera": "cam", "clip_far": 6.0}).clip_far == 6.0


def test_realsense_d435_points_imply_depth_and_reproject_into_the_optical_frame():
    engine = _stepped(D435, points=True)
    assert _endpoint(engine, "depth") is not None, "`points` implies `depth`"
    ep = _endpoint(engine, "points")
    cloud = ep.read()
    assert ep.backend["ros2"]["topic"] == "camera/depth/color/points"
    assert cloud.points.dtype == np.float32 and cloud.points.shape[1] == 3
    assert 0 < len(cloud.points) <= 64 * 48
    assert np.isfinite(cloud.points).all()  # "no return" pixels are dropped, not sentinel-valued

    # Geometry check against the fixture: the camera sits 0.5 m above the ground plane and looks
    # horizontally, so every return in frame is a floor point -- and a floor point is 0.5 m BELOW the
    # camera whatever its distance. In the ROS optical frame (x right, y DOWN, z forward) that fixes
    # the whole cloud to y ~ +0.5: the sign pins the convention (a flipped y would put the floor
    # overhead) and the magnitude pins the reprojection scale at every depth, not just one.
    assert np.allclose(cloud.points[:, 1], 0.5, atol=0.05)
    # z is the distance along the view direction: the nearest visible floor point is where the bottom
    # of a 45 deg vertical FOV meets the ground, 0.5 / tan(22.5 deg), and nothing past clip_far
    # survives (the plane runs on well beyond 3 m).
    assert cloud.points[:, 2].min() == pytest.approx(0.5 / math.tan(math.radians(22.5)), abs=0.05)
    assert cloud.points[:, 2].max() <= 3.0
    # The cloud is exactly the depth image's valid pixels -- one is the other reprojected.
    assert len(cloud.points) == int(np.isfinite(_endpoint(engine, "depth").read()).sum())


def test_d455_model_fov_matches_datasheet():
    """The bundled d455 model must reproduce the D455f colour FOV (87 deg H x 62 deg V).

    MuJoCo stores only fovy (vertical); the horizontal FOV falls out of fovy + the resolution
    aspect, so this locks BOTH: fovy == 62 and the derived horizontal FOV ~= 87 deg."""
    cfg = {"sim": {}, "plugins": [{"spawn_sensor": {"model": "d455", "name": "d455"}}]}
    engine = Engine(load_config_from_dict(cfg))
    engine.setup()
    engine.reset()
    m = engine.ctx.model
    cid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "d455_color")
    fovy = float(m.cam_fovy[cid])
    w, h = (int(v) for v in m.cam_resolution[cid])
    fovx = math.degrees(2.0 * math.atan((w / h) * math.tan(math.radians(fovy) / 2.0)))
    assert fovy == 62.0
    assert abs(fovx - 87.0) < 1.5  # datasheet colour horizontal FOV, from the 1.6 aspect


# -- subscriber-gated rendering (unit-level; no ctx/render needed) --------------------------


class _FakeEndpoint:
    def __init__(self, has_subscribers=None):
        self.has_subscribers = has_subscribers


class _FakeCtx:
    def __init__(self, sim_time: float):
        self.sim_time = sim_time


def test_due_gates_on_rate_and_has_subscribers():
    plugin = RealsenseD435Plugin({"rate_hz": 10.0})  # period = 0.1s
    plugin._last_capture = 0.0
    plugin._image_ep = _FakeEndpoint(has_subscribers=None)
    assert plugin._due(_FakeCtx(0.05)) is False  # too soon
    assert plugin._due(_FakeCtx(0.2)) is True  # due, subscriber count unknown -> assume yes

    plugin._image_ep.has_subscribers = lambda: False
    assert plugin._due(_FakeCtx(0.2)) is False  # due but nobody's listening -> skip

    plugin._image_ep.has_subscribers = lambda: True
    assert plugin._due(_FakeCtx(0.2)) is True


def test_due_gates_on_every_endpoint_the_render_feeds_not_just_colour():
    """A depth-only or cloud-only consumer must keep the renderer running.

    This is a regression: gating on ``image`` alone starved exactly the consumer that never subscribes
    to colour -- MoveIt's octomap updater takes the point cloud -- and the symptom was not an error
    but an empty world, published forever at the configured rate.
    """
    plugin = RealsenseD435Plugin({"rate_hz": 10.0})
    plugin._last_capture = 0.0
    plugin._image_ep = _FakeEndpoint(has_subscribers=lambda: False)
    assert plugin._due(_FakeCtx(0.2)) is False  # colour only, unsubscribed -> nothing to render for

    depth_ep = _FakeEndpoint(has_subscribers=lambda: False)
    points_ep = _FakeEndpoint(has_subscribers=lambda: True)
    plugin._extra_outputs = [depth_ep, points_ep]
    assert plugin._due(_FakeCtx(0.2)) is True  # cloud subscriber alone justifies the render

    points_ep.has_subscribers = lambda: False
    depth_ep.has_subscribers = lambda: True
    assert plugin._due(_FakeCtx(0.2)) is True  # so does a depth subscriber alone

    depth_ep.has_subscribers = lambda: False
    assert plugin._due(_FakeCtx(0.2)) is False  # nobody on any render-fed endpoint -> skip


def test_camera_info_is_not_a_render_gate():
    # Checked on a configured plugin, so the gate list is the one the real endpoints produce: every
    # output fed by the render pass gates it, and camera_info -- which needs no render, and which an
    # rviz panel subscribes to on its own -- does not.
    engine = _stepped(D435, depth=True, points=True)
    plugin = next(p for p in engine.plugins if isinstance(p, RealsenseD435Plugin))
    gated = {ep.name for ep in plugin._gate_endpoints()}
    assert gated == {"image", "image_compressed", "depth", "points"}
    assert _endpoint(engine, "camera_info") is not None  # registered, just not a gate
    assert _endpoint(engine, "depth_camera_info") is not None


# -- depth intrinsics ----------------------------------------------------------------------------
def test_depth_camera_info_follows_realsense_ros_naming_and_only_exists_with_depth():
    """A consumer that rectifies or reprojects depth subscribes to the DEPTH info topic; given only
    the colour one it waits forever. Eager and un-gated on purpose: it needs no render, so a lone
    info subscriber must not switch the renderer on."""
    engine = _stepped(D435, depth=True)
    ep = _endpoint(engine, "depth_camera_info")
    assert ep.backend["ros2"]["topic"] == "camera/depth/camera_info"
    assert ep.backend["ros2"]["frame_id"] == "camera_depth_optical_frame"
    assert ep.lazy is False
    plugin = next(p for p in engine.plugins if isinstance(p, RealsenseD435Plugin))
    assert ep not in plugin._gate_endpoints()
    # Off by default, because depth itself is.
    assert _endpoint(_stepped(D435), "depth_camera_info") is None


def test_depth_camera_info_topic_can_be_hardwired():
    engine = _stepped(D435, depth=True, topics={"depth_camera_info": "/cam/depth/info"})
    assert _endpoint(engine, "depth_camera_info").backend["ros2"]["topic"] == "/cam/depth/info"


# -- compressed colour ---------------------------------------------------------------------------
def test_compressed_endpoint_is_offered_by_default_on_the_conventional_topic():
    """``<image topic>/compressed``, which is what makes a driver-shaped consumer find it."""
    engine = _stepped(D435)
    image = _endpoint(engine, "image")
    compressed = _endpoint(engine, "image_compressed")
    assert compressed is not None
    hints, image_hints = compressed.backend["ros2"], image.backend["ros2"]
    assert hints["type"] == "sensor_msgs.msg.CompressedImage"
    assert hints["topic"] == image_hints["topic"] + "/compressed"
    assert (hints["format"], hints["quality"], hints["encoding"]) == ("jpeg", 95, "rgb8")
    # Same frame as the raw stream: it is the same pixels, so a consumer that time-syncs on one and
    # projects with the other must not see two frame ids.
    assert hints["frame_id"] == image_hints["frame_id"]


def test_compressed_topic_follows_a_hardwired_image_topic():
    """The point of deriving from the *resolved* topic: a world matching an external driver's names
    gets the matching compressed topic without naming it twice."""
    engine = _stepped(D435, topics={"image": "/camera_1/camera_1/color/image_raw"})
    compressed = _endpoint(engine, "image_compressed")
    assert compressed.backend["ros2"]["topic"] == "/camera_1/camera_1/color/image_raw/compressed"


def test_compressed_topic_can_be_hardwired_on_its_own():
    engine = _stepped(D435, topics={"image_compressed": "/elsewhere/compressed"})
    assert _endpoint(engine, "image_compressed").backend["ros2"]["topic"] == "/elsewhere/compressed"


def test_compressed_can_be_switched_off():
    engine = _stepped(D435, compressed=False)
    assert _endpoint(engine, "image_compressed") is None
    assert _endpoint(engine, "image") is not None


def test_a_compressed_subscriber_alone_justifies_the_render():
    """The tb4-shaped consumer: it subscribes to compressed and never to raw. Gating on `image`
    alone would hand it an endless stream of nothing -- the same failure as the point-cloud case."""
    plugin = RealsenseD435Plugin({"rate_hz": 10.0})
    plugin._last_capture = 0.0
    plugin._image_ep = _FakeEndpoint(has_subscribers=lambda: False)
    plugin._compressed_ep = _FakeEndpoint(has_subscribers=lambda: True)
    assert plugin._due(_FakeCtx(0.2)) is True

    plugin._compressed_ep.has_subscribers = lambda: False
    assert plugin._due(_FakeCtx(0.2)) is False


def test_expensive_endpoints_are_lazy_and_cheap_ones_are_not():
    engine = _stepped(D435, depth=True, points=True)
    for name in ("image", "image_compressed", "depth", "points"):
        assert _endpoint(engine, name).lazy is True, name
    for name in ("camera_info", "depth_camera_info"):
        assert _endpoint(engine, name).lazy is False, name


def test_jpeg_quality_is_validated():
    plugin = RealsenseD435Plugin({})
    for bad in (0, 101):
        assert any("jpeg_quality" in e for e in plugin.validate_config({"jpeg_quality": bad})), bad
    assert plugin.validate_config({"jpeg_quality": 95}) == []


# -- depth encoding ------------------------------------------------------------------------------
def test_depth_is_float_metres_by_default():
    """32FC1 is lossless in the unit the renderer produces, so it stays the default: a world that
    says nothing must not change what it publishes."""
    engine = _stepped(D435, depth=True)
    ep = _endpoint(engine, "depth")
    assert ep.backend["ros2"]["encoding"] == "32FC1"
    assert ep.read().dtype == np.float32


def test_depth_encoding_16uc1_publishes_uint16_millimetres_with_zero_for_no_return():
    """What a real RealSense driver puts on `depth/image_rect_raw`: millimetres, 0 for invalid."""
    engine = _stepped(D435, depth=True, depth_encoding="16UC1")
    ep = _endpoint(engine, "depth")
    assert ep.backend["ros2"]["encoding"] == "16UC1"
    depth = ep.read()
    assert depth.shape == (48, 64) and depth.dtype == np.uint16

    # The same geometry the cloud test pins, in the other unit: the camera is 0.5 m above the floor
    # with a 45 deg vertical FOV, so the nearest visible floor point is 0.5 / tan(22.5 deg) away.
    valid = depth[depth > 0]
    assert valid.min() == pytest.approx(1000 * 0.5 / math.tan(math.radians(22.5)), abs=50)
    assert valid.max() <= 3000  # the D435's clip_far, in millimetres
    # The plane runs on well past clip_far, so there ARE unseen pixels -- and they read 0, the
    # device's marker for "no return", not a clamp to 65535 (which would be a surface 65.5 m away).
    assert (depth == 0).any()


def test_16uc1_rounds_rather_than_truncates():
    """A plain cast biases every reading down by up to a millimetre, systematically."""
    engine = _stepped(D435, depth=True, depth_encoding="16UC1")
    plugin = next(p for p in engine.plugins if isinstance(p, RealsenseD435Plugin))
    metres, millimetres = plugin._depth, _endpoint(engine, "depth").read()
    seen = np.isfinite(metres)
    assert np.array_equal(millimetres[seen], np.rint(metres[seen] * 1000.0).astype(np.uint16))


def test_the_cloud_stays_in_metres_when_depth_is_published_in_millimetres():
    """Two consumers, two units, one buffer: the reprojection reads the float metres regardless of
    what the depth topic advertises."""
    engine = _stepped(D435, points=True, depth_encoding="16UC1")
    cloud = _endpoint(engine, "points").read()
    assert cloud.points.dtype == np.float32
    assert cloud.points[:, 2].min() == pytest.approx(0.5 / math.tan(math.radians(22.5)), abs=0.05)
    assert _endpoint(engine, "depth").read().dtype == np.uint16


def test_the_encoded_frame_is_converted_once_per_capture():
    """Both depth topics read the same frame, so the conversion is cached until the next capture --
    and a new capture must not keep serving the old frame's copy."""
    engine = _stepped(D435, depth=True, depth_encoding="16UC1")
    plugin = next(p for p in engine.plugins if isinstance(p, RealsenseD435Plugin))
    ep = _endpoint(engine, "depth")
    assert ep.read() is ep.read()
    plugin._capture_extra(engine.ctx, plugin._frames.raw)
    assert plugin._depth_wire is None


def test_depth_encoding_is_validated():
    plugin = RealsenseD435Plugin({})
    errors = plugin.validate_config({"depth_encoding": "mono16"})
    assert any("depth_encoding" in e and "32FC1" in e for e in errors)
    assert plugin.validate_config({"depth_encoding": "16UC1"}) == []  # D435 clips at 3 m


def test_16uc1_refuses_a_range_uint16_millimetres_cannot_carry():
    """The OAK-D's own 100 m default is the case: millimetres end at 65.535 m, and saturating there
    would publish a wall 65.5 m away rather than an error."""
    from roqsim_sensors.plugins.oakd_camera import OakDCameraPlugin

    errors = OakDCameraPlugin({}).validate_config({"depth_encoding": "16UC1"})
    assert any("clip_far" in e and "65.535" in e for e in errors)
    assert OakDCameraPlugin({}).validate_config({"depth_encoding": "16UC1", "clip_far": 10.0}) == []


def test_d455_carries_the_depth_encoding_option_too():
    from roqsim_sensors.plugins.realsense_d455 import RealsenseD455Plugin

    plugin = RealsenseD455Plugin({"depth": True, "depth_encoding": "16UC1"})
    assert plugin.depth_encoding == "16UC1"
    assert plugin.validate_config(plugin.config) == []  # its 6 m range fits


# -- the compressed depth companion --------------------------------------------------------------
def test_compressed_depth_is_offered_for_16uc1_on_the_transport_s_own_topic():
    """`<depth topic>/compressedDepth`, which is the transport a RealSense driver advertises -- and
    the same payload as the raw topic, so the plugin owns no codec."""
    engine = _stepped(D435, depth=True, depth_encoding="16UC1")
    raw, compressed = _endpoint(engine, "depth"), _endpoint(engine, "depth_compressed")
    assert compressed is not None
    hints, raw_hints = compressed.backend["ros2"], raw.backend["ros2"]
    assert hints["type"] == "sensor_msgs.msg.CompressedImage"
    assert hints["topic"] == raw_hints["topic"] + "/compressedDepth"
    assert (hints["encoding"], hints["format"]) == ("16UC1", "rvl")
    assert hints["frame_id"] == raw_hints["frame_id"]
    assert compressed.read() is raw.read()  # one array, two wire formats
    assert compressed.lazy is True


def test_compressed_depth_follows_a_hardwired_depth_topic():
    """A world matching the rig's topic names gets the rig's compressed topic without naming it."""
    engine = _stepped(
        D435,
        depth=True,
        depth_encoding="16UC1",
        topics={"depth": "/camera_1/camera_1/depth/image_rect_raw"},
    )
    assert (
        _endpoint(engine, "depth_compressed").backend["ros2"]["topic"]
        == "/camera_1/camera_1/depth/image_rect_raw/compressedDepth"
    )


def test_no_compressed_depth_under_float_metres_or_with_compression_off():
    """RVL is 16-bit: under 32FC1 the topic cannot exist, and `compressed: false` opts out of the
    compressed companions of both streams."""
    assert _endpoint(_stepped(D435, depth=True), "depth_compressed") is None
    engine = _stepped(D435, depth=True, depth_encoding="16UC1", compressed=False)
    assert _endpoint(engine, "depth_compressed") is None
    assert _endpoint(engine, "image_compressed") is None
    assert _endpoint(engine, "depth") is not None


def test_a_compressed_depth_subscriber_alone_justifies_the_render():
    engine = _stepped(D435, depth=True, depth_encoding="16UC1")
    plugin = next(p for p in engine.plugins if isinstance(p, RealsenseD435Plugin))
    assert "depth_compressed" in {ep.name for ep in plugin._gate_endpoints()}


def test_asking_for_a_compressed_depth_topic_that_cannot_exist_is_an_error():
    plugin = RealsenseD435Plugin({})
    errors = plugin.validate_config({"topics": {"depth_compressed": "/somewhere"}})
    assert any("depth_compressed" in e and "16UC1" in e for e in errors)


def test_a_range_past_the_codec_s_own_limit_refuses_the_pair_rather_than_disagreeing():
    """compressedDepth's encoder drops returns past 10 m, so a camera that sees further would
    publish a raw and a compressed depth topic that disagree beyond it."""
    plugin = RealsenseD435Plugin({})
    errors = plugin.validate_config({"depth_encoding": "16UC1", "clip_far": 30.0})
    assert any("compressedDepth" in e and "compressed: false" in e for e in errors)
    assert (
        plugin.validate_config({"depth_encoding": "16UC1", "clip_far": 30.0, "compressed": False})
        == []
    )
