"""The ``spot_body_camera`` device on its own: its frames, lenses and the spot_ros2 interface.

Where the numbers come from is in the device's MJCF header and in
``roqsim_quadruped/tests/test_spot_cameras.py``, which checks the five stations on the robot.
"""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from roqsim.config import PluginError, load_config_from_dict
from roqsim.engine import Engine


def _mounted(device: str = "left", **camera) -> Engine:
    entry = {
        "spawn_sensor": {
            "model": "spot_body_camera",
            "device_name": device,
            "pose": {"position": {"z": 1.0}},
        },
        "name": "cam",
    }
    if camera:
        entry["components"] = [{"spot_body_camera": camera}]
    engine = Engine(load_config_from_dict({"sim": {}, "components": [entry]}))
    engine.ctx.seed = 1
    engine.setup()
    engine.reset()
    return engine


def _endpoints(engine: Engine) -> dict:
    return {
        ep.name: ep for ep in engine.ctx.interface.all() if ep.producer == "cam.spot_body_camera"
    }


def test_the_frames_are_the_station_and_its_fisheye():
    engine = _mounted("frontleft")
    chain = {
        t.child: (t.parent, tuple(t.translation))
        for ep in engine.ctx.interface.all()
        if ep.name == "frames"
        for t in ep.read().transforms
    }
    assert chain["frontleft_fisheye"] == ("frontleft", pytest.approx((-0.01, 0.0, 0.0)))
    engine.shutdown()


def test_each_camera_renders_from_the_frame_its_stream_is_stamped_in():
    engine = _mounted("back")
    m, d = engine.ctx.model, engine.ctx.data
    prefix = next(p for p in engine.plugins if p.address == "cam.spot_body_camera")._prefix
    for cam, site in (("spot_fisheye", "back_fisheye"), ("spot_depth", "back")):
        c = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, prefix + cam)
        s = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, prefix + site)
        assert c >= 0 and s >= 0, (cam, site)
        np.testing.assert_allclose(d.cam_xpos[c], d.site_xpos[s], atol=1e-9)
        np.testing.assert_allclose(
            d.cam_xmat[c].reshape(3, 3),
            d.site_xmat[s].reshape(3, 3) @ np.diag([1, -1, -1]),
            atol=1e-9,
        )
    engine.shutdown()


def test_the_lenses_render_the_published_intrinsics():
    eps = _endpoints(_mounted())
    image, depth = eps["camera_info"].read(), eps["depth_camera_info"].read()
    assert (image.width, image.height, image.fx, image.fy, image.cx, image.cy) == pytest.approx(
        (640, 480, 257.487, 257.487, 319.5, 239.5)
    )
    assert (depth.width, depth.height, depth.fx, depth.fy, depth.cx, depth.cy) == pytest.approx(
        (424, 240, 170.6, 170.6, 211.5, 119.5)
    )
    assert eps["registered_depth_camera_info"].read() == image


def test_the_streams_are_spot_ros2s():
    engine = _mounted("right")
    eps = _endpoints(engine)
    want = {
        "image": ("camera/right/image", "right_fisheye"),
        "camera_info": ("camera/right/camera_info", "right_fisheye"),
        "depth": ("depth/right/image", "right"),
        "depth_camera_info": ("depth/right/camera_info", "right"),
        "registered_depth": ("depth_registered/right/image", "right_fisheye"),
        "registered_depth_camera_info": ("depth_registered/right/camera_info", "right_fisheye"),
    }
    assert set(eps) == set(want)  # no compressed streams: spot_ros2's default
    for name, (topic, frame) in want.items():
        ros = eps[name].backend["ros2"]
        assert (ros["topic"], ros["frame_id"]) == (topic, frame), name
        assert ros["qos"] == {
            "reliability": "reliable",
            "durability": "transient_local",
            "depth": 10,
        }
        assert eps[name].rate_hz == 15.0
    engine.step()
    assert eps["image"].read().encoding == "mono8"
    assert eps["depth"].read().encoding == "16UC1"
    engine.shutdown()


def test_compressed_is_spot_ros2s_sibling_of_image():
    eps = _endpoints(_mounted("left", compressed=True))
    assert eps["image_compressed"].backend["ros2"]["topic"] == "camera/left/compressed"
    assert "depth_compressed" in eps  # 16UC1 within the codec's range: compressedDepth is offered


def test_colour_bodies_publish_bgr8():
    engine = _mounted("left", color_encoding="bgr8")
    engine.step()
    frame = _endpoints(engine)["image"].read()
    assert frame.encoding == "bgr8" and frame.data.shape == (480, 640, 3)
    engine.shutdown()


def test_a_station_must_be_named():
    with pytest.raises(PluginError, match="station"):
        _mounted("left", station="")
