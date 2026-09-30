"""The OAK-D Pro is mounted by its vendor ``oakd_link``.

* **The chain is the vendor's** (``turtlebot4_description/urdf/sensors/oakd.urdf.xacro``): the mount
  publishes ``oakd_link -> oakd_rgb_camera_frame -> oakd_rgb_camera_optical_frame``, the stereo pair at
  +-baseline/2 and the IMU frame, and the camera the image is rendered from IS the RGB optical frame.
* **The device carries the vendor inertial** rather than a mass MuJoCo derives from the box.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np
import pytest

from roqsim.config import load_config_from_dict
from roqsim.engine import Engine
from roqsim.pose import pose_mapping

OPTICAL_RPY = (-math.pi / 2, 0.0, -math.pi / 2)
BASELINE = 0.075
# oakd.urdf.xacro: `mass` and the base link's inertia.
MASS = 0.061
DIAGINERTIA = (0.00000202475, 0.00001527320, 0.00001605536)

POSES = [
    ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
    ([-1.5, -1.5, 0.6], [0.0, 0.0, -0.7854]),
    ([2.0, 1.0, 3.0], [-0.6, 0.2, 2.4]),
]


def _compiled(model, pos, rpy):
    cfg = load_config_from_dict(
        {
            "sim": {},
            "components": [
                {
                    "spawn_sensor": {
                        "model": model,
                        "prefix": "cam_",
                        "pose": pose_mapping(pos, rpy),
                        "default_plugins": False,
                    },
                    "name": "cam",
                }
            ],
        }
    )
    engine = Engine(cfg)
    engine.setup()
    mujoco.mj_forward(engine.ctx.model, engine.ctx.data)
    return engine


def _pose(engine, objtype, name):
    m, d = engine.ctx.model, engine.ctx.data
    ident = mujoco.mj_name2id(m, objtype, name)
    assert ident >= 0, name
    if objtype == mujoco.mjtObj.mjOBJ_CAMERA:
        return d.cam_xpos[ident].copy(), d.cam_xmat[ident].reshape(3, 3).copy()
    if objtype == mujoco.mjtObj.mjOBJ_SITE:
        return d.site_xpos[ident].copy(), d.site_xmat[ident].reshape(3, 3).copy()
    return d.geom_xpos[ident].copy(), d.geom_xmat[ident].reshape(3, 3).copy()


def test_the_mount_publishes_the_vendor_chain():
    engine = _compiled("oakd_pro", [0.0, 0.0, 1.0], [0.0, 0.0, 0.0])
    frames = next(e for e in engine.ctx.interface.all() if e.name == "frames")
    tfs = {(t["parent"], t["child"]): t for t in [vars(t) for t in frames.read().transforms]}
    assert set(tfs) == {
        ("world", "oakd_link"),
        ("oakd_link", "oakd_rgb_camera_frame"),
        ("oakd_rgb_camera_frame", "oakd_rgb_camera_optical_frame"),
        ("oakd_link", "oakd_left_camera_frame"),
        ("oakd_left_camera_frame", "oakd_left_camera_optical_frame"),
        ("oakd_link", "oakd_right_camera_frame"),
        ("oakd_right_camera_frame", "oakd_right_camera_optical_frame"),
        ("oakd_link", "oakd_imu_frame"),
    }
    assert np.allclose(
        tfs[("oakd_link", "oakd_left_camera_frame")]["translation"], [0, BASELINE / 2, 0]
    )
    assert np.allclose(
        tfs[("oakd_link", "oakd_right_camera_frame")]["translation"], [0, -BASELINE / 2, 0]
    )
    want = np.zeros(4)
    mujoco.mju_euler2Quat(want, np.asarray(OPTICAL_RPY), "XYZ")
    optical = tfs[("oakd_rgb_camera_frame", "oakd_rgb_camera_optical_frame")]["rotation"]
    assert abs(abs(float(np.dot(optical, want))) - 1.0) < 1e-9


@pytest.mark.parametrize("pos, rpy", POSES)
def test_the_image_is_rendered_from_the_frame_it_is_stamped_in(pos, rpy):
    engine = _compiled("oakd_pro", pos, rpy)
    site_pos, site_rot = _pose(
        engine, mujoco.mjtObj.mjOBJ_SITE, "cam_oakd_rgb_camera_optical_frame"
    )
    cam_pos, cam_rot = _pose(engine, mujoco.mjtObj.mjOBJ_CAMERA, "cam_oakd_rgb")
    assert np.allclose(site_pos, cam_pos, atol=1e-9)
    # MuJoCo's camera looks down -z with +y up; the optical frame down +z with +y down.
    assert np.allclose(site_rot, cam_rot @ np.diag([1.0, -1.0, -1.0]), atol=1e-9)


def test_the_device_carries_the_vendor_inertial():
    """Without it MuJoCo derives ~65 g from the box at 1000 kg/m^3, which is not the device."""
    m = _compiled("oakd_pro", [0, 0, 0], [0, 0, 0]).ctx.model
    body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "cam_mount")
    assert float(m.body_mass[body]) == pytest.approx(MASS, abs=1e-12)
    assert np.allclose(sorted(m.body_inertia[body]), sorted(DIAGINERTIA), rtol=1e-6)


def test_the_images_are_stamped_in_the_vendor_optical_frame():
    cfg = load_config_from_dict(
        {"sim": {}, "components": [{"spawn_sensor": {"model": "oakd_pro"}, "name": "cam"}]}
    )
    camera = next(s for s in cfg.plugins if s.ref == "oakd_camera")
    assert camera.config["frame_id"] == "oakd_rgb_camera_optical_frame"
