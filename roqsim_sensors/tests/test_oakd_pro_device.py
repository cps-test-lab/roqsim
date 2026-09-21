"""The OAK-D Pro is mounted by its vendor ``oakd_link``, and the move changed no camera.

* **The chain is the vendor's** (``turtlebot4_description/urdf/sensors/oakd.urdf.xacro``): the mount
  publishes ``oakd_link -> oakd_rgb_camera_frame -> oakd_rgb_camera_optical_frame``, the stereo pair at
  +-baseline/2 and the IMU frame, and the camera the image is rendered from IS the RGB optical frame.
* **The re-seat moved names, not the camera.** The retired ``oakd`` model pre-rotated ``mount`` by +90
  deg about z so a mount at ``rpy [0, 0, 0]`` looked along +y; a mount of it at ``T_old`` and one of
  ``oakd_pro`` at ``T_old * Rq`` (``external/convert/build_oakd_pro.py --rewrite-mounts``) put the
  camera and the housing at the same world pose.
* **The device carries the vendor inertial** rather than a mass MuJoCo derives from the box.
* **The old name is refused**, naming the new one.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np
import pytest

from roqsim.config import load_config_from_dict
from roqsim.engine import Engine
from roqsim.models import ModelError, resolve_model

OPTICAL_RPY = (-math.pi / 2, 0.0, -math.pi / 2)
BASELINE = 0.075
# oakd.urdf.xacro: `mass` and the base link's inertia.
MASS = 0.061
DIAGINERTIA = (0.00000202475, 0.00001527320, 0.00001605536)

# The retired model, as it was: every geom in oakd_link, the body pre-rotated by +90 deg about z.
RETIRED_QUAT = (0.70710678, 0.0, 0.0, 0.70710678)
RETIRED_MJCF = """<mujoco>
  <worldbody>
    <body name="mount" quat="0.70710678 0 0 0.70710678">
      <geom name="oakd_visual" type="box" pos="-0.011 0 -0.005" size="0.01125 0.0485 0.015"/>
      <camera name="oakd_rgb" pos="0 0 0" xyaxes="0 -1 0 0 0 1" fovy="56.84" resolution="320 240"/>
    </body>
  </worldbody>
</mujoco>"""

POSES = [
    ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
    ([-1.5, -1.5, 0.6], [0.0, 0.0, -0.7854]),
    ([2.0, 1.0, 3.0], [-0.6, 0.2, 2.4]),
]


def _rot(rpy) -> np.ndarray:
    quat = np.zeros(4)
    mujoco.mju_euler2Quat(quat, np.asarray(rpy, dtype=float), "XYZ")
    mat = np.zeros(9)
    mujoco.mju_quat2Mat(mat, quat)
    return mat.reshape(3, 3)


def _rpy(m: np.ndarray) -> list[float]:
    pitch = math.asin(max(-1.0, min(1.0, -m[2, 0])))
    return [math.atan2(m[2, 1], m[2, 2]), pitch, math.atan2(m[1, 0], m[0, 0])]


def _re_expressed(pos, rpy):
    q = np.asarray(RETIRED_QUAT) / np.linalg.norm(RETIRED_QUAT)
    rq = np.zeros(9)
    mujoco.mju_quat2Mat(rq, q)
    return list(pos), _rpy(_rot(rpy) @ rq.reshape(3, 3))


def _compiled(model, pos, rpy):
    cfg = load_config_from_dict(
        {
            "sim": {},
            "components": [
                {
                    "spawn_sensor": {
                        "model": model,
                        "prefix": "cam_",
                        "pos": list(pos),
                        "rpy": list(rpy),
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


@pytest.mark.parametrize("pos, rpy", POSES)
def test_a_re_expressed_mount_puts_the_camera_where_the_retired_one_did(tmp_path, pos, rpy):
    retired = tmp_path / "retired_oakd.xml"
    retired.write_text(RETIRED_MJCF)
    old = _compiled(str(retired), pos, rpy)
    new = _compiled("oakd_pro", *_re_expressed(pos, rpy))
    for objtype, name in (
        (mujoco.mjtObj.mjOBJ_CAMERA, "cam_oakd_rgb"),
        (mujoco.mjtObj.mjOBJ_GEOM, "cam_oakd_visual"),
    ):
        (p_old, r_old), (p_new, r_new) = _pose(old, objtype, name), _pose(new, objtype, name)
        assert np.allclose(p_old, p_new, atol=1e-9), name
        assert np.allclose(r_old, r_new, atol=1e-7), name  # the retired quat is 8 digits


def test_the_mount_publishes_the_vendor_chain():
    engine = _compiled("oakd_pro", [0.0, 0.0, 1.0], [0.0, 0.0, 0.0])
    frames = next(e for e in engine.ctx.interface.all() if e.name == "frames")
    tfs = {(t["parent"], t["child"]): t for t in frames.backend["ros2"]["static_tf"]}
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


def test_the_retired_name_is_refused_naming_the_new_one():
    with pytest.raises(ModelError) as exc:
        resolve_model("oakd")
    assert str(exc.value) == (
        "spawn_sensor: model 'oakd' — renamed to 'oakd_pro' when its mount frame became the vendor "
        "link (it was a display convention pointing the lens along +y). Update the name, and "
        "re-express this mount's pos/rpy against the vendor link; see roqsim_sensors/README.md."
    )
