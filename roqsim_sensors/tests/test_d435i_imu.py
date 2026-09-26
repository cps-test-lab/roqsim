"""The D435i's integrated IMU: it arrives with the model, where the vendor says it sits.

The mount is the vendor ``camera_link``, so the vendor's own extrinsics hold as written: the IMU is
the gyro optical frame of the manifest's ``frames:`` chain, and ``d435_color`` is the colour optical
frame. Both are checked here against the published ``realsense2_description`` constants, so a chain
that drifted from the vendor would fail. And a device on a fixed mount must read 1 g: MuJoCo
computes no acceleration for a body welded to the world, so without the plugin's closed-form branch
a tripod-mounted camera would report free fall forever.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np
import pytest
from roqsim_sensors.plugins.imu import ImuPlugin

from roqsim.config import load_config_from_dict
from roqsim.engine import Engine

# -- the vendor's own numbers, and only these ------------------------------------------------
# realsense2_description/urdf/_d435.urdf.xacro
D435_CAM_DEPTH_TO_COLOR_OFFSET = 0.015
# realsense2_description/urdf/_d435i_imu_modules.urdf.xacro -- accel and gyro frames are co-located
D435I_IMU_XYZ = (-0.01174, -0.00552, 0.0051)
# Every optical joint of the macro: rpy (-pi/2, 0, -pi/2) from its parent frame.
OPTICAL_RPY = (-math.pi / 2, 0.0, -math.pi / 2)


def _rot(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Fixed-axis XYZ rotation matrix, the convention a URDF ``rpy`` uses."""
    quat = np.zeros(4)
    mujoco.mju_euler2Quat(quat, np.asarray([roll, pitch, yaw], dtype=float), "XYZ")
    mat = np.zeros(9)
    mujoco.mju_quat2Mat(mat, quat)
    return mat.reshape(3, 3)


def _spawn(*, overrides=None):
    cfg = load_config_from_dict(
        {
            "sim": {},
            "components": [
                {
                    "spawn_sensor": {
                        "model": "realsense_d435",
                        "prefix": "d435_",
                        "pos": [1.0, 0.0, 0.5],
                    },
                    "name": "cam",
                }
            ],
        },
        overrides=overrides,
    )
    engine = Engine(cfg)
    engine.setup()
    engine.reset()
    for _ in range(20):
        engine.step()
    return engine


def _imu(engine) -> ImuPlugin | None:
    return next((p for p in engine.plugins if isinstance(p, ImuPlugin)), None)


def _in_mount(model, data, site_or_cam_pos, site_or_cam_mat):
    """A world pose expressed in the ``d435_mount`` body (the vendor ``camera_link``)."""
    mount = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "d435_mount")
    r = data.xmat[mount].reshape(3, 3)
    return r.T @ (site_or_cam_pos - data.xpos[mount]), r.T @ site_or_cam_mat.reshape(3, 3)


# -- provenance ------------------------------------------------------------------------------


def test_the_imu_is_the_vendors_gyro_optical_frame_in_camera_link():
    engine = _spawn()
    model, data = engine.ctx.model, engine.ctx.data
    site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, _imu(engine)._resolved_site)
    assert site >= 0
    body = int(model.site_bodyid[site])
    assert mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body) == "d435_mount"
    pos, rot = _in_mount(model, data, data.site_xpos[site], data.site_xmat[site])
    assert np.allclose(pos, D435I_IMU_XYZ, atol=1e-9)
    assert np.allclose(rot, _rot(*OPTICAL_RPY), atol=1e-9)


def test_the_colour_camera_is_the_vendors_colour_optical_frame():
    """MuJoCo's camera looks down -z with +y up, the optical frame down +z with +y down."""
    engine = _spawn()
    model, data = engine.ctx.model, engine.ctx.data
    cam = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "d435_d435_color")
    assert cam >= 0
    pos, rot = _in_mount(model, data, data.cam_xpos[cam], data.cam_xmat[cam])
    assert np.allclose(pos, [0.0, D435_CAM_DEPTH_TO_COLOR_OFFSET, 0.0], atol=1e-9)
    assert np.allclose(rot @ np.diag([1.0, -1.0, -1.0]), _rot(*OPTICAL_RPY), atol=1e-9)


def test_the_reported_frame_is_the_optical_one_the_driver_stamps():
    engine = _spawn()
    endpoint = next(e for e in engine.ctx.interface.all() if e.name == "imu")
    hints = endpoint.backend["ros2"]
    # realsense2_camera's own names: `<camera>/imu` (with unite_imu_method set) in
    # camera_imu_optical_frame -- an unmodified RealSense consumer finds it there or not at all.
    assert hints["topic"] == "camera/imu"
    assert hints["frame_id"] == "camera_imu_optical_frame"
    assert endpoint.lazy is True


def test_the_imu_frame_is_published_on_the_chain():
    engine = _spawn()
    frames = next(e for e in engine.ctx.interface.all() if e.name == "frames")
    links = {(t["parent"], t["child"]) for t in frames.backend["ros2"]["static_tf"]}
    assert ("camera_link", "camera_gyro_frame") in links
    assert ("camera_gyro_frame", "camera_gyro_optical_frame") in links
    assert ("camera_gyro_optical_frame", "camera_imu_optical_frame") in links


# -- what it reports -------------------------------------------------------------------------


def test_a_fixed_mount_reads_one_g_along_the_optical_down_axis():
    """The trap this closes: MuJoCo computes no acceleration for a body welded to the world."""
    reading = _imu(_spawn()).read()
    accel = np.asarray(reading.linear_acceleration)
    assert np.linalg.norm(accel) == pytest.approx(9.81, rel=1e-6)
    # Proper acceleration points UP, and the ROS optical convention has +y pointing DOWN, so the
    # whole of g lands on -y. That the sensor's own axes put it there is the mount quaternion being
    # right, checked from the reading rather than from the number that produced it.
    assert accel[1] == pytest.approx(-9.81, rel=1e-6)
    assert abs(accel[0]) < 1e-9 and abs(accel[2]) < 1e-9
    # A welded body cannot turn, so no rate is the correct reading rather than a missing one.
    assert np.allclose(reading.angular_velocity, 0.0, atol=1e-12)


def test_the_hardware_reports_no_attitude_and_says_so():
    """Accelerometer + gyroscope, no magnetometer, no fusion on board."""
    reading = _imu(_spawn()).read()
    assert reading.orientation_valid is False
    assert reading.orientation_variance == 0.0


# -- opting out ------------------------------------------------------------------------------


def test_a_world_that_models_a_plain_d435_can_switch_the_imu_off():
    """`enable_gyro`/`enable_accel` default false in the real driver, so the opt-out is documented.

    A flag rather than a second model: it stays addressable, the run's record says what was turned
    off, and "does this device have an IMU" becomes a campaign factor instead of a file edit.
    """
    engine = _spawn(overrides={"components": {"cam.imu": {"enabled": False}}})
    assert _imu(engine) is None
    assert not [e for e in engine.ctx.interface.all() if e.name == "imu"]
    # The camera is untouched: switching one manifest component off is not opting out of the model.
    assert [e for e in engine.ctx.interface.all() if e.name == "image"]
