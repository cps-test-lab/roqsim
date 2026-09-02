"""The D435i's integrated IMU: it arrives with the model, where the vendor says it sits.

Two things are pinned here that no reader can check by eye. The mount pose in
``d435.manifest.yaml`` is re-derived from the published ``realsense2_description`` constants -- and
so is the ``d435_color`` camera pose the model has carried all along, which is what shows the same
transform chain reproduces a number nobody is arguing about. And a device on a fixed mount must read
1 g: MuJoCo computes no acceleration for a body welded to the world, so without the plugin's
closed-form branch a tripod-mounted camera would report free fall forever.
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
D435_ZERO_DEPTH_TO_GLASS = 4.2e-3
D435_GLASS_TO_FRONT = 0.1e-3
D435_CAM_DEPTH_PY = 0.0175
D435_CAM_DEPTH_TO_COLOR_OFFSET = 0.015
# realsense2_description/urdf/_d435i_imu_modules.urdf.xacro -- accel and gyro frames are co-located
D435I_IMU_XYZ = (-0.01174, -0.00552, 0.0051)


def _rot(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Fixed-axis XYZ rotation matrix, the convention a URDF ``rpy`` uses."""
    quat = np.zeros(4)
    mujoco.mju_euler2Quat(quat, np.asarray([roll, pitch, yaw], dtype=float), "XYZ")
    mat = np.zeros(9)
    mujoco.mju_quat2Mat(mat, quat)
    return mat.reshape(3, 3)


def _link_to_mesh(p_link) -> np.ndarray:
    """A point in ``camera_link`` expressed in the frame this MJCF's ``mount`` body uses.

    The mesh is placed in ``camera_link`` at ``(zero_depth_to_glass + glass_to_front, -cam_depth_py,
    0)`` with ``rpy = (pi/2, 0, pi/2)``, and this model's body-local axes are the mesh's own -- so the
    inverse of that placement is the whole conversion.
    """
    offset = np.array([D435_ZERO_DEPTH_TO_GLASS + D435_GLASS_TO_FRONT, -D435_CAM_DEPTH_PY, 0.0])
    return _rot(math.pi / 2, 0.0, math.pi / 2).T @ (np.asarray(p_link, dtype=float) - offset)


def _spawn(*, overrides=None):
    cfg = load_config_from_dict(
        {
            "sim": {},
            "components": [
                {
                    "spawn_sensor": {"model": "d435", "prefix": "d435_", "pos": [1.0, 0.0, 0.5]},
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


# -- provenance ------------------------------------------------------------------------------


def test_the_manifest_pose_is_the_vendors_extrinsic_put_through_the_models_own_transform():
    engine = _spawn()
    plugin = _imu(engine)
    assert plugin is not None, "the d435 manifest must ship the D435i's IMU"
    assert np.allclose(plugin.config["pos"], _link_to_mesh(D435I_IMU_XYZ), atol=1e-5)


def test_the_same_chain_reproduces_the_camera_pose_the_model_already_had():
    """The check that makes the one above worth trusting: a number nobody derived for this test."""
    model = _spawn().ctx.model
    cam = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "d435_d435_color")
    assert cam >= 0
    expected = _link_to_mesh([0.0, D435_CAM_DEPTH_TO_COLOR_OFFSET, 0.0])
    assert np.allclose(model.cam_pos[cam], expected, atol=1e-6)


def test_the_site_is_built_on_the_mount_at_that_offset():
    engine = _spawn()
    model = engine.ctx.model
    site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, _imu(engine)._resolved_site)
    assert site >= 0
    body = int(model.site_bodyid[site])
    assert mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body) == "d435_mount"
    assert np.allclose(model.site_pos[site], _link_to_mesh(D435I_IMU_XYZ), atol=1e-5)


def test_the_reported_frame_is_the_optical_one_the_driver_stamps():
    engine = _spawn()
    endpoint = next(e for e in engine.ctx.interface.all() if e.name == "imu")
    hints = endpoint.backend["ros2"]
    # realsense2_camera's own names: `<camera>/imu` (with unite_imu_method set) in
    # camera_imu_optical_frame -- an unmodified RealSense consumer finds it there or not at all.
    assert hints["topic"] == "camera/imu"
    assert hints["frame_id"] == "camera_imu_optical_frame"
    assert endpoint.lazy is True


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
