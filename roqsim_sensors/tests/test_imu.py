"""``imu`` checks: what the three channels read, in which frame, and how the reading is degraded.

The load-bearing assertion is the first one. A simulated accelerometer is only useful to a stack if
it reports **proper** acceleration -- gravity included, as REP 145 requires -- and that property
lives in MuJoCo rather than in the plugin, so nothing in ``imu.py`` shows it to a reader. A level,
stationary robot reading ``+9.81`` on z and a falling one reading ``0`` are what would catch either a
future "helpful" gravity compensation here or a change of convention upstream.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np
import pytest
from roqsim_sensors.plugins.imu import ImuPlugin

from roqsim.config import PluginError, load_config_from_dict
from roqsim.context import Entity, SimContext
from roqsim.plugin import Plugin

GRAVITY = 9.81


class _RobotScene(Plugin):
    """A minimal entity: a free-floating box resting on the floor, registered as a robot.

    The floor and lighting come from the default ``empty_room`` world, so this builds only the body.
    """

    provides_entity = True
    #: Height the base spawns at. Above the floor by more than settling distance -> free fall.
    spawn_z = 0.15
    #: Whether the scene ships its own IMU site + sensor triple (a vendor MJCF that already has one).
    ships_own_imu = False

    def build(self, spec: mujoco.MjSpec, ctx: SimContext) -> None:
        base = spec.worldbody.add_body(name="base_link", pos=[0, 0, self.spawn_z])
        base.add_freejoint()
        base.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.15, 0.15, 0.1], mass=5.0)
        if self.ships_own_imu:
            base.add_site(name="vendor_imu", pos=[0, 0, 0.05])
            for suffix, kind in (
                ("accel", mujoco.mjtSensor.mjSENS_ACCELEROMETER),
                ("gyro", mujoco.mjtSensor.mjSENS_GYRO),
                ("quat", mujoco.mjtSensor.mjSENS_FRAMEQUAT),
            ):
                s = spec.add_sensor()
                s.name = f"vendor_imu_{suffix}"
                s.type = kind
                s.objtype = mujoco.mjtObj.mjOBJ_SITE
                s.objname = "vendor_imu"

    def configure(self, ctx: SimContext) -> None:
        ctx.entities.add(
            Entity(
                name=self.name,
                kind="robot",
                body="base_link",
                meta={"prefix": "", "namespace": ""},
            )
        )


class _FallingScene(_RobotScene):
    """The same robot dropped from high enough that it is still in free fall when read."""

    spawn_z = 20.0


class _VendorImuScene(_RobotScene):
    ships_own_imu = True


def _engine(scene: str = f"{__name__}:_RobotScene", *, steps: int = 400, **imu_config):
    """An engine with one IMU nested under the robot, stepped *steps* times."""
    from roqsim.engine import Engine

    cfg = load_config_from_dict(
        {
            "sim": {},
            "components": [
                {
                    scene: {},
                    "name": "robot",
                    "components": [{"imu": dict(imu_config)}],
                }
            ],
        }
    )
    engine = Engine(cfg)
    engine.setup()
    engine.reset()
    for _ in range(steps):
        engine.step()
    return engine


def _plugin(engine) -> ImuPlugin:
    return next(p for p in engine.plugins if isinstance(p, ImuPlugin))


# -- what it reads ---------------------------------------------------------------------------


def test_a_resting_robot_reads_proper_acceleration_not_zero():
    """Gravity is IN the reading. Every ROS consumer of sensor_msgs/Imu assumes this (REP 145)."""
    reading = _plugin(_engine()).read()
    accel = np.asarray(reading.linear_acceleration)
    assert accel[2] == pytest.approx(GRAVITY, rel=2e-2)
    assert abs(accel[0]) < 0.1 and abs(accel[1]) < 0.1
    # Resting, so no rate; and the attitude is the true one, which here is level.
    assert np.allclose(reading.angular_velocity, 0.0, atol=1e-3)
    assert np.allclose(reading.orientation, [1.0, 0.0, 0.0, 0.0], atol=1e-3)


def test_a_falling_robot_reads_zero():
    """The other half of the same convention: an accelerometer in free fall measures nothing."""
    accel = _plugin(_engine(f"{__name__}:_FallingScene", steps=50)).read().linear_acceleration
    assert np.allclose(accel, 0.0, atol=1e-3)


def test_the_mount_orientation_rotates_the_reading_into_the_sensor_frame():
    """A strap-down device measures in its own frame, so the mount's rotation is observable.

    Rolled 90 deg about x, gravity that is +z in the world lies along the site's own y axis. Getting
    this wrong silently puts a robot's pitch rate on its roll channel, which a filter absorbs as
    plausible motion rather than rejecting.
    """
    reading = _plugin(_engine(rpy=[math.pi / 2, 0.0, 0.0])).read()
    accel = np.asarray(reading.linear_acceleration)
    assert accel[1] == pytest.approx(GRAVITY, rel=2e-2)
    assert abs(accel[0]) < 0.2 and abs(accel[2]) < 0.2
    # The attitude follows the mount too -- it is the SITE's world orientation, so a rolled IMU on a
    # level robot reports the roll. A consumer fuses attitude and rates from one frame or neither.
    assert reading.orientation[0] == pytest.approx(math.cos(math.pi / 4), abs=1e-6)
    assert reading.orientation[1] == pytest.approx(math.sin(math.pi / 4), abs=1e-6)


def test_the_mount_offset_reaches_the_published_static_transform():
    """The frame a bridge publishes comes off the same site the sensors read."""
    engine = _engine(pos=[0.1, 0.0, 0.2], frame_id="imu_link")
    endpoint = next(e for e in engine.ctx.interface.all() if e.name == "imu")
    hints = endpoint.backend["ros2"]
    assert hints["type"] == "sensor_msgs.msg.Imu"
    assert hints["topic"] == "imu/data" and hints["frame_id"] == "imu_link"
    assert hints["static_tf"]["parent"] == "base_link"
    assert np.allclose(hints["static_tf"]["translation"], [0.1, 0.0, 0.2], atol=1e-9)


# -- degradation -----------------------------------------------------------------------------


def test_bias_is_systematic_and_noise_is_not_folded_into_the_covariance():
    clean = _plugin(_engine()).read()
    biased = _plugin(_engine(accel_bias=[0.0, 0.0, 0.5], gyro_bias=[0.02, 0.0, 0.0])).read()
    assert biased.linear_acceleration[2] == pytest.approx(clean.linear_acceleration[2] + 0.5)
    assert biased.angular_velocity[0] == pytest.approx(clean.angular_velocity[0] + 0.02)
    # A bias does not average out, so reporting it as variance would tell a filter it does.
    assert biased.linear_acceleration_variance == 0.0

    noisy = _plugin(_engine(accel_stddev=0.2, gyro_stddev=0.01)).read()
    assert noisy.linear_acceleration_variance == pytest.approx(0.04)
    assert noisy.angular_velocity_variance == pytest.approx(1e-4)


def test_noise_is_the_same_for_two_readers_in_one_step():
    """The endpoint and the blackboard reader are two readers by construction.

    With a stateful generator each read would advance the stream, so a controller and the recorded
    signal would disagree about the rate at one instant -- indistinguishable from a controller bug.
    """
    engine = _engine(accel_stddev=0.3, gyro_stddev=0.1)
    reader = engine.ctx.blackboard.get("imu:robot.imu")
    assert reader is not None and reader.frame == "imu_link"
    endpoint = next(e for e in engine.ctx.interface.all() if e.name == "imu")
    first, second = reader.read(), endpoint.read()
    assert np.allclose(first.linear_acceleration, second.linear_acceleration)
    assert np.allclose(first.angular_velocity, second.angular_velocity)


def test_attitude_noise_is_a_rotation_not_four_perturbed_numbers():
    """Perturbing the components and renormalising would bias the attitude towards level."""
    reading = _plugin(_engine(orientation_stddev=0.05, yaw_stddev=0.1)).read()
    quat = np.asarray(reading.orientation)
    assert np.linalg.norm(quat) == pytest.approx(1.0, abs=1e-9)
    assert not np.allclose(quat, [1.0, 0.0, 0.0, 0.0], atol=1e-4)
    assert reading.orientation_variance == pytest.approx(0.05**2 + 0.1**2)


def test_orientation_false_marks_the_channel_absent():
    """A rate-only IMU must say so; an identity quaternion reads as a level robot."""
    reading = _plugin(_engine(orientation=False)).read()
    assert reading.orientation_valid is False
    # The rates are unaffected: only the attitude channel is withheld.
    assert np.allclose(reading.angular_velocity, 0.0, atol=1e-3)


def test_a_fault_degrades_the_sensor_mid_run_and_a_reset_restores_it():
    engine = _engine(gyro_stddev=0.0, fault={"gyro_stddev": 0.5})
    imu = _plugin(engine)
    imu.set_fault_active(True, engine.ctx.sim_time)
    assert imu.gyro_stddev == 0.5
    assert imu.read().angular_velocity_variance == pytest.approx(0.25)
    engine.reset()
    assert imu.gyro_stddev == 0.0, "a fault must not survive into the next trial of one process"


# -- wiring ----------------------------------------------------------------------------------


def test_an_existing_site_and_sensor_triple_are_reused_not_duplicated():
    engine = _engine(f"{__name__}:_VendorImuScene", site="vendor_imu")
    m = engine.ctx.model
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_SENSOR, i) for i in range(m.nsensor)]
    assert names.count("vendor_imu_accel") == 1
    assert names.count("vendor_imu_quat") == 1
    # And it reads through the model's own triple.
    assert _plugin(engine).read().linear_acceleration[2] == pytest.approx(GRAVITY, rel=2e-2)


def test_two_imus_on_one_robot_get_their_own_sites():
    """Two devices on one link is a real configuration (a redundant pair), and MuJoCo would refuse
    two sites of one name -- so the site is named off the LABEL, not off the plugin."""
    from roqsim.engine import Engine

    cfg = load_config_from_dict(
        {
            "sim": {},
            "components": [
                {
                    f"{__name__}:_RobotScene": {},
                    "name": "robot",
                    "components": [
                        {"imu": {"pos": [0.1, 0, 0]}, "name": "front_imu"},
                        {"imu": {"pos": [-0.1, 0, 0]}, "name": "rear_imu"},
                    ],
                }
            ],
        }
    )
    engine = Engine(cfg)
    engine.setup()
    engine.reset()
    sites = [
        mujoco.mj_id2name(engine.ctx.model, mujoco.mjtObj.mjOBJ_SITE, i)
        for i in range(engine.ctx.model.nsite)
    ]
    assert "base_link_front_imu_site" in sites and "base_link_rear_imu_site" in sites


def test_an_imu_at_the_top_of_a_document_is_refused():
    """It measures the motion of the body it is bolted to, so it belongs to an entity."""
    with pytest.raises(PluginError):
        load_config_from_dict({"sim": {}, "components": [{"imu": {}}]})


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        ({"site": "vendor_imu", "pos": [0, 0, 1]}, "would be ignored"),
        ({"quat": [1, 0, 0, 0], "rpy": [0, 0, 0]}, "not both"),
        ({"seed": 7}, "not an imu setting"),
        ({"rate_hz": 0}, "must be > 0"),
        ({"accel_stddev": -1}, "must be >= 0"),
        ({"gyro_bias": [0, 0]}, "three numbers"),
        ({"fault": {"rate_hz": 50}}, "cannot be written"),
        ({"fault": {}}, "empty"),
    ],
)
def test_config_errors_are_reported_by_name(config, expected):
    errors = ImuPlugin(config, entity="robot", label="imu").validate_config(config)
    assert any(expected in e for e in errors), errors
