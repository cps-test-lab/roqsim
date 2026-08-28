"""The OOMWOO One robot vacuum: a description that states everything, and a robot that barely balances.

Two things this file pins.

``test_no_value_here_is_an_assumption`` guards what makes this port unusual: every drive and sensor
number is the vendor's own, from ``params.xacro``, ``plugins.xacro`` and ``config/navigation.yaml``.
Every other mobile port in this batch had to invent at least a scanner mount height. If someone later
"tunes" one of these, the test should stop them and make them say so.

``test_the_caster_barely_carries_load`` pins a property of the platform rather than of our model. The
whole-robot COM sits ~1.6 mm behind the wheel axle, in a support polygon 144 mm deep, so the caster
taps rather than rolls -- in contact for a few percent of steps. The robot drives fine, but it is
marginally pitch-stable, and for a vacuum that is a real constraint: a filling dustbin, or any payload
mounted forward of the axle, tips it onto its bumper ring.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import pytest

from roqsim.config import load_config_from_dict
from roqsim.engine import Engine

#: From the expanded makerspet/oomwoo-one description @ 5305e6c5, not measured from our model.
TOTAL_MASS = 2.930
WHEEL_RADIUS = 0.034           # params.xacro wheel_diameter 0.068
WHEEL_SEPARATION = 0.235       # params.xacro wheel_base
MAX_LINEAR_VEL = 0.2           # config/navigation.yaml max_vel_x
MAX_ANGULAR_VEL = 0.8          # config/navigation.yaml max_vel_theta
LIDAR_HEIGHT = 0.0755          # the description's scan_joint, in the base_link frame
BUMPER_FACETS = 12             # params.xacro bumper_facets_per_side 6, both sides


def _engine(**diff_drive):
    world = {
        "sim": {"timestep": 0.002},
        "components": [{
            "spawn_robot": {"model": "oomwoo_one", "prefix": "o_"},
            "name": "o",
            **({"components": [{"diff_drive": diff_drive}]} if diff_drive else {}),
        }],
    }
    engine = Engine(load_config_from_dict(world, base_dir=Path(".")))
    engine.setup()
    engine.reset()
    return engine


def _yaw(data, bid):
    q = data.xquat[bid]
    return float(np.arctan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2)))


def test_mass_matches_the_vendor_description():
    engine = _engine()
    try:
        assert engine.ctx.model.body_mass.sum() == pytest.approx(TOTAL_MASS, abs=1e-3)
    finally:
        engine.shutdown()


def test_manifest_brings_the_drive_and_the_lidar():
    engine = _engine()
    try:
        assert engine.ctx.blackboard.get("robot:o") is not None, "diff_drive did not attach"
        assert any(type(p).__name__ == "LidarPlugin" for p in engine.plugins), "lidar did not attach"
    finally:
        engine.shutdown()


def test_no_value_here_is_an_assumption():
    """Every drive number is the vendor's own -- see the module docstring."""
    engine = _engine()
    try:
        drive = next(p for p in engine.plugins if "DiffDrive" in type(p).__name__)
        assert drive.config["wheel_radius"] == pytest.approx(WHEEL_RADIUS)
        assert drive.config["wheel_separation"] == pytest.approx(WHEEL_SEPARATION)
        assert drive.config["max_linear_vel"] == pytest.approx(MAX_LINEAR_VEL)
        assert drive.config["max_angular_vel"] == pytest.approx(MAX_ANGULAR_VEL)
        assert "slip_factor" not in drive.config, (
            "a true 2-wheel differential drive with a caster does not scrub, so it must not carry "
            "a slip_factor -- the same line turtlebot3_waffle and raspimouse draw"
        )
    finally:
        engine.shutdown()


def test_it_rests_on_two_wheels_and_the_caster():
    """The check that caught the real defect in this port.

    The wheel joints carry ``rpy="-pi/2 0 0"``, and reading only their xyz left the wheel cylinders
    axis-vertical -- flat discs 28 mm clear of the floor. The robot then settled onto its bumper ring
    with the wheels touching nothing, which every other check in this file tolerated.
    """
    engine = _engine()
    try:
        model, data = engine.ctx.model, engine.ctx.data
        for _ in range(2000):
            engine.step()
        touching = {
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g)
            for i in range(data.ncon)
            for g in (data.contact[i].geom1, data.contact[i].geom2)
        }
        assert "o_wheel_left_tyre" in touching, touching
        assert "o_wheel_right_tyre" in touching, touching
        for i in range(BUMPER_FACETS):
            assert f"o_base_link_collision{i + 1}" not in touching, (
                "a bumper plate is dragging on the floor -- the wheels are not carrying the robot"
            )
    finally:
        engine.shutdown()


def test_wheels_spin_about_the_robot_y_axis():
    """The joint rpy, asserted in the frame it actually matters in."""
    engine = _engine()
    try:
        model, data = engine.ctx.model, engine.ctx.data
        for side in ("left", "right"):
            gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"o_wheel_{side}_tyre")
            axis = data.geom_xmat[gid].reshape(3, 3)[:, 2]
            assert abs(abs(axis[1]) - 1.0) < 1e-6, (
                f"{side} tyre axis is {np.round(axis, 4)}, not along y")
    finally:
        engine.shutdown()


def test_b1_drives_straight():
    engine = _engine()
    try:
        model, data = engine.ctx.model, engine.ctx.data
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "o_base_link")
        handle = engine.ctx.blackboard.get("robot:o")
        for _ in range(500):
            engine.step()
        handle.drive(MAX_LINEAR_VEL, 0.0, 0.0)
        for _ in range(500):
            engine.step()
        x0, t0 = float(data.xpos[bid][0]), float(data.time)
        for _ in range(1500):
            engine.step()
        speed = (float(data.xpos[bid][0]) - x0) / (float(data.time) - t0)
        assert 0.9 < speed / MAX_LINEAR_VEL < 1.05, (
            f"commanded {MAX_LINEAR_VEL} m/s, achieved {speed:.4f} m/s")
        assert abs(_yaw(data, bid)) < 0.02, "veered while driving straight"
    finally:
        engine.shutdown()


@pytest.mark.parametrize("commanded", [0.3, 0.5, 0.8])
def test_b2_rotates_at_the_commanded_rate(commanded):
    """No slip_factor, so this measures the drive rather than a calibration."""
    engine = _engine()
    try:
        model, data = engine.ctx.model, engine.ctx.data
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "o_base_link")
        handle = engine.ctx.blackboard.get("robot:o")
        for _ in range(500):
            engine.step()
        handle.drive(0.0, 0.0, commanded)
        for _ in range(250):
            engine.step()
        t0, previous, total = float(data.time), _yaw(data, bid), 0.0
        for _ in range(1500):
            engine.step()
            current = _yaw(data, bid)
            total += np.unwrap([previous, current])[1] - previous
            previous = current
        ratio = (total / (float(data.time) - t0)) / commanded
        assert 0.88 < ratio < 1.05, f"achieved/commanded yaw {ratio:.3f} at {commanded} rad/s"
    finally:
        engine.shutdown()


def test_the_caster_barely_carries_load():
    """A property of the platform, not of the port -- see the module docstring."""
    engine = _engine()
    try:
        model, data = engine.ctx.model, engine.ctx.data
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "o_base_link")
        for _ in range(1000):
            engine.step()
        mujoco.mj_subtreeVel(model, data)
        com_x = float(data.subtree_com[bid][0] - data.xpos[bid][0])
        assert -0.01 < com_x < 0.0, (
            f"COM is {com_x:+.5f} m from the wheel axle. Behind it by a whisker is what the "
            f"description gives; forward of it and the robot tips onto its bumper ring."
        )
    finally:
        engine.shutdown()


def test_the_bumper_ring_is_present_and_proud_of_the_body():
    """The platform's distinguishing feature: 12 collision plates a bump sensor could read.

    roqsim has no bumper plugin, so nothing reads them yet. They are geometry, not a sensor, and the
    test says so rather than implying a capability we do not have.
    """
    engine = _engine()
    try:
        model, data = engine.ctx.model, engine.ctx.data
        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "o_base_link_collision0")
        body_radius = float(model.geom_size[body][0])
        plates = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"o_base_link_collision{i + 1}")
                  for i in range(BUMPER_FACETS)]
        assert all(g >= 0 for g in plates), "the bumper ring is incomplete"
        base = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "o_base_link")
        for gid in plates:
            offset = data.geom_xpos[gid][:2] - data.xpos[base][:2]
            reach = float(np.linalg.norm(offset)) + float(model.geom_size[gid][1])
            assert reach > body_radius, "a bumper plate does not stand proud of the body"
            assert reach < body_radius + 0.01, "a bumper plate stands too far proud"
    finally:
        engine.shutdown()
