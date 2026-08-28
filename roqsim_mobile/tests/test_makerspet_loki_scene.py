"""Maker's Pet Loki: what a caster costs when MuJoCo will not let you make it slippery.

The finding this file pins is ``test_the_caster_needs_priority_not_just_low_friction``. Loki carries
0.42 kg of its 1.72 on a *front* caster -- a quarter of the robot -- and a real caster swivels, so it
should offer almost no lateral resistance. Setting a low ``friction`` on the caster geom does
nothing at all, because MuJoCo takes the **maximum** of the two contacting geoms' friction unless one
of them sets ``priority``. The floor's 1.0 wins, the caster scrubs, and yaw tracks 0.77-0.87 of
commanded. With ``priority`` it tracks 0.92-0.93.

The second lesson is in ``test_wheels_track_the_commanded_speed``: a velocity servo's steady-state
error is (required torque)/kv, so at the gain inherited from the vacuum these wheels ran 21% slow
while drawing a quarter of their torque limit. Neither symptom looks like its cause -- one looks like
slip, the other like saturation -- which is why both are measured rather than tuned by eye.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import pytest

from mobile_scene_utils import named

from roqsim.config import load_config_from_dict
from roqsim.engine import Engine

#: From the expanded makerspet/makerspet_loki description @ e778ecde, not measured from our model.
TOTAL_MASS = 1.720
WHEEL_RADIUS = 0.0335          # the description's wheel_diameter 0.067
WHEEL_SEPARATION = 0.159063    # the description's wheel_base
MAX_LINEAR_VEL = 0.26          # config/navigation.yaml max_vel_x
MAX_ANGULAR_VEL = 1.0          # config/navigation.yaml max_vel_theta
LIDAR_HEIGHT = 0.13135         # the description's scan_joint, in the base_link frame


def _engine(**diff_drive):
    world = {
        "sim": {"timestep": 0.002},
        "components": [{
            "spawn_robot": {"model": "makerspet_loki", "prefix": "l_"},
            "name": "l",
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


def _yaw_ratio(engine, commanded):
    model, data = engine.ctx.model, engine.ctx.data
    bid = named(model, mujoco.mjtObj.mjOBJ_BODY, "l_base_link")
    handle = engine.ctx.blackboard.get("robot:l")
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
    return (total / (float(data.time) - t0)) / commanded


def test_mass_matches_the_vendor_description():
    engine = _engine()
    try:
        assert engine.ctx.model.body_mass.sum() == pytest.approx(TOTAL_MASS, abs=1e-3)
    finally:
        engine.shutdown()


def test_manifest_brings_the_drive_and_the_lidar():
    engine = _engine()
    try:
        assert engine.ctx.blackboard.get("robot:l") is not None, "diff_drive did not attach"
        assert any(type(p).__name__ == "LidarPlugin" for p in engine.plugins), "lidar did not attach"
    finally:
        engine.shutdown()


def test_drive_geometry_is_the_vendors():
    engine = _engine()
    try:
        drive = next(p for p in engine.plugins if "DiffDrive" in type(p).__name__)
        assert drive.config["wheel_radius"] == pytest.approx(WHEEL_RADIUS)
        assert drive.config["wheel_separation"] == pytest.approx(WHEEL_SEPARATION)
        assert drive.config["max_linear_vel"] == pytest.approx(MAX_LINEAR_VEL)
        assert "slip_factor" not in drive.config, (
            "a true 2-wheel differential drive with a caster does not scrub"
        )
    finally:
        engine.shutdown()


def test_it_rests_on_two_wheels_and_the_caster():
    """Catches the joint-rpy class of defect: the wheel joints carry rpy="-pi/2 0 0"."""
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
        # named() first: a "not in touching" assertion passes for free when the geom is absent,
        # which is exactly how the root link's missing body cylinder hid during this port.
        for geom in ("l_wheel_left_link_collision0", "l_wheel_right_link_collision0",
                     "l_caster_link_collision0", "l_base_link_collision0"):
            named(model, mujoco.mjtObj.mjOBJ_GEOM, geom)
        assert "l_wheel_left_link_collision0" in touching, touching
        assert "l_wheel_right_link_collision0" in touching, touching
        assert "l_caster_link_collision0" in touching, touching
        assert "l_base_link_collision0" not in touching, "the body is dragging on the floor"
    finally:
        engine.shutdown()


def test_wheels_spin_about_the_robot_y_axis():
    engine = _engine()
    try:
        model, data = engine.ctx.model, engine.ctx.data
        for side in ("left", "right"):
            gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM,
                                    f"l_wheel_{side}_link_collision0")
            axis = data.geom_xmat[gid].reshape(3, 3)[:, 2]
            assert abs(abs(axis[1]) - 1.0) < 1e-6, (
                f"{side} wheel axis is {np.round(axis, 4)}, not along y")
    finally:
        engine.shutdown()


def test_the_scanner_is_inverted_between_the_decks():
    """The description mounts a 360-degree puck upside down (scan_joint rpy -pi) to fit under the
    head. The height is the vendor's; only the scan parameters are ours."""
    engine = _engine()
    try:
        model, data = engine.ctx.model, engine.ctx.data
        base = named(model, mujoco.mjtObj.mjOBJ_BODY, "l_base_link")
        sid = named(model, mujoco.mjtObj.mjOBJ_SITE, "l_lidar")
        height = float(data.site_xpos[sid][2] - data.xpos[base][2])
        assert height == pytest.approx(LIDAR_HEIGHT, abs=1e-3)
        puck = named(model, mujoco.mjtObj.mjOBJ_BODY, "l_base_scan")
        z_axis = data.xmat[puck].reshape(3, 3)[:, 2]
        assert z_axis[2] < -0.99, f"the scanner puck is not inverted: z axis {np.round(z_axis, 3)}"
    finally:
        engine.shutdown()


def test_b1_drives_straight():
    engine = _engine()
    try:
        model, data = engine.ctx.model, engine.ctx.data
        bid = named(model, mujoco.mjtObj.mjOBJ_BODY, "l_base_link")
        handle = engine.ctx.blackboard.get("robot:l")
        for _ in range(500):
            engine.step()
        handle.drive(MAX_LINEAR_VEL, 0.0, 0.0)
        for _ in range(500):
            engine.step()
        x0, t0 = float(data.xpos[bid][0]), float(data.time)
        for _ in range(1500):
            engine.step()
        speed = (float(data.xpos[bid][0]) - x0) / (float(data.time) - t0)
        assert 0.93 < speed / MAX_LINEAR_VEL < 1.05, (
            f"commanded {MAX_LINEAR_VEL} m/s, achieved {speed:.4f} m/s")
        assert abs(_yaw(data, bid)) < 0.02, "veered while driving straight"
    finally:
        engine.shutdown()


@pytest.mark.parametrize("commanded", [0.3, 0.6, 1.0])
def test_b2_rotates_at_the_commanded_rate(commanded):
    engine = _engine()
    try:
        ratio = _yaw_ratio(engine, commanded)
        assert 0.88 < ratio < 1.05, f"achieved/commanded yaw {ratio:.3f} at {commanded} rad/s"
    finally:
        engine.shutdown()


def test_the_caster_needs_priority_not_just_low_friction():
    """The finding -- see the module docstring.

    Asserted structurally rather than by re-running the drive, because the point is exactly that the
    friction number alone is inert: a test that only checked ``friction`` would have passed while the
    caster scrubbed.
    """
    engine = _engine()
    try:
        model = engine.ctx.model
        caster = named(model, mujoco.mjtObj.mjOBJ_GEOM, "l_caster_link_collision0")
        assert model.geom_priority[caster] > 0, (
            "the caster carries no contact priority, so MuJoCo takes the MAXIMUM of its friction "
            "and the floor's -- the low friction below is then inert and the caster scrubs"
        )
        assert model.geom_friction[caster][0] < 0.2, "the caster is not slippery enough to swivel"
        for side in ("left", "right"):
            wheel = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM,
                                      f"l_wheel_{side}_link_collision0")
            assert model.geom_friction[wheel][0] >= 1.0, "the driven wheels must keep their grip"
    finally:
        engine.shutdown()


def test_wheels_track_the_commanded_speed():
    """A velocity servo's steady-state error is (required torque)/kv -- see the module docstring."""
    engine = _engine()
    try:
        model, data = engine.ctx.model, engine.ctx.data
        handle = engine.ctx.blackboard.get("robot:l")
        dof = model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT,
                                                 "l_wheel_left_joint")]
        for _ in range(500):
            engine.step()
        handle.drive(0.0, 0.0, MAX_ANGULAR_VEL)
        for _ in range(1200):
            engine.step()
        ideal = MAX_ANGULAR_VEL * WHEEL_SEPARATION / 2 / WHEEL_RADIUS
        assert abs(float(data.qvel[dof])) / ideal > 0.9, (
            f"wheel runs at {abs(float(data.qvel[dof])):.3f} of an ideal {ideal:.3f} rad/s; the "
            f"servo gain is not tracking"
        )
    finally:
        engine.shutdown()
