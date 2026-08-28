"""The Neobotix MPO-700: the substrate's first swerve base, and a vendor that ships it welded shut.

Two findings this file pins.

``test_the_wheels_aim`` is the swerve inverse kinematics, asserted as the invariant -- each steer
angle is ``atan2`` of its corner's contact velocity ``v + w x r`` -- rather than against a table of
angles. The decisive case is a pure spin, where every wheel sits tangential to its own corner radius
and the two diagonal pairs therefore end up at *different* angles (126.87 and 53.13 degrees for
offsets of ±0.24, ±0.18). Forward, strafe and diagonal all put every wheel at one shared angle and
are easy to pass by accident; a first draft of this file hardcoded four identical values and failed
against a model that was correct.

``test_the_wheels_do_not_fight_the_body`` pins an exclusion. The vendor's base collision *is* the full
body mesh, and MuJoCo convex-hulls a collision mesh, so the hull closes over the wheel arches and
overlaps the wheels inside them. A wheel is a grandchild of ``base_link`` via its steering link, so
MuJoCo's automatic parent-child exclusion does not cover the pair and the robot rests fighting itself.

Worth knowing while reading: the vendor's own description has **every joint fixed**. This model
instantiates the steer/roll chain its xacro macros can already describe, which is a deliberate
improvement on upstream and comes with no vendor steer limits to calibrate against.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import pytest
from mobile_scene_utils import named

from roqsim.config import load_config_from_dict
from roqsim.engine import Engine

#: From the expanded neo_simulation2 humble xacro @ 83204145, not measured from our model.
TOTAL_MASS = 196.8010
WHEEL_RADIUS = 0.09
#: Steering axis offsets from base_link, (x, y), in omni_drive's WHEEL_ORDER.
CORNERS = ((0.24, 0.18), (0.24, -0.18), (-0.24, 0.18), (-0.24, -0.18))
CORNER_NAMES = ("front_left", "front_right", "back_left", "back_right")
MAX_VEL = 1.0


def _engine():
    world = {
        "sim": {"timestep": 0.002},
        "components": [{"spawn_robot": {"model": "mpo_700", "prefix": "n_"}, "name": "n"}],
    }
    engine = Engine(load_config_from_dict(world, base_dir=Path(".")))
    engine.setup()
    engine.reset()
    return engine


def _steer_angles(engine):
    model, data = engine.ctx.model, engine.ctx.data
    return [
        float(data.qpos[model.jnt_qposadr[
            named(model, mujoco.mjtObj.mjOBJ_JOINT, f"n_mpo_700_caster_{c}_joint")]])
        for c in CORNER_NAMES
    ]


def _drive(engine, vx, vy, wz, steps=2600):
    """Command a body-frame twist and return the achieved body-frame twist.

    2600 steps = 5.2 s, because the vendor's acceleration limit is 0.25 m/s^2: reaching their 0.8 m/s
    top speed alone takes 3.2 s. A shorter window measures the ramp, not the steady state -- which is
    what this test did until the limits were corrected from a remembered datasheet figure to the
    vendor's own Nav2 profile.
    """
    model, data = engine.ctx.model, engine.ctx.data
    bid = named(model, mujoco.mjtObj.mjOBJ_BODY, "n_base_link")
    dof = model.jnt_dofadr[named(model, mujoco.mjtObj.mjOBJ_JOINT, "n_base_free")]
    engine.ctx.blackboard.get("robot:n").drive(vx, vy, wz)
    for _ in range(steps):
        engine.step()
    rot = data.xmat[bid].reshape(3, 3)
    local = rot.T @ np.array(data.qvel[dof:dof + 3])
    return float(local[0]), float(local[1]), float(data.qvel[dof + 5])


def test_the_limits_are_not_isotropic():
    """The vendor's Nav2 profile allows 0.8 m/s forward and only 0.5 sideways.

    Pinned because the first version of this port used 1.0 for both, from a remembered datasheet
    figure rather than the file -- overstating the lateral direction by 60%.
    """
    engine = _engine()
    try:
        drive = next(p for p in engine.plugins if "OmniDrive" in type(p).__name__)
        assert drive.config["max_linear_vel"] == pytest.approx(0.8)
        assert drive.config["max_lateral_vel"] == pytest.approx(0.5)
        assert drive.config["max_lateral_vel"] < drive.config["max_linear_vel"]
    finally:
        engine.shutdown()


def test_mass_matches_the_vendor_description():
    engine = _engine()
    try:
        assert engine.ctx.model.body_mass.sum() == pytest.approx(TOTAL_MASS, abs=1e-3)
    finally:
        engine.shutdown()


def test_manifest_brings_a_swerve_drive_and_two_scanners():
    """The first model here to declare more than one lidar, because the vendor ships two."""
    engine = _engine()
    try:
        drive = next(p for p in engine.plugins if "OmniDrive" in type(p).__name__)
        assert drive.config["steer_joints"], "swerve IK not selected -- no steer_joints"
        assert len(drive.config["steer_joints"]) == 4
        assert "slip_factor" not in drive.config, "a swerve base does not turn by scrubbing"
        scans = [p for p in engine.plugins if type(p).__name__ == "LidarPlugin"]
        assert len(scans) == 2, f"expected the vendor's two S300s, got {len(scans)}"
        assert {p.config["site"] for p in scans} == {"lidar_1", "lidar_2"}
    finally:
        engine.shutdown()


def test_it_rests_on_four_wheels():
    engine = _engine()
    try:
        model, data = engine.ctx.model, engine.ctx.data
        for _ in range(2500):
            engine.step()
        touching = {
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g)
            for i in range(data.ncon)
            for g in (data.contact[i].geom1, data.contact[i].geom2)
        }
        for name in CORNER_NAMES:
            named(model, mujoco.mjtObj.mjOBJ_GEOM, f"n_mpo_700_wheel_{name}_link_tyre")
            assert f"n_mpo_700_wheel_{name}_link_tyre" in touching, touching
        named(model, mujoco.mjtObj.mjOBJ_GEOM, "n_base_link_collision")
        assert "n_base_link_collision" not in touching, "the body is on the floor"
    finally:
        engine.shutdown()


def test_the_wheels_do_not_fight_the_body():
    """The exclusion -- see the module docstring. Asserted as an absence of self-contact at rest."""
    engine = _engine()
    try:
        model, data = engine.ctx.model, engine.ctx.data
        for _ in range(1500):
            engine.step()
        for i in range(data.ncon):
            g1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, data.contact[i].geom1) or ""
            g2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, data.contact[i].geom2) or ""
            pair = {g1, g2}
            assert not (any("tyre" in n for n in pair) and any("base_link_collision" in n
                                                              for n in pair)), (
                f"a wheel is in contact with the robot's own body hull ({g1} <-> {g2}); the "
                f"<contact><exclude> pairs are missing or renamed"
            )
    finally:
        engine.shutdown()


@pytest.mark.parametrize("command", [(0.6, 0.0, 0.0), (0.0, 0.6, 0.0), (0.4, 0.4, 0.0),
                                     (0.0, 0.0, 0.6), (0.3, -0.2, 0.4)])
def test_the_wheels_aim(command):
    """Swerve IK, asserted as the invariant rather than against a table of angles.

    Each steer angle must be ``atan2`` of that corner's contact velocity, ``v + w x r``. Writing the
    formula rather than four numbers is what makes the test meaningful: a hardcoded table got the
    pure-spin case wrong here, because the tangential angle **alternates** between corners (126.87
    and 53.13 degrees for these offsets) rather than being one value, and a wrong table is
    indistinguishable from a wrong model until the geometry is written down.

    Compared modulo pi, because ``omni_drive`` resolves each target to the nearer of the two
    equivalent headings and negates the roll rate for the flipped one -- so -90 and +90 are the same
    instruction, and pinning the branch would pin where the wheel happened to start.
    """
    vx, vy, wz = command
    engine = _engine()
    try:
        for _ in range(500):
            engine.step()
        _drive(engine, *command)
        for angle, (x, y), name in zip(_steer_angles(engine), CORNERS, CORNER_NAMES, strict=True):
            want = np.degrees(np.arctan2(vy + wz * x, vx - wz * y)) % 180.0
            got = np.degrees(angle) % 180.0
            gap = min(abs(got - want), abs(got - want - 180.0), abs(got - want + 180.0))
            assert gap < 2.0, (
                f"{name} at ({x}, {y}) aimed {np.degrees(angle):+.1f} deg; the corner velocity "
                f"({vx - wz * y:+.3f}, {vy + wz * x:+.3f}) wants {want:.1f} (mod 180)")
    finally:
        engine.shutdown()


def test_a_pure_spin_aims_each_wheel_tangentially():
    """The one case worth stating in closed form, because it is where a plausible guess goes wrong.

    Under a pure spin every wheel is tangential to its own corner radius, so the two diagonal pairs
    sit at *different* angles -- 126.87 and 53.13 degrees for offsets of (0.24, 0.18). Forward,
    strafe and diagonal all put every wheel at one shared angle and are easy to pass by accident.
    """
    engine = _engine()
    try:
        for _ in range(500):
            engine.step()
        _drive(engine, 0.0, 0.0, 0.6)
        angles = [np.degrees(a) % 180.0 for a in _steer_angles(engine)]
        assert angles[0] == pytest.approx(126.87, abs=2.0), angles
        assert angles[1] == pytest.approx(53.13, abs=2.0), angles
        assert angles[2] == pytest.approx(53.13, abs=2.0), angles
        assert angles[3] == pytest.approx(126.87, abs=2.0), angles
    finally:
        engine.shutdown()


#: Commands stay inside the vendor's own per-axis limits (0.8 forward, 0.5 sideways). The lateral
#: cases are 0.5, not 0.6: this base is deliberately NOT isotropic, and a first draft of this test
#: commanded 0.6 sideways and failed against a model that was correctly clamping.
@pytest.mark.parametrize("command", [(0.8, 0.0, 0.0), (0.0, 0.5, 0.0), (0.0, 0.0, 0.6),
                                     (0.4, 0.4, 0.0)])
def test_it_tracks_a_holonomic_twist(command):
    engine = _engine()
    try:
        for _ in range(500):
            engine.step()
        achieved = _drive(engine, *command)
        for got, want, axis in zip(achieved, command, "xyw", strict=True):
            if abs(want) < 1e-9:
                assert abs(got) < 0.02, f"{axis}: commanded 0, got {got:+.4f}"
            else:
                assert 0.9 < got / want < 1.06, f"{axis}: commanded {want}, got {got:+.4f}"
    finally:
        engine.shutdown()


def test_the_wheels_roll_at_the_right_rate():
    """Roll rate must be the corner speed over the wheel radius, not the base speed."""
    engine = _engine()
    try:
        model, data = engine.ctx.model, engine.ctx.data
        for _ in range(500):
            engine.step()
        _drive(engine, 0.6, 0.0, 0.0)
        for name in CORNER_NAMES:
            jid = named(model, mujoco.mjtObj.mjOBJ_JOINT, f"n_mpo_700_wheel_{name}_joint")
            rate = abs(float(data.qvel[model.jnt_dofadr[jid]]))
            assert rate == pytest.approx(0.6 / WHEEL_RADIUS, rel=0.15), (
                f"{name} rolls at {rate:.3f} rad/s; 0.6 m/s over a {WHEEL_RADIUS} m wheel is "
                f"{0.6 / WHEEL_RADIUS:.3f}")
    finally:
        engine.shutdown()
