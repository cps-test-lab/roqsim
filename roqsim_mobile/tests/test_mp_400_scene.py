"""The Neobotix MP-400: a differential Neobotix, and a description whose masses cannot be trusted.

Two findings this file pins.

``test_the_caster_masses_are_upstream_nonsense`` records a defect in the source, asserted rather than
merely noted. Each 38 mm caster sphere is declared at **12.7 kg** — exactly the mass of the MPO-700's
steering modules in the same repository — so 50.8 kg of this 84.4 kg robot sits in four casters. That
is provably copy-paste, not a plausible figure, and it is why this model is fit for navigation and not
for dynamics. The test exists so nobody "fixes" the mass audit by quietly substituting our own number:
the audit's value is that it checks the vendor's.

``test_the_wheel_axis_matches_the_plugin_convention`` pins a sign. ``diff_drive`` writes the commanded
wheel rate straight to the actuator with no sign derivation, so it requires wheels whose axis is +y in
the base frame — which all five existing ``diff_drive`` models happen to have. This description's axis
is -y, and with it the robot drives and turns *backwards* (measured: -0.984 of a forward command).
``omni_drive`` derives the sign off the model instead and would not have cared. That asymmetry between
the two drive plugins is recorded in the port log as a substrate follow-up.
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
TOTAL_MASS = 84.4482
BASE_MASS = 30.0
CASTER_MASS = 12.7        # each, for a 38 mm sphere. See the module docstring.
WHEEL_MASS = 1.82362
WHEEL_RADIUS = 0.0765
WHEEL_SEPARATION = 0.52
MAX_LINEAR_VEL = 0.8      # configs/mp_400/navigation.yaml max_vel_x
WHEELS = ("left", "right")
CASTERS = ("front_left", "front_right", "back_left", "back_right")


def _engine():
    world = {
        "sim": {"timestep": 0.002},
        "components": [{"spawn_robot": {"model": "mp_400", "prefix": "q_"}, "name": "q"}],
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


def test_the_caster_masses_are_upstream_nonsense():
    """A defect in the source, asserted so it cannot be silently "corrected" -- see the docstring."""
    engine = _engine()
    try:
        model = engine.ctx.model
        for corner in CASTERS:
            bid = named(model, mujoco.mjtObj.mjOBJ_BODY, f"q_mp_400_caster_wheel_{corner}_link")
            assert model.body_mass[bid] == pytest.approx(CASTER_MASS, abs=1e-3), (
                "the caster masses are the vendor's, absurd though 12.7 kg for a 38 mm sphere is. "
                "If this now differs, someone substituted a plausible number -- which breaks the "
                "mass audit's whole purpose, since it exists to check the description."
            )
        casters = 4 * CASTER_MASS
        assert casters / TOTAL_MASS > 0.5, (
            f"{casters} kg of {TOTAL_MASS} kg is in the casters; if that share has changed the "
            f"port log's 'navigation, not dynamics' boundary needs revisiting")
    finally:
        engine.shutdown()


def test_the_wheel_axis_matches_the_plugin_convention():
    """diff_drive has no sign derivation, so the model must supply +y -- see the docstring."""
    engine = _engine()
    try:
        model, data = engine.ctx.model, engine.ctx.data
        base = named(model, mujoco.mjtObj.mjOBJ_BODY, "q_base_link")
        rot = data.xmat[base].reshape(3, 3)
        for side in WHEELS:
            jid = named(model, mujoco.mjtObj.mjOBJ_JOINT, f"q_mp_400_fixed_wheel_{side}_joint")
            axis = rot.T @ (data.xmat[model.jnt_bodyid[jid]].reshape(3, 3) @ model.jnt_axis[jid])
            assert axis[1] > 0.99, (
                f"{side} wheel axis is {np.round(axis, 4)}; diff_drive writes the wheel rate with no "
                f"sign derivation and needs +y, or the robot drives backwards"
            )
    finally:
        engine.shutdown()


def test_manifest_brings_a_differential_drive_and_one_scanner():
    """One lidar here, unlike both holonomic Neobotix siblings' two."""
    engine = _engine()
    try:
        drive = next(p for p in engine.plugins if "DiffDrive" in type(p).__name__)
        assert drive.config["wheel_radius"] == pytest.approx(WHEEL_RADIUS)
        assert drive.config["wheel_separation"] == pytest.approx(WHEEL_SEPARATION)
        assert "slip_factor" not in drive.config, (
            "two driven wheels with passive casters do not scrub, so no slip_factor -- the same line "
            "turtlebot3_waffle, raspimouse and oomwoo_one draw"
        )
        scans = [p for p in engine.plugins if type(p).__name__ == "LidarPlugin"]
        assert len(scans) == 1, f"the MP-400 ships one front S300, got {len(scans)}"
    finally:
        engine.shutdown()


def test_it_rests_on_its_wheels_and_casters():
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
        for side in WHEELS:
            geom = f"q_mp_400_fixed_wheel_{side}_link_tyre"
            named(model, mujoco.mjtObj.mjOBJ_GEOM, geom)
            assert geom in touching, touching
        named(model, mujoco.mjtObj.mjOBJ_GEOM, "q_base_link_collision")
        assert "q_base_link_collision" not in touching, "the body is dragging on the floor"
    finally:
        engine.shutdown()


def test_the_casters_slide_rather_than_grip():
    """They are FIXED spheres, not articulated wheels, so they stand in for casters via friction.

    `priority` is what makes the low friction apply at all: MuJoCo otherwise takes the maximum of
    the two contacting geoms' friction and the floor's value wins, at which point four loaded
    spheres fight every turn.
    """
    engine = _engine()
    try:
        model = engine.ctx.model
        for corner in CASTERS:
            gid = named(model, mujoco.mjtObj.mjOBJ_GEOM,
                        f"q_mp_400_caster_wheel_{corner}_link_tyre")
            assert model.geom_priority[gid] > 0
            assert model.geom_friction[gid][0] < 0.2
        for side in WHEELS:
            gid = named(model, mujoco.mjtObj.mjOBJ_GEOM, f"q_mp_400_fixed_wheel_{side}_link_tyre")
            assert model.geom_friction[gid][0] >= 1.0, "the driven wheels must keep their grip"
    finally:
        engine.shutdown()


def test_b1_drives_straight():
    """Windows are sized to the room, not to patience.

    The vendor's acceleration limit is 0.25 m/s^2, so reaching 0.8 m/s takes 3.2 s -- and at that
    speed the robot crosses `empty_room`'s 5 m half-extent in about six more. A first draft measured
    over eight seconds and read 0.82 of commanded from a model whose steady state is 0.985, because
    the window ended against a wall. That is the third time this batch; measure the ramp out, then
    take a short sample.
    """
    engine = _engine()
    try:
        model, data = engine.ctx.model, engine.ctx.data
        bid = named(model, mujoco.mjtObj.mjOBJ_BODY, "q_base_link")
        handle = engine.ctx.blackboard.get("robot:q")
        for _ in range(500):
            engine.step()
        handle.drive(MAX_LINEAR_VEL, 0.0, 0.0)
        for _ in range(1750):        # 3.5 s, just past the 3.2 s acceleration ramp
            engine.step()
        x0, t0 = float(data.xpos[bid][0]), float(data.time)
        for _ in range(600):         # 1.2 s, about 0.95 m
            engine.step()
        speed = (float(data.xpos[bid][0]) - x0) / (float(data.time) - t0)
        assert 0.94 < speed / MAX_LINEAR_VEL < 1.03, (
            f"commanded {MAX_LINEAR_VEL} m/s, achieved {speed:.4f} m/s")
        assert abs(_yaw(data, bid)) < 0.02, "veered while driving straight"
    finally:
        engine.shutdown()


def test_the_wheels_grip_rather_than_slip():
    """With 60% of the declared mass on low-friction casters, traction is worth checking."""
    engine = _engine()
    try:
        model, data = engine.ctx.model, engine.ctx.data
        jid = named(model, mujoco.mjtObj.mjOBJ_JOINT, "q_mp_400_fixed_wheel_left_joint")
        handle = engine.ctx.blackboard.get("robot:q")
        for _ in range(500):
            engine.step()
        handle.drive(MAX_LINEAR_VEL, 0.0, 0.0)
        for _ in range(1750):
            engine.step()
        rolling = float(data.qvel[model.jnt_dofadr[jid]]) * WHEEL_RADIUS
        assert abs(1 - float(data.qvel[0]) / rolling) < 0.05, (
            f"base {float(data.qvel[0]):.4f} m/s against a rolling speed of {rolling:.4f} -- the "
            f"wheels are slipping")
    finally:
        engine.shutdown()


@pytest.mark.parametrize("commanded", [0.3, 0.6, 1.0])
def test_b2_rotates_at_the_commanded_rate(commanded):
    """No slip_factor, so this measures the drive rather than a calibration."""
    engine = _engine()
    try:
        model, data = engine.ctx.model, engine.ctx.data
        bid = named(model, mujoco.mjtObj.mjOBJ_BODY, "q_base_link")
        handle = engine.ctx.blackboard.get("robot:q")
        for _ in range(500):
            engine.step()
        handle.drive(0.0, 0.0, commanded)
        for _ in range(600):
            engine.step()
        t0, previous, total = float(data.time), _yaw(data, bid), 0.0
        for _ in range(1500):
            engine.step()
            current = _yaw(data, bid)
            total += np.unwrap([previous, current])[1] - previous
            previous = current
        ratio = (total / (float(data.time) - t0)) / commanded
        assert 0.94 < ratio < 1.04, f"achieved/commanded yaw {ratio:.3f} at {commanded} rad/s"
    finally:
        engine.shutdown()
