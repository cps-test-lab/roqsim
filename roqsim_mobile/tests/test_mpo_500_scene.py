"""The Neobotix MPO-500: omni wheels, and a vendor profile that is not isotropic.

The finding this file pins is ``test_the_limits_are_not_isotropic``. It is tempting to give a
holonomic base one top speed and use it for every axis; this vendor does not, and neither does its
sibling. ``configs/mpo_500/navigation.yaml`` allows 0.6 m/s forward and **0.5 sideways**, and the
MPO-700's allows 0.8 and 0.5. Assuming isotropy would have overstated the MPO-700 by 60% in the
lateral direction, and it did in this port's first draft, where "Neobotix publishes 1.0 m/s" was a
recollection rather than a file.

``test_it_is_not_a_swerve_base`` guards the drive-type verdict the other direction: the MPO-700 in the
same repository steers, this one does not, and their models must not converge by copy-paste. The
ledger's unknown here was "mecanum or Swedish-roller … the macro was listed rather than opened";
opened, the only wheel macro is ``mpo_500_omni_wheel`` and there is no caster macro at all.
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
TOTAL_MASS = 72.8002
WHEEL_RADIUS = 0.117
WHEEL_SEPARATION = 0.56
AXIS_SEPARATION = 0.50
#: configs/mpo_500/navigation.yaml. Deliberately different per axis -- see the module docstring.
MAX_VEL_X = 0.6
MAX_VEL_Y = 0.5
CORNERS = ("front_left", "front_right", "back_left", "back_right")


def _engine():
    world = {
        "sim": {"timestep": 0.002},
        "components": [{"spawn_robot": {"model": "mpo_500", "prefix": "p_"}, "name": "p"}],
    }
    engine = Engine(load_config_from_dict(world, base_dir=Path(".")))
    engine.setup()
    engine.reset()
    return engine


def _drive(engine, vx, vy, wz, steps=2000):
    model, data = engine.ctx.model, engine.ctx.data
    bid = named(model, mujoco.mjtObj.mjOBJ_BODY, "p_base_link")
    dof = model.jnt_dofadr[named(model, mujoco.mjtObj.mjOBJ_JOINT, "p_base_free")]
    engine.ctx.blackboard.get("robot:p").drive(vx, vy, wz)
    for _ in range(steps):
        engine.step()
    rot = data.xmat[bid].reshape(3, 3)
    local = rot.T @ np.array(data.qvel[dof:dof + 3])
    return float(local[0]), float(local[1]), float(data.qvel[dof + 5])


def test_mass_matches_the_vendor_description():
    engine = _engine()
    try:
        assert engine.ctx.model.body_mass.sum() == pytest.approx(TOTAL_MASS, abs=1e-3)
    finally:
        engine.shutdown()


def test_it_is_not_a_swerve_base():
    """The MPO-700 in the same repository steers; this one does not. See the module docstring."""
    engine = _engine()
    try:
        drive = next(p for p in engine.plugins if "OmniDrive" in type(p).__name__)
        assert not drive.config.get("steer_joints"), (
            "the MPO-500 has no steering: its only wheel macro is mpo_500_omni_wheel and there is "
            "no caster macro. Swerve IK here would aim wheels that cannot turn."
        )
        assert "slip_factor" not in drive.config, "an omni base does not turn by scrubbing"
        assert drive.config["wheel_radius"] == pytest.approx(WHEEL_RADIUS)
        assert drive.config["wheel_separation"] == pytest.approx(WHEEL_SEPARATION)
        assert drive.config["axis_separation"] == pytest.approx(AXIS_SEPARATION)
        model = engine.ctx.model
        for corner in CORNERS:
            named(model, mujoco.mjtObj.mjOBJ_JOINT, f"p_mpo_500_omni_wheel_{corner}_joint")
        assert model.njnt == 5, f"expected one free joint plus four wheels, got {model.njnt}"
    finally:
        engine.shutdown()


def test_the_limits_are_not_isotropic():
    """The finding -- see the module docstring."""
    engine = _engine()
    try:
        drive = next(p for p in engine.plugins if "OmniDrive" in type(p).__name__)
        assert drive.config["max_linear_vel"] == pytest.approx(MAX_VEL_X)
        assert drive.config["max_lateral_vel"] == pytest.approx(MAX_VEL_Y)
        assert drive.config["max_lateral_vel"] < drive.config["max_linear_vel"], (
            "the vendor's own Nav2 profile allows less sideways than forward; collapsing the two "
            "to one figure overstates the platform"
        )
    finally:
        engine.shutdown()


def test_manifest_brings_two_scanners():
    engine = _engine()
    try:
        scans = [p for p in engine.plugins if type(p).__name__ == "LidarPlugin"]
        assert len(scans) == 2, f"expected the vendor's two MicroScan3s, got {len(scans)}"
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
        for corner in CORNERS:
            geom = f"p_mpo_500_omni_wheel_{corner}_link_tyre"
            named(model, mujoco.mjtObj.mjOBJ_GEOM, geom)
            assert geom in touching, touching
        named(model, mujoco.mjtObj.mjOBJ_GEOM, "p_base_link_collision")
        assert "p_base_link_collision" not in touching, "the body is on the floor"
    finally:
        engine.shutdown()


def test_the_wheel_contact_is_a_sphere():
    """The vendor's own choice, and the right one: an omni wheel has no preferred rolling direction.

    Also asserts `priority`, without which the low friction is inert -- MuJoCo takes the maximum of
    the two contacting geoms' friction and the floor's 1.0 would win.
    """
    engine = _engine()
    try:
        model = engine.ctx.model
        for corner in CORNERS:
            gid = named(model, mujoco.mjtObj.mjOBJ_GEOM,
                        f"p_mpo_500_omni_wheel_{corner}_link_tyre")
            assert model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_SPHERE
            assert float(model.geom_size[gid][0]) == pytest.approx(WHEEL_RADIUS)
            assert model.geom_priority[gid] > 0
    finally:
        engine.shutdown()


@pytest.mark.parametrize("command", [(0.6, 0.0, 0.0), (0.0, 0.5, 0.0), (0.0, 0.0, 1.0),
                                     (0.4, 0.3, 0.0)])
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
                assert 0.94 < got / want < 1.04, f"{axis}: commanded {want}, got {got:+.4f}"
    finally:
        engine.shutdown()


def test_the_wheels_turn_and_turn_differently_when_strafing():
    """Observational wheel spin, which is the whole reason omni_drive drives these servos.

    Compared as the **physical** spin about the base's y axis, not as raw joint velocity. This
    vendor mirrors its right-hand wheel joints (their axes are -y in the base frame), so the two
    sides carry opposite ``qvel`` signs for the *same* rotation -- which is exactly what
    ``omni_drive``'s derived roll sign exists to absorb, and its docstring warns about. A first
    draft of this test compared raw signs and failed against a correct model.

    A forward command must spin all four the same way; a strafe must split them, because that is
    what an omni/mecanum roller layout does. If a strafe did not split them the wheel IK would have
    degraded to one shared sign -- which looks like a working drive and a broken strafe.
    """
    engine = _engine()
    try:
        model, data = engine.ctx.model, engine.ctx.data
        base = named(model, mujoco.mjtObj.mjOBJ_BODY, "p_base_link")
        jids = [named(model, mujoco.mjtObj.mjOBJ_JOINT, f"p_mpo_500_omni_wheel_{c}_joint")
                for c in CORNERS]

        def spins():
            """Physical roll rate about the base's +y, per corner."""
            rb = data.xmat[base].reshape(3, 3)
            out = []
            for jid in jids:
                axis = rb.T @ (data.xmat[model.jnt_bodyid[jid]].reshape(3, 3)
                               @ model.jnt_axis[jid])
                out.append(float(data.qvel[model.jnt_dofadr[jid]]) * float(axis[1]))
            return out

        for _ in range(500):
            engine.step()
        _drive(engine, 0.4, 0.0, 0.0, steps=1200)
        forward = spins()
        assert all(np.sign(v) == np.sign(forward[0]) for v in forward), (
            f"a forward command must spin all four wheels the same way; physical rates {forward}")
        engine.ctx.blackboard.get("robot:p").drive(0.0, 0.4, 0.0)
        for _ in range(1200):
            engine.step()
        strafe = spins()
        signs = {int(np.sign(v)) for v in strafe if abs(v) > 0.05}
        assert signs == {-1, 1}, f"a strafe must split the wheel directions; physical rates {strafe}"
    finally:
        engine.shutdown()
