"""The LGDXRobot2: a description that is visualisation-only, and a sim description that is not.

The finding this file pins is ``test_the_physics_came_from_the_sim_description``. This vendor ships
two URDFs. ``lgdxrobot2.urdf`` has nine links, twelve meshes and **no inertial or collision elements
whatsoever** -- it sums to 0.0 kg. Everything physical is in ``lgdxrobot2_sim.urdf``. Porting from
the obvious file would have produced a robot that loads, renders correctly, and is massless; the mass
audit is the only check that would have noticed, and only because it compares against a number read
from the *other* file.

The platform ledger recorded "skid-steer or mecanum" as this port's biggest unknown and priced most
of the cost against it, because the answer decides whether a ``slip_factor`` calibration is needed.
It did not need guessing: the sim description declares ``gz::sim::systems::MecanumDrive``, so this is
``omni_drive`` and carries no calibration at all.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import pytest
from mobile_scene_utils import named

from roqsim.config import load_config_from_dict
from roqsim.engine import Engine

#: From the expanded lgdxrobot2_SIM.urdf @ b8210400. The plain lgdxrobot2.urdf sums to 0.0 kg.
TOTAL_MASS = 2.6840
BASE_MASS = 2.2768
WHEEL_MASS = 0.1018
#: From the gz::sim::systems::MecanumDrive plugin's own configuration.
WHEEL_RADIUS = 0.0375
WHEEL_SEPARATION = 0.208
AXIS_SEPARATION = 0.164
#: From lgdxrobot2_bringup/param/loc/nav2.yaml -- the vendor's REAL-robot profile.
MAX_LINEAR_VEL = 0.33
MAX_ANGULAR_VEL = 0.3
WHEELS = ("wheel1_link", "wheel2_link", "wheel3_link", "wheel4_link")


def _engine(**omni):
    world = {
        "sim": {"timestep": 0.002},
        "components": [{
            "spawn_robot": {"model": "lgdxrobot2", "prefix": "g_"},
            "name": "g",
            **({"components": [{"omni_drive": omni}]} if omni else {}),
        }],
    }
    engine = Engine(load_config_from_dict(world, base_dir=Path(".")))
    engine.setup()
    engine.reset()
    return engine


def _yaw(data, bid):
    q = data.xquat[bid]
    return float(np.arctan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2)))


def _achieved(engine, vx, vy, wz):
    """Steady-state **body-frame** (vx, vy, wz) actually achieved.

    Measured in the body frame, not from world displacement. A commanded twist is body-frame, so
    when wz is non-zero the robot curves and averaging world displacement over a window returns a
    rotated average rather than the velocity that was asked for -- it read 1.27x on a diagonal-plus-
    yaw command from a model that was tracking correctly. MuJoCo's free joint gives linear velocity
    in the global frame and angular velocity in the body frame, so only the linear part is rotated.
    """
    model, data = engine.ctx.model, engine.ctx.data
    bid = named(model, mujoco.mjtObj.mjOBJ_BODY, "g_base_link")
    dof = model.jnt_dofadr[named(model, mujoco.mjtObj.mjOBJ_JOINT, "g_base_free")]
    handle = engine.ctx.blackboard.get("robot:g")
    for _ in range(500):
        engine.step()
    handle.drive(vx, vy, wz)
    for _ in range(500):
        engine.step()
    samples = []
    for _ in range(1500):
        engine.step()
        rot = data.xmat[bid].reshape(3, 3)
        local = rot.T @ np.array(data.qvel[dof:dof + 3])
        samples.append((local[0], local[1], float(data.qvel[dof + 5])))
    mean = np.mean(samples, axis=0)
    return float(mean[0]), float(mean[1]), float(mean[2])


def test_the_physics_came_from_the_sim_description():
    """The mass audit is the check that catches a port from the wrong URDF -- see the docstring."""
    engine = _engine()
    try:
        model = engine.ctx.model
        assert model.body_mass.sum() == pytest.approx(TOTAL_MASS, abs=1e-3), (
            "total mass is wrong. 0.0 would mean this was built from lgdxrobot2.urdf, which carries "
            "no inertial elements at all; the physics is only in lgdxrobot2_sim.urdf"
        )
        base = named(model, mujoco.mjtObj.mjOBJ_BODY, "g_base_link")
        assert model.body_mass[base] == pytest.approx(BASE_MASS, abs=1e-3)
        for wheel in WHEELS:
            bid = named(model, mujoco.mjtObj.mjOBJ_BODY, f"g_{wheel}")
            assert model.body_mass[bid] == pytest.approx(WHEEL_MASS, abs=1e-4)
    finally:
        engine.shutdown()


def test_manifest_brings_the_drive_and_the_lidar():
    engine = _engine()
    try:
        assert engine.ctx.blackboard.get("robot:g") is not None, "omni_drive did not attach"
        assert any(type(p).__name__ == "LidarPlugin" for p in engine.plugins), "lidar did not attach"
        drive = next(p for p in engine.plugins if "OmniDrive" in type(p).__name__)
        assert drive.config["wheel_radius"] == pytest.approx(WHEEL_RADIUS)
        assert drive.config["wheel_separation"] == pytest.approx(WHEEL_SEPARATION)
        assert drive.config["axis_separation"] == pytest.approx(AXIS_SEPARATION)
        assert "slip_factor" not in drive.config, (
            "a mecanum base does not turn by scrubbing, so it must not carry a slip_factor -- the "
            "same line the ridgeback draws, and the reason this port needed no calibration"
        )
    finally:
        engine.shutdown()


def test_it_rests_on_four_wheels():
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
        for geom in [f"g_{w}_tyre" for w in WHEELS] + ["g_base_collision"]:
            named(model, mujoco.mjtObj.mjOBJ_GEOM, geom)
        for wheel in WHEELS:
            assert f"g_{wheel}_tyre" in touching, touching
        assert "g_base_collision" not in touching, "the chassis is dragging on the floor"
    finally:
        engine.shutdown()


def test_the_wheel_collision_is_the_vendors_sphere():
    """A mecanum wheel has no preferred rolling direction, and the vendor models that as a sphere.

    Asserted because a cylinder would look like the obvious 'fix' to someone reading the model, and
    would quietly give the base a rolling axis it does not have.
    """
    engine = _engine()
    try:
        model = engine.ctx.model
        for wheel in WHEELS:
            gid = named(model, mujoco.mjtObj.mjOBJ_GEOM, f"g_{wheel}_tyre")
            assert model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_SPHERE
            assert float(model.geom_size[gid][0]) == pytest.approx(WHEEL_RADIUS)
            assert model.geom_priority[gid] > 0, (
                "without priority MuJoCo takes the MAXIMUM of this geom's friction and the floor's, "
                "so the low friction an omni wheel needs is inert"
            )
    finally:
        engine.shutdown()


def test_it_drives_forward():
    engine = _engine()
    try:
        vx, vy, wz = _achieved(engine, MAX_LINEAR_VEL, 0.0, 0.0)
        assert vx / MAX_LINEAR_VEL > 0.9, f"commanded {MAX_LINEAR_VEL} m/s, achieved {vx:.4f}"
        assert abs(vy) < 0.01 and abs(wz) < 0.02, f"drifted: vy={vy:.4f} wz={wz:.4f}"
    finally:
        engine.shutdown()


def test_it_strafes():
    """The behaviour only this and the ridgeback can produce here."""
    engine = _engine()
    try:
        vx, vy, wz = _achieved(engine, 0.0, MAX_LINEAR_VEL, 0.0)
        assert vy / MAX_LINEAR_VEL > 0.9, f"commanded {MAX_LINEAR_VEL} m/s sideways, got {vy:.4f}"
        assert abs(vx) < 0.01 and abs(wz) < 0.02, f"drifted: vx={vx:.4f} wz={wz:.4f}"
    finally:
        engine.shutdown()


@pytest.mark.parametrize("command", [(0.0, 0.0, 0.3), (0.2, 0.2, 0.0), (0.15, -0.15, 0.15)])
def test_it_tracks_a_holonomic_twist(command):
    engine = _engine()
    try:
        achieved = _achieved(engine, *command)
        for got, want, axis in zip(achieved, command, "xyw", strict=True):
            if abs(want) < 1e-9:
                assert abs(got) < 0.02, f"{axis}: commanded 0, got {got:+.4f}"
            else:
                assert 0.88 < got / want < 1.08, f"{axis}: commanded {want}, got {got:+.4f}"
    finally:
        engine.shutdown()


def test_wheels_spin_about_the_robot_y_axis():
    """The joint rpy, which this description puts on the joint (-pi/2) rather than the visual."""
    engine = _engine()
    try:
        model, data = engine.ctx.model, engine.ctx.data
        for wheel in WHEELS:
            jid = named(model, mujoco.mjtObj.mjOBJ_JOINT, f"g_{wheel}_joint")
            axis = data.xaxis[jid]
            assert abs(abs(axis[1]) - 1.0) < 1e-6, (
                f"{wheel} spins about {np.round(axis, 4)}, not the robot's y")
    finally:
        engine.shutdown()


def test_the_parts_kept_their_colours():
    """dae2obj split each Collada by bound material so the MJCF can name them.

    Without the split MuJoCo reads no OBJ material and every part renders flat grey. The PCB green
    and the black mecanum rollers are the visible evidence that the split happened.
    """
    engine = _engine()
    try:
        model = engine.ctx.model
        # spawn_robot prefixes every asset name, so the material is g_lgdx_* here.
        wanted = {f"g_{n}" for n in
                  ("lgdx_aluminium", "lgdx_black", "lgdx_pcb", "lgdx_red", "lgdx_steel")}
        have = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MATERIAL, i)
                for i in range(model.nmat)}
        assert wanted <= {n for n in have if n}, f"missing materials: {wanted - have}"
        used = {model.geom_matid[g] for g in range(model.ngeom) if model.geom_group[g] == 2}
        assert len(used) >= 5, "the visual geoms collapsed onto fewer materials than were declared"
    finally:
        engine.shutdown()
