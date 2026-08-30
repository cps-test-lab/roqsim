"""The Husarion ROSbot: Husarion's numbers unchanged, and a skid-steer calibration that is ours.

Two tests carry the port's real risk.

``test_body_mesh_is_in_metres`` guards the trap this port actually hit. The URDF carries
``scale="0.001 0.001 0.001"`` on ``body.glb`` and ``1.0`` on the wheels, so a converter that treats
the tree uniformly ships a chassis 1000x too large around correctly sized wheels. Every other check
here passes in that state -- the model compiles, the masses are right, and only looking at it (or
this test) catches it.

``test_in_place_rotation_ratio`` pins the ``slip_factor`` calibration. MuJoCo reproduces a
skid-steer's lateral scrub badly: uncompensated, this base yaws at ~66% of what was commanded, so a
planner under-rotates and reads as a tuning problem rather than a model one. The factor is a
calibration against *this* model's friction, mass and timestep, and it must be re-measured if any of
those change -- which is exactly what this test is for.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import pytest

from roqsim.config import load_config_from_dict
from roqsim.engine import Engine
from roqsim.models import resolve_model

#: From Husarion's expanded rosbot_description @ 41fad021, not measured from our model.
TOTAL_MASS = 2.024
WHEEL_RADIUS = 0.0425
#: rosbot_controller/config/rosbot/controllers.yaml. Note this is NOT the URDF's geometric track
#: (2 x 0.096 = 0.192): the vendor's controller uses the smaller value its odometry is calibrated to.
WHEEL_SEPARATION = 0.186
#: ROSbot 2R published body envelope, for the mesh scale check.
BODY_EXTENT_M = 0.25


def _engine(**diff_drive):
    world = {
        "sim": {"timestep": 0.002},
        "components": [{
            "spawn_robot": {"model": "rosbot", "prefix": "rb_"},
            "name": "rb",
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


def test_body_mesh_is_in_metres():
    """The URDF scales body.glb by 0.001 and the wheels by 1.0; the OBJs must already be metres."""
    meshes = resolve_model("roqsim_mobile:rosbot").path.parent / "meshes"
    # Per-material files: reduce-mesh --split-materials writes <stem>__<material>.obj.
    for prefix, limit in (("body__", BODY_EXTENT_M), ("wheel_", 0.1)):
        found = sorted(meshes.glob(f"{prefix}*.obj"))
        assert found, f"no {prefix}*.obj meshes -- the split did not run"
        for obj in found:
            verts = np.array([
                [float(v) for v in line.split()[1:4]]
                for line in obj.read_text().splitlines()
                if line.startswith("v ")
            ])
            extent = (verts.max(axis=0) - verts.min(axis=0)).max()
            assert extent < limit, (
                f"{obj.name} spans {extent:.3f} m; it is almost certainly still in millimetres"
            )


def test_manifest_is_expanded():
    # A directory beside the world sharing this model's name used to shadow the packaged model, and
    # the manifest then vanished silently -- the robot spawned with no drive and no lidar while the
    # world loaded and ran. resolve_model now requires a file; this asserts the components arrive.
    engine = _engine()
    try:
        assert engine.ctx.blackboard.get("robot:rb") is not None, "diff_drive did not attach"
        assert any(type(p).__name__ == "LidarPlugin" for p in engine.plugins), "lidar did not attach"
    finally:
        engine.shutdown()


def test_mass_matches_the_vendor_description():
    engine = _engine()
    try:
        assert engine.ctx.model.body_mass.sum() == pytest.approx(TOTAL_MASS, abs=1e-3)
    finally:
        engine.shutdown()


def test_rests_on_its_wheels():
    engine = _engine()
    try:
        model, data = engine.ctx.model, engine.ctx.data
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "rb_base_link")
        for _ in range(750):
            engine.step()
        # base_link is at ground level on this platform: the wheels' radius lifts body_link, not it.
        assert abs(float(data.xpos[bid][2])) < 0.005
        assert np.abs(data.qvel).max() < 1e-3, "did not settle"
    finally:
        engine.shutdown()


def test_drives_straight():
    engine = _engine()
    try:
        model, data = engine.ctx.model, engine.ctx.data
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "rb_base_link")
        handle = engine.ctx.blackboard.get("robot:rb")
        for _ in range(500):
            engine.step()
        start, t0 = np.array(data.xpos[bid]).copy(), float(data.time)
        handle.drive(0.5, 0.0, 0.0)
        for _ in range(2000):
            engine.step()
        travelled = float(np.linalg.norm(np.array(data.xpos[bid])[:2] - start[:2]))
        speed = travelled / (float(data.time) - t0)
        assert 0.42 < speed < 0.52, f"commanded 0.5 m/s, achieved {speed:.3f} m/s"
        assert abs(_yaw(data, bid)) < 0.08, "veered while driving straight"
    finally:
        engine.shutdown()


@pytest.mark.parametrize("commanded", [0.3, 0.8])
def test_in_place_rotation_ratio(commanded):
    """The slip_factor calibration: achieved yaw must track the command within 10%."""
    engine = _engine()
    try:
        model, data = engine.ctx.model, engine.ctx.data
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "rb_base_link")
        handle = engine.ctx.blackboard.get("robot:rb")
        for _ in range(500):
            engine.step()
        handle.drive(0.0, 0.0, commanded)
        for _ in range(250):          # spin-up, excluded from the measurement
            engine.step()
        t0, previous, total = float(data.time), _yaw(data, bid), 0.0
        for _ in range(1500):
            engine.step()
            current = _yaw(data, bid)
            total += np.unwrap([previous, current])[1] - previous
            previous = current
        ratio = (total / (float(data.time) - t0)) / commanded
        assert 0.9 < ratio < 1.1, (
            f"achieved/commanded yaw {ratio:.3f} at {commanded} rad/s. Uncompensated this base sits "
            f"at ~0.66; the slip_factor needs re-measuring against the current friction/mass/timestep."
        )
    finally:
        engine.shutdown()


def test_uncompensated_rotation_is_why_slip_factor_exists():
    """Guards the calibration's premise, so it cannot quietly become unnecessary."""
    engine = _engine(slip_factor=1.0)
    try:
        model, data = engine.ctx.model, engine.ctx.data
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "rb_base_link")
        handle = engine.ctx.blackboard.get("robot:rb")
        for _ in range(500):
            engine.step()
        handle.drive(0.0, 0.0, 0.8)
        for _ in range(250):
            engine.step()
        t0, previous, total = float(data.time), _yaw(data, bid), 0.0
        for _ in range(1500):
            engine.step()
            current = _yaw(data, bid)
            total += np.unwrap([previous, current])[1] - previous
            previous = current
        ratio = (total / (float(data.time) - t0)) / 0.8
        assert ratio < 0.8, (
            f"uncompensated yaw ratio is {ratio:.3f}; if MuJoCo now reproduces skid-steer scrub "
            f"faithfully the slip_factor should be revisited rather than left in"
        )
    finally:
        engine.shutdown()


def test_wheels_are_upright_and_coloured():
    """Wheel axles on Y, and the vendor's materials present.

    Two bugs this caught, both of which every other test passed through. The GLB source is Y-up and
    `reduce-mesh` already exports Z-up, so re-applying the URDF's own rpy to the visual mesh rotated
    the wheels a second time and laid them flat -- while the collision cylinders, which genuinely do
    need that rotation because MuJoCo cylinders run along local z, stayed correct. Visual and
    collision disagreeing is invisible to a drive test: the robot drove perfectly on invisible
    upright cylinders while its wheels rendered as plates.

    And the meshes were converted joined, so MuJoCo -- which loads one mesh per OBJ and reads no OBJ
    material -- rendered a uniformly grey robot instead of a black one with red fenders.
    """
    engine = _engine()
    try:
        model, data = engine.ctx.model, engine.ctx.data
        for _ in range(750):
            engine.step()

        visuals = 0
        for g in range(model.ngeom):
            body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[g]) or ""
            if "wheel" not in body or model.geom_dataid[g] < 0:
                continue
            visuals += 1
            mid = model.geom_dataid[g]
            adr, num = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
            verts = model.mesh_vert[adr:adr + num].reshape(-1, 3)
            rot = np.zeros(9)
            mujoco.mju_quat2Mat(rot, model.geom_quat[g])
            local = verts @ rot.reshape(3, 3).T + model.geom_pos[g]
            world = local @ np.array(data.xmat[model.geom_bodyid[g]]).reshape(3, 3).T
            extent = world.max(axis=0) - world.min(axis=0)
            assert int(np.argmin(extent)) == 1, (
                f"{body}: wheel mesh is thinnest along {'xyz'[int(np.argmin(extent))]}, not y -- "
                f"the axle is not aligned with the joint and the wheel is lying flat"
            )
        assert visuals == 8, f"expected 8 wheel visual sub-meshes (4 wheels x 2 materials), got {visuals}"

        # Collision cylinders must point the same way, or the robot drives on geometry it does not
        # look like.
        for g in range(model.ngeom):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
            if not name.endswith("_wheel_geom"):
                continue
            rot = np.zeros(9)
            mujoco.mju_quat2Mat(rot, model.geom_quat[g])
            axis = np.array(data.xmat[model.geom_bodyid[g]]).reshape(3, 3) @ (
                rot.reshape(3, 3) @ np.array([0.0, 0.0, 1.0])
            )
            assert abs(axis[1]) > 0.99, f"{name}: cylinder axis is {axis}, not along y"

        # spawn_robot prefixes every name it splices in, materials included.
        for material in ("body_black", "fenders_red", "rim_metallic", "tire_rubber"):
            assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MATERIAL, f"rb_{material}") >= 0, (
                f"material {material!r} missing -- the meshes were joined instead of split, so the "
                f"robot renders flat grey"
            )
    finally:
        engine.shutdown()
