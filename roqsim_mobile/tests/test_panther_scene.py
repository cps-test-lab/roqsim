"""The Husarion Panther: Husarion's numbers, and a calibration the vendor's own number does not give.

The finding this file pins is ``test_vendor_multiplier_is_not_the_sim_factor``. Husarion publishes
``wheel_separation_multiplier: 1.5`` in its controller config -- the ICR compensation a skid-steer
needs on the real robot, and the same *quantity* as our ``slip_factor``. The assessment expected that
to make this port cheaper than the husky's blind calibration. It did not: at 1.5 this base achieves
only 0.40 of commanded yaw in MuJoCo, because point-contact scrub is far worse than a real tyre's.
The simulator needs 3.4. A vendor's real-robot correction is a starting point, not an answer, and
that is worth a test rather than a sentence.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import pytest

from roqsim.config import load_config_from_dict
from roqsim.engine import Engine
from roqsim.models import resolve_model

#: From Husarion's expanded husarion_ugv_description @ 559e784b, not measured from our model.
TOTAL_MASS = 55.0
WHEEL_RADIUS = 0.1825          # config/WH01.yaml
WHEEL_SEPARATION = 0.697       # WH01_controller.yaml, and the URDF geometry agrees (2 x 0.3485)


def _engine(**diff_drive):
    world = {
        "sim": {"timestep": 0.002},
        "components": [{
            "spawn_robot": {"model": "panther", "prefix": "p_"},
            "name": "p",
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
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "p_base_link")
    handle = engine.ctx.blackboard.get("robot:p")
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


def test_manifest_is_expanded():
    engine = _engine()
    try:
        assert engine.ctx.blackboard.get("robot:p") is not None, "diff_drive did not attach"
        assert any(type(p).__name__ == "LidarPlugin" for p in engine.plugins), "lidar did not attach"
    finally:
        engine.shutdown()


def test_rests_on_its_wheels():
    engine = _engine()
    try:
        model, data = engine.ctx.model, engine.ctx.data
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "p_base_link")
        for _ in range(1000):
            engine.step()
        # body_link rides at the wheel radius, which is where base_footprint puts it.
        assert float(data.xpos[bid][2]) == pytest.approx(WHEEL_RADIUS, abs=0.005)
        assert np.abs(data.qvel).max() < 1e-3, "did not settle"
    finally:
        engine.shutdown()


def test_uses_the_vendor_collision_hull():
    """Husarion ships a real simplified hull; it must be what we collide against.

    base_collision.stl is 9.7 kB against the 1.4 MB visual mesh -- unlike Doosan, whose *_collision
    files are byte-for-byte copies of its visual CAD and had to be replaced.
    """
    meshes = resolve_model("roqsim_mobile:panther").path.parent / "meshes"
    assert (meshes / "base_collision.stl").is_file()
    assert (meshes / "base_collision.stl").stat().st_size < 100_000


def test_drives_straight():
    engine = _engine()
    try:
        model, data = engine.ctx.model, engine.ctx.data
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "p_base_link")
        handle = engine.ctx.blackboard.get("robot:p")
        for _ in range(500):
            engine.step()
        start, t0 = np.array(data.xpos[bid]).copy(), float(data.time)
        handle.drive(0.8, 0.0, 0.0)
        for _ in range(2000):
            engine.step()
        speed = float(np.linalg.norm(np.array(data.xpos[bid])[:2] - start[:2])) / (
            float(data.time) - t0
        )
        assert 0.68 < speed < 0.88, f"commanded 0.8 m/s, achieved {speed:.3f} m/s"
        assert abs(_yaw(data, bid)) < 0.08, "veered while driving straight"
    finally:
        engine.shutdown()


@pytest.mark.parametrize("commanded", [0.3, 0.8])
def test_in_place_rotation_ratio(commanded):
    """The slip_factor calibration: achieved yaw must track the command within 15%."""
    engine = _engine()
    try:
        ratio = _yaw_ratio(engine, commanded)
        assert 0.85 < ratio < 1.15, (
            f"achieved/commanded yaw {ratio:.3f} at {commanded} rad/s; re-measure slip_factor "
            f"against the current friction/mass/timestep"
        )
    finally:
        engine.shutdown()


def test_vendor_multiplier_is_not_the_sim_factor():
    """Husarion's published 1.5 must NOT be mistaken for the simulator's slip_factor.

    Guards the finding, not just the value: if MuJoCo's skid-steer scrub ever improves enough that
    the vendor's real-robot number works here, this fails and the calibration should be revisited
    rather than left at 3.4 out of habit.
    """
    engine = _engine(slip_factor=1.5)
    try:
        ratio = _yaw_ratio(engine, 0.8)
        assert ratio < 0.7, (
            f"at the vendor's wheel_separation_multiplier of 1.5 this base now achieves {ratio:.3f} "
            f"of commanded yaw. It measured 0.40 when ported; if that has changed, re-derive "
            f"slip_factor instead of keeping 3.4."
        )
    finally:
        engine.shutdown()


def test_wheels_are_upright_and_coloured():
    """Wheel axles on y, and the vendor's materials present.

    The same regression the ROSbot needed: a mesh can be rotated 90 degrees or stripped of every
    colour without moving a number any drive test measures, because the robot drives on its
    collision cylinders.
    """
    engine = _engine()
    try:
        model, data = engine.ctx.model, engine.ctx.data
        for _ in range(1000):
            engine.step()
        visuals = 0
        for g in range(model.ngeom):
            body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[g]) or ""
            if "wheel" not in body or model.geom_dataid[g] < 0:
                continue
            visuals += 1
            mid = model.geom_dataid[g]
            adr, num = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
            rot = np.zeros(9)
            mujoco.mju_quat2Mat(rot, model.geom_quat[g])
            local = model.mesh_vert[adr:adr + num].reshape(-1, 3) @ rot.reshape(3, 3).T + \
                model.geom_pos[g]
            world = local @ np.array(data.xmat[model.geom_bodyid[g]]).reshape(3, 3).T
            extent = world.max(axis=0) - world.min(axis=0)
            assert int(np.argmin(extent)) == 1, (
                f"{body}: wheel mesh is thinnest along {'xyz'[int(np.argmin(extent))]}, not y"
            )
        assert visuals >= 4, f"expected at least one visual sub-mesh per wheel, got {visuals}"

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
    finally:
        engine.shutdown()
