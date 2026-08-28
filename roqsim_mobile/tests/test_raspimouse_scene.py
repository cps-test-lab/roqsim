"""The Raspberry Pi Mouse: the substrate's smallest robot, and what its scale changes.

At 0.74 kg and 117 mm it is an order of magnitude below anything else here, and the two tests worth
having are both about that rather than about kinematics.

``test_rotation_and_straight_line_track_the_command`` pins a servo gain that had to be calibrated for
the scale. At the ``kv=0.05`` first written -- a plausible-looking number for a tiny robot -- the
velocity servo needs a large error before it makes any torque at all, and the base reached 0.62 of
commanded yaw. Nothing else noticed: it drove, it rested, its mass was right.

``test_rests_on_wheels_and_chassis`` pins the fact that this base has **no caster in the
description**. It tips ~2 degrees onto its chassis box and drives on that edge, as the real robot
does on a smooth skid. The contact pair giving that edge skid friction is why rotation works at all;
at the geom default of 1.0 it costs 38% of commanded yaw.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import pytest

from roqsim.config import load_config_from_dict
from roqsim.engine import Engine

#: From RT Corporation's expanded raspimouse_description @ ed2c8b7a, not measured from our model.
TOTAL_MASS = 0.7412
WHEEL_RADIUS = 0.024
WHEEL_SEPARATION = 0.085


def _engine():
    engine = Engine(load_config_from_dict(
        {"sim": {"timestep": 0.001}, "components": [
            {"spawn_robot": {"model": "raspimouse", "prefix": "r_"}, "name": "r"}]},
        base_dir=Path(".")))
    engine.setup()
    engine.reset()
    return engine


def _bid(model):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "r_base_link")


def _yaw(data, bid):
    q = data.xquat[bid]
    return float(np.arctan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2)))


def test_mass_matches_the_vendor_description():
    engine = _engine()
    try:
        assert engine.ctx.model.body_mass.sum() == pytest.approx(TOTAL_MASS, abs=1e-4)
    finally:
        engine.shutdown()


def test_manifest_is_expanded():
    engine = _engine()
    try:
        assert engine.ctx.blackboard.get("robot:r") is not None, "diff_drive did not attach"
        assert any(type(p).__name__ == "LidarPlugin" for p in engine.plugins), "lidar did not attach"
    finally:
        engine.shutdown()


def test_rests_on_wheels_and_chassis():
    """Two driven wheels and no caster: it tips onto the chassis edge, and must settle there."""
    engine = _engine()
    try:
        model, data = engine.ctx.model, engine.ctx.data
        bid = _bid(model)
        for _ in range(2000):
            engine.step()
        assert abs(float(data.xpos[bid][2])) < 0.01, "base did not settle near the floor"
        assert np.abs(data.qvel).max() < 5e-3, "did not settle"
        touching = {
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g)
            for c in range(data.ncon)
            for g in (data.contact[c].geom1, data.contact[c].geom2)
        }
        assert "r_chassis_geom" in touching, (
            "the chassis is not touching the floor -- with no caster in the description it is a "
            "bearing surface, and the skid-friction contact pair depends on it"
        )
    finally:
        engine.shutdown()


def test_has_no_slip_factor():
    """A true two-wheel differential drive does not scrub, so it must not carry one.

    Guards the distinction from husky_a200 / clearpath_jackal / rosbot / panther, all of which do.
    """
    from roqsim.models import resolve_model
    import yaml

    manifest = resolve_model("roqsim_mobile:raspimouse").path.parent / "raspimouse.manifest.yaml"
    drive = next(c["diff_drive"] for c in yaml.safe_load(manifest.read_text())["components"]
                 if "diff_drive" in c)
    assert "slip_factor" not in drive
    assert drive["wheel_radius"] == pytest.approx(WHEEL_RADIUS)
    assert drive["wheel_separation"] == pytest.approx(WHEEL_SEPARATION)


@pytest.mark.parametrize("commanded", [0.3, 1.0])
def test_rotation_and_straight_line_track_the_command(commanded):
    engine = _engine()
    try:
        model, data = engine.ctx.model, engine.ctx.data
        bid = _bid(model)
        handle = engine.ctx.blackboard.get("robot:r")
        for _ in range(1000):
            engine.step()
        handle.drive(0.0, 0.0, commanded)
        for _ in range(500):
            engine.step()
        t0, previous, total = float(data.time), _yaw(data, bid), 0.0
        for _ in range(3000):
            engine.step()
            current = _yaw(data, bid)
            total += np.unwrap([previous, current])[1] - previous
            previous = current
        ratio = (total / (float(data.time) - t0)) / commanded
        assert 0.85 < ratio < 1.1, (
            f"achieved/commanded yaw {ratio:.3f}. At this scale the wheel servo's kv is the usual "
            f"cause: too low and it needs a large error before making any torque."
        )
    finally:
        engine.shutdown()


def test_drives_straight():
    engine = _engine()
    try:
        model, data = engine.ctx.model, engine.ctx.data
        bid = _bid(model)
        handle = engine.ctx.blackboard.get("robot:r")
        for _ in range(1000):
            engine.step()
        start, t0 = np.array(data.xpos[bid]).copy(), float(data.time)
        handle.drive(0.25, 0.0, 0.0)
        for _ in range(3000):
            engine.step()
        speed = float(np.linalg.norm(np.array(data.xpos[bid])[:2] - start[:2])) / (
            float(data.time) - t0
        )
        assert 0.22 < speed < 0.28, f"commanded 0.25 m/s, achieved {speed:.3f} m/s"
    finally:
        engine.shutdown()


def test_wheels_are_upright():
    engine = _engine()
    try:
        model, data = engine.ctx.model, engine.ctx.data
        for _ in range(1000):
            engine.step()
        for g in range(model.ngeom):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
            if not name.endswith("_wheel_geom"):
                continue
            rot = np.zeros(9)
            mujoco.mju_quat2Mat(rot, model.geom_quat[g])
            axis = np.array(data.xmat[model.geom_bodyid[g]]).reshape(3, 3) @ (
                rot.reshape(3, 3) @ np.array([0.0, 0.0, 1.0])
            )
            assert abs(axis[1]) > 0.98, f"{name}: wheel cylinder axis is {axis}, not along y"
    finally:
        engine.shutdown()


def test_lidar_sits_where_the_vendor_puts_it():
    """The scan height must be RT's own, not a guess.

    The port first invented a mount height of 0.089 m and a bare site marker, which rendered as a red
    ball floating above the robot -- spotted by a human in the viewer, not by any test. The
    description supports the scanner as `lidar:=lds`, placing the multi-lidar mount at base_link +
    0.0855 and the LDS-01 0.0345 above that. Both meshes are now shipped, so the sensor the manifest
    declares is geometry rather than a marker hanging in the air.
    """
    engine = _engine()
    try:
        model, data = engine.ctx.model, engine.ctx.data
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "r_lidar")
        assert sid >= 0
        mujoco.mj_forward(model, data)
        bid = _bid(model)
        height = float(data.site_xpos[sid][2] - data.xpos[bid][2])
        assert height == pytest.approx(0.0855 + 0.0345, abs=1e-4), (
            f"scan plane {height:.4f} m above base_link; the vendor's own offsets give 0.1200"
        )
        for mesh in ("r_RasPiMouse_MultiLiDARMount", "r_robotis_lds01"):
            assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MESH, mesh) >= 0, (
                f"{mesh} missing -- the site would be a marker floating over nothing"
            )
    finally:
        engine.shutdown()
