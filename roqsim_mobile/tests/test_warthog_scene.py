"""The Clearpath Warthog: how far a vendor's own ICR compensation is from the simulator's.

The finding this file pins is ``test_vendor_compensation_is_not_the_sim_factor``. Clearpath states
its compensation more legibly than any other vendor in this package: ``diff_4wd.yaml`` declares
``wheel_separation: 1.5`` for a robot whose URDF track is 1.13642 m and then applies
``wheel_separation_multiplier: 1.125``, an effective 1.6875 m -- the geometric track inflated by 48%.
That is the same *quantity* as our ``slip_factor``, expressed as a fictitious axle width.

It buys 0.31 of commanded yaw here. The Panther taught this lesson once (vendor 1.5, simulator 3.4);
the Warthog is the replication, and it adds the size trend: husky 3.0, panther 3.4, and this 260 kg
base on a 1.136 m track needs 5.25. MuJoCo's point-contact scrub does not merely fail to match a
real tyre, it fails *worse the larger the machine*, which is why a slip factor cannot be inherited
from a sibling platform however similar the drive.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import pytest

from mobile_scene_utils import named

from roqsim.config import load_config_from_dict
from roqsim.engine import Engine

#: From Clearpath's expanded w200 xacro @ b0f6d920, not measured from our model. The description's
#: own sum is 260.001 kg; the missing gram is imu_0_link, which is a site here rather than a body.
TOTAL_MASS = 260.0
WHEEL_RADIUS = 0.3             # diff_4wd.yaml, and the URDF collision cylinder agrees
WHEEL_SEPARATION = 1.13642     # the URDF geometry (2 x 0.56821)
SLIP_FACTOR = 5.25             # calibrated against this model -- see the manifest
#: diff_4wd.yaml's wheel_separation 1.5 x wheel_separation_multiplier 1.125, over the real track.
VENDOR_COMPENSATION = 1.5 * 1.125 / WHEEL_SEPARATION
#: Fender top and scanner height, both in the base_link frame. The scanner must clear the fenders.
FENDER_TOP = 0.533
LIDAR_HEIGHT = 0.625


def _engine(**diff_drive):
    world = {
        "sim": {"timestep": 0.002},
        "components": [{
            "spawn_robot": {"model": "warthog", "prefix": "w_"},
            "name": "w",
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
    """Achieved / commanded steady-state yaw rate.

    The engine is settled, then commanded, then measured over a fixed window -- and is never reset
    inside that window. A harness that reset mid-loop reported both a wrong magnitude and a wrong
    yaw sign for a correct ridgeback, which cost that port an iteration.
    """
    model, data = engine.ctx.model, engine.ctx.data
    bid = named(model, mujoco.mjtObj.mjOBJ_BODY, "w_base_link")
    handle = engine.ctx.blackboard.get("robot:w")
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
    """spawn_robot must expand the manifest: a Warthog with no drive is a 260 kg paperweight."""
    engine = _engine()
    try:
        assert engine.ctx.blackboard.get("robot:w") is not None, "diff_drive did not attach"
        assert any(type(p).__name__ == "LidarPlugin" for p in engine.plugins), "lidar did not attach"
    finally:
        engine.shutdown()


def test_it_rests_on_four_tyres():
    engine = _engine()
    try:
        model, data = engine.ctx.model, engine.ctx.data
        for _ in range(1500):
            engine.step()
        touching = {
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g)
            for i in range(data.ncon)
            for g in (data.contact[i].geom1, data.contact[i].geom2)
        }
        for end in ("front", "rear"):
            for side in ("left", "right"):
                assert f"w_{end}_{side}_wheel_tyre" in touching, touching
        assert "w_chassis_collision" not in touching, "the chassis is dragging on the floor"
    finally:
        engine.shutdown()


def test_b1_drives_straight():
    engine = _engine()
    try:
        model, data = engine.ctx.model, engine.ctx.data
        bid = named(model, mujoco.mjtObj.mjOBJ_BODY, "w_base_link")
        handle = engine.ctx.blackboard.get("robot:w")
        for _ in range(500):
            engine.step()
        handle.drive(1.5, 0.0, 0.0)
        for _ in range(500):
            engine.step()
        x0, t0 = float(data.xpos[bid][0]), float(data.time)
        # 1000 steps = 2 s = 3 m, which with the ~0.75 m spent accelerating leaves the 0.68 m nose
        # short of empty_room's wall at x = 5. This robot is 1.35 m long and does 5 m/s, so the
        # default room is barely three of its own lengths ahead of it -- measure any longer and the
        # test is measuring a collision.
        for _ in range(1000):
            engine.step()
        speed = (float(data.xpos[bid][0]) - x0) / (float(data.time) - t0)
        assert 1.3 < speed < 1.7, f"commanded 1.5 m/s, achieved {speed:.3f} m/s"
        assert abs(_yaw(data, bid)) < 0.08, "veered while driving straight"
    finally:
        engine.shutdown()


@pytest.mark.parametrize("commanded", [0.3, 0.5, 0.8, 1.0])
def test_b2_rotates_at_the_commanded_rate(commanded):
    """The slip_factor calibration: achieved yaw must track the command within 15%."""
    engine = _engine()
    try:
        ratio = _yaw_ratio(engine, commanded)
        assert 0.85 < ratio < 1.15, (
            f"achieved/commanded yaw {ratio:.3f} at {commanded} rad/s; re-measure slip_factor "
            f"rather than widening this band."
        )
    finally:
        engine.shutdown()


def test_vendor_compensation_is_not_the_sim_factor():
    """Clearpath's effective 1.6875 m axle must NOT be mistaken for the simulator's slip_factor.

    It is the same quantity -- an inflated track that buys back the yaw scrub steals -- and it is
    the closest thing to prior art a vendor publishes. It still lands nowhere near, and pinning that
    stops the next Clearpath port (a300, dd100, do100) from adopting the config value and calling it
    calibrated.
    """
    assert VENDOR_COMPENSATION == pytest.approx(1.485, abs=0.001)
    engine = _engine(slip_factor=VENDOR_COMPENSATION)
    try:
        ratio = _yaw_ratio(engine, 0.8)
        assert ratio < 0.5, (
            f"Clearpath's own compensation now achieves {ratio:.3f} of commanded yaw. It measured "
            f"0.31 when ported; if that has changed, re-derive slip_factor instead of keeping "
            f"{SLIP_FACTOR}."
        )
    finally:
        engine.shutdown()


def test_slip_factor_is_the_calibrated_one():
    """The geometry the calibration was measured against, asserted so it cannot drift silently."""
    engine = _engine()
    try:
        drive = next(p for p in engine.plugins if "DiffDrive" in type(p).__name__)
        assert drive.config["slip_factor"] == pytest.approx(SLIP_FACTOR)
        assert drive.config["wheel_radius"] == pytest.approx(WHEEL_RADIUS)
        assert drive.config["wheel_separation"] == pytest.approx(WHEEL_SEPARATION)
    finally:
        engine.shutdown()


def test_wheels_are_upright_and_the_scanner_clears_the_fenders():
    """Geometry a dynamics battery cannot see.

    Three ports in this batch shipped a defect a green battery missed and a person caught in the
    viewer, so the checks a test *can* make are made here. The scanner one is specific to this
    platform: the fenders stand 0.533 m above base_link, which is taller than every other robot in
    roqsim_mobile, so a deck-height scan would return fender at every bearing.
    """
    engine = _engine()
    try:
        model, data = engine.ctx.model, engine.ctx.data
        for end in ("front", "rear"):
            for side in ("left", "right"):
                gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM,
                                        f"w_{end}_{side}_wheel_tyre")
                axis = data.geom_xmat[gid].reshape(3, 3)[:, 2]
                assert abs(abs(axis[1]) - 1.0) < 1e-6, (
                    f"{end}_{side} tyre axis is {np.round(axis, 4)}, not along y")
        base = named(model, mujoco.mjtObj.mjOBJ_BODY, "w_base_link")
        sid = named(model, mujoco.mjtObj.mjOBJ_SITE, "w_lidar")
        height = float(data.site_xpos[sid][2] - data.xpos[base][2])
        assert height == pytest.approx(LIDAR_HEIGHT, abs=1e-3)
        assert height > FENDER_TOP + 0.05, (
            f"scanner at {height:.3f} m does not clear the {FENDER_TOP} m fenders")
    finally:
        engine.shutdown()
