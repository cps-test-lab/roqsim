"""The Raspberry Pi Mouse: the substrate's smallest robot, and what its scale changes.

At 0.74 kg and 117 mm it is an order of magnitude below anything else here, and the two tests worth
having are both about that rather than about kinematics.

``test_rotation_and_straight_line_track_the_command`` pins a servo gain calibrated for the scale. At
``kv=0.05`` -- a plausible-looking number for a tiny robot -- the velocity servo needs a large error
before it makes any torque at all, and the base reaches 0.62 of commanded yaw. Nothing else notices:
it drives, it rests, its mass is right.

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
import scan_mount_utils as scan_mount

from roqsim.config import load_config_from_dict
from roqsim.engine import Engine

#: From RT Corporation's expanded raspimouse_description @ ed2c8b7a, not measured from our model:
#: base_link 0.7186 kg + two 0.0113 kg wheels. The multi-lidar mount's own 0.04 kg is not carried.
ROBOT_MASS = 0.7412
#: The lds01 device model's inertial, from turtlebot3_waffle.urdf:226 @ 0c0be84 -- not RT's 0.160 kg
#: for its `laser` link (urdf/sensors/lidar.urdf.xacro:88 @ ed2c8b7), which the device does not take.
LDS_MASS = 0.114
TOTAL_MASS = ROBOT_MASS + LDS_MASS
WHEEL_RADIUS = 0.024
WHEEL_SEPARATION = 0.085

#: base_link -> lds_multi_mount_link, raspimouse_description @ ed2c8b7 urdf/raspimouse.urdf.xacro:86-88.
LDS_MOUNT = ((0.0, 0.0, 0.0855), (0.0, 0.0, 0.0))
#: lds_multi_mount_link -> laser, the same file :89-91 -- the vendor's yaw of 3.14, not pi.
LASER_JOINT = ((0.0, 0.0, 0.0345), (0.0, 0.0, 3.14))


def _engine():
    engine = Engine(load_config_from_dict(
        {"sim": {"timestep": 0.001}, "components": [
            {"spawn_robot": {"model": "raspimouse", "prefix": "r_"}, "name": "r"}]},
        base_dir=Path(".")))
    engine.ctx.seed = 0  # this test is the driver, and the scanner's range noise draws from it
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


# -- the mounted scanner ----------------------------------------------------------------------
#
# The description supports the scanner as `lidar:=lds`: RT's multi-lidar mount on base_link, and an
# LDS-01 on that mount turned by 3.14 rad. The mount is robot geometry and ships in the MJCF; the
# scanner is the lds01 device model, which the manifest hangs from the declared mount frame.


@pytest.fixture(scope="module")
def mounted():
    """The robot spawned as a world spawns it, prefixed and namespaced, its LDS-01 cast once."""
    engine = scan_mount.spawn("raspimouse", ["lds01"])
    lidar = scan_mount.lidar(engine, "robot.lds01")
    yield engine, lidar
    engine.shutdown()


def test_the_mount_is_robot_geometry_and_the_scanner_is_the_device(mounted):
    engine, _ = mounted
    model = engine.ctx.model
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MESH, "r_RasPiMouse_MultiLiDARMount") >= 0
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MESH, "r_robotis_lds01") < 0
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "r_lidar") < 0
    lds = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "r_lds01_mount")
    assert lds >= 0 and float(model.body_mass[lds]) == pytest.approx(LDS_MASS)


def test_scan_frame_is_the_vendor_chain(mounted):
    """base_link -> lds_multi_mount_link -> laser, as the expanded xacro chains it, 3.14 yaw included;
    the frame the mount publishes is that same pose."""
    engine, _ = mounted
    want_pos, want_rot = scan_mount.chain(LDS_MOUNT, LASER_JOINT)
    np.testing.assert_allclose(want_pos, [0.0, 0.0, 0.12], atol=1e-12)
    for site in ("r_lds01_scan", "r_lds01_laser"):
        pos, rot = scan_mount.pose_in_base(engine, site)
        np.testing.assert_allclose(pos, want_pos, atol=1e-9, err_msg=site)
        np.testing.assert_allclose(rot, want_rot, atol=1e-9, err_msg=site)


def test_the_forward_ray_reads_the_true_wall_distance(mounted):
    """Bearing 0 is laser +x, which the 3.14 yaw turns to face base_link -x."""
    engine, lidar = mounted
    origin, dirs, bearings = scan_mount.world_rays(engine, lidar)
    fwd = scan_mount.forward_index(bearings)
    np.testing.assert_allclose(
        dirs[fwd], scan_mount.base_rotation(engine) @ [np.cos(3.14), np.sin(3.14), 0.0], atol=1e-9
    )
    assert lidar.latest.ranges[fwd] == pytest.approx(
        scan_mount.wall_distance(origin, dirs[fwd]), abs=1e-6
    )


def test_no_ray_starts_inside_the_robot_or_returns_from_its_own_mount(mounted):
    """Every ray meets its first surface from outside, and the housing is the only exclusion."""
    engine, lidar = mounted
    mount_body = scan_mount.mount_body(engine, "lds01")
    dirs, hits = scan_mount.recast(engine, lidar, bodyexclude=mount_body)
    assert lidar._bodyexclude == mount_body
    hit = hits.geomid >= 0
    assert hit.all(), "a closed room leaves no ray without a return"
    facing = np.einsum("ij,ij->i", hits.normal[hit], dirs[hit])
    assert not np.any(facing > 0), f"{int((facing > 0).sum())} ray(s) start inside robot geometry"
    own = lidar._hits.geomid
    assert not np.any(engine.ctx.model.geom_bodyid[own[own >= 0]] == mount_body)


def test_the_scan_sees_nothing_of_the_robot(mounted):
    """The scan plane clears the mount and the legs the scanner stands on."""
    engine, lidar = mounted
    _, hits = scan_mount.recast(engine, lidar)
    assert scan_mount.robot_returns(engine, hits) == (set(), set())


def test_the_static_tf_chain_is_published(mounted):
    """base_link -> lds_multi_mount_link from the robot, lds_multi_mount_link -> laser from the mount."""
    engine, _ = mounted
    (mount,) = scan_mount.static_tf(engine, "robot")
    (laser,) = scan_mount.static_tf(engine, "robot.lds01")
    assert (mount["parent"], mount["child"]) == ("base_link", "lds_multi_mount_link")
    assert (laser["parent"], laser["child"]) == ("lds_multi_mount_link", "laser")
    for tf, joint in ((mount, LDS_MOUNT), (laser, LASER_JOINT)):
        np.testing.assert_allclose(tf["translation"], joint[0], atol=1e-9)
        assert scan_mount.same_rotation(tf["rotation"], scan_mount.chain(joint)[1])


def test_the_scan_topic_is_the_robots(mounted):
    """`scan` (urdf/sensors/lidar.gazebo.xacro:10), relative, in the robot's namespace, stamped laser."""
    engine, _ = mounted
    scan = scan_mount.scan_endpoint(engine)
    assert scan.owner == "robot.lds01" and scan.namespace == scan_mount.NAMESPACE
    assert scan.backend["ros2"]["topic"] == "scan"
    assert scan.backend["ros2"]["frame_id"] == "laser"
    assert "static_tf" not in scan.backend["ros2"]
