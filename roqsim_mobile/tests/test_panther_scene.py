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

import math
from pathlib import Path

import mujoco
import numpy as np
import pytest
import yaml
from mobile_scene_utils import named
from scan_mount_utils import (
    chain,
    endpoint,
    forward_range,
    lidar,
    pose_in_base,
    robot_hits,
    same_rotation,
    spawn,
    static_tf,
    urdf_rotation,
)

from roqsim.config import load_config_from_dict
from roqsim.engine import Engine
from roqsim.models import resolve_model

#: From Husarion's expanded husarion_ugv_description @ 559e784b, not measured from our model.
#: The mounted RPLIDAR S3 adds Husarion's 0.115033 kg (slamtec_rplidar.urdf.xacro:58 @ 5f783f8).
TOTAL_MASS = 55.0 + 0.115033
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
    engine.ctx.seed = 1  # a test driving an Engine is the driver; the scanner draws range noise
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


# --------------------------------------------------------------------------- C. scanner
#
# The fixtures are Husarion's numbers, not the model's. husarion_ugv_description config/components.yaml
# :5-14 @ 559e784b names an RPLIDAR S3 (LDR06) on mount_link at xyz (0, -0.1, 0) called
# `second_lidar`, and the chain to its scan frame is
#   urdf/panther/base.urdf.xacro:87-91         body_link -> cover_link (0, 0, 0.14)
#   base.urdf.xacro:95-99                      cover_link -> mount_link (0, 0, 0.0315)
#   husarion_components_description urdf/slamtec_rplidar.urdf.xacro:249-253 @ 5f783f8
#                                              mount_link -> second_lidar_link (0, -0.1, 0)
#   slamtec_rplidar.urdf.xacro:63-65, 272-276  second_lidar_link -> second_lidar_laser (0, 0, 0.0305), yaw pi

LABEL = "second_lidar"
SCAN_FRAME = "second_lidar_laser"
TOPIC = "second_lidar/scan"
OWNER, PREFIX, NAMESPACE = "pt", "pt_", "panther1"
COVER = ((0.0, 0.0, 0.14), (0.0, 0.0, 0.0))
MOUNT = ((0.0, 0.0, 0.0315), (0.0, 0.0, 0.0))
COMPONENT = ((0.0, -0.1, 0.0), (0.0, 0.0, 0.0))
LASER = ((0.0, 0.0, 0.0305), (0.0, 0.0, math.pi))
#: What the S3's fan meets of the robot from outside: the Husarion hull, at 0.254 m and beyond.
OUTSIDE_RETURNS = {"body_link": 48}


@pytest.fixture(scope="module")
def scan():
    """The robot spawned with a prefix and namespace in a room of known walls, its scanner cast once."""
    engine = spawn("panther", {LABEL: None}, owner=OWNER, prefix=PREFIX, namespace=NAMESPACE)
    yield engine
    engine.shutdown()


def test_c1_the_scan_frame_is_husarions_chain(scan):
    """C1: 0.202 m above body_link, 0.1 m to the right of the centreline, turned half a revolution."""
    want_pos, want_rot = chain(COVER, MOUNT, COMPONENT, LASER)
    assert np.allclose(want_pos, (0.0, -0.1, 0.202), atol=1e-9)
    for site in (f"{PREFIX}{LABEL}_scan", f"{PREFIX}{LABEL}_{SCAN_FRAME}"):
        pos, rot = pose_in_base(scan, site, PREFIX)
        assert np.allclose(pos, want_pos, atol=1e-6), f"{site} at {pos}"
        assert np.allclose(rot, want_rot, atol=1e-6), f"{site} rotation {rot}"


def test_c2_the_forward_ray_reads_the_wall(scan):
    published, true = forward_range(scan, lidar(scan, f"{OWNER}.{LABEL}"))
    assert published == pytest.approx(true, abs=1e-3), (
        f"reads {published:.4f} m, wall at {true:.4f} m"
    )


def test_c3_the_scan_skips_its_own_mount_and_nothing_else(scan):
    model = scan.ctx.model
    scanner = lidar(scan, f"{OWNER}.{LABEL}")
    mount = named(model, mujoco.mjtObj.mjOBJ_BODY, f"{PREFIX}{LABEL}_mount")
    assert scanner._bodyexclude == mount, "the scanner excludes something other than its housing"


def test_c4_no_ray_starts_inside_robot_geometry(scan):
    """The former embedded site excluded base_link; the device mount starts no ray inside the robot."""
    scanner = lidar(scan, f"{OWNER}.{LABEL}")
    inside, _ = robot_hits(scan, scanner, PREFIX)
    assert not inside, {body: len(d) for body, d in inside.items()}
    assert np.asarray(scanner.latest.ranges).min() > scanner.range_min, "a ray reads too close"


def test_c5_the_robot_returns_are_the_hull_seen_from_outside(scan):
    """C5: real returns of a scanner that sees part of its own robot, pinned by body and ray count."""
    _, outside = robot_hits(scan, lidar(scan, f"{OWNER}.{LABEL}"), PREFIX)
    assert {body: len(d) for body, d in outside.items()} == OUTSIDE_RETURNS


def test_c6_the_tf_chain_and_topic(scan):
    address = f"{OWNER}.{LABEL}"
    tf = static_tf(scan, address, NAMESPACE)
    assert [(t["parent"], t["child"]) for t in tf] == [(f"{LABEL}_link", SCAN_FRAME)], tf
    assert np.allclose(tf[0]["translation"], LASER[0], atol=1e-6)
    assert same_rotation(tf[0]["rotation"], urdf_rotation(LASER[1]))
    robot = static_tf(scan, OWNER, NAMESPACE)
    want = [
        ("body_link", "cover_link", COVER[0]),
        ("cover_link", "mount_link", MOUNT[0]),
        ("mount_link", f"{LABEL}_link", COMPONENT[0]),
    ]
    assert [(t["parent"], t["child"]) for t in robot] == [(p, c) for p, c, _ in want], robot
    for t, (_, _, xyz) in zip(robot, want, strict=True):
        assert np.allclose(t["translation"], xyz, atol=1e-9), t
    hints = endpoint(scan, "scan", address).backend["ros2"]
    assert (hints["frame_id"], hints["topic"]) == (SCAN_FRAME, TOPIC)
    assert "static_tf" not in hints, "the mount owns the chain; the scan publishes none"


def test_c7_the_scan_is_the_s3_devices(scan):
    """C7: the robot publishes what sllidar_ros2 publishes for an S3, with Husarion's range noise."""
    from roqsim_sensors.models import MODELS_DIR

    manifest = yaml.safe_load((MODELS_DIR / "rplidar_s3" / "rplidar_s3.manifest.yaml").read_text())
    (device,) = [c["lidar"] for c in manifest["components"] if "lidar" in c]
    scanner = lidar(scan, f"{OWNER}.{LABEL}")
    for key in ("rays", "angle_min", "angle_max", "range_min", "max_range", "too_close",
                "no_return", "rate_hz"):
        assert scanner.config[key] == device[key], f"{key}: {scanner.config[key]} != {device[key]}"
    # husarion_components_description slamtec_rplidar.urdf.xacro:48 @ 5f783f8
    assert scanner.config["range_stddev"] == 0.015
    assert scanner.num_rays == 3240
