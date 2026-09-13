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

The scanner fixtures are Clearpath's numbers, not the model's. clearpath_config
sample/w200/w200_dual_laser.yaml @ b2a64ba mounts two `hokuyo_ust` on vertical PACS brackets, one at
the front of each diff unit, and the chain to each scan frame is
  w200.urdf.xacro:52-56                     base_link -> chassis_link (0, 0, 0.025)
  diff_unit.urdf.xacro:253-260, w200:39     chassis_link -> <side>_diff_unit_link (0, +-0.56821, 0)
  w200_dual_laser.yaml:14-21                <side>_diff_unit_link -> front_left_link / rear_right_link
  clearpath_mounts_description urdf/pacs/bracket.urdf.xacro:59-64, 100-110 @ 33e4b31
                                            -> bracket_<i>_link -> bracket_<i>_mount (0, 0, 0.010125)
                                            -> bracket_<i>_vertical_mount (0.0518, 0, 0.086875), pitch -pi/2
  w200_dual_laser.yaml:33-35, 46-48         -> lidar2d_<i>_link (identity)
  clearpath_sensors_description urdf/hokuyo_ust.urdf.xacro:39-44 @ b0f6d92
                                            lidar2d_<i>_link -> lidar2d_<i>_laser (0, 0, 0.0474)
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
    quat_matrix,
    robot_hits,
    spawn,
    static_tf,
    urdf_rotation,
)

from roqsim.config import load_config_from_dict
from roqsim.engine import Engine

#: From Clearpath's expanded w200 xacro @ b0f6d920, not measured from our model. The description's
#: own sum is 260.001 kg; the missing gram is imu_0_link, which is a site here rather than a body.
#: The two scanner brackets carry no inertial in Clearpath's macro and add no mass here; the two
#: UST-10LX devices add their specification weight of 0.13 kg each.
TOTAL_MASS = 260.0 + 2 * 0.13
WHEEL_RADIUS = 0.3  # diff_4wd.yaml, and the URDF collision cylinder agrees
WHEEL_SEPARATION = 1.13642  # the URDF geometry (2 x 0.56821)
SLIP_FACTOR = 5.25  # calibrated against this model -- see the manifest
#: diff_4wd.yaml's wheel_separation 1.5 x wheel_separation_multiplier 1.125, over the real track.
VENDOR_COMPENSATION = 1.5 * 1.125 / WHEEL_SEPARATION

#: The scanner mounts, as Clearpath chains them (fixed joints ``(xyz, rpy)``, see the module docstring).
OWNER, PREFIX, NAMESPACE = "wh", "wh_", "warthog1"
CHASSIS = ((0.0, 0.0, 0.025), (0.0, 0.0, 0.0))
BRACKET_LINK = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
BRACKET_MOUNT = ((0.0, 0.0, 0.010125), (0.0, 0.0, 0.0))
VERTICAL_MOUNT = ((0.0518, 0.0, 0.086875), (0.0, -math.pi / 2, 0.0))
SENSOR = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
LASER = ((0.0, 0.0, 0.0474), (0.0, 0.0, 0.0))
#: label -> (diff unit, unit joint, frame link, frame joint, bracket, scan frame, topic)
SCANNERS = {
    "lidar2d_0": (
        "left_diff_unit_link",
        ((0.0, 0.56821, 0.0), (0.0, 0.0, 0.0)),
        "front_left_link",
        ((0.68, -0.03, 0.35), (0.0, 1.5707, 0.0)),  # w200_dual_laser.yaml:14-17
        "bracket_0",
        "lidar2d_0_laser",
        "sensors/lidar2d_0/scan",
    ),
    "lidar2d_1": (
        "right_diff_unit_link",
        ((0.0, -0.56821, 0.0), (0.0, 0.0, 0.0)),
        "rear_right_link",
        ((-0.68, 0.03, 0.35), (3.1415, 1.5707, 0.0)),  # w200_dual_laser.yaml:18-21
        "bracket_1",
        "lidar2d_1_laser",
        "sensors/lidar2d_1/scan",
    ),
}
#: urg_node's request, w200_dual_laser.yaml:41-42 and 54-55.
REQUESTED_ANGLE = 1.5707
#: The UST-10LX's steps per turn: 0.25 deg (Hokuyo specification C-42-04077, 2-2).
AREA_RESOLUTION = 1440


def _engine(**diff_drive):
    world = {
        "sim": {"timestep": 0.002},
        "components": [
            {
                "spawn_robot": {"model": "warthog", "prefix": "w_"},
                "name": "w",
                **({"components": [{"diff_drive": diff_drive}]} if diff_drive else {}),
            }
        ],
    }
    engine = Engine(load_config_from_dict(world, base_dir=Path(".")))
    engine.ctx.seed = 1  # a test driving an Engine is the driver; the scanners draw range noise
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


def test_manifest_brings_the_drive_and_both_scanners():
    """spawn_robot must expand the manifest: a Warthog with no drive is a 260 kg paperweight."""
    engine = _engine()
    try:
        assert engine.ctx.blackboard.get("robot:w") is not None, "diff_drive did not attach"
        scanners = sorted(p.address for p in engine.plugins if type(p).__name__ == "LidarPlugin")
        assert scanners == ["w.lidar2d_0.lidar", "w.lidar2d_1.lidar"], scanners
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


def test_wheels_are_upright():
    """Geometry a dynamics battery cannot see: all four tyre axes along y."""
    engine = _engine()
    try:
        data = engine.ctx.data
        for end in ("front", "rear"):
            for side in ("left", "right"):
                gid = named(
                    engine.ctx.model, mujoco.mjtObj.mjOBJ_GEOM, f"w_{end}_{side}_wheel_tyre"
                )
                axis = data.geom_xmat[gid].reshape(3, 3)[:, 2]
                assert abs(abs(axis[1]) - 1.0) < 1e-6, (
                    f"{end}_{side} tyre axis is {np.round(axis, 4)}, not along y"
                )
    finally:
        engine.shutdown()


# --------------------------------------------------------------------------- C. scanners


def _vendor_chain(label: str):
    unit, unit_joint, _, frame_joint, *_ = SCANNERS[label]
    return chain(
        CHASSIS, unit_joint, frame_joint, BRACKET_LINK, BRACKET_MOUNT, VERTICAL_MOUNT, SENSOR, LASER
    )


@pytest.fixture(scope="module")
def scan():
    """The robot spawned with a prefix and namespace in a room of known walls, both scanners cast once."""
    engine = spawn("warthog", SCANNERS, owner=OWNER, prefix=PREFIX, namespace=NAMESPACE)
    yield engine
    engine.shutdown()


@pytest.mark.parametrize("label", SCANNERS)
def test_c1_the_scan_frame_is_clearpaths_chain(scan, label):
    """C1: front-left and rear-right, 0.3706 m above base_link, the rear one facing backwards."""
    want_pos, want_rot = _vendor_chain(label)
    sign = 1.0 if label == "lidar2d_0" else -1.0
    assert np.allclose(want_pos, (sign * 0.777, sign * 0.53821, 0.3706), atol=5e-5), want_pos
    assert np.allclose(want_rot[:, 0], (sign, 0.0, 0.0), atol=2e-4), "the scan faces the wrong way"
    for site in (f"{PREFIX}{label}_scan", f"{PREFIX}{label}_{SCANNERS[label][5]}"):
        pos, rot = pose_in_base(scan, site, PREFIX)
        assert np.allclose(pos, want_pos, atol=1e-6), f"{site} at {pos}"
        assert np.allclose(rot, want_rot, atol=1e-6), f"{site} rotation {rot}"


@pytest.mark.parametrize("label", SCANNERS)
def test_c2_the_forward_ray_reads_the_wall(scan, label):
    published, true = forward_range(scan, lidar(scan, f"{OWNER}.{label}"))
    assert published == pytest.approx(true, abs=1e-3), (
        f"reads {published:.4f} m, wall at {true:.4f} m"
    )


@pytest.mark.parametrize("label", SCANNERS)
def test_c3_the_scan_skips_its_own_mount_and_nothing_else(scan, label):
    model = scan.ctx.model
    scanner = lidar(scan, f"{OWNER}.{label}")
    mount = named(model, mujoco.mjtObj.mjOBJ_BODY, f"{PREFIX}{label}_mount")
    assert scanner._bodyexclude == mount, "the scanner excludes something other than its housing"


@pytest.mark.parametrize("label", SCANNERS)
def test_c4_no_ray_starts_inside_and_none_returns_from_the_robot(scan, label):
    """C4: at the front of its diff unit, 0.0474 m above the bracket, each 180 deg fan looks away from
    the fender, the bracket and the chassis: nothing of the robot is in it."""
    scanner = lidar(scan, f"{OWNER}.{label}")
    inside, outside = robot_hits(scan, scanner, PREFIX)
    assert not inside, {body: len(d) for body, d in inside.items()}
    assert outside == {}, {body: len(d) for body, d in outside.items()}
    assert np.asarray(scanner.latest.ranges).min() > scanner.range_min, "a ray reads too close"


@pytest.mark.parametrize("label", SCANNERS)
def test_c5_the_scanner_stands_on_its_brackets_upright_plate(scan, label):
    """C5: the vertical mount lies on the inner face of the bracket's upright plate, whose collision box
    is fitted to Clearpath's mesh, and the base plate sits on the diff unit's frame link."""
    m, d = scan.ctx.model, scan.ctx.data
    unit, _, frame, _, bracket, *_ = SCANNERS[label]
    mount = named(m, mujoco.mjtObj.mjOBJ_SITE, f"{PREFIX}{bracket}_vertical_mount")
    for plate, face, axis in (
        ("upright", mount, 0),
        ("base", named(m, mujoco.mjtObj.mjOBJ_SITE, f"{PREFIX}{bracket}_link"), 2),
    ):
        gid = named(m, mujoco.mjtObj.mjOBJ_GEOM, f"{PREFIX}{bracket}_{plate}_collision")
        assert m.geom_bodyid[gid] == named(m, mujoco.mjtObj.mjOBJ_BODY, f"{PREFIX}{unit}")
        local = d.geom_xmat[gid].reshape(3, 3).T @ (d.site_xpos[face] - d.geom_xpos[gid])
        assert local[axis] == pytest.approx(-m.geom_size[gid][axis], abs=1e-6), (plate, local)


def test_c6_the_tf_chain_and_topics(scan):
    robot = static_tf(scan, OWNER, NAMESPACE)
    want = []
    for unit, _, frame, frame_joint, bracket, *_ignored in SCANNERS.values():
        want += [
            (unit, frame, frame_joint),
            (frame, f"{bracket}_link", BRACKET_LINK),
            (f"{bracket}_link", f"{bracket}_mount", BRACKET_MOUNT),
            (f"{bracket}_mount", f"{bracket}_vertical_mount", VERTICAL_MOUNT),
        ]
    assert [(t["parent"], t["child"]) for t in robot] == [(p, c) for p, c, _ in want], robot
    for t, (_, _, (xyz, rpy)) in zip(robot, want, strict=True):
        assert np.allclose(t["translation"], xyz, atol=1e-6), t
        assert np.allclose(quat_matrix(t["rotation"]), urdf_rotation(rpy), atol=1e-6), t
    for label, (*_, bracket, frame_id, topic) in SCANNERS.items():
        address = f"{OWNER}.{label}"
        tf = static_tf(scan, address, NAMESPACE)
        assert [(t["parent"], t["child"]) for t in tf] == [(f"{bracket}_vertical_mount", frame_id)]
        assert np.allclose(tf[0]["translation"], LASER[0], atol=1e-6)
        assert np.allclose(quat_matrix(tf[0]["rotation"]), np.eye(3), atol=1e-9)
        hints = endpoint(scan, "scan", address).backend["ros2"]
        assert (hints["frame_id"], hints["topic"]) == (frame_id, topic)
        assert "static_tf" not in hints, "the mount owns the chain; the scan publishes none"


def _urg_step(radian: float) -> int:
    """urg_c's rad2step for a UST-10LX: round to nearest at its steps per turn (urg_utils.c:144-174)."""
    return math.floor(AREA_RESOLUTION * radian / (2 * math.pi) + 0.5)


@pytest.mark.parametrize("label", SCANNERS)
def test_c7_the_field_is_the_one_clearpaths_configuration_asks_urg_node_for(scan, label):
    """C7: angle_min/angle_max +-1.5707 become steps -360 .. +360, published as exactly +-pi/2 with
    721 rays; every other value is the UST-10LX device's."""
    from roqsim_sensors.models import MODELS_DIR

    first, last = _urg_step(-REQUESTED_ANGLE), _urg_step(REQUESTED_ANGLE)
    assert (first, last) == (-360, 360)
    scanner = lidar(scan, f"{OWNER}.{label}")
    assert scanner.num_rays == last - first + 1
    assert scanner.angle_min == pytest.approx(2 * math.pi * first / AREA_RESOLUTION, abs=1e-9)
    assert scanner.angle_max == pytest.approx(2 * math.pi * last / AREA_RESOLUTION, abs=1e-9)
    manifest = yaml.safe_load((MODELS_DIR / "hokuyo_ust" / "hokuyo_ust.manifest.yaml").read_text())
    (device,) = [c["lidar"] for c in manifest["components"] if "lidar" in c]
    for key in (
        "range_min",
        "max_range",
        "detection_min",
        "detection_max",
        "too_close",
        "no_return",
        "rate_hz",
        "range_stddev",
    ):
        assert scanner.config[key] == device[key], f"{key}: {scanner.config[key]} != {device[key]}"
