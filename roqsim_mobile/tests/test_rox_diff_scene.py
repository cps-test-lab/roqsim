"""The Neobotix ROX-Diff: a differential Neobotix from a second repository, and two findings.

``test_the_wheel_inertia_is_four_times_the_vendors_own_geometry`` records a defect in the source,
asserted rather than merely noted. ``diff_wheel.xacro`` declares ``izz = 0.05625`` at ``mass = 5.0``,
which is exactly one half m r^2 for **r = 0.15 m** — the wheel's DIAMETER, used where its radius
belongs. Three independent things in the same description say the radius is 0.075 m: the collision
sphere, the joint's mounting height (a 0.15 m wheel would put the axle below the floor) and the
visual mesh, which measures 0.1499 m across. The vendor's tensor is shipped unchanged and pinned
here so nobody "fixes" the audit by substituting a plausible number: the audit's value is that it
checks the vendor's.

``test_the_casters_carry_the_robot_and_the_drive_tyres_barely_touch`` pins the other one, which is
about this geometry rather than the vendor's arithmetic. All six contacts are coplanar and rigid, so
the vertical load split is statically indeterminate, and MuJoCo settles it with ~97% of the weight on
the four casters and almost none on the two drive tyres. That makes the caster friction load-bearing:
at the 0.04 its siblings carry, caster drag exceeds the traction the tyres can generate and the robot
under-rotates by 11%. The model carries 0.01 — a swivel caster's rolling resistance on a hard floor —
and this test pins the split so a later change to either number is made knowingly.

Every mass here is the vendor's placeholder: each inertial in this description is commented "These
are not accurate value", and the ROX data sheet publishes payload but no dead weight, so there is
nothing to calibrate the 140 kg body against. Navigation, not dynamics.

The scanners are two ``sick_nanoscan3`` device models at the vendor's two lidar joints, at opposite
corners and each turned 45 degrees outward, so the pair covers 360 degrees between them.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import pytest
from mobile_scene_utils import named
from scan_mount_utils import (
    assert_scan_frames,
    assert_tf_chain,
    forward_range,
    lidar,
    recast,
    robot_hits,
    spawn,
)

from roqsim.config import load_config_from_dict
from roqsim.engine import Engine

#: From the expanded rox jazzy xacro @ c865076d, not measured from our model.
VENDOR_MASS = 155.602
#: lidar_{1,2}_link as the vendor declares them. Those links are not in the MJCF: the two
#: sick_nanoscan3 devices mounted there carry the vendor's same 1 g placeholder, so the sum is equal.
VENDOR_LIDAR_LINK_MASS = 0.001
NANOSCAN3_MASS = 0.001
TOTAL_MASS = VENDOR_MASS - 2 * VENDOR_LIDAR_LINK_MASS + 2 * NANOSCAN3_MASS
BASE_MASS = 140.0
CASTER_MASS = 1.4         # each, for a 124 mm sphere
WHEEL_MASS = 5.0          # each
WHEEL_RADIUS = 0.075
WHEEL_SEPARATION = 0.634
MAX_LINEAR_VEL = 0.8      # rox_navigation/configs/navigation_diff.yaml FollowPath max_vel_x
MAX_ANGULAR_VEL = 0.8     # the same file's max_rot_vel

WHEELS = ("left", "right")
CASTERS = ("front_left", "front_right", "back_left", "back_right")


def _engine():
    world = {
        "sim": {"timestep": 0.002},
        "components": [{"spawn_robot": {"model": "rox_diff", "prefix": "q_"}, "name": "q"}],
    }
    engine = Engine(load_config_from_dict(world, base_dir=Path(".")))
    # A test driving an Engine is the driver, and `ctx.seed` is driver-owned: the scanner's range
    # noise refuses to draw without one.
    engine.ctx.seed = 0
    engine.setup()
    engine.reset()
    return engine


def _yaw(data, bid):
    q = data.xquat[bid]
    return float(np.arctan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2)))


def _normal_forces(engine) -> dict[str, float]:
    model, data = engine.ctx.model, engine.ctx.data
    out: dict[str, float] = {}
    for i in range(data.ncon):
        contact = data.contact[i]
        force = np.zeros(6)
        mujoco.mj_contactForce(model, data, i, force)
        for gid in (contact.geom1, contact.geom2):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
            if name.endswith("_tyre"):
                out[name] = out.get(name, 0.0) + float(force[0])
    return out


def test_mass_matches_the_vendor_description():
    engine = _engine()
    try:
        model = engine.ctx.model
        bid = named(model, mujoco.mjtObj.mjOBJ_BODY, "q_base_link")
        assert model.body_subtreemass[bid] == pytest.approx(TOTAL_MASS, abs=1e-3)
    finally:
        engine.shutdown()


def test_the_wheel_inertia_is_four_times_the_vendors_own_geometry():
    """The vendor computed 1/2 m r^2 with the DIAMETER in the radius's place. See the docstring.

    If this now differs, someone substituted a plausible number -- which breaks the audit's whole
    purpose, since it exists to check the description rather than to agree with it.
    """
    engine = _engine()
    try:
        model = engine.ctx.model
        for side in WHEELS:
            bid = named(model, mujoco.mjtObj.mjOBJ_BODY, f"q_wheel_{side}_link")
            assert model.body_mass[bid] == pytest.approx(WHEEL_MASS)
            izz = float(model.body_inertia[bid][2])
            assert izz == pytest.approx(0.05625, abs=1e-6), "the vendor's tensor was altered"
            # What the vendor wrote, against what its own geometry says.
            assert izz == pytest.approx(0.5 * WHEEL_MASS * (2 * WHEEL_RADIUS) ** 2, abs=1e-6)
            honest = 0.5 * WHEEL_MASS * WHEEL_RADIUS**2
            assert izz == pytest.approx(4 * honest, abs=1e-6)

        # The three places that agree the radius is 0.075 m, so the error is in the tensor.
        gid = named(model, mujoco.mjtObj.mjOBJ_GEOM, "q_wheel_left_link_tyre")
        assert float(model.geom_size[gid][0]) == pytest.approx(WHEEL_RADIUS)
        bid = named(model, mujoco.mjtObj.mjOBJ_BODY, "q_wheel_left_link")
        assert float(model.body_pos[bid][2]) == pytest.approx(WHEEL_RADIUS)
    finally:
        engine.shutdown()


def test_the_casters_carry_the_robot_and_the_drive_tyres_barely_touch():
    """The coplanar six-contact load split, pinned because the caster friction depends on it."""
    engine = _engine()
    try:
        for _ in range(500):
            engine.step()
        forces = _normal_forces(engine)
        casters = sum(v for k, v in forces.items() if "caster" in k)
        wheels = sum(v for k, v in forces.items() if "caster" not in k)
        assert casters + wheels == pytest.approx(TOTAL_MASS * 9.81, rel=0.02)
        assert casters / (casters + wheels) > 0.9, (
            f"casters carry {casters:.0f} N of {casters + wheels:.0f} N; if this has changed, "
            "re-derive the caster friction -- it is set against this split")
    finally:
        engine.shutdown()


def test_it_rests_on_its_wheels_and_casters():
    engine = _engine()
    try:
        model, data = engine.ctx.model, engine.ctx.data
        bid = named(model, mujoco.mjtObj.mjOBJ_BODY, "q_base_link")
        for _ in range(500):
            engine.step()
        assert abs(float(data.xpos[bid][2])) < 1e-3, "settled off its rest height"
        assert float(np.linalg.norm(data.qvel[:3])) < 1e-3, "still drifting"
    finally:
        engine.shutdown()


def test_the_casters_slide_rather_than_grip():
    engine = _engine()
    try:
        model = engine.ctx.model
        for corner in CASTERS:
            gid = named(model, mujoco.mjtObj.mjOBJ_GEOM, f"q_caster_wheel_{corner}_link_tyre")
            assert float(model.geom_friction[gid][0]) < 0.02, "a caster grips like a driven tyre"
            assert int(model.geom_priority[gid]) > 0, (
                "without priority MuJoCo takes the MAXIMUM friction and the floor's value wins")
    finally:
        engine.shutdown()


def test_manifest_brings_a_differential_drive_and_two_scanners():
    engine = _engine()
    try:
        handle = engine.ctx.blackboard.get("robot:q")
        assert handle is not None, "no diff_drive on the blackboard"
        model = engine.ctx.model
        for side in WHEELS:
            assert named(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"q_wheel_{side}_motor") >= 0
        for name in ("scan_front_left", "scan_back_right"):
            assert named(model, mujoco.mjtObj.mjOBJ_BODY, f"q_{name}_mount") >= 0
    finally:
        engine.shutdown()


def test_b1_drives_straight():
    """Ramp out first: at 0.25 m/s^2 reaching 0.8 m/s takes 3.2 s, then take a short sample."""
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
        assert 0.96 < speed / MAX_LINEAR_VEL < 1.03, (
            f"commanded {MAX_LINEAR_VEL} m/s, achieved {speed:.4f} m/s")
        assert abs(_yaw(data, bid)) < 0.02, "veered while driving straight"
    finally:
        engine.shutdown()


@pytest.mark.parametrize("commanded", [0.3, 0.6, MAX_ANGULAR_VEL])
def test_b2_rotates_at_the_commanded_rate(commanded):
    """The window is this model's, not its siblings': see the load split in the module docstring."""
    engine = _engine()
    try:
        model, data = engine.ctx.model, engine.ctx.data
        bid = named(model, mujoco.mjtObj.mjOBJ_BODY, "q_base_link")
        handle = engine.ctx.blackboard.get("robot:q")
        for _ in range(500):
            engine.step()
        handle.drive(0.0, 0.0, commanded)
        for _ in range(1500):
            engine.step()
        previous, t0, total = _yaw(data, bid), float(data.time), 0.0
        for _ in range(1000):
            engine.step()
            current = _yaw(data, bid)
            step = (current - previous + np.pi) % (2 * np.pi) - np.pi
            total += step
            previous = current
        rate = total / (float(data.time) - t0)
        assert 0.95 < rate / commanded < 1.03, (
            f"commanded {commanded} rad/s, achieved {rate:.4f} rad/s")
    finally:
        engine.shutdown()


def test_odometry_agrees_with_ground_truth():
    """The wheel-axis sign must reach the odometry read as well as the command.

    Applying it in only one place is the subtle failure: the robot drives correctly and reports
    itself going backwards, which looks like a bridge or frame problem rather than a drive one.
    """
    engine = _engine()
    try:
        data = engine.ctx.data
        handle = engine.ctx.blackboard.get("robot:q")
        for _ in range(400):
            engine.step()
        handle.drive(0.2, 0.0, 0.0)
        for _ in range(1800):
            engine.step()
        truth = float(data.qvel[0])
        reported = float(handle.read_odom()[3])
        assert reported * truth > 0, (
            f"odometry reports {reported:+.4f} m/s while the base does {truth:+.4f} -- opposite "
            f"signs mean the wheel-axis sign reached the command but not the odometry read"
        )
        assert abs(reported - truth) < 0.05, f"odometry {reported:+.4f} vs truth {truth:+.4f}"
    finally:
        engine.shutdown()


MOUNTS = {
    "scan_front_left": ("sick_nanoscan3", "lidar_1_link", (0.2995, 0.269, 0.189),
                        (3.14159265, 0.0, 0.78539816), "scan"),
    "scan_back_right": ("sick_nanoscan3", "lidar_2_link", (-0.2995, -0.269, 0.189),
                        (3.14159265, 0.0, -2.35619449), "scan2"),
}
NAMESPACE = "neo"
#: Each scanner sits in a chamfered corner of the frame, so the chassis stands in the inboard part of
#: its 275 degree field -- 540 of 1651 rays, which is the ~90 degrees of robot a corner mount sees.
#: Nothing else: the casters clear the scan plane and the drive tyres are below it.
OUTSIDE_HITS = {"scan_front_left": {"base_link": 540}, "scan_back_right": {"base_link": 540}}
HIT_DISTANCE = (0.003, 0.12)


@pytest.fixture(scope="module")
def scan():
    engine = spawn("rox_diff", MOUNTS, owner="q", prefix="q_", namespace=NAMESPACE)
    yield engine
    engine.shutdown()


def test_the_scan_frames_are_the_vendor_joint_origins(scan):
    """Both mounts carry the vendor's roll of pi, so lidar_*_link is z-down on this robot."""
    assert_scan_frames(scan, "q_", MOUNTS)


@pytest.mark.parametrize("label", MOUNTS)
def test_each_scan_skips_its_own_mount_and_nothing_else(scan, label):
    model = scan.ctx.model
    scanner = lidar(scan, f"q.{label}")
    mount = named(model, mujoco.mjtObj.mjOBJ_BODY, f"q_{label}_mount")
    assert scanner._bodyexclude == mount, "the device excludes its own housing, not the robot"
    _, hits = recast(scan, scanner)
    assert mount not in set(model.geom_bodyid[hits.geomid[hits.geomid >= 0]].tolist())


@pytest.mark.parametrize("label", MOUNTS)
def test_the_forward_ray_reads_the_wall(scan, label):
    published, true = forward_range(scan, lidar(scan, f"q.{label}"))
    assert published == pytest.approx(true, abs=1e-3), (
        f"{label} reads {published:.4f} m against a wall at {true:.4f} m")


@pytest.mark.parametrize("label", MOUNTS)
def test_no_ray_starts_inside_robot_geometry(scan, label):
    """The reason the chassis collision is a CLIPPED hull and the casters are the vendor's wheel.

    Hull the chassis without cutting the chamfers and this scanner is inside its own robot; keep the
    vendor's 0.124 m caster sphere and the front-left one is inside a caster. Either way every ray
    starts in collision geometry and the scan reads its too-close value in all 1651 directions --
    which no other test here would notice, because the ranges stay self-consistent.
    """
    scanner = lidar(scan, f"q.{label}")
    inside, outside = robot_hits(scan, scanner, "q_")
    assert not inside, {body: len(d) for body, d in inside.items()}

    # The chassis beside a corner-mounted scanner is nearer than the device's own 0.1 m minimum
    # range, so those rays are published as too_close rather than as a distance -- which is the
    # device being honest, not a ray starting in geometry. Every one of them is accounted for.
    ranges = np.asarray(scanner.latest.ranges)
    near = int(np.isneginf(ranges).sum())
    blocked = sum(len(d) for d in outside.values())
    assert near <= blocked, f"{near} rays read too close but only {blocked} meet the robot"
    assert near == sum(1 for d in outside.values() for v in d if v < scanner.range_min)
    assert ranges[np.isfinite(ranges)].min() > scanner.range_min


@pytest.mark.parametrize("label", MOUNTS)
def test_the_robot_the_scan_sees_is_the_chassis_beside_it(scan, label):
    _, outside = robot_hits(scan, lidar(scan, f"q.{label}"), "q_")
    assert {body: len(d) for body, d in outside.items()} == OUTSIDE_HITS[label], sorted(outside)
    low, high = HIT_DISTANCE
    for body, distances in outside.items():
        assert low < min(distances) and max(distances) < high, (
            body, min(distances), max(distances))


def test_the_tf_chain_and_topics_are_the_vendors(scan):
    assert_tf_chain(scan, "q", NAMESPACE, MOUNTS)
