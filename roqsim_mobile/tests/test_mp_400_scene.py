"""The Neobotix MP-400: a differential Neobotix, and a description whose masses cannot be trusted.

Two findings this file pins.

``test_the_caster_masses_are_upstream_nonsense`` records a defect in the source, asserted rather than
merely noted. Each 38 mm caster sphere is declared at **12.7 kg** — exactly the mass of the MPO-700's
steering modules in the same repository — so 50.8 kg of this 84.4 kg robot sits in four casters. That
is provably copy-paste, not a plausible figure, and it is why this model is fit for navigation and not
for dynamics. The test exists so nobody "fixes" the mass audit by quietly substituting our own number:
the audit's value is that it checks the vendor's.

``test_the_vendors_wheel_axis_is_kept_and_still_drives_forward`` pins a sign. ``diff_drive`` used to write the commanded
wheel rate straight to the actuator with no sign derivation, so it silently required wheels whose axis
is +y in the base frame — a convention every other model satisfied only because our own generators
wrote them. This description's axis is -y and the robot drove *backwards* (-0.984 of a forward
command). The plugin now derives the sign off the model, so the vendor's axis is kept and the tests
pin that the plugin copes rather than that the model was bent to suit it.

The scanner is the ``sick_s300`` device model at the vendor's ``lidar_1_joint``, upside down as the
vendor mounts it, at the height Neobotix's hardware documentation gives (110 mm above the floor)
rather than the joint's 0.141, and the scan tests at the end check it in a closed room. At that height
it sits in the body cover's own scanner pocket, open to the front, so the body meshes are the
vendor's unmodified; the pocket's side and rear walls and the two drive wheels stand in the scan, as
they do on the real robot.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import pytest
from mobile_scene_utils import named
from scan_mount_utils import (
    assert_mounts,
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

#: From the expanded neo_simulation2 humble xacro @ 83204145, not measured from our model.
VENDOR_MASS = 84.4482
#: lidar_1_link as the vendor declares it (mp_400_body.urdf.xacro:45). That link is not in the MJCF:
#: the sick_s300 device mounted there carries the scanner at its datasheet 1.2 kg instead.
VENDOR_LIDAR_LINK_MASS = 0.001
S300_MASS = 1.2
TOTAL_MASS = VENDOR_MASS - VENDOR_LIDAR_LINK_MASS + S300_MASS
BASE_MASS = 30.0
CASTER_MASS = 12.7        # each, for a 38 mm sphere. See the module docstring.
WHEEL_MASS = 1.82362
WHEEL_RADIUS = 0.0765
WHEEL_SEPARATION = 0.52
MAX_LINEAR_VEL = 0.8      # configs/mp_400/navigation.yaml max_vel_x
WHEELS = ("left", "right")
CASTERS = ("front_left", "front_right", "back_left", "back_right")


def _engine():
    world = {
        "sim": {"timestep": 0.002},
        "components": [{"spawn_robot": {"model": "mp_400", "prefix": "q_"}, "name": "q"}],
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


def test_mass_matches_the_vendor_description():
    """The description's sum, with its 1 g lidar_1_link replaced by the 1.2 kg S300 device."""
    engine = _engine()
    try:
        assert engine.ctx.model.body_mass.sum() == pytest.approx(TOTAL_MASS, abs=1e-3)
    finally:
        engine.shutdown()


def test_the_caster_masses_are_upstream_nonsense():
    """A defect in the source, asserted so it cannot be silently "corrected" -- see the docstring."""
    engine = _engine()
    try:
        model = engine.ctx.model
        for corner in CASTERS:
            bid = named(model, mujoco.mjtObj.mjOBJ_BODY, f"q_mp_400_caster_wheel_{corner}_link")
            assert model.body_mass[bid] == pytest.approx(CASTER_MASS, abs=1e-3), (
                "the caster masses are the vendor's, absurd though 12.7 kg for a 38 mm sphere is. "
                "If this now differs, someone substituted a plausible number -- which breaks the "
                "mass audit's whole purpose, since it exists to check the description."
            )
        casters = 4 * CASTER_MASS
        assert casters / TOTAL_MASS > 0.5, (
            f"{casters} kg of {TOTAL_MASS} kg is in the casters; if that share has changed the "
            f"port log's 'navigation, not dynamics' boundary needs revisiting")
    finally:
        engine.shutdown()


def test_the_vendors_wheel_axis_is_kept_and_still_drives_forward():
    """This description's wheel axis is -y, the opposite of every other model here, and that is fine.

    A revolute axis is arbitrary up to sign, so a vendor may express the same wheel either way. The
    axis was briefly flipped in this port because ``diff_drive`` wrote the commanded rate straight to
    the actuator and so silently required +y -- a convention the other models satisfied only because
    our own generators wrote them, and which drove this robot backwards. The plugin now derives the
    sign off the model, as ``omni_drive`` does, so the vendor's axis is kept and this test pins that
    the *plugin* copes rather than that the model was bent to suit it.
    """
    engine = _engine()
    try:
        model, data = engine.ctx.model, engine.ctx.data
        base = named(model, mujoco.mjtObj.mjOBJ_BODY, "q_base_link")
        rot = data.xmat[base].reshape(3, 3)
        for side in WHEELS:
            jid = named(model, mujoco.mjtObj.mjOBJ_JOINT, f"q_mp_400_fixed_wheel_{side}_joint")
            axis = rot.T @ (data.xmat[model.jnt_bodyid[jid]].reshape(3, 3) @ model.jnt_axis[jid])
            assert axis[1] < -0.99, (
                f"{side} wheel axis is {np.round(axis, 4)}; the vendor's is -y, and keeping it is the "
                f"point -- if this is now +y someone re-applied the old workaround"
            )
        handle = engine.ctx.blackboard.get("robot:q")
        for _ in range(400):
            engine.step()
        handle.drive(0.2, 0.0, 0.0)
        for _ in range(1800):
            engine.step()
        assert float(data.qvel[0]) > 0.15, (
            f"drives at {float(data.qvel[0]):+.4f} m/s on a +0.2 command -- the plugin's roll-sign "
            f"derivation is not absorbing the vendor's -y axis"
        )
    finally:
        engine.shutdown()


def test_odometry_agrees_with_ground_truth():
    """The roll sign must be applied to the odometry read as well as to the command.

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
            f"signs mean the roll sign reached the command but not the odometry read"
        )
        assert abs(reported - truth) < 0.05, f"odometry {reported:+.4f} vs truth {truth:+.4f}"
    finally:
        engine.shutdown()


def test_manifest_brings_a_differential_drive_and_one_scanner():
    """One lidar here, unlike both holonomic Neobotix siblings' two."""
    engine = _engine()
    try:
        drive = next(p for p in engine.plugins if "DiffDrive" in type(p).__name__)
        assert drive.config["wheel_radius"] == pytest.approx(WHEEL_RADIUS)
        assert drive.config["wheel_separation"] == pytest.approx(WHEEL_SEPARATION)
        assert "slip_factor" not in drive.config, (
            "two driven wheels with passive casters do not scrub, so no slip_factor -- the same line "
            "turtlebot3_waffle, raspimouse and oomwoo_one draw"
        )
        assert_mounts(engine, "q", MOUNTS)
    finally:
        engine.shutdown()


def test_it_rests_on_its_wheels_and_casters():
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
        for side in WHEELS:
            geom = f"q_mp_400_fixed_wheel_{side}_link_tyre"
            named(model, mujoco.mjtObj.mjOBJ_GEOM, geom)
            assert geom in touching, touching
        named(model, mujoco.mjtObj.mjOBJ_GEOM, "q_base_link_collision")
        assert "q_base_link_collision" not in touching, "the body is dragging on the floor"
    finally:
        engine.shutdown()


def test_the_casters_slide_rather_than_grip():
    """They are FIXED spheres, not articulated wheels, so they stand in for casters via friction.

    `priority` is what makes the low friction apply at all: MuJoCo otherwise takes the maximum of
    the two contacting geoms' friction and the floor's value wins, at which point four loaded
    spheres fight every turn.
    """
    engine = _engine()
    try:
        model = engine.ctx.model
        for corner in CASTERS:
            gid = named(model, mujoco.mjtObj.mjOBJ_GEOM,
                        f"q_mp_400_caster_wheel_{corner}_link_tyre")
            assert model.geom_priority[gid] > 0
            assert model.geom_friction[gid][0] < 0.2
        for side in WHEELS:
            gid = named(model, mujoco.mjtObj.mjOBJ_GEOM, f"q_mp_400_fixed_wheel_{side}_link_tyre")
            assert model.geom_friction[gid][0] >= 1.0, "the driven wheels must keep their grip"
    finally:
        engine.shutdown()


def test_b1_drives_straight():
    """Windows are sized to the room, not to patience.

    The vendor's acceleration limit is 0.25 m/s^2, so reaching 0.8 m/s takes 3.2 s -- and at that
    speed the robot crosses `empty_room`'s 5 m half-extent in about six more. A first draft measured
    over eight seconds and read 0.82 of commanded from a model whose steady state is 0.985, because
    the window ended against a wall. That is the third time this batch; measure the ramp out, then
    take a short sample.
    """
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
        assert 0.94 < speed / MAX_LINEAR_VEL < 1.03, (
            f"commanded {MAX_LINEAR_VEL} m/s, achieved {speed:.4f} m/s")
        assert abs(_yaw(data, bid)) < 0.02, "veered while driving straight"
    finally:
        engine.shutdown()


def test_the_wheels_grip_rather_than_slip():
    """With 60% of the declared mass on low-friction casters, traction is worth checking."""
    engine = _engine()
    try:
        model, data = engine.ctx.model, engine.ctx.data
        jid = named(model, mujoco.mjtObj.mjOBJ_JOINT, "q_mp_400_fixed_wheel_left_joint")
        handle = engine.ctx.blackboard.get("robot:q")
        for _ in range(500):
            engine.step()
        handle.drive(MAX_LINEAR_VEL, 0.0, 0.0)
        for _ in range(1750):
            engine.step()
        # Signed by the wheel's axis in the base frame: this vendor's axis is -y, so the raw joint
        # velocity is negative while the robot drives forward. Comparing the raw number to the base
        # speed is the frame error this batch has now made four times.
        base = named(model, mujoco.mjtObj.mjOBJ_BODY, "q_base_link")
        rot = data.xmat[base].reshape(3, 3)
        axis_y = float((rot.T @ (data.xmat[model.jnt_bodyid[jid]].reshape(3, 3)
                                 @ model.jnt_axis[jid]))[1])
        rolling = float(data.qvel[model.jnt_dofadr[jid]]) * np.sign(axis_y) * WHEEL_RADIUS
        assert abs(1 - float(data.qvel[0]) / rolling) < 0.05, (
            f"base {float(data.qvel[0]):.4f} m/s against a rolling speed of {rolling:.4f} -- the "
            f"wheels are slipping")
    finally:
        engine.shutdown()


@pytest.mark.parametrize("commanded", [0.3, 0.6, 1.0])
def test_b2_rotates_at_the_commanded_rate(commanded):
    """No slip_factor, so this measures the drive rather than a calibration."""
    engine = _engine()
    try:
        model, data = engine.ctx.model, engine.ctx.data
        bid = named(model, mujoco.mjtObj.mjOBJ_BODY, "q_base_link")
        handle = engine.ctx.blackboard.get("robot:q")
        for _ in range(500):
            engine.step()
        handle.drive(0.0, 0.0, commanded)
        for _ in range(600):
            engine.step()
        t0, previous, total = float(data.time), _yaw(data, bid), 0.0
        for _ in range(1500):
            engine.step()
            current = _yaw(data, bid)
            total += np.unwrap([previous, current])[1] - previous
            previous = current
        ratio = (total / (float(data.time) - t0)) / commanded
        assert 0.94 < ratio < 1.04, f"achieved/commanded yaw {ratio:.3f} at {commanded} rad/s"
    finally:
        engine.shutdown()


# -- the scanner: the sick_s300 device at the vendor's lidar_1_joint ------------------------------

#: neo_simulation2 @ 832041452c1a: robots/mp_400/urdf/mp_400_body.urdf.xacro:38 (the lidar_1_joint
#: rotation, written as the vendor writes it) and mp_400_gazebo.urdf.xacro:41,50 (the scan's link and
#: topic). The position is Neobotix's hardware documentation (MP-400 Mechanical Properties, Positions of
#: Sensors) instead of the joint's (0.244, 0, 0.141): LS1 at X 230 from the origin under the drive axle
#: and Z 110 mm above the floor, with base_link 1 mm above it (150 mm drive wheels, wheel joints at
#: x 0, z 0.074).
#: ``{label: (device, scan frame, xyz, rpy, topic)}``, parent frame base_link.
MOUNTS = {"scan_front": ("sick_s300", "lidar_1_link", (0.230, 0.0, 0.109), (3.14, 0.0, 0.0), "scan")}
NAMESPACE = "neo"
#: The robot bodies the scan meets from outside, with their ray counts and distance windows (m): the
#: side walls of the body cover's scanner pocket either side, and the two driven wheels, whose tyres
#: stand in the scan plane at the edges of the field. The rays start at the S300's
#: physical scan plane, 4.1 mm above lidar_1_link on this upside-down mount (z 0.1131).
OUTSIDE_HITS = {
    "base_link": 100,
    "mp_400_fixed_wheel_left_link": 29,
    "mp_400_fixed_wheel_right_link": 29,
}
HIT_DISTANCE = {
    "base_link": (0.127, 0.143),
    "mp_400_fixed_wheel_left_link": (0.280, 0.325),
    "mp_400_fixed_wheel_right_link": (0.280, 0.325),
}


@pytest.fixture(scope="module")
def scan():
    engine = spawn("mp_400", MOUNTS, owner="q", prefix="q_", namespace=NAMESPACE)
    yield engine
    engine.shutdown()


def test_the_scan_frame_is_the_vendor_joint_origin(scan):
    """Upside down (roll 3.14, as written), so the scan's +y bearing points to the robot's right."""
    assert_scan_frames(scan, "q_", MOUNTS)


def test_the_scan_skips_its_own_mount_and_nothing_else(scan):
    model = scan.ctx.model
    scanner = lidar(scan, "q.scan_front")
    mount = named(model, mujoco.mjtObj.mjOBJ_BODY, "q_scan_front_mount")
    assert scanner._bodyexclude == mount, "the device excludes its own housing, not the robot"
    _, hits = recast(scan, scanner)
    assert mount not in set(model.geom_bodyid[hits.geomid[hits.geomid >= 0]].tolist())


def test_the_forward_ray_reads_the_wall(scan):
    published, true = forward_range(scan, lidar(scan, "q.scan_front"))
    assert published == pytest.approx(true, abs=1e-3), (
        f"reads {published:.4f} m against a wall at {true:.4f} m")


def test_no_ray_starts_inside_robot_geometry(scan):
    scanner = lidar(scan, "q.scan_front")
    inside, _ = robot_hits(scan, scanner, "q_")
    assert not inside, {body: len(d) for body, d in inside.items()}
    assert np.asarray(scanner.latest.ranges).min() > scanner.range_min, "a ray reads too close"


def test_the_robot_parts_the_scan_sees_are_the_pocket_walls_and_drive_wheels(scan):
    _, outside = robot_hits(scan, lidar(scan, "q.scan_front"), "q_")
    assert {body: len(d) for body, d in outside.items()} == OUTSIDE_HITS, sorted(outside)
    for body, distances in outside.items():
        low, high = HIT_DISTANCE[body]
        assert low < min(distances) and max(distances) < high, (body, min(distances), max(distances))


def test_the_tf_chain_and_topic_are_the_vendors(scan):
    assert_tf_chain(scan, "q", NAMESPACE, MOUNTS)
