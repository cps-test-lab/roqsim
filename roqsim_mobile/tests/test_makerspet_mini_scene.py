"""Maker's Pet Mini: the substrate's smallest wheeled robot, and a description that states everything.

Three findings this file pins.

``test_no_value_here_is_an_assumption`` guards what makes this port unusual. Every drive number is the
vendor's own -- ``params.xacro`` for the geometry, ``config/navigation.yaml`` for the limits -- and the
scan is the real robot's: its LD14P as Maker's Pet's ``config/telem.yaml`` has ``kaiaai_telemetry``
publish it (720 bins of 0.5 deg, 0.1-8 m, nothing measured published as 0.0), at 6 Hz, the LD14P's
default speed.

``test_the_head_mesh_is_scaled_correctly`` pins the trap this vendor's descriptions set. The head is
the model's only mesh and it carries a **non-uniform** scale of ``0.000124 0.000124 7.76e-05``. Emit
the mesh at 1:1 and it comes out ~1000x too large -- and *no physics check notices*, because the head's
collision is a cylinder and only its visual is the mesh. ``urdf_source.mesh_scales`` carries the
scale through, and this test pins that it does.

``test_the_scan_sees_the_wall_through_the_head_gap`` pins the one deviation from the description. The
lidar skips only its own housing (``base_scan``), and the description's head encloses the scan plane.
The real Mini carries its LiDAR clear above its body, so the model's head ends at the bottom face of the
scanner puck, collision and visual alike, with the head's mass and inertia unchanged: every ray leaves
the robot, and no robot geometry is left in the scan plane. The scan plane is the manufacturer's CAD
height, not the description's.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import pytest
from mobile_scene_utils import named

from roqsim import raycast
from roqsim.config import load_config_from_dict
from roqsim.engine import Engine
from roqsim.plugin import Plugin

#: From the expanded makerspet/makerspet_mini description @ 77d196b6, not measured from our model.
TOTAL_MASS = 0.800
WHEEL_RADIUS = 0.0215          # params.xacro wheel_diameter 0.043
WHEEL_SEPARATION = 0.105043    # params.xacro wheel_base
MAX_LINEAR_VEL = 0.1           # config/navigation.yaml max_vel_x
MAX_ANGULAR_VEL = 0.5          # config/navigation.yaml max_vel_theta
#: The LD14P's optical block centre in the manufacturer's CAD, 72.25 mm above the floor (makerspet/store
#: @ e338516e, MINI-BDC30P-BODY v1.0.1 STEP, GJZJ_ASM 68.0-76.5 mm), less base_link's 14.9 mm above the
#: floor (wheel radius 0.0215 less the wheel joints' z 0.0066). The description's scan_joint says 0.0704.
LIDAR_HEIGHT = 0.07225 - (0.0215 - 0.0066)
#: kaiaai_telemetry @ 7ea0d663 config/telem.yaml:13-19 for LDROBOT-LD14P, as makerspet_mini's
#: config/telem.yaml selects it: 720 bins from 0 deg, range 0.1-8.0 m.
SCAN_RAYS = 720
SCAN_RANGE = (0.1, 8.0)
SCAN_RATE_HZ = 6.0             # LDROBOT LD14P Development Manual V0.2, 1.2: default 6 Hz
BODY_RADIUS = 0.062            # params.xacro base_diameter 0.124
HEAD_HEIGHT = 0.0388           # params.xacro head_height
HEAD_JOINT_Z = 0.032           # params.xacro lower_cylinder_height, head_joint's origin
#: The bottom face of the scanner puck, where the model's head ends: the scan plane less half of
#: params.xacro laser_puck_height 0.016.
SCAN_GAP_BOTTOM = LIDAR_HEIGHT - 0.008
HEAD_MASS = 0.200              # params.xacro head_mass
#: head_link's solid_semi_ellipsoid_inertia over base_diameter/2 and head_height, as expanded.
HEAD_DIAGINERTIA = (0.0001069888, 0.0001069888, 0.00015376)


def _engine(**diff_drive):
    world = {
        "sim": {"timestep": 0.002},
        "components": [{
            "spawn_robot": {"model": "makerspet_mini", "prefix": "k_"},
            "name": "k",
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


def test_mass_matches_the_vendor_description():
    engine = _engine()
    try:
        assert engine.ctx.model.body_mass.sum() == pytest.approx(TOTAL_MASS, abs=1e-3)
    finally:
        engine.shutdown()


def test_no_value_here_is_an_assumption():
    """Every drive number is the vendor's own -- see the module docstring."""
    engine = _engine()
    try:
        drive = next(p for p in engine.plugins if "DiffDrive" in type(p).__name__)
        assert drive.config["wheel_radius"] == pytest.approx(WHEEL_RADIUS)
        assert drive.config["wheel_separation"] == pytest.approx(WHEEL_SEPARATION)
        assert drive.config["max_linear_vel"] == pytest.approx(MAX_LINEAR_VEL)
        assert drive.config["max_angular_vel"] == pytest.approx(MAX_ANGULAR_VEL)
        assert "slip_factor" not in drive.config, (
            "a true 2-wheel differential drive with a caster does not scrub, so it must not carry a "
            "slip_factor -- the same line turtlebot3_waffle, raspimouse and oomwoo_one draw"
        )
        scan = next(p for p in engine.plugins if type(p).__name__ == "LidarPlugin")
        assert scan.num_rays == SCAN_RAYS
        assert scan.angle_min == pytest.approx(0.0)
        assert scan.angle_increment == pytest.approx(np.radians(0.5))
        assert (scan.range_min, scan.range_max) == pytest.approx(SCAN_RANGE)
        assert (scan.detection_min, scan.detection_max) == pytest.approx(SCAN_RANGE)
        assert (scan.too_close, scan.no_return) == (0.0, 0.0), "kaiaai_telemetry publishes 0.0"
        assert scan.rate_hz == pytest.approx(SCAN_RATE_HZ)
    finally:
        engine.shutdown()


def test_the_head_mesh_is_scaled_correctly():
    """The trap this vendor's descriptions set -- see the module docstring.

    Checked against the vendor's own params rather than a remembered number: the head must be as wide
    as the body, and as tall as it stands below the scan gap (``head_height`` less the part the gap
    cuts off). A mesh emitted at 1:1 fails by three orders of magnitude and a mesh emitted at a
    *uniform* scale fails on one axis.
    """
    engine = _engine()
    try:
        model = engine.ctx.model
        head = [g for g in range(model.ngeom)
                if model.geom_dataid[g] >= 0
                and mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH,
                                      model.geom_dataid[g]) == "k_hemisphere_scan_gap"]
        assert head, "the head mesh is missing -- it is this model's only mesh"
        half = model.geom_aabb[head[0]][3:]
        extents = sorted(2 * float(v) for v in half)
        expected = SCAN_GAP_BOTTOM - HEAD_JOINT_Z
        assert expected < HEAD_HEIGHT
        assert extents[0] == pytest.approx(expected, abs=2e-3), (
            f"head is {extents[0]:.4f} m on its short axis, expected {expected:.4f}: the vendor's "
            f"head_height {HEAD_HEIGHT} cut at the scan gap. A 1:1 emit would be ~1000x this; a "
            f"uniform scale would be wrong on one axis only."
        )
        assert extents[-1] == pytest.approx(2 * BODY_RADIUS, abs=2e-3), (
            f"head is {extents[-1]:.4f} m across, expected the body's {2 * BODY_RADIUS}")
    finally:
        engine.shutdown()


def test_it_rests_on_two_wheels_and_the_caster():
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
        for geom in ("k_wheel_left_link_collision0", "k_wheel_right_link_collision0",
                     "k_caster_link_collision0", "k_base_link_collision0"):
            named(model, mujoco.mjtObj.mjOBJ_GEOM, geom)
        assert "k_wheel_left_link_collision0" in touching, touching
        assert "k_wheel_right_link_collision0" in touching, touching
        assert "k_caster_link_collision0" in touching, touching
        assert "k_base_link_collision0" not in touching, "the body is dragging on the floor"
    finally:
        engine.shutdown()


def test_wheels_spin_about_the_robot_y_axis():
    """The joint rpy: this description rotates the JOINT (-pi/2), not the visual."""
    engine = _engine()
    try:
        model, data = engine.ctx.model, engine.ctx.data
        for side in ("left", "right"):
            gid = named(model, mujoco.mjtObj.mjOBJ_GEOM, f"k_wheel_{side}_link_collision0")
            axis = data.geom_xmat[gid].reshape(3, 3)[:, 2]
            assert abs(abs(axis[1]) - 1.0) < 1e-6, (
                f"{side} wheel axis is {np.round(axis, 4)}, not along y")
    finally:
        engine.shutdown()


def test_the_scanner_is_inverted_under_the_head():
    """The scan frame is the description's inverted base_scan, at the CAD's optical height."""
    engine = _engine()
    try:
        model, data = engine.ctx.model, engine.ctx.data
        base = named(model, mujoco.mjtObj.mjOBJ_BODY, "k_base_link")
        sid = named(model, mujoco.mjtObj.mjOBJ_SITE, "k_lidar")
        height = float(data.site_xpos[sid][2] - data.xpos[base][2])
        assert height == pytest.approx(LIDAR_HEIGHT, abs=1e-3)
        puck = named(model, mujoco.mjtObj.mjOBJ_BODY, "k_base_scan")
        z_axis = data.xmat[puck].reshape(3, 3)[:, 2]
        assert z_axis[2] < -0.99, f"the scanner puck is not inverted: z axis {np.round(z_axis, 3)}"
    finally:
        engine.shutdown()


def test_the_lidar_motor_hangs_off_the_puck():
    """A nested link, which is why this port needs the recursive emitter and not the flat one.

    Unlike the OOMWOO, whose links all hang off base_link, the Mini is nested: `scan_motor` is a
    child of `base_scan`, not of `base_link`.
    """
    engine = _engine()
    try:
        model = engine.ctx.model
        motor = named(model, mujoco.mjtObj.mjOBJ_BODY, "k_scan_motor")
        puck = named(model, mujoco.mjtObj.mjOBJ_BODY, "k_base_scan")
        assert int(model.body_parentid[motor]) == puck, (
            "scan_motor should hang off base_scan; a flat emitter would have parented it to base_link"
        )
    finally:
        engine.shutdown()


def test_b1_drives_straight():
    engine = _engine()
    try:
        model, data = engine.ctx.model, engine.ctx.data
        bid = named(model, mujoco.mjtObj.mjOBJ_BODY, "k_base_link")
        handle = engine.ctx.blackboard.get("robot:k")
        for _ in range(500):
            engine.step()
        handle.drive(MAX_LINEAR_VEL, 0.0, 0.0)
        for _ in range(600):
            engine.step()
        x0, t0 = float(data.xpos[bid][0]), float(data.time)
        for _ in range(1500):
            engine.step()
        speed = (float(data.xpos[bid][0]) - x0) / (float(data.time) - t0)
        assert 0.94 < speed / MAX_LINEAR_VEL < 1.05, (
            f"commanded {MAX_LINEAR_VEL} m/s, achieved {speed:.4f} m/s")
        assert abs(_yaw(data, bid)) < 0.02, "veered while driving straight"
    finally:
        engine.shutdown()


@pytest.mark.parametrize("commanded", [0.2, 0.35, 0.5])
def test_b2_rotates_at_the_commanded_rate(commanded):
    engine = _engine()
    try:
        model, data = engine.ctx.model, engine.ctx.data
        bid = named(model, mujoco.mjtObj.mjOBJ_BODY, "k_base_link")
        handle = engine.ctx.blackboard.get("robot:k")
        for _ in range(500):
            engine.step()
        handle.drive(0.0, 0.0, commanded)
        for _ in range(400):
            engine.step()
        t0, previous, total = float(data.time), _yaw(data, bid), 0.0
        for _ in range(1500):
            engine.step()
            current = _yaw(data, bid)
            total += np.unwrap([previous, current])[1] - previous
            previous = current
        ratio = (total / (float(data.time) - t0)) / commanded
        assert 0.93 < ratio < 1.05, f"achieved/commanded yaw {ratio:.3f} at {commanded} rad/s"
    finally:
        engine.shutdown()


def test_the_caster_carries_priority():
    """A caster swivels; a fixed sphere cannot, so it stands in for one via friction.

    `priority` is what makes the low friction apply at all -- MuJoCo otherwise takes the MAXIMUM of
    the two contacting geoms' friction and the floor's value wins. Measured on this vendor's 200 mm
    sibling: without it, yaw tracked 0.77-0.87 of commanded instead of 0.92-0.93.
    """
    engine = _engine()
    try:
        model = engine.ctx.model
        gid = named(model, mujoco.mjtObj.mjOBJ_GEOM, "k_caster_link_collision0")
        assert model.geom_priority[gid] > 0
        assert model.geom_friction[gid][0] < 0.2
        for side in ("left", "right"):
            wheel = named(model, mujoco.mjtObj.mjOBJ_GEOM, f"k_wheel_{side}_link_collision0")
            assert model.geom_friction[wheel][0] >= 1.0, "the driven wheels must keep their grip"
    finally:
        engine.shutdown()


# -- the scanner mount: the vendor's frame, and only its own housing excluded --------------------

WALL_FACE = 1.0  # near face of the probe wall, along world +x from the robot's origin


class _WallAhead(Plugin):
    """A wall across world +x, ``WALL_FACE`` ahead of the spawned robot, spanning the scan plane."""

    def build(self, spec: mujoco.MjSpec, ctx) -> None:
        spec.worldbody.add_geom(name="scan_probe_wall", type=mujoco.mjtGeom.mjGEOM_BOX,
                                pos=[WALL_FACE + 0.05, 0.0, 0.5], size=[0.05, 3.0, 0.5])


def _scan_engine():
    world = {
        "sim": {"timestep": 0.002},
        "components": [
            {f"{__name__}:_WallAhead": {}},
            {"spawn_robot": {"model": "makerspet_mini", "prefix": "k_"}, "name": "k"},
        ],
    }
    engine = Engine(load_config_from_dict(world, base_dir=Path(".")))
    engine.setup()
    engine.reset()
    engine.step()  # the rate gate starts open, so the first step casts
    return engine


def _lidar(engine):
    return next(p for p in engine.plugins if type(p).__name__ == "LidarPlugin")


def _recast(engine, **cast):
    """The published scan's own rays, recast with normals: ``(world directions, hits)``."""
    model, data = engine.ctx.model, engine.ctx.data
    lidar = _lidar(engine)
    scan = lidar.latest
    sid = named(model, mujoco.mjtObj.mjOBJ_SITE, "k_lidar")
    angles = scan.angle_min + scan.angle_increment * np.arange(len(scan.ranges))
    local = np.stack([np.cos(angles), np.sin(angles), np.zeros_like(angles)], axis=1)
    dirs = local @ data.site_xmat[sid].reshape(3, 3).T
    hits = raycast.buffers(len(dirs), normals=True)
    raycast.cast(model, data, data.site_xpos[sid], dirs, cutoff=lidar.range_max,
                 bodyexclude=named(model, mujoco.mjtObj.mjOBJ_BODY, "k_base_scan"), out=hits, **cast)
    return dirs, hits


def test_the_scan_is_stamped_in_the_vendors_base_scan_frame():
    """plugins.xacro stamps the scan in base_scan; scan_joint hangs it off base_link, rpy 0 -pi 0."""
    engine = _scan_engine()
    try:
        model, data = engine.ctx.model, engine.ctx.data
        lidar = _lidar(engine)
        assert lidar.exclude_body == "base_scan", "a lidar excludes its own housing and nothing else"
        hints = next(e for e in engine.ctx.interface.all() if e.name == "scan").backend["ros2"]
        assert hints["frame_id"] == "base_scan"
        tf = hints["static_tf"]
        assert tf["parent"] == "base_link"
        assert np.allclose(tf["translation"], [0.0, 0.0, LIDAR_HEIGHT], atol=1e-9)
        assert np.allclose(np.abs(tf["rotation"]), [0.0, 0.0, 1.0, 0.0], atol=1e-9)  # pi about y
        sid = named(model, mujoco.mjtObj.mjOBJ_SITE, "k_lidar")
        puck = named(model, mujoco.mjtObj.mjOBJ_BODY, "k_base_scan")
        assert np.allclose(data.site_xpos[sid], data.xpos[puck], atol=1e-9)
        assert np.allclose(data.site_xmat[sid], data.xmat[puck], atol=1e-9), (
            "the lidar site must be base_scan's frame, rotation included, or every bearing is "
            "measured in a frame the scan is not stamped in")
    finally:
        engine.shutdown()


def test_the_head_ends_at_the_scan_gap_and_keeps_its_mass():
    """The head's collision cylinder tops out at the puck's bottom face; its inertial is the vendor's."""
    engine = _engine()
    try:
        model, data = engine.ctx.model, engine.ctx.data
        base = named(model, mujoco.mjtObj.mjOBJ_BODY, "k_base_link")
        head = named(model, mujoco.mjtObj.mjOBJ_BODY, "k_head_link")
        gid = named(model, mujoco.mjtObj.mjOBJ_GEOM, "k_head_link_collision0")
        top = float(data.geom_xpos[gid][2] + model.geom_size[gid][1] - data.xpos[base][2])
        assert top == pytest.approx(SCAN_GAP_BOTTOM, abs=1e-6)
        assert top < LIDAR_HEIGHT
        assert model.body_mass[head] == pytest.approx(HEAD_MASS)
        assert np.allclose(model.body_inertia[head], HEAD_DIAGINERTIA, rtol=1e-6)
    finally:
        engine.shutdown()


def test_the_scan_sees_the_wall_through_the_head_gap():
    """No robot geometry is left in the scan plane, and the wall 1 m ahead reads at its true range.

    Cast against every visible group, the head's visual included. Rays that miss the probe wall end
    on the world's own geometry.
    """
    engine = _scan_engine()
    try:
        model = engine.ctx.model
        lidar = _lidar(engine)
        ranges = np.asarray(lidar.latest.ranges)
        assert ranges.shape == (SCAN_RAYS,)
        dirs, hits = _recast(engine)
        np.testing.assert_array_equal(hits.geomid, lidar._hits.geomid)
        struck = hits.geomid >= 0
        robot = struck & (model.geom_bodyid[np.maximum(hits.geomid, 0)] != 0)
        assert not robot.any(), "a ray meets robot geometry"
        wall = named(model, mujoco.mjtObj.mjOBJ_GEOM, "scan_probe_wall")
        on_wall = hits.geomid == wall
        assert on_wall.sum() > SCAN_RAYS // 4, "the wall ahead spans well over a quarter of the turn"
        # The wall's face is the plane x = WALL_FACE in the world; the site stands at x = 0.
        np.testing.assert_allclose(ranges[on_wall], WALL_FACE / dirs[on_wall, 0], atol=1e-6)
        assert ranges[SCAN_RAYS // 2] == pytest.approx(WALL_FACE, abs=1e-6), "the forward ray (bearing pi)"
    finally:
        engine.shutdown()


def test_the_inverted_mount_mirrors_bearings_as_the_vendor_frame_does():
    """base_scan's x points backwards, so bearing 0 looks behind the robot and a wall ahead is at pi.

    Read against world geometry only (group 0), so the bearing check stands on its own of the robot's
    geometry.
    """
    engine = _scan_engine()
    try:
        model = engine.ctx.model
        dirs, hits = _recast(engine, geomgroup=np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8))
        assert np.allclose(dirs[0], [-1.0, 0.0, 0.0], atol=1e-9)
        assert np.allclose(dirs[SCAN_RAYS // 4], [0.0, 1.0, 0.0], atol=1e-9)
        wall = named(model, mujoco.mjtObj.mjOBJ_GEOM, "scan_probe_wall")
        on_wall = np.flatnonzero(hits.geomid == wall)
        nearest = int(on_wall[np.argmin(hits.dist[on_wall])])
        assert nearest == SCAN_RAYS // 2
        assert hits.dist[nearest] == pytest.approx(WALL_FACE, abs=1e-3)
    finally:
        engine.shutdown()
