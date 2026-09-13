"""Maker's Pet Mini: the substrate's smallest wheeled robot, and a description that states everything.

Three findings this file pins.

``test_no_value_here_is_an_assumption`` guards what makes this port unusual. Every drive and sensor
number is the vendor's own -- ``params.xacro`` for the geometry, ``config/navigation.yaml`` for the
limits, ``plugins.xacro`` for the scan (360 samples over a full turn, 0.1-10 m, 5 Hz). Most mobile
ports here have to invent at least a scanner mount height. This vendor's other model in the substrate,
the OOMWOO vacuum, has the same property.

``test_the_head_mesh_is_scaled_correctly`` pins the trap this vendor's descriptions set. The head is
the model's only mesh and it carries a **non-uniform** scale of ``0.000124 0.000124 7.76e-05``. Emit
the mesh at 1:1 and it comes out ~1000x too large -- and *no physics check notices*, because the head's
collision is a cylinder and only its visual is the mesh. That is exactly what happened on this
vendor's 200 mm sibling before ``urdf_source.mesh_scales`` existed; the guard is why it did not happen
again here.

``test_the_vendor_head_occludes_every_ray`` pins a property of the description, not a wish. The lidar
skips only its own housing (``base_scan``) and never other robot geometry, and its site is the vendor's
``base_scan`` frame. The description puts that scan plane inside the head, so every ray returns from
inside the head and reads ``range_min``. As written, this robot's scan sees nothing; the port log
records it.
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
LIDAR_HEIGHT = 0.0704          # the description's scan_joint, in the base_link frame
BODY_RADIUS = 0.062            # params.xacro base_diameter 0.124
HEAD_HEIGHT = 0.0388           # params.xacro head_height


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
        assert scan.config["rays"] == 360
        assert scan.config["rate_hz"] == pytest.approx(5.0)
        assert scan.config["max_range"] == pytest.approx(10.0)
    finally:
        engine.shutdown()


def test_the_head_mesh_is_scaled_correctly():
    """The trap this vendor's descriptions set -- see the module docstring.

    Checked against the vendor's own params rather than a remembered number: the head must be
    ``head_height`` tall and as wide as the body, so a mesh emitted at 1:1 fails by three orders of
    magnitude and a mesh emitted at a *uniform* scale fails on one axis.
    """
    engine = _engine()
    try:
        model = engine.ctx.model
        head = [g for g in range(model.ngeom)
                if model.geom_dataid[g] >= 0
                and mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH,
                                      model.geom_dataid[g]) == "k_hemisphere"]
        assert head, "the head mesh is missing -- it is this model's only mesh"
        half = model.geom_aabb[head[0]][3:]
        extents = sorted(2 * float(v) for v in half)
        assert extents[0] == pytest.approx(HEAD_HEIGHT, abs=2e-3), (
            f"head is {extents[0]:.4f} m on its short axis, expected the vendor's head_height "
            f"{HEAD_HEIGHT}. A 1:1 emit would be ~1000x this; a uniform scale would be wrong on one "
            f"axis only."
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
    """A 360-degree puck mounted upside down is how this design fits one under a head."""
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

    The ledger's recorded unknown for this row was whether the Mini wants the flat emitter (like the
    OOMWOO, whose links all hang off base_link) or the nested one. It is nested: `scan_motor` is a
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


def test_the_vendor_head_occludes_every_ray():
    """Pinned as the description is, not as we would like it -- see the manifest and the port log.

    The head is vendor geometry, so the lidar does not exclude it. The scan plane lies inside it, so
    each ray returns 8.4 mm out from inside the head's hemisphere and is clamped to range_min: the
    wall 1 m ahead is invisible. If this fails, the description or the mount has changed.
    """
    engine = _scan_engine()
    try:
        model = engine.ctx.model
        lidar = _lidar(engine)
        ranges = np.asarray(lidar.latest.ranges)
        assert ranges.shape == (360,)
        assert np.allclose(ranges, lidar.range_min), "a ray sees past the head"
        dirs, hits = _recast(engine)
        head = named(model, mujoco.mjtObj.mjOBJ_BODY, "k_head_link")
        assert np.all(hits.geomid >= 0)
        assert set(model.geom_bodyid[hits.geomid].tolist()) == {head}
        assert np.allclose(hits.dist, 0.0084, atol=1e-4)
        assert np.all(np.einsum("ij,ij->i", hits.normal, dirs) > 0), "every return is from inside"
    finally:
        engine.shutdown()


def test_the_inverted_mount_mirrors_bearings_as_the_vendor_frame_does():
    """base_scan's x points backwards, so bearing 0 looks behind the robot and a wall ahead is at pi.

    Read against world geometry only (group 0): a diagnostic through the robot, since the head
    occludes the scan itself.
    """
    engine = _scan_engine()
    try:
        model = engine.ctx.model
        dirs, hits = _recast(engine, geomgroup=np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8))
        assert np.allclose(dirs[0], [-1.0, 0.0, 0.0], atol=1e-9)
        assert np.allclose(dirs[90], [0.0, 1.0, 0.0], atol=1e-9)
        wall = named(model, mujoco.mjtObj.mjOBJ_GEOM, "scan_probe_wall")
        on_wall = np.flatnonzero(hits.geomid == wall)
        nearest = int(on_wall[np.argmin(hits.dist[on_wall])])
        assert nearest == 180
        assert hits.dist[nearest] == pytest.approx(WALL_FACE, abs=1e-3)
    finally:
        engine.shutdown()
