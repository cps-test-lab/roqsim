"""The ROSbot's RPLIDAR C1, mounted as Husarion mounts it: vendor chain, frame, bearings and returns.

The robot is spawned through ``spawn_robot`` with a prefix and a namespace, in a closed room whose
inner wall faces are known planes, so each ray's true range is analytic. Fixtures are the vendor's
numbers, not the model's:

* ``husarion_rosbot`` @ 41fad02196ee200a39e01579c75391385cc9b714 (rosbot_description):
  ``config/rosbot/basic.yaml:4-7`` mounts ``rplidar_c1`` on ``cover_link`` at xyz (0.02, 0, 0);
  ``urdf/rosbot/body.urdf.xacro:9-13`` puts ``body_link`` at z = wheel_radius (0.0425,
  ``rosbot_macro.urdf.xacro:10``) and ``:39-43`` ``cover_link`` 0.0603 above it.
* ``husarion_components_description`` @ 5f783f89961bb16098184f5381b1a76058cec19e:
  ``urdf/slamtec_rplidar.urdf.xacro:142-144,249-278`` chains ``rplidar_link`` -> ``laser`` at
  (0, 0, 0.032), yaw pi, and ``:286`` publishes ``<ns>/scan``.

The scan sees the robot's own body mesh where it rises beside the scanner (the camera-mount post), and
those are real returns: the C1 skips only its own housing. Which bodies return is pinned exactly.
"""

from __future__ import annotations

import math
from pathlib import Path

import mujoco
import numpy as np
from mobile_scene_utils import named

from roqsim import raycast
from roqsim.config import load_config_from_dict
from roqsim.context import SimContext
from roqsim.engine import Engine
from roqsim.plugin import Plugin

PREFIX = "rb_"
NAMESPACE = "rosbot1"
#: The vendor chain, base_link -> laser.
BODY_Z = 0.0425  # body.urdf.xacro:10, wheel_radius (rosbot_macro.urdf.xacro:10)
COVER_Z = 0.0603  # body.urdf.xacro:40
MOUNT_XYZ = (0.02, 0.0, 0.0)  # basic.yaml:6
LASER_XYZ = (0.0, 0.0, 0.032)  # slamtec_rplidar.urdf.xacro:143
LASER_YAW = math.pi  # slamtec_rplidar.urdf.xacro:143
#: The laser origin in base_link, composed from the four joints above.
LASER_IN_BASE = np.array([0.02, 0.0, BODY_Z + COVER_Z + 0.032])

#: Inner wall faces at x, y = +-HALF around the spawn pose.
HALF = 1.5
#: Spawned off the origin and yawed, so neither pose nor bearing is a special value.
SPAWN = (0.2, -0.1, 0.3)

#: Robot bodies whose geometry returns a ray, from outside: the body mesh's camera-mount post.
OUTSIDE_HIT_BODIES = {PREFIX + "body_link"}


class _Room(Plugin):
    def build(self, spec: mujoco.MjSpec, ctx: SimContext) -> None:
        t = 0.05
        for axis in (0, 1):
            for sign in (-1.0, 1.0):
                pos = [SPAWN[0], SPAWN[1], 0.5]
                pos[axis] += sign * (HALF + t)
                size = [HALF + 2 * t, HALF + 2 * t, 0.5]
                size[axis] = t
                spec.worldbody.add_geom(
                    name=f"room_wall_{axis}_{int(sign)}",
                    type=mujoco.mjtGeom.mjGEOM_BOX,
                    pos=pos,
                    size=size,
                )


def _engine() -> Engine:
    world = {
        "sim": {"timestep": 0.002},
        "components": [
            {f"{__name__}:_Room": {}},
            {
                "spawn_robot": {
                    "model": "rosbot",
                    "prefix": PREFIX,
                    "namespace": NAMESPACE,
                    "pose": {
                        "position": {"x": SPAWN[0], "y": SPAWN[1]},
                        "orientation": {"yaw": SPAWN[2]},
                    },
                },
                "name": "rb",
            },
        ],
    }
    engine = Engine(load_config_from_dict(world, base_dir=Path(".")))
    engine.ctx.seed = 1
    engine.setup()
    engine.reset()
    return engine


def _lidar(engine):
    (lidar,) = [p for p in engine.plugins if type(p).__name__ == "LidarPlugin"]
    return lidar


def _scan(engine):
    """One noise-free scan, with the ray origin, world directions and bearings it was cast with."""
    lidar = _lidar(engine)
    lidar.range_stddev = 0.0
    engine.step()
    m, d = engine.ctx.model, engine.ctx.data
    sid = named(m, mujoco.mjtObj.mjOBJ_SITE, PREFIX + "rplidar_scan")
    dirs = lidar._build_directions() @ d.site_xmat[sid].reshape(3, 3).T
    bearings = lidar.angle_min + np.arange(lidar.num_rays) * lidar._angle_increment
    return lidar, lidar.latest, d.site_xpos[sid].copy(), dirs, bearings


def _vendor_laser_pose(engine):
    """World pose of `laser`, composed from the vendor joints on the robot's base_link."""
    m, d = engine.ctx.model, engine.ctx.data
    base = named(m, mujoco.mjtObj.mjOBJ_BODY, PREFIX + "base_link")
    rbase = d.xmat[base].reshape(3, 3)
    c, s = math.cos(LASER_YAW), math.sin(LASER_YAW)
    return d.xpos[base] + rbase @ LASER_IN_BASE, rbase @ np.array(
        [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    )


def test_the_scan_frame_is_the_vendor_chain():
    engine = _engine()
    try:
        m, d = engine.ctx.model, engine.ctx.data
        mujoco.mj_forward(m, d)
        pos, rot = _vendor_laser_pose(engine)
        for site in ("rplidar_scan", "rplidar_laser"):
            sid = named(m, mujoco.mjtObj.mjOBJ_SITE, PREFIX + site)
            np.testing.assert_allclose(d.site_xpos[sid], pos, atol=1e-9)
            np.testing.assert_allclose(d.site_xmat[sid].reshape(3, 3), rot, atol=1e-9)
    finally:
        engine.shutdown()


def test_the_static_tf_chain_is_published_in_the_robots_namespace():
    engine = _engine()
    try:
        endpoints = {(e.owner, e.name): e for e in engine.ctx.interface.all()}
        robot = endpoints[("rb", "frames")]
        device = endpoints[("rb.rplidar", "frames")]
        assert robot.namespace == device.namespace == NAMESPACE
        links = robot.backend["ros2"]["static_tf"] + device.backend["ros2"]["static_tf"]
        assert [(t["parent"], t["child"]) for t in links] == [
            ("body_link", "cover_link"),
            ("cover_link", "rplidar_link"),
            ("rplidar_link", "laser"),
        ]
        for tf, xyz in zip(links, [(0.0, 0.0, COVER_Z), MOUNT_XYZ, LASER_XYZ], strict=True):
            np.testing.assert_allclose(tf["translation"], xyz, atol=1e-9)
        np.testing.assert_allclose(np.abs(links[0]["rotation"]), [1, 0, 0, 0], atol=1e-9)
        np.testing.assert_allclose(np.abs(links[1]["rotation"]), [1, 0, 0, 0], atol=1e-9)
        np.testing.assert_allclose(np.abs(links[2]["rotation"]), [0, 0, 0, 1], atol=1e-9)  # yaw pi
    finally:
        engine.shutdown()


def test_topic_frame_and_scan_values():
    engine = _engine()
    try:
        scan = next(
            e for e in engine.ctx.interface.all() if e.name == "scan" and e.owner == "rb.rplidar"
        )
        assert scan.namespace == NAMESPACE
        assert scan.backend["ros2"]["topic"] == "scan"  # relative: <ns>/scan
        assert scan.backend["ros2"]["frame_id"] == "laser"
        assert "static_tf" not in scan.backend["ros2"]
        lidar = _lidar(engine)
        assert lidar.address == "rb.rplidar.lidar"
        # RPLIDAR C1 datasheet (the device) and Husarion's std_dev 0.02 override.
        assert (lidar.num_rays, lidar.range_min, lidar.range_max, lidar.rate_hz) == (
            500,
            0.05,
            12.0,
            10.0,
        )
        assert lidar.range_stddev == 0.02
        assert (lidar.angle_min, lidar.angle_max) == (-3.141592654, 3.141592654)
    finally:
        engine.shutdown()


def test_bearings_follow_the_vendor_laser_frame():
    """`laser` is yawed pi: bearing 0 looks backwards along the robot, +pi/2 to the robot's right."""
    engine = _engine()
    try:
        lidar, scan, origin, dirs, bearings = _scan(engine)
        _, rot = _vendor_laser_pose(engine)
        yaw = SPAWN[2]
        backward = -np.array([math.cos(yaw), math.sin(yaw), 0.0])
        right = np.array([math.sin(yaw), -math.cos(yaw), 0.0])
        for bearing, expect in ((0.0, backward), (math.pi / 2, right)):
            i = int(np.argmin(np.abs(np.angle(np.exp(1j * (bearings - bearing))))))
            assert abs(bearings[i] - bearing) < 1e-9, "a ray lands exactly on this bearing"
            np.testing.assert_allclose(dirs[i], expect, atol=1e-9)
            np.testing.assert_allclose(
                rot @ [math.cos(bearing), math.sin(bearing), 0], expect, atol=1e-9
            )
        # The wall behind the robot, straight down bearing 0: its face lies on the plane through
        # the room centre offset -HALF along world x, reached along `backward`.
        centre = np.array([SPAWN[0], SPAWN[1]])
        t = np.inf
        for axis in (0, 1):
            for sign in (-1.0, 1.0):
                if backward[axis] * sign > 1e-12:
                    t = min(t, (centre[axis] + sign * HALF - origin[axis]) / backward[axis])
        fwd = int(np.argmin(np.abs(bearings)))
        assert abs(scan.ranges[fwd] - t) < 1e-4, (scan.ranges[fwd], t)
    finally:
        engine.shutdown()


def _cast(engine, origin, dirs):
    m, d = engine.ctx.model, engine.ctx.data
    mount = named(m, mujoco.mjtObj.mjOBJ_BODY, PREFIX + "rplidar_mount")
    return raycast.cast(
        m,
        d,
        origin,
        dirs,
        cutoff=12.0,
        bodyexclude=mount,
        out=raycast.buffers(len(dirs), normals=True),
    )


def test_no_ray_starts_inside_the_robot_and_the_mount_never_returns():
    """The scan origin is outside every robot geom; the robot bodies that return are pinned.

    The body mesh is an open shell (Husarion's GLB, reduced), so a ray that grazes one of its open
    edges can meet a back face, and a ray between two of its faces can pass through to the room.
    Such a back-face ray is an isolated one: a neighbour meets the same geom from outside at nearly
    the same range. A scan origin buried in geometry would meet it from inside on a run of rays.
    """
    engine = _engine()
    try:
        m = engine.ctx.model
        lidar, scan, origin, dirs, _ = _scan(engine)
        hits = _cast(engine, origin, dirs)
        mount = named(m, mujoco.mjtObj.mjOBJ_BODY, PREFIX + "rplidar_mount")
        own = lidar._hits.geomid
        assert not np.any(m.geom_bodyid[own[own >= 0]] == mount), "a return from the C1's own mount"

        hit = hits.geomid >= 0
        assert hit.all(), "a closed room leaves no ray without a return"
        robot = np.array([m.geom_bodyid[g] != 0 for g in hits.geomid])
        facing = np.einsum("ij,ij->i", hits.normal, dirs)
        n = len(dirs)
        for i in np.flatnonzero(facing >= 0):
            assert robot[i], f"ray {i} meets a world geom from inside"
            assert any(
                hits.geomid[j] == hits.geomid[i]
                and facing[j] < 0
                and abs(hits.dist[j] - hits.dist[i]) < 0.005
                for j in ((i - 1) % n, (i + 1) % n)
            ), (
                f"ray {i} meets geom {hits.geomid[i]} from inside and no neighbour meets it from "
                f"outside: the scan origin is inside robot geometry"
            )
        assert int((facing >= 0).sum()) == 3, "the open-edge rays of the body mesh changed"

        bodies = {
            mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[g]))
            for g in hits.geomid[robot]
        }
        assert bodies == OUTSIDE_HIT_BODIES
        assert int(robot.sum()) == 36
        # Where the body returns, the published scan reads it -- pushed out to range_min where it is
        # inside the C1's 0.05 m blind zone -- and everywhere else it reads the room.
        np.testing.assert_allclose(scan.ranges, np.maximum(hits.dist, lidar.range_min), atol=1e-4)
        assert int((robot & (hits.dist < lidar.range_min)).sum()) == 6
    finally:
        engine.shutdown()
