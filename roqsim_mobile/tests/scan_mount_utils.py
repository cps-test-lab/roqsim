"""Checks for a 2D scanner a robot manifest mounts as a device model, in a room of known walls.

A robot manifest mounts a scanner with a nested ``spawn_sensor`` at the vendor's parent frame and joint
origin, directly on ``base_link`` or from a flattened vendor link its ``frames:`` declares. What such a
mount has to get right is the same on every robot: the scan frame sits at the vendor origin, a ray
reads the true distance to a wall, no ray starts inside robot geometry, the device skips its own
housing and nothing else, and the published TF chain is the vendor's. These helpers spawn, cast and
read each back from the compiled model, so a robot's test states only its vendor fixtures.

The robot is spawned through ``spawn_robot`` with a non-empty prefix and namespace, so a mount that
resolves a name without the carrier's prefix, or publishes outside its namespace, fails here rather
than only in a single-robot world.

Every expectation is computed independently of the code under test: the rotation is the URDF
fixed-axis convention written out here, and a wall distance is a ray-plane intersection.

Not ``conftest.py``, for the reason ``mobile_scene_utils`` gives.
"""

from __future__ import annotations

import mujoco
import numpy as np
from mobile_scene_utils import named

from roqsim import raycast
from roqsim.config import load_config_from_dict
from roqsim.engine import Engine
from roqsim.plugin import Plugin

#: Inner half-width of the square room :func:`spawn` builds around the spawn origin (m). Small enough
#: that the shortest-range scanner mounted here (the LDS-01, 3.5 m) still reaches every corner.
ROOM_HALF = 2.0
WALL_HEIGHT = 2.0
_WALL_THICKNESS = 0.1

#: The spawn's defaults, for a test that has no reason to choose its own.
OWNER = "robot"
PREFIX = "r_"
NAMESPACE = "rb"


class Room(Plugin):
    """Four walls whose inner faces are at +-``ROOM_HALF`` on x and y, taller than any base's scan."""

    def build(self, spec: mujoco.MjSpec, ctx) -> None:
        t, h, span = _WALL_THICKNESS / 2, WALL_HEIGHT / 2, ROOM_HALF + _WALL_THICKNESS
        for pos, size in (
            ([ROOM_HALF + t, 0.0, h], [t, span, h]),
            ([-ROOM_HALF - t, 0.0, h], [t, span, h]),
            ([0.0, ROOM_HALF + t, h], [span, t, h]),
            ([0.0, -ROOM_HALF - t, h], [span, t, h]),
        ):
            spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, pos=pos, size=size)


def spawn(
    model: str,
    mounts,
    *,
    owner: str = OWNER,
    prefix: str = PREFIX,
    namespace: str = NAMESPACE,
    disabled: tuple[str, ...] = (),
) -> Engine:
    """*model* spawned at the room's centre as *owner*, set up, reset and stepped once so every scanner
    has cast.

    *mounts* are the labels of the scanners the manifest mounts (a ``{label: ...}`` fixture works);
    each one's range noise and quantisation are switched off on the running scanner only (all three
    keys are live-writable), because the checks compare a published range with a wall's exact
    distance, while its config keeps the datasheet noise the manifest states. *disabled* are
    component addresses switched off (a camera that would need a GL context).
    """
    world = {
        "sim": {"timestep": 0.002},
        "components": [
            {f"{__name__}:Room": {}},
            {
                "spawn_robot": {"model": model, "prefix": prefix, "namespace": namespace},
                "name": owner,
            },
        ],
    }
    overrides = {"components": {a: {"enabled": False} for a in disabled}} if disabled else None
    engine = Engine(load_config_from_dict(world, overrides=overrides))
    # A test driving an Engine is the driver, and `ctx.seed` is driver-owned.
    engine.ctx.seed = 0
    engine.setup()
    engine.reset()
    for label in mounts:
        scanner = lidar(engine, f"{owner}.{label}")
        scanner.range_stddev = 0.0
        scanner.range_stddev_relative = 0.0
        scanner.range_resolution = 0.0
    engine.step()  # the rate gate starts open, so the first step casts
    return engine


# -- plugins and endpoints ---------------------------------------------------------------------


def mount_plugins(engine: Engine, owner: str) -> dict:
    """``{label: spawn_sensor plugin}`` for every device mounted on *owner*."""
    return {
        p.address.rsplit(".", 1)[-1]: p
        for p in engine.plugins
        if type(p).__name__ == "SpawnSensorPlugin" and p.entity == owner
    }


def lidar(engine: Engine, mount_address: str):
    return next(
        p for p in engine.plugins if type(p).__name__ == "LidarPlugin" and p.entity == mount_address
    )


def mount_body(engine: Engine, label: str, prefix: str = PREFIX) -> int:
    """The id of the device's own housing body, found by name rather than asked of the scanner."""
    return named(engine.ctx.model, mujoco.mjtObj.mjOBJ_BODY, f"{prefix}{label}_mount")


def endpoint(engine: Engine, name: str, owner: str):
    return next(e for e in engine.ctx.interface.all() if e.name == name and e.owner == owner)


def scan_endpoint(engine: Engine):
    """The one ``scan`` endpoint of a single-scanner robot."""
    (scan,) = [e for e in engine.ctx.interface.all() if e.name == "scan"]
    return scan


def static_tf(engine: Engine, owner: str, namespace: str = NAMESPACE) -> list[dict]:
    """The static transforms *owner* publishes on its ``frames`` endpoint, in the robot's namespace."""
    (frames,) = [e for e in engine.ctx.interface.all() if e.name == "frames" and e.owner == owner]
    assert frames.namespace == namespace
    return frames.backend["ros2"]["static_tf"]


# -- rotations and poses -----------------------------------------------------------------------


def urdf_rotation(rpy) -> np.ndarray:
    """The rotation a URDF ``<origin rpy>`` states: fixed axes x, y, z, i.e. ``Rz(y) Ry(p) Rx(r)``."""
    r, p, y = (float(v) for v in rpy)
    rx = np.array([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]])
    ry = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])
    rz = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])
    return rz @ ry @ rx


def chain(*joints) -> tuple[np.ndarray, np.ndarray]:
    """Position and rotation at the end of fixed joints ``(xyz, rpy)``, in the first joint's parent."""
    pos, rot = np.zeros(3), np.eye(3)
    for xyz, rpy in joints:
        pos = pos + rot @ np.asarray(xyz, dtype=np.float64)
        rot = rot @ urdf_rotation(rpy)
    return pos, rot


def quat_matrix(wxyz) -> np.ndarray:
    mat = np.zeros(9)
    mujoco.mju_quat2Mat(mat, np.asarray(wxyz, dtype=np.float64))
    return mat.reshape(3, 3)


def same_rotation(quat_wxyz, rot: np.ndarray) -> bool:
    """Whether a ``(w, x, y, z)`` quaternion is the rotation matrix *rot* (either sign)."""
    expected = np.zeros(4)
    mujoco.mju_mat2Quat(expected, np.ascontiguousarray(rot).reshape(-1))
    return abs(abs(float(np.dot(expected, quat_wxyz))) - 1.0) < 1e-9


def _pose(engine: Engine, objtype, name: str) -> tuple[np.ndarray, np.ndarray]:
    m, d = engine.ctx.model, engine.ctx.data
    ident = named(m, objtype, name)
    if objtype == mujoco.mjtObj.mjOBJ_SITE:
        return d.site_xpos[ident].copy(), d.site_xmat[ident].reshape(3, 3).copy()
    return d.xpos[ident].copy(), d.xmat[ident].reshape(3, 3).copy()


def pose_in_base(engine: Engine, site: str, prefix: str = PREFIX) -> tuple[np.ndarray, np.ndarray]:
    """A site's pose relative to the robot's ``base_link`` body."""
    pb, rb = _pose(engine, mujoco.mjtObj.mjOBJ_BODY, f"{prefix}base_link")
    ps, rs = _pose(engine, mujoco.mjtObj.mjOBJ_SITE, site)
    return rb.T @ (ps - pb), rb.T @ rs


def base_rotation(engine: Engine, prefix: str = PREFIX) -> np.ndarray:
    return _pose(engine, mujoco.mjtObj.mjOBJ_BODY, f"{prefix}base_link")[1]


# -- rays and walls ----------------------------------------------------------------------------


def world_rays(engine: Engine, scanner) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(origin, unit world directions, bearings)`` of the rays the scanner's latest scan was cast along."""
    scan = scanner.latest
    bearings = scan.angle_min + scan.angle_increment * np.arange(len(scan.ranges))
    local = np.stack([np.cos(bearings), np.sin(bearings), np.zeros_like(bearings)], axis=1)
    d = engine.ctx.data
    return (
        d.site_xpos[scanner._site_id].copy(),
        local @ d.site_xmat[scanner._site_id].reshape(3, 3).T,
        bearings,
    )


def forward_index(bearings: np.ndarray) -> int:
    """Index of the ray whose bearing, wrapped to (-pi, pi], is closest to 0 in the scan frame."""
    return int(np.argmin(np.abs(np.angle(np.exp(1j * np.asarray(bearings))))))


def wall_distance(origin: np.ndarray, direction: np.ndarray) -> float:
    """Distance along a horizontal *direction* from *origin* to the first inner wall face of :class:`Room`."""
    steps = [
        (np.sign(direction[i]) * ROOM_HALF - origin[i]) / direction[i]
        for i in (0, 1)
        if abs(direction[i]) > 1e-12
    ]
    return float(min(s for s in steps if s > 0))


def forward_range(engine: Engine, scanner) -> tuple[float, float]:
    """``(published range, true wall distance)`` of the ray nearest the scan's zero bearing."""
    origin, dirs, bearings = world_rays(engine, scanner)
    fwd = forward_index(bearings)
    return float(np.asarray(scanner.latest.ranges)[fwd]), wall_distance(origin, dirs[fwd])


def recast(
    engine: Engine, scanner, bodyexclude: int | None = None
) -> tuple[np.ndarray, raycast.RayHits]:
    """The scan's own rays cast again with normals: ``(world directions, hits)``.

    Skips *bodyexclude*, by default the body the scanner itself excludes.
    """
    origin, dirs, _ = world_rays(engine, scanner)
    hits = raycast.buffers(len(dirs), normals=True)
    raycast.cast(
        engine.ctx.model,
        engine.ctx.data,
        origin,
        dirs,
        cutoff=scanner.range_max,
        bodyexclude=scanner._bodyexclude if bodyexclude is None else bodyexclude,
        out=hits,
    )
    return dirs, hits


def robot_rays(engine: Engine, hits) -> np.ndarray:
    """Mask of the rays whose first surface belongs to a body other than the world (the walls)."""
    geomid = hits.geomid
    return (geomid >= 0) & (engine.ctx.model.geom_bodyid[np.maximum(geomid, 0)] != 0)


def robot_returns(engine: Engine, hits) -> tuple[set[str], set[str]]:
    """``(bodies, meshes)`` of the robot that the rays meet first, with their prefixed names."""
    m = engine.ctx.model
    geoms = [int(g) for g in hits.geomid[robot_rays(engine, hits)]]
    bodies = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[g])) for g in geoms}
    meshes = {
        mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MESH, int(m.geom_dataid[g]))
        for g in geoms
        if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH
    }
    return bodies, meshes


def robot_hits(engine: Engine, scanner, prefix: str) -> tuple[dict, dict]:
    """``(inside, outside)``: robot bodies the scan's rays hit first, each ``{body: [distances]}``.

    Names are unprefixed. A hit is from inside when the surface normal points along the ray
    (``normal . ray > 0``); the room's walls and the floor belong to the world body and are not
    counted.
    """
    m = engine.ctx.model
    dirs, hits = recast(engine, scanner)
    inside: dict[str, list[float]] = {}
    outside: dict[str, list[float]] = {}
    for i in np.flatnonzero(robot_rays(engine, hits)):
        body = int(m.geom_bodyid[hits.geomid[i]])
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, body).removeprefix(prefix)
        side = inside if float(hits.normal[i] @ dirs[i]) > 0 else outside
        side.setdefault(name, []).append(float(hits.dist[i]))
    return inside, outside


def uncovered_bearings(
    engine: Engine, scanners, radius: float, step_deg: float = 0.5
) -> np.ndarray:
    """World bearings (deg) about the base whose point at *radius* lies in no scanner's field.

    A point is in a scanner's field when its bearing in that scanner's site frame lies within the
    scan's ``[angle_min, angle_max]``. Geometry only: occlusion is what :func:`robot_hits` checks.
    """
    d = engine.ctx.data
    base = scanners[0].robot.split(".")[0]
    bid = named(engine.ctx.model, mujoco.mjtObj.mjOBJ_BODY, engine.ctx.entities.get(base).body)
    centre = d.xpos[bid]
    missed = []
    for phi in np.radians(np.arange(0.0, 360.0, step_deg)):
        seen = False
        for s in scanners:
            origin = d.site_xpos[s._site_id]
            point = centre + radius * np.array([np.cos(phi), np.sin(phi), 0.0])
            point[2] = origin[2]
            local = d.site_xmat[s._site_id].reshape(3, 3).T @ (point - origin)
            if s.angle_min <= np.arctan2(local[1], local[0]) <= s.angle_max:
                seen = True
                break
        if not seen:
            missed.append(np.degrees(phi))
    return np.asarray(missed)


# -- whole-mount assertions --------------------------------------------------------------------


def assert_mounts(engine: Engine, owner: str, mounts) -> None:
    """*owner*'s manifest mounts exactly *mounts*: device, parent frame, origin, scan frame and topic.

    *mounts* is ``{label: (model, frame_id, pos, rpy, topic)}``; the parent frame is ``base_link``.
    """
    found = mount_plugins(engine, owner)
    assert set(found) == set(mounts), f"mounted {sorted(found)}, the vendor ships {sorted(mounts)}"
    for label, (model, frame, pos, rpy, topic) in mounts.items():
        cfg = found[label].config
        assert cfg["model"] == model, f"{label}: {cfg['model']}"
        assert cfg["parent_frame"] == "base_link", f"{label}: {cfg['parent_frame']}"
        assert cfg["frame_id"] == frame, f"{label}: {cfg['frame_id']}"
        assert [float(v) for v in cfg["pos"]] == list(pos), f"{label}: {cfg['pos']}"
        assert [float(v) for v in cfg["rpy"]] == list(rpy), f"{label}: {cfg['rpy']}"
        assert lidar(engine, f"{owner}.{label}").config["topics"] == {"scan": topic}
    topics = [spec[4] for spec in mounts.values()]
    assert len(set(topics)) == len(topics), "two scanners on one topic publish over each other"


#: Where a device's data sheet puts its physical scan plane relative to its vendor scan frame, in that
#: frame (m). The rays start there; the scan is stamped in the vendor frame. A device not listed casts
#: from the vendor frame itself.
SCAN_PLANE_OFFSET = {
    # SICK data sheet S30B-2011BA, dimensional drawing: the plane 36.4 mm below the housing top.
    "sick_s300": (0.0, 0.0, -0.0041),
}


def assert_scan_frames(engine: Engine, prefix: str, mounts) -> None:
    """The device's frame site at the base pose composed with the vendor origin, and its scan site
    that pose composed with the device's declared scan-plane offset.

    *mounts* is ``{label: (model, frame_id, pos, rpy, topic)}`` with ``pos``/``rpy`` as the vendor's
    joint origin relative to ``base_link``.
    """
    pb, rb = _pose(engine, mujoco.mjtObj.mjOBJ_BODY, f"{prefix}base_link")
    for label, (model, frame, pos, rpy, _topic) in mounts.items():
        want_p, want_r = pb + rb @ np.asarray(pos, dtype=np.float64), rb @ urdf_rotation(rpy)
        offset = np.asarray(SCAN_PLANE_OFFSET.get(model, (0.0, 0.0, 0.0)), dtype=np.float64)
        for site, site_p in (
            (f"{prefix}{label}_{frame}", want_p),
            (f"{prefix}{label}_scan", want_p + want_r @ offset),
        ):
            got_p, got_r = _pose(engine, mujoco.mjtObj.mjOBJ_SITE, site)
            assert np.allclose(got_p, site_p, atol=1e-6), f"{site} at {got_p}, expected {site_p}"
            assert np.allclose(got_r, want_r, atol=1e-6), (
                f"{site} rotation {got_r}, vendor {want_r}"
            )


def assert_tf_chain(engine: Engine, owner: str, namespace: str, mounts) -> None:
    """Each mount publishes ``base_link -> <vendor scan frame>`` at the vendor origin, and the scan is
    stamped in that frame, on the vendor topic, under the robot's namespace, with no TF of its own."""
    for label, (_model, frame, pos, rpy, topic) in mounts.items():
        address = f"{owner}.{label}"
        tf = static_tf(engine, address, namespace)
        assert [(t["parent"], t["child"]) for t in tf] == [("base_link", frame)], tf
        assert np.allclose(tf[0]["translation"], pos, atol=1e-6)
        assert np.allclose(quat_matrix(tf[0]["rotation"]), urdf_rotation(rpy), atol=1e-6)
        scan = endpoint(engine, "scan", address)
        hints = scan.backend["ros2"]
        assert scan.namespace == namespace
        assert (hints["frame_id"], hints["topic"]) == (frame, topic)
        assert "static_tf" not in hints, "the mount owns the chain; the scan publishes none"
