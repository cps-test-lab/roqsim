"""TIAGo Pro's two SICK TIM571s, mounted as PAL mounts them: vendor frames, bearings, topics, returns.

Built through ``spawn_robot`` with a prefix and a namespace, in a closed room whose inner wall faces
are known planes. The robot's manifest mounts are taken from the manifest itself; the drive, torso and
arm controllers are left out (``default_plugins: false``), because the scanners need none of them and
this module then runs on its own.

Fixtures are PAL's numbers, not the model's:

* ``pal_omni_base`` @ 251ecc4cf57ed1870e84dcc135be917676e318f2,
  ``omni_base_description/urdf/base/base_sensors.urdf.xacro``: ``laser_height`` 0.13244 (:62); rear
  laser at (-0.27512, 0.18297), rpy (-180, 0, 135) deg, topic ``scan_rear_raw`` (:65-67); front laser
  at (0.27512, -0.18297), rpy (-180, 0, -45) deg, topic ``scan_front_raw`` (:70-72); both on
  ``base_link``, ``update_rate`` 10 (:28). ``urdf/base/base.urdf.xacro:64-69``: base_link's
  0.58 x 0.39 x 0.03 m collision box at z 0.132. ``meshes/base/base_link.stl``: across that box's
  z-range the visual body is a 0.500 x 0.318 m waist.
* ``pal_urdf_utils`` @ 775cdd6886296e6c00f17dbdfd9bcdd20e0e6622,
  ``urdf/laser/sick_tim571_laser.gazebo.xacro``: scan stamped in ``${name}_link`` (:25); 818 samples
  over 270 deg (:32) from min + 1 deg (:34) to max; 0.05-25 m (:39-40); stddev 0.01 (:46), PAL's
  value for the TiM571 on this robot, over the device's data sheet 0.02.

**The scanners look out through the base's waist.** PAL's collision box at the scan height encloses
both scan origins; PAL's visual body is recessed there, and both scanners stand outside that recess.
The model's box takes the visual waist's extent (a deviation from PAL's collision, recorded in the
port log), so every ray of both scans leaves the base and reads the room's wall at its true range, with
no robot geometry anywhere in either fan.
"""

from __future__ import annotations

import math
from pathlib import Path

import mujoco
import numpy as np
import pytest
import yaml

from roqsim import raycast
from roqsim.config import load_config_from_dict
from roqsim.context import SimContext
from roqsim.engine import Engine
from roqsim.models import resolve_model
from roqsim.plugin import Plugin
from roqsim.pose import rpy_to_quat

PREFIX = "tp_"
NAMESPACE = "tiago1"
DEG = math.pi / 180.0
LASER_HEIGHT = 0.13244  # base_sensors.urdf.xacro:62
#: label -> (xyz in base_link, rpy, scan frame, topic), base_sensors.urdf.xacro:65-72.
LASERS = {
    "base_front_laser": (
        (0.27512, -0.18297, LASER_HEIGHT),
        (-180 * DEG, 0.0, -45 * DEG),
        "base_front_laser_link",
        "scan_front_raw",
    ),
    "base_rear_laser": (
        (-0.27512, 0.18297, LASER_HEIGHT),
        (-180 * DEG, 0.0, 135 * DEG),
        "base_rear_laser_link",
        "scan_rear_raw",
    ),
}
#: base.urdf.xacro:64-69: PAL's scanner-band collision box, which encloses both scan origins (half
#: extents), and the half extents of base_link.stl's waist across that band, which the model's box
#: takes instead. Centre z is PAL's for both.
PAL_BAND_BOX_HALF = (0.29, 0.195, 0.015)
WAIST_HALF = (0.25, 0.159, 0.015)
BASE_BOX_Z = 0.132
#: base.urdf.xacro: base_link's explicit inertial mass, which the box change must leave alone.
BASE_LINK_MASS = 34.047

#: Inner wall faces at +-HALF around the spawn position.
HALF = 2.0
#: Off the origin and yawed, so neither pose nor bearing is a special value.
SPAWN = (0.3, -0.2, 0.4)
#: Every geom group but the collision one (3): the world's walls (0) and the vendor visuals (2).
VISUAL_AND_WORLD = np.array([1, 1, 1, 0, 1, 1], dtype=np.uint8)


class _Room(Plugin):
    def build(self, spec: mujoco.MjSpec, ctx: SimContext) -> None:
        t = 0.05
        for axis in (0, 1):
            for sign in (-1.0, 1.0):
                pos = [SPAWN[0], SPAWN[1], 1.0]
                pos[axis] += sign * (HALF + t)
                size = [HALF + 2 * t, HALF + 2 * t, 1.0]
                size[axis] = t
                spec.worldbody.add_geom(
                    name=f"room_wall_{axis}_{int(sign)}",
                    type=mujoco.mjtGeom.mjGEOM_BOX,
                    pos=pos,
                    size=size,
                )


def _manifest_mounts() -> list[dict]:
    manifest = resolve_model("tiago_pro").path.with_name("tiago_pro.manifest.yaml")
    mounts = [c for c in yaml.safe_load(manifest.read_text())["components"] if "spawn_sensor" in c]
    assert [m["name"] for m in mounts] == list(LASERS)
    return mounts


@pytest.fixture(scope="module")
def engine():
    world = {
        "sim": {},
        "components": [
            {f"{__name__}:_Room": {}},
            {
                "spawn_robot": {
                    "model": "tiago_pro",
                    "prefix": PREFIX,
                    "namespace": NAMESPACE,
                    "default_plugins": False,
                    "pose": {
                        "position": {"x": SPAWN[0], "y": SPAWN[1]},
                        "orientation": {"yaw": SPAWN[2]},
                    },
                },
                "name": "tp",
                "components": _manifest_mounts(),
            },
        ],
    }
    eng = Engine(load_config_from_dict(world, base_dir=Path(".")))
    eng.ctx.seed = 1
    eng.setup()
    eng.reset()
    for lidar in _lidars(eng).values():
        lidar.range_stddev = 0.0
    eng.step()
    yield eng
    eng.shutdown()


def _lidars(eng) -> dict:
    return {p.address.split(".")[1]: p for p in eng.plugins if type(p).__name__ == "LidarPlugin"}


def _id(m, objtype, name) -> int:
    ident = mujoco.mj_name2id(m, objtype, name)
    assert ident >= 0, f"{name!r} is not in the model"
    return ident


def _rot(rpy) -> np.ndarray:
    mat = np.zeros(9)
    mujoco.mju_quat2Mat(mat, np.asarray(rpy_to_quat(*rpy), dtype=float))
    return mat.reshape(3, 3)


def _vendor_pose(eng, label):
    """World pose of PAL's `<label>_link`: base_link's pose composed with the vendor joint."""
    m, d = eng.ctx.model, eng.ctx.data
    base = _id(m, mujoco.mjtObj.mjOBJ_BODY, PREFIX + "base_link")
    rbase = d.xmat[base].reshape(3, 3)
    xyz, rpy, _, _ = LASERS[label]
    return d.xpos[base] + rbase @ np.asarray(xyz), rbase @ _rot(rpy)


def _rays(eng, label):
    lidar = _lidars(eng)[label]
    m, d = eng.ctx.model, eng.ctx.data
    sid = _id(m, mujoco.mjtObj.mjOBJ_SITE, f"{PREFIX}{label}_scan")
    dirs = lidar._build_directions() @ d.site_xmat[sid].reshape(3, 3).T
    bearings = lidar.angle_min + np.arange(lidar.num_rays) * lidar._angle_increment
    return lidar, d.site_xpos[sid].copy(), dirs, bearings


def _room_range(origin, dirs) -> np.ndarray:
    centre = np.array([SPAWN[0], SPAWN[1]])
    out = np.full(len(dirs), np.inf)
    for axis in (0, 1):
        for sign in (-1.0, 1.0):
            with np.errstate(divide="ignore", invalid="ignore"):
                t = (centre[axis] + sign * HALF - origin[axis]) / dirs[:, axis]
            out = np.where((dirs[:, axis] * sign > 1e-12) & (t < out), t, out)
    return out


@pytest.mark.parametrize("label", list(LASERS))
def test_the_scan_frame_is_pals_joint_origin(engine, label):
    m, d = engine.ctx.model, engine.ctx.data
    pos, rot = _vendor_pose(engine, label)
    frame = LASERS[label][2]
    for site in ("scan", frame):
        sid = _id(m, mujoco.mjtObj.mjOBJ_SITE, f"{PREFIX}{label}_{site}")
        np.testing.assert_allclose(d.site_xpos[sid], pos, atol=1e-9)
        np.testing.assert_allclose(d.site_xmat[sid].reshape(3, 3), rot, atol=1e-9)
    assert rot[2, 2] == pytest.approx(-1.0), "mounted upside down"


@pytest.mark.parametrize("label", list(LASERS))
def test_the_static_tf_and_topic_are_pals(engine, label):
    xyz, rpy, frame, topic = LASERS[label]
    endpoints = {(e.owner, e.name): e for e in engine.ctx.interface.all()}
    tfs = endpoints[(f"tp.{label}", "frames")]
    assert tfs.namespace == NAMESPACE
    (tf,) = tfs.backend["ros2"]["static_tf"]
    assert (tf["parent"], tf["child"]) == ("base_link", frame)
    np.testing.assert_allclose(tf["translation"], xyz, atol=1e-9)
    assert abs(abs(float(np.dot(tf["rotation"], rpy_to_quat(*rpy)))) - 1.0) < 1e-9
    scan = endpoints[(f"tp.{label}", "scan")]
    assert scan.namespace == NAMESPACE
    assert scan.backend["ros2"]["topic"] == topic  # relative: <ns>/scan_*_raw
    assert scan.backend["ros2"]["frame_id"] == frame
    assert "static_tf" not in scan.backend["ros2"]


@pytest.mark.parametrize("label", list(LASERS))
def test_scan_values_are_pals_where_pal_differs_from_the_datasheet(engine, label):
    lidar = _lidars(engine)[label]
    assert lidar.address == f"tp.{label}.lidar"
    assert lidar.num_rays == 808  # -134 .. +135 deg at 1/3 deg, both edges sampled
    assert lidar.angle_min == pytest.approx(-134 * DEG)
    assert lidar.angle_max == pytest.approx(135 * DEG)
    assert lidar.angle_increment == pytest.approx(DEG / 3)
    # sick_tim's header, where the device's sick_scan_xd would publish 0.0 / 100.0.
    assert (lidar.range_min, lidar.range_max, lidar.rate_hz) == (0.05, 25.0, 10.0)
    assert (lidar.detection_min, lidar.detection_max) == (0.05, 25.0)
    assert (lidar.too_close, lidar.no_return) == (-np.inf, np.inf)
    # PAL's stddev 0.01, not the device's data sheet 0.02; zeroed on the instance by the fixture.
    assert lidar.config["range_stddev"] == 0.01
    assert lidar.exclude_body == "mount" and not lidar.emit_static_tf


@pytest.mark.parametrize("label", list(LASERS))
def test_bearings_follow_the_upside_down_vendor_frame(engine, label):
    """Upside down, a positive bearing turns clockwise seen from above: +90 deg is the scanner's right.

    Cast against the world and the vendor visuals, the wall points land where PAL's frame predicts, at
    their true range; the forward ray included.
    """
    _, origin, dirs, bearings = _rays(engine, label)
    _, rot = _vendor_pose(engine, label)
    forward, left = rot @ [1.0, 0.0, 0.0], np.cross([0.0, 0.0, 1.0], rot @ [1.0, 0.0, 0.0])
    for bearing, expect in ((0.0, forward), (90 * DEG, -left), (-90 * DEG, left)):
        i = int(np.argmin(np.abs(bearings - bearing)))
        assert abs(bearings[i] - bearing) < lidar_step(bearings) / 2
        expect_dir = math.cos(bearings[i] - bearing) * expect + math.sin(bearings[i] - bearing) * (
            np.cross(expect, [0.0, 0.0, 1.0])
        )
        np.testing.assert_allclose(dirs[i], expect_dir, atol=1e-9)

    m, d = engine.ctx.model, engine.ctx.data
    mount = _id(m, mujoco.mjtObj.mjOBJ_BODY, f"{PREFIX}{label}_mount")
    hits = raycast.cast(
        m,
        d,
        origin,
        dirs,
        cutoff=25.0,
        bodyexclude=mount,
        geomgroup=VISUAL_AND_WORLD,
        out=raycast.buffers(len(dirs), normals=True),
    )
    assert np.all(m.geom_bodyid[hits.geomid] == 0), "vendor visual geometry in the scan plane"
    np.testing.assert_allclose(hits.dist, _room_range(origin, dirs), atol=1e-4)
    fwd = int(np.argmin(np.abs(bearings)))
    assert hits.dist[fwd] == pytest.approx(_room_range(origin, dirs[fwd : fwd + 1])[0], abs=1e-4)


def lidar_step(bearings) -> float:
    return float(bearings[1] - bearings[0])


def test_the_scanner_band_box_is_the_visual_waist(engine):
    """base_link's scanner-band collision box has the visual waist's extent, and the mass is PAL's.

    Its z-range is PAL's; both scan origins lie outside it, in the open band around the waist.
    """
    m, d = engine.ctx.model, engine.ctx.data
    base = _id(m, mujoco.mjtObj.mjOBJ_BODY, PREFIX + "base_link")
    boxes = {
        tuple(np.round(m.geom_size[g], 6))
        for g in range(m.ngeom)
        if m.geom_bodyid[g] == base and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX
    }
    assert WAIST_HALF in boxes and PAL_BAND_BOX_HALF not in boxes, boxes
    assert m.body_mass[base] == pytest.approx(BASE_LINK_MASS)
    for label in LASERS:
        _, origin, _, _ = _rays(engine, label)
        local = d.xmat[base].reshape(3, 3).T @ (origin - d.xpos[base])
        assert abs(local[2] - BASE_BOX_Z) < WAIST_HALF[2], "the scan plane runs through the band"
        assert np.any(np.abs(local[:2]) > WAIST_HALF[:2]), f"{label} origin inside the waist box"


@pytest.mark.parametrize("label", list(LASERS))
def test_every_ray_leaves_the_base_and_reads_the_wall(engine, label):
    """Cast against every group, collision included: no robot geometry in the fan, from either side.

    The published scan (noise zeroed by the fixture) reads each ray's true wall range, the forward
    ray included.
    """
    m, d = engine.ctx.model, engine.ctx.data
    lidar, origin, dirs, bearings = _rays(engine, label)
    mount = _id(m, mujoco.mjtObj.mjOBJ_BODY, f"{PREFIX}{label}_mount")
    hits = raycast.cast(
        m,
        d,
        origin,
        dirs,
        cutoff=25.0,
        bodyexclude=mount,
        out=raycast.buffers(len(dirs), normals=True),
    )
    np.testing.assert_array_equal(hits.geomid, lidar._hits.geomid)
    robot = {
        mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[g]))
        for g in hits.geomid[hits.geomid >= 0]
        if m.geom_bodyid[g] != 0
    }
    assert robot == set(), f"robot bodies in the fan: {robot}"
    assert np.all(hits.geomid >= 0), "a ray missed the room"
    walls = _room_range(origin, dirs)
    np.testing.assert_allclose(lidar.latest.ranges, walls, atol=1e-4)
    fwd = int(np.argmin(np.abs(bearings)))
    assert lidar.latest.ranges[fwd] == pytest.approx(walls[fwd], abs=1e-4)
