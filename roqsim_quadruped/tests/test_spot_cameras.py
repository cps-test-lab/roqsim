"""Spot's five body camera stations: where they are, what they image, and what they publish.

Every expected value is a fixture naming its source, at the pins below. The robot is posed at its
standing stance without stepping, so the tests need no walking policy (``spot_locomotion`` is
switched off) and run anywhere a camera can render.

Sources:

* SDK  -- github.com/boston-dynamics/spot-sdk @ 04537ddafe0641d1751af3eb9e76cb785fc18594 (5.2.0)
* DRV  -- github.com/bdaiinstitute/spot_ros2 @ 949feb7b0d702d1923e5b5dd5c0a4a7728ddc88f
* SUBT -- github.com/osrf/subt @ 083c6a8acb0c25aa1a35e34517e628cd3b52ae77,
  submitted_models/bosdyn_spot/model.sdf: Boston Dynamics publishes no camera extrinsics or
  intrinsics (the robot reports them at runtime, SDK docs/concepts/robot_services.md), and this
  model's were estimated from real-robot data.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np
import pytest

from roqsim.config import load_config_from_dict
from roqsim.context import SimContext
from roqsim.engine import Engine
from roqsim.plugin import Plugin

SUBT = "SUBT model.sdf link camera_{station} (pose), converted to the optical frame"
ROLL = "SDK python/examples/get_image/get_image.py:21-27 ROTATION_ANGLE"

#: Each station's optical frame (z out of the lens, x right, y down) relative to `body`.
#: `level` is the rotation (degrees, counter-clockwise as scipy.ndimage.rotate takes it) that levels
#: the raw image; `view` the optical axis's azimuth from body +x towards +y and its elevation.
STATIONS = {
    "frontleft": {
        "position": (0.415, 0.037, -0.010),
        "quat_xyzw": (0.140744, 0.80838, -0.212978, 0.530427),
        "view": (-31.75, -20.28),
        "level": -78.0,
        "parent": "head",
        "source": (SUBT.format(station="frontleft"), ROLL),
    },
    "frontright": {
        "position": (0.415, -0.037, -0.010),
        "quat_xyzw": (-0.140744, 0.80838, 0.212978, 0.530427),
        "view": (31.75, -20.28),
        "level": -102.0,
        "parent": "head",
        "source": (SUBT.format(station="frontright"), ROLL),
    },
    "left": {
        "position": (-0.125, 0.120, 0.035),
        "quat_xyzw": (-0.774167, 0.0, 0.0, 0.632981),
        "view": (90.0, -11.46),
        "level": 0.0,
        "parent": "body",
        "source": (SUBT.format(station="left"), ROLL),
    },
    "right": {
        "position": (-0.125, -0.120, 0.035),
        "quat_xyzw": (0.774167, 0.0, 0.0, 0.632981),
        "view": (-90.0, -11.46),
        "level": 180.0,
        "parent": "body",
        "source": (SUBT.format(station="right"), ROLL),
    },
    "back": {
        "position": (-0.425, 0.0, 0.010),
        "quat_xyzw": (-0.569105, -0.569105, 0.419666, 0.419666),
        "view": (180.0, -17.19),
        "level": 0.0,
        "parent": "body",
        "source": (SUBT.format(station="back"), ROLL),
    },
}

#: `<station>_fisheye` in `<station>`: SUBT `<sensor name="<station>_fisheye_image"><pose>0 0.01 0`
#: in its camera link, whose +y is the optical -x.
FISHEYE_IN_STATION = (-0.01, 0.0, 0.0)

#: The streams of one station: size, K and wire format.
STREAMS = {
    # SUBT <station>_fisheye_image: 640x480, <lens><intrinsics> fx fy 257.487, cx 319.5, cy 239.5.
    "image": {"size": (640, 480), "k": (257.487, 257.487, 319.5, 239.5), "encoding": "mono8"},
    # SUBT <station>_depth: 424x240, fx fy 170.6, cx 211.5, cy 119.5.
    "depth": {"size": (424, 240), "k": (170.6, 170.6, 211.5, 119.5), "encoding": "16UC1"},
    # depth_image_proc's registration (DRV spot_driver/launch/spot_image_publishers.launch.py:35-57)
    # puts depth into the fisheye image: its size and K.
    "registered_depth": {
        "size": (640, 480),
        "k": (257.487, 257.487, 319.5, 239.5),
        "encoding": "16UC1",
    },
}
#: DRV: mono8 for greyscale bodies (src/conversions/decompress_images.cpp:72-78, README L56),
#: 16UC1 for depth (:87-93), in millimetres (SDK protos/bosdyn/api/image.proto:49-50).

#: DRV's topics and frames for a station (src/api/spot_image_sources.cpp:76-87,
#: src/images/images_middleware_handle.cpp:34-46, src/api/default_image_client.cpp:60,100).
#: The frame names are Spot's (src/object_sync/object_synchronizer.cpp:80-94); which one each
#: stream is stamped in is inferred from the image source's sensor frame.
DRV_TOPICS = {
    "image": ("camera/{s}/image", "{s}_fisheye"),
    "camera_info": ("camera/{s}/camera_info", "{s}_fisheye"),
    "depth": ("depth/{s}/image", "{s}"),
    "depth_camera_info": ("depth/{s}/camera_info", "{s}"),
    "registered_depth": ("depth_registered/{s}/image", "{s}_fisheye"),
    "registered_depth_camera_info": ("depth_registered/{s}/camera_info", "{s}_fisheye"),
}
#: DRV include/spot_driver/api/middleware_handle_base.hpp:14-20 and
#: src/images/images_middleware_handle.cpp:12.
DRV_IMAGE_QOS = {"reliability": "reliable", "durability": "transient_local", "depth": 10}
#: DRV image_rate default (include/spot_driver/interfaces/parameter_interface_base.hpp:62-79).
DRV_IMAGE_RATE_HZ = 15.0
#: DRV publish_compressed_images default false (same file).
DRV_COMPRESSED = False

#: The standing stance the policy holds (policy/spot.yaml default_angles) and the base height the
#: MJCF rests at in it.
STANCE = {"fl_hx": 0.1, "fr_hx": -0.1, "hl_hx": 0.1, "hr_hx": -0.1}
STANCE.update({"fl_hy": 0.9, "fr_hy": 0.9, "hl_hy": 1.1, "hr_hy": 1.1})
STANCE.update({f"{leg}_kn": -1.5 for leg in ("fl", "fr", "hl", "hr")})
STANDING_Z = 0.52


class _Markers(Plugin):
    """White boxes and poles placed in the world frame of the default room (Spot stands at the
    origin, facing +x)."""

    def build(self, spec: mujoco.MjSpec, ctx: SimContext) -> None:
        for i, (kind, pos, size) in enumerate(self.config.get("markers", [])):
            spec.worldbody.add_geom(
                name=f"marker_{i}",
                type=getattr(mujoco.mjtGeom, f"mjGEOM_{kind.upper()}"),
                pos=pos,
                size=size,
                rgba=[1, 1, 1, 1],
            )


def _spot(*markers) -> Engine:
    cfg = load_config_from_dict(
        {
            "sim": {"timestep": 0.002},
            "components": [
                {f"{__name__}:_Markers": {"markers": [list(m) for m in markers]}},
                {
                    "spawn_robot": {"model": "spot"},
                    "name": "robot",
                    "components": [{"spot_locomotion": {}, "enabled": False}],
                },
            ],
        }
    )
    engine = Engine(cfg)
    engine.ctx.seed = 1
    engine.setup()
    engine.reset()
    m, d = engine.ctx.model, engine.ctx.data
    for joint, angle in STANCE.items():
        d.qpos[m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, joint)]] = angle
    adr = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "base_free")]
    d.qpos[adr : adr + 7] = [0, 0, STANDING_Z, 1, 0, 0, 0]
    mujoco.mj_forward(m, d)
    return engine


@pytest.fixture(scope="module")
def spot():
    engine = _spot()
    yield engine
    engine.shutdown()


def _endpoints(engine: Engine, station: str) -> dict:
    return {
        ep.name: ep
        for ep in engine.ctx.interface.all()
        if ep.producer == f"robot.{station}.spot_body_camera"
    }


def _static_tf(engine: Engine) -> dict[str, tuple[str, np.ndarray, np.ndarray]]:
    """child -> (parent, translation, rotation matrix) over every published static transform."""
    out = {}
    for ep in engine.ctx.interface.all():
        if ep.name != "frames":
            continue
        for t in ep.read().transforms:
            mat = np.zeros(9)
            mujoco.mju_quat2Mat(mat, np.asarray(t.rotation, dtype=float))
            out[t.child] = (t.parent, np.asarray(t.translation, dtype=float), mat.reshape(3, 3))
    return out


def _in_body(tf: dict, frame: str) -> tuple[np.ndarray, np.ndarray]:
    """*frame*'s pose in `body`, composed from the published static transforms."""
    pos, rot = np.zeros(3), np.eye(3)
    while frame != "body":
        parent, t, r = tf[frame]
        pos, rot = r @ pos + t, r @ rot
        frame = parent
    return pos, rot


def _quat_matrix(xyzw) -> np.ndarray:
    x, y, z, w = xyzw
    q = np.array([w, x, y, z], dtype=float)
    mat = np.zeros(9)
    mujoco.mju_quat2Mat(mat, q / np.linalg.norm(q))  # the fixture is rounded to 6 decimals
    return mat.reshape(3, 3)


def _plugin(engine: Engine, station: str):
    return next(p for p in engine.plugins if p.address == f"robot.{station}.spot_body_camera")


def _capture(engine: Engine, station: str):
    """Every stream of *station*, rendered now. The stations render on demand, so each stream is
    claimed first, as an in-process consumer would."""
    plugin = _plugin(engine, station)
    eps = _endpoints(engine, station)
    names = ("image", "depth", "registered_depth")
    for name in names:
        eps[name].claim()
    plugin.capture(engine.ctx)
    frames = {name: eps[name].read() for name in names}
    for name in names:
        eps[name].release()
    return frames


def _camera(engine: Engine, station: str, which: str) -> int:
    plugin = _plugin(engine, station)
    return plugin._cam_id if which == "fisheye" else plugin._depth_cam_id


def _site(engine: Engine, station: str, frame: str) -> int:
    name = _plugin(engine, station)._prefix + frame
    return mujoco.mj_name2id(engine.ctx.model, mujoco.mjtObj.mjOBJ_SITE, name)


def _base(engine: Engine) -> int:
    prefix = engine.ctx.entities.get("robot").meta.get("prefix", "")
    return mujoco.mj_name2id(engine.ctx.model, mujoco.mjtObj.mjOBJ_BODY, prefix + "base_link")


# -- the frame tree ---------------------------------------------------------------------------


def test_body_is_base_link_and_head_carries_the_front_pair(spot):
    """spot_description urdf/spot_macro.xacro:37-41 @ 156d180: base_link is an identity child of
    body, so `body` coincides with this MJCF's base_link; `head` is the parent of the front pair in
    Spot's frame tree (DRV object_synchronizer.cpp:80-94)."""
    tf = _static_tf(spot)
    parent, t, r = tf["body"]
    assert parent == "base_link"
    np.testing.assert_allclose(t, 0.0, atol=1e-9)
    np.testing.assert_allclose(r, np.eye(3), atol=1e-9)
    assert tf["head"][0] == "body"


@pytest.mark.parametrize("station", sorted(STATIONS))
def test_each_station_is_where_the_fixture_puts_it_relative_to_body(spot, station):
    want = STATIONS[station]
    tf = _static_tf(spot)
    assert tf[station][0] == want["parent"]
    pos, rot = _in_body(tf, station)
    np.testing.assert_allclose(pos, want["position"], atol=1e-6)
    # The angle of the relative rotation, from its sine half rather than an acos of the trace, which
    # loses all precision this close to identity.
    rel = rot.T @ _quat_matrix(want["quat_xyzw"])
    skew = np.array([rel[2, 1] - rel[1, 2], rel[0, 2] - rel[2, 0], rel[1, 0] - rel[0, 1]])
    angle = math.degrees(math.atan2(np.linalg.norm(skew) / 2, (np.trace(rel) - 1) / 2))
    assert angle < 0.01, f"{station}: off by {angle:.4f} deg"


@pytest.mark.parametrize("station", sorted(STATIONS))
def test_the_fisheye_frame_hangs_from_its_station(spot, station):
    parent, t, r = _static_tf(spot)[f"{station}_fisheye"]
    assert parent == station
    np.testing.assert_allclose(t, FISHEYE_IN_STATION, atol=1e-9)
    np.testing.assert_allclose(r, np.eye(3), atol=1e-9)


@pytest.mark.parametrize("station", sorted(STATIONS))
def test_each_station_looks_where_the_fixture_says(spot, station):
    """The front pair looks across, each to the far side; every station looks down."""
    _, rot = _in_body(_static_tf(spot), station)
    axis = rot[:, 2]
    azimuth = math.degrees(math.atan2(axis[1], axis[0]))
    elevation = math.degrees(math.asin(axis[2]))
    want_az, want_el = STATIONS[station]["view"]
    assert abs((azimuth - want_az + 180) % 360 - 180) < 0.05
    assert elevation == pytest.approx(want_el, abs=0.05)


@pytest.mark.parametrize("station", sorted(STATIONS))
def test_the_sdk_rotation_angle_levels_the_raw_image(spot, station):
    """World-up, seen from the station, points where rotating by ROTATION_ANGLE turns it upright."""
    _, rot = _in_body(_static_tf(spot), station)
    up = rot.T @ np.array([0.0, 0.0, 1.0])  # body is level standing: its +z is world up
    level = math.degrees(math.atan2(up[0], -up[1]))
    assert abs((level - STATIONS[station]["level"] + 180) % 360 - 180) < 0.01


@pytest.mark.parametrize("station", sorted(STATIONS))
def test_each_camera_renders_from_the_frame_its_stream_is_stamped_in(spot, station):
    """MuJoCo's camera is the published optical frame turned into its own convention."""
    d = spot.ctx.data
    for which, site in (("fisheye", f"{station}_fisheye"), ("depth", station)):
        c, s = _camera(spot, station, which), _site(spot, station, site)
        assert c >= 0 and s >= 0
        np.testing.assert_allclose(d.cam_xpos[c], d.site_xpos[s], atol=1e-9)
        np.testing.assert_allclose(
            d.cam_xmat[c].reshape(3, 3),
            d.site_xmat[s].reshape(3, 3) @ np.diag([1, -1, -1]),
            atol=1e-9,
        )


# -- what each station publishes --------------------------------------------------------------


@pytest.mark.parametrize("station", sorted(STATIONS))
def test_topics_frames_qos_and_rate_are_spot_ros2s(spot, station):
    eps = _endpoints(spot, station)
    for name, (topic, frame) in DRV_TOPICS.items():
        ros = eps[name].backend["ros2"]
        assert ros["topic"] == topic.format(s=station), name
        assert ros["frame_id"] == frame.format(s=station), name
        assert ros["qos"] == DRV_IMAGE_QOS, name
        assert eps[name].rate_hz == DRV_IMAGE_RATE_HZ, name
    assert ("image_compressed" in eps) is DRV_COMPRESSED
    assert "depth_compressed" not in eps  # spot_ros2 has no compressed depth


@pytest.mark.parametrize("station", sorted(STATIONS))
def test_intrinsics_and_sizes_are_the_fixtures(spot, station):
    eps = _endpoints(spot, station)
    infos = {
        "image": eps["camera_info"].read(),
        "depth": eps["depth_camera_info"].read(),
        "registered_depth": eps["registered_depth_camera_info"].read(),
    }
    for stream, info in infos.items():
        want = STREAMS[stream]
        assert (info.width, info.height) == want["size"], stream
        assert (info.fx, info.fy, info.cx, info.cy) == pytest.approx(want["k"], abs=1e-3), stream
        assert list(info.d) == [0.0] * 5, stream  # DRV default_image_client.cpp:57-65


@pytest.mark.parametrize("station", sorted(STATIONS))
def test_each_stream_renders_at_its_size_and_encoding(spot, station):
    frames = _capture(spot, station)
    for stream, frame in frames.items():
        w, h = STREAMS[stream]["size"]
        assert frame.data.shape == (h, w), stream
        assert frame.encoding == STREAMS[stream]["encoding"], stream
        assert frame.stamp == pytest.approx(spot.ctx.sim_time)
    # Millimetres with 0 for no return: the floor is in view of every station, within 4 m.
    depth = frames["depth"].data
    assert (depth > 0).any() and depth.max() <= 4000


# -- what each station sees -------------------------------------------------------------------


def _project(engine: Engine, station: str, world_point) -> tuple[float, float, float]:
    """(u, v, z) of *world_point* in the station's fisheye image, through the published TF and K."""
    d = engine.ctx.data
    base = _base(engine)
    pos, rot = _in_body(_static_tf(engine), f"{station}_fisheye")
    body_rot = d.xmat[base].reshape(3, 3)
    cam_pos, cam_rot = d.xpos[base] + body_rot @ pos, body_rot @ rot
    p = cam_rot.T @ (np.asarray(world_point, dtype=float) - cam_pos)
    fx, fy, cx, cy = STREAMS["image"]["k"]
    return fx * p[0] / p[2] + cx, fy * p[1] / p[2] + cy, p[2]


def _in_image(u: float, v: float, z: float) -> bool:
    w, h = STREAMS["image"]["size"]
    return z > 0 and 0 <= u < w and 0 <= v < h


def test_the_front_pair_looks_across_each_other():
    """A marker ahead and to the right is in frontleft's view and not in frontright's, and the
    mirror: the trap of mounting each front camera to look out on its own side."""
    right_marker, left_marker = (1.4, -0.9, 0.15), (1.4, 0.9, 0.15)
    engine = _spot(
        ("box", right_marker, (0.15, 0.15, 0.15)), ("box", left_marker, (0.15, 0.15, 0.15))
    )
    assert _in_image(*_project(engine, "frontleft", right_marker))
    assert not _in_image(*_project(engine, "frontleft", left_marker))
    assert _in_image(*_project(engine, "frontright", left_marker))
    assert not _in_image(*_project(engine, "frontright", right_marker))
    engine.shutdown()


@pytest.mark.parametrize("station", sorted(STATIONS))
def test_a_marker_renders_where_tf_and_camera_info_project_it(station):
    """The render agrees with the published pose and intrinsics: a box placed on the station's
    optical axis, 1.5 m out, shows up as a bright patch at the projected pixel and, in registered
    depth, at the range the geometry gives."""
    probe = _spot()
    pos, rot = _in_body(_static_tf(probe), f"{station}_fisheye")
    probe.shutdown()
    centre = np.array([0, 0, STANDING_Z]) + pos + rot[:, 2] * 1.5
    engine = _spot(("box", tuple(centre), (0.12, 0.12, 0.12)))
    u, v, z = _project(engine, station, centre)
    assert _in_image(u, v, z)
    frames = _capture(engine, station)
    patch = frames["image"].data[int(v) - 3 : int(v) + 4, int(u) - 3 : int(u) + 4]
    assert patch.mean() > 150, f"{station}: the marker is not where the camera_info projects it"
    near_face = frames["registered_depth"].data[int(v), int(u)] / 1000.0
    assert 1.5 - 0.2 < near_face < 1.5, station
    engine.shutdown()


@pytest.mark.parametrize("station", sorted(STATIONS))
def test_rotating_by_the_sdk_angle_stands_a_pole_upright(station):
    """A vertical pole on the station's line of sight, after the raw image is rotated by
    ROTATION_ANGLE, runs up and down the picture: the image arrives rotated as the real one does."""
    pil = pytest.importorskip("PIL.Image")
    probe = _spot()
    pos, rot = _in_body(_static_tf(probe), f"{station}_fisheye")
    probe.shutdown()
    ahead = rot[:, 2].copy()
    ahead[2] = 0.0
    ahead /= np.linalg.norm(ahead)
    foot = np.array([pos[0], pos[1], 0.0]) + ahead * 1.6
    empty = _spot()
    without = _capture(empty, station)["image"].data.astype(int)
    empty.shutdown()
    engine = _spot(("cylinder", (foot[0], foot[1], 0.5), (0.03, 0.5, 0)))
    image = _capture(engine, station)["image"].data.astype(int)
    engine.shutdown()
    pole = (np.abs(image - without) > 20).astype(np.uint8) * 255
    level = np.asarray(pil.fromarray(pole).rotate(STATIONS[station]["level"], expand=True))
    ys, xs = np.nonzero(level > 127)
    assert len(xs) > 50, f"{station}: the pole is not in view"
    cov = np.cov(np.stack([xs, ys]).astype(float))
    major = np.linalg.eigh(cov)[1][:, -1]
    tilt = math.degrees(math.atan2(abs(major[0]), abs(major[1])))
    assert tilt < 15.0, f"{station}: the levelled pole leans {tilt:.1f} deg"


#: How far a station may sit from the body shell: on its housing, in the window recess.
ON_HOUSING_M = 0.01
#: spot_body_camera's clip_near: robot geometry nearer than this is not imaged.
CLIP_NEAR_M = 0.1


def _robot_hit(m, d, origin, direction, robot: int) -> float:
    """Distance along *direction* to the first rendered geom of *robot*'s base, else inf. What a
    camera sees through is what renders: the visual groups 0-2, not the convex collision hull
    (group 3), a simplification the real camera windows are cut out of."""
    visible = np.array([1, 1, 1, 0, 0, 0], dtype=np.uint8)
    geomid = np.array([-1], dtype=np.int32)
    origin = np.asarray(origin, dtype=float).copy()
    travelled = 0.0
    while True:
        dist = mujoco.mj_ray(m, d, origin, direction, visible, 1, -1, geomid)
        if dist < 0:
            return math.inf
        if m.geom_bodyid[geomid[0]] == robot:
            return travelled + dist
        step = dist + 1e-6
        origin += direction * step
        travelled += step


@pytest.mark.parametrize("station", sorted(STATIONS))
def test_each_station_sits_on_its_housing_and_looks_out(spot, station):
    """Each camera is at the body shell -- within a centimetre of it, on the housing -- and nothing
    of the body is imaged along its axis past the near clip."""
    m, d = spot.ctx.model, spot.ctx.data
    robot = _base(spot)
    for which in ("fisheye", "depth"):
        c = _camera(spot, station, which)
        origin = d.cam_xpos[c]
        axis = -d.cam_xmat[c].reshape(3, 3)[:, 2]
        ahead, behind = (
            _robot_hit(m, d, origin, axis, robot),
            _robot_hit(m, d, origin, -axis, robot),
        )
        assert min(ahead, behind) < ON_HOUSING_M, f"{station} {which}: not on the shell"
        if ahead < math.inf:
            assert ahead < CLIP_NEAR_M, f"{station} {which}: the body is in view {ahead:.3f} m out"
