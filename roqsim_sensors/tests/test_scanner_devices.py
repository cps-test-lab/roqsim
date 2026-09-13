"""The nine scanner device models: each scans from its vendor frame, past its own housing, as declared.

Every device is mounted the way a world mounts it -- `spawn_sensor` with the manifest's own lidar --
inside a closed room whose walls are at known planes, so each ray's true range is known analytically.
What this pins, per device:

* the scan reads the walls at their true range from the scan site, on every ray -- the forward one
  included, so an offset site shows up as a range error;
* no ray stops on the device's own `mount` body, and no ray's first surface is met from inside a geom
  (hit normal against the ray), which is what a scan origin buried in geometry `exclude_body` does not
  cover would look like;
* the scan's ray count, field and header are the manifest's, its last ray sits at `angle_max`, and the
  manifest's `fov:` sector is its field and physical detection limits;
* a too-close surface and an empty bearing publish the values the manifest declares for them, which
  are what the device's driver publishes (pinned in ``DECLARED_OUTPUTS``);
* the `scan` site in the MJCF sits where the manifest's `frames:` entry says, so the frame published
  for the scan and the point it is cast from cannot drift apart.

Needs nothing from roqsim_mobile: the room is a plugin defined here.
"""

from __future__ import annotations

import mujoco
import numpy as np
import pytest
import yaml
from roqsim_sensors.models import MODELS_DIR
from roqsim_sensors.plugins.lidar import LidarPlugin

from roqsim import raycast
from roqsim.config import load_config_from_dict
from roqsim.context import SimContext
from roqsim.engine import Engine
from roqsim.frames import parse_frames, substitute
from roqsim.manifest import manifest_frame_id
from roqsim.plugin import Plugin, PluginError
from roqsim.pose import rpy_to_quat

DEVICES = [
    "hokuyo_ust",
    "lds01",
    "rplidar_a1",
    "rplidar_c1",
    "sick_lms1xx",
    "sick_microscan3",
    "sick_s300",
    "sick_tim571",
    "velodyne_vlp16",
]

#: Each vendor's default scan-frame name; the TiM571's macro takes its link name from the robot.
VENDOR_FRAME = {
    "hokuyo_ust": "lidar2d_0_laser",
    "sick_lms1xx": "lidar2d_0_laser",
    "lds01": "base_scan",
    "rplidar_a1": "rplidar_link",
    "rplidar_c1": "laser",
    "sick_microscan3": "lidar_1_link",
    "sick_s300": "lidar_1_link",
    "sick_tim571": None,
    "velodyne_vlp16": "velodyne",
}

#: `(too_close, no_return)` each device declares, from its driver (see the manifests' citations).
#: Every value its driver's source leaves unverified is the REP 117 default.
DECLARED_OUTPUTS = {
    "hokuyo_ust": ("0.004", "65.533"),
    "lds01": ("-inf", "+inf"),
    "rplidar_a1": ("-inf", "+inf"),
    "rplidar_c1": ("raw", "+inf"),
    "sick_lms1xx": ("-inf", "+inf"),
    "sick_microscan3": ("-inf", "+inf"),
    "sick_s300": ("-inf", "+inf"),
    "sick_tim571": ("-inf", "+inf"),
    "velodyne_vlp16": ("+inf", "+inf"),
}
_WORDS = {"-inf": -np.inf, "+inf": np.inf, "nan": np.nan}

FRAME_ID = "scanner_frame"
#: Inner faces of the room walls sit at x = +-HALF and y = +-HALF.
HALF = 2.0
WALL_TOP = 2.0
#: Off-centre and yawed, so neither the pose nor the bearing of a ray is a special value.
MOUNT_POS = [0.3, -0.2, 0.8]
MOUNT_RPY = [0.0, 0.0, 0.4]


class _Room(Plugin):
    """Four walls around the origin, inner faces at +-HALF, floor to WALL_TOP."""

    def build(self, spec: mujoco.MjSpec, ctx: SimContext) -> None:
        t = 0.05
        for axis in (0, 1):
            for sign in (-1.0, 1.0):
                pos = [0.0, 0.0, WALL_TOP / 2]
                pos[axis] = sign * (HALF + t)
                size = [HALF + 2 * t, HALF + 2 * t, WALL_TOP / 2]
                size[axis] = t
                spec.worldbody.add_geom(
                    name=f"wall_{axis}_{int(sign)}",
                    type=mujoco.mjtGeom.mjGEOM_BOX,
                    pos=pos,
                    size=size,
                )


def _manifest(device: str) -> dict:
    return yaml.safe_load((MODELS_DIR / device / f"{device}.manifest.yaml").read_text())


def _lidar_config(device: str) -> dict:
    (lidar,) = [c["lidar"] for c in _manifest(device)["components"] if "lidar" in c]
    return lidar


def _engine(device: str, **spawn) -> Engine:
    cfg = {
        "sim": {},
        "plugins": [
            {f"{__name__}:_Room": {}},
            {
                # A key passed as None is left out, so a test can mount without it.
                "spawn_sensor": {
                    key: value
                    for key, value in {
                        "model": device,
                        "pos": MOUNT_POS,
                        "rpy": MOUNT_RPY,
                        "frame_id": FRAME_ID,
                        **spawn,
                    }.items()
                    if value is not None
                },
                "name": device,
            },
        ],
    }
    engine = Engine(load_config_from_dict(cfg))
    engine.ctx.seed = 1  # this test is the driver; the noise test below draws
    engine.setup()
    return engine


def _lidar(engine: Engine) -> LidarPlugin:
    (lidar,) = [p for p in engine.plugins if isinstance(p, LidarPlugin)]
    return lidar


def _room_range(origin: np.ndarray, dirs: np.ndarray) -> np.ndarray:
    """Distance along each horizontal ray from *origin* to the first inner wall face."""
    with np.errstate(divide="ignore", invalid="ignore"):
        tx = np.where(
            dirs[:, 0] > 0, (HALF - origin[0]) / dirs[:, 0], (-HALF - origin[0]) / dirs[:, 0]
        )
        ty = np.where(
            dirs[:, 1] > 0, (HALF - origin[1]) / dirs[:, 1], (-HALF - origin[1]) / dirs[:, 1]
        )
    tx = np.where(np.abs(dirs[:, 0]) < 1e-12, np.inf, tx)
    ty = np.where(np.abs(dirs[:, 1]) < 1e-12, np.inf, ty)
    return np.minimum(tx, ty)


def _world_dirs(engine: Engine, lidar: LidarPlugin) -> tuple[np.ndarray, np.ndarray]:
    d = engine.ctx.data
    sid = mujoco.mj_name2id(engine.ctx.model, mujoco.mjtObj.mjOBJ_SITE, "scan")
    rot = d.site_xmat[sid].reshape(3, 3)
    return d.site_xpos[sid].copy(), lidar._build_directions() @ rot.T


@pytest.mark.parametrize("device", DEVICES)
def test_scan_reads_the_walls_at_their_true_range(device):
    engine = _engine(device)
    lidar = _lidar(engine)
    # Noise and quantisation off for this one check (live-writable keys): the range must be exact.
    lidar.range_stddev = 0.0
    lidar.range_stddev_relative = 0.0
    lidar.range_resolution = 0.0
    engine.reset()
    engine.step()
    scan = lidar.latest
    origin, dirs = _world_dirs(engine, lidar)
    assert abs(dirs[:, 2]).max() < 1e-9, "the device is upright, so its scan plane is horizontal"
    expected = _room_range(origin, dirs)
    assert np.all(np.isfinite(scan.ranges)), "a closed room leaves no ray without a return"
    np.testing.assert_allclose(scan.ranges, expected, atol=1e-4)
    # The forward ray (bearing closest to 0 in the scan frame) on its own, for a readable failure.
    bearings = np.linspace(lidar.angle_min, lidar.angle_max, lidar.num_rays)
    fwd = int(np.argmin(np.abs(np.angle(np.exp(1j * bearings)))))
    assert abs(scan.ranges[fwd] - expected[fwd]) < 1e-4


@pytest.mark.parametrize("device", DEVICES)
def test_no_ray_stops_on_the_housing_or_starts_inside_a_geom(device):
    engine = _engine(device)
    lidar = _lidar(engine)
    engine.reset()
    engine.step()
    m, d = engine.ctx.model, engine.ctx.data
    mount = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "mount")
    assert mount >= 0 and lidar._bodyexclude == mount, "exclude_body resolves to the device's mount"

    origin, dirs = _world_dirs(engine, lidar)
    hits = raycast.cast(
        m,
        d,
        origin,
        dirs,
        cutoff=lidar.range_max,
        bodyexclude=mount,
        out=raycast.buffers(len(dirs), normals=True),
    )
    hit = hits.geomid >= 0
    assert hit.all()
    assert not np.any(m.geom_bodyid[hits.geomid[hit]] == mount)
    # A surface met from outside faces the ray; one met from inside faces away from it.
    facing = np.einsum("ij,ij->i", hits.normal[hit], dirs[hit])
    assert np.all(facing < 0), f"{int((facing >= 0).sum())} ray(s) start inside a geom"
    # The rays the plugin itself cast agree: none of them stopped on the housing either.
    own = lidar._hits.geomid
    assert not np.any(m.geom_bodyid[own[own >= 0]] == mount)


@pytest.mark.parametrize("device", DEVICES)
def test_scan_window_is_the_manifest_and_fov_matches_it(device):
    cfg = _lidar_config(device)
    engine = _engine(device)
    lidar = _lidar(engine)
    engine.reset()
    engine.step()
    scan = lidar.latest
    assert len(scan.ranges) == cfg["rays"] == lidar.num_rays
    assert scan.angle_min == pytest.approx(cfg["angle_min"])
    assert scan.angle_max == pytest.approx(cfg["angle_max"])
    assert scan.angle_increment == pytest.approx(
        (cfg["angle_max"] - cfg["angle_min"]) / (cfg["rays"] - 1)
    )
    last = lidar._build_directions()[-1]
    np.testing.assert_allclose(
        last, [np.cos(cfg["angle_max"]), np.sin(cfg["angle_max"]), 0.0], atol=1e-9
    )
    assert scan.range_min == pytest.approx(cfg["range_min"])
    assert scan.range_max == pytest.approx(cfg["max_range"])
    near = cfg.get("detection_min", cfg["range_min"])
    far = cfg.get("detection_max", cfg["max_range"])
    assert (lidar.detection_min, lidar.detection_max) == pytest.approx((near, far))
    assert lidar.rate_hz == pytest.approx(cfg["rate_hz"])
    assert lidar.range_stddev == pytest.approx(cfg["range_stddev"])
    assert lidar.frame_id == FRAME_ID
    assert lidar.exclude_body == "mount" and not lidar.emit_static_tf

    fov = _manifest(device)["fov"]
    assert (fov["near"], fov["far"]) == pytest.approx((near, far))
    # Each ray stands for one increment of azimuth: rays that fill the turn are a full-turn sector.
    span = cfg["angle_max"] - cfg["angle_min"]
    if span + scan.angle_increment >= 2 * np.pi - 1e-9:
        assert fov["h_min"] == pytest.approx(cfg["angle_min"])
        assert fov["h_max"] - fov["h_min"] == pytest.approx(2 * np.pi)
    else:
        assert (fov["h_min"], fov["h_max"]) == pytest.approx((cfg["angle_min"], cfg["angle_max"]))
    assert fov["v_min"] == fov["v_max"] == 0.0


class _Plate(Plugin):
    """A lidar site at the origin and one small plate straight ahead of it, 1 cm away."""

    DISTANCE = 0.01

    def build(self, spec: mujoco.MjSpec, ctx: SimContext) -> None:
        spec.worldbody.add_site(name="lidar", pos=[0.0, 0.0, 0.5])
        t = 0.002
        spec.worldbody.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[self.DISTANCE + t, 0.0, 0.5],
            size=[t, self.DISTANCE, 0.05],
        )


def _declared(value: str) -> float | None:
    """The published value a declared word or number stands for; ``None`` for ``raw``."""
    if value == "raw":
        return None
    return _WORDS[value] if value in _WORDS else float(value)


@pytest.mark.parametrize("device", DEVICES)
def test_the_device_publishes_its_declared_too_close_and_no_return(device):
    cfg = _lidar_config(device)
    assert (str(cfg["too_close"]), str(cfg["no_return"])) == DECLARED_OUTPUTS[device]
    near = cfg.get("detection_min", cfg["range_min"])
    assert _Plate.DISTANCE < near, "the plate must be inside every device's detection_min"
    lidar_cfg = {
        **cfg,
        "site": "lidar",
        "frame_id": "scan",
        "exclude_body": "",
        "emit_static_tf": True,
        "range_stddev": 0.0,
        "range_stddev_relative": 0.0,
        "range_resolution": 0.0,
        # Short of the default world's walls, so the ray behind the plate meets nothing in range.
        "max_range": 1.0,
        "detection_max": 1.0,
    }
    engine = Engine(
        load_config_from_dict(
            {"sim": {}, "plugins": [{f"{__name__}:_Plate": {}}, {"lidar": lidar_cfg}]}
        )
    )
    engine.ctx.seed = 1
    engine.setup()
    engine.reset()
    engine.step()
    lidar = _lidar(engine)
    ranges = np.asarray(lidar.latest.ranges)
    bearings = np.angle(np.exp(1j * np.linspace(lidar.angle_min, lidar.angle_max, lidar.num_rays)))
    ahead = int(np.argmin(np.abs(bearings)))
    behind = int(np.argmin(np.abs(np.angle(np.exp(1j * (bearings - np.pi))))))

    too_close = _declared(DECLARED_OUTPUTS[device][0])
    if too_close is None:
        assert ranges[ahead] == pytest.approx(_Plate.DISTANCE / np.cos(bearings[ahead]), abs=1e-6)
    else:
        assert ranges[ahead] == too_close
    no_return = _declared(DECLARED_OUTPUTS[device][1])
    assert np.isnan(ranges[behind]) if np.isnan(no_return) else ranges[behind] == no_return


@pytest.mark.parametrize("device", DEVICES)
def test_show_fov_draws_the_sector(device):
    engine = _engine(device, show_fov=True)
    assert mujoco.mj_name2id(engine.ctx.model, mujoco.mjtObj.mjOBJ_GEOM, "scan_fov") >= 0


@pytest.mark.parametrize("device", DEVICES)
def test_scan_site_is_the_manifest_frame(device):
    frames = parse_frames(
        substitute(
            _manifest(device)["frames"], {"frame_id": FRAME_ID, "parent_frame": "world"}, device
        ),
        device,
    )
    # The chain from the mount body to the scan frame, which is the last link declared: a device's
    # intermediate vendor links (the RPLIDAR C1's rplidar_link) come before it.
    assert frames[0].parent == "mount" and frames[-1].name == FRAME_ID
    by_name = {f.name: f for f in frames}
    pos, quat = np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0])
    chain, link = [], frames[-1]
    while True:
        chain.append(link)
        if link.parent == "mount":
            break
        link = by_name[link.parent]
    for link in reversed(chain):
        step = np.zeros(3)
        mujoco.mju_rotVecQuat(step, np.asarray(link.pos, dtype=float), quat)
        pos = pos + step
        composed = np.zeros(4)
        mujoco.mju_mulQuat(composed, quat, np.asarray(rpy_to_quat(*link.rpy), dtype=float))
        quat = composed

    # In the MJCF as written: the site hangs directly off the mount at the composed chain's pose.
    spec = mujoco.MjSpec.from_file(str(MODELS_DIR / device / f"{device}.xml"))
    site = spec.site("scan")
    assert site is not None and site.parent.name == "mount"
    np.testing.assert_allclose(site.pos, pos, atol=1e-9)
    assert abs(abs(float(np.asarray(site.quat) @ quat)) - 1.0) < 1e-9

    # In the compiled world: the frame site spawn_sensor adds coincides with the scan site.
    engine = _engine(device)
    m, d = engine.ctx.model, engine.ctx.data
    mujoco.mj_forward(m, d)
    scan = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "scan")
    framed = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, FRAME_ID)
    assert scan >= 0 and framed >= 0
    np.testing.assert_allclose(d.site_xpos[framed], d.site_xpos[scan], atol=1e-9)
    np.testing.assert_allclose(d.site_xmat[framed], d.site_xmat[scan], atol=1e-9)


def _chain(engine: Engine, device: str) -> list[tuple[str, str]]:
    (frames,) = [e for e in engine.ctx.interface.all() if e.name == "frames" and e.owner == device]
    return [(t["parent"], t["child"]) for t in frames.backend["ros2"]["static_tf"]]


@pytest.mark.parametrize("device", DEVICES)
def test_the_manifest_declares_the_vendor_default_frame(device):
    assert manifest_frame_id(MODELS_DIR / device / f"{device}.xml") == VENDOR_FRAME[device]


@pytest.mark.parametrize("device", [d for d in DEVICES if VENDOR_FRAME[d]])
def test_a_mount_without_frame_id_takes_the_vendor_default(device):
    engine = _engine(device, frame_id=None)
    assert _lidar(engine).frame_id == VENDOR_FRAME[device]
    assert _chain(engine, device)[-1][1] == VENDOR_FRAME[device]


def test_the_rplidar_c1_mounts_without_a_frame_id():
    """Its chain has a vendor mount link before the scan frame, and the scan frame is Husarion's."""
    engine = _engine("rplidar_c1", frame_id=None)
    assert _chain(engine, "rplidar_c1") == [("world", "rplidar_link"), ("rplidar_link", "laser")]


@pytest.mark.parametrize("device", DEVICES)
def test_an_explicit_frame_id_wins_over_the_vendor_default(device):
    engine = _engine(device)
    assert _lidar(engine).frame_id == FRAME_ID
    assert _chain(engine, device)[-1][1] == FRAME_ID


def test_a_device_with_no_vendor_default_refuses_a_mount_without_frame_id():
    with pytest.raises(PluginError, match="declares no default 'frame_id'"):
        _engine("sick_tim571", frame_id=None)
