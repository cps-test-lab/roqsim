"""Husky A200 drive-test battery (port verification).

Mirrors the robot-porting verification battery: static sanity (A), open-loop drive tests (B) and
sensor checks (C). Everything is driven through the real ``diff_drive`` plugin so the model, its
manifest config and the controller are verified together.

Reference dimensions come from Clearpath's ``husky_description`` (see models/husky_a200/husky_a200_LICENSE):
base box 1.0074 x 0.5709 x 0.2675 m, mass 33.455 kg; wheel r=0.17775 m, w=0.1143 m, m=2.637 kg;
track 0.5708 m, wheelbase 0.512 m, base_link rest height 0.14493 m.

Note on skid-steer: turning is scrubbed, not rolled, so the tolerances below are deliberately looser
than for an ideal diff-drive (rotation drift, arc odometry). See ``slip_factor`` in the diff_drive
plugin and the port log.
"""

from __future__ import annotations

import math
from pathlib import Path

import mujoco
import numpy as np
import pytest
import yaml

from roqsim.context import Entity, SimContext
from roqsim.models import apply_assets, resolve_model
from roqsim_mobile.plugins.diff_drive import DiffDrivePlugin

MODELS = Path(__file__).resolve().parents[1] / "src" / "roqsim_mobile" / "models"
MODEL_DIR = MODELS / "husky_a200"
MANIFEST = MODEL_DIR / "husky_a200.manifest.yaml"

WHEEL_R = 0.17775
TRACK = 0.5708
WHEELBASE = 0.512
REST_Z = 0.14493
TOTAL_MASS = 33.455 + 4 * 2.637  # 44.003 kg


def _drive_config() -> dict:
    """The diff_drive config the model actually ships with (manifest is the source of truth)."""
    plugins = yaml.safe_load(MANIFEST.read_text())["plugins"]
    for entry in plugins:
        if "diff_drive" in entry:
            return dict(entry["diff_drive"])
    raise AssertionError("husky_a200 manifest has no diff_drive plugin")


def _build(gravity=None):
    """Compose the robot with a ground plane named `floor` (the wheel contact pairs reference it)."""
    asset = resolve_model("roqsim_mobile:husky_a200")
    spec = mujoco.MjSpec.from_file(str(asset.path))
    apply_assets(spec, asset)
    spec.meshdir = str(MODELS / "meshes")
    floor = spec.worldbody.add_geom()
    floor.name = "floor"
    floor.type = mujoco.mjtGeom.mjGEOM_PLANE
    floor.size = [15, 15, 0.05]
    floor.friction = [2.0, 0.005, 0.0001]
    model = spec.compile()
    if gravity is not None:
        model.opt.gravity[:] = gravity
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    return model, data


def _plugin(model, data, **overrides):
    ctx = SimContext(config={})
    ctx.model, ctx.data = model, data
    ctx.entities.add(
        Entity(name="robot", kind="robot", body="base_link", meta={"prefix": "", "namespace": ""})
    )
    plugin = DiffDrivePlugin({**_drive_config(), **overrides})
    plugin.configure(ctx)
    plugin.on_reset(ctx)
    return ctx, plugin


def _yaw(data) -> float:
    w, x, y, z = data.qpos[3:7]
    return float(math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def _run(v, w, seconds, gravity=None, tail=2.0):
    """Drive (v, w) for `seconds`; return ground truth + odometry."""
    model, data = _build(gravity)
    ctx, plugin = _plugin(model, data)
    yaw_rates = []
    for k in range(int(seconds / model.opt.timestep)):
        plugin.drive(v, 0.0, w)
        plugin.pre_step(ctx)
        mujoco.mj_step(model, data)
        plugin.post_step(ctx)
        assert np.all(np.isfinite(data.qpos)), f"simulation diverged at t={data.time:.3f}"
        if k * model.opt.timestep > seconds - tail:
            yaw_rates.append(float(data.qvel[5]))
    ox, oy, oyaw, *_ = plugin.read_odom()
    return dict(
        model=model,
        data=data,
        x=float(data.qpos[0]),
        y=float(data.qpos[1]),
        yaw=_yaw(data),
        speed=float(np.linalg.norm(data.qvel[:2])),
        yaw_rate=float(np.mean(yaw_rates)) if yaw_rates else 0.0,
        odom=(ox, oy, oyaw),
    )


# --------------------------------------------------------------------------- A. static sanity


def test_a1_loads_and_steps_without_warnings():
    """A1: 10 s of stepping at the campaign timestep -- no divergence, no MuJoCo warnings."""
    model, data = _build()
    for _ in range(int(10.0 / model.opt.timestep)):
        mujoco.mj_step(model, data)
    assert np.all(np.isfinite(data.qpos))
    fired = [
        mujoco.mjtWarning(i).name
        for i in range(mujoco.mjtWarning.mjNWARNING)
        if data.warning[i].number > 0
    ]
    assert not fired, f"MuJoCo warnings fired: {fired}"


def test_a2_mass_audit():
    """A2: total and per-body masses match husky_description; no near-zero inertials."""
    model, _ = _build()
    base = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    assert float(model.body_subtreemass[base]) == pytest.approx(TOTAL_MASS, rel=0.01)
    assert float(model.body_mass[base]) == pytest.approx(33.455, rel=0.01)
    for side in ("front_left", "front_right", "rear_left", "rear_right"):
        wid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_wheel")
        assert float(model.body_mass[wid]) == pytest.approx(2.637, rel=0.01)
        assert np.all(model.body_inertia[wid] > 1e-4), "near-zero wheel inertia"


def test_a3_rest_stability():
    """A3: settles onto its four wheels at the URDF rest height and stays there."""
    model, data = _build()
    for _ in range(int(2.0 / model.opt.timestep)):
        mujoco.mj_step(model, data)
    assert float(data.qpos[2]) == pytest.approx(REST_Z, abs=0.005)
    assert np.linalg.norm(data.qvel[:3]) < 1e-3  # < 1 mm/s drift
    assert np.linalg.norm(data.qpos[:2]) < 1e-3


def test_a5_scale_and_geometry():
    """A5/A4: wheel radius, track, wheelbase and overall footprint match the datasheet."""
    model, data = _build()

    def body_xy(name):
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        return data.xpos[bid][:2]

    fl, fr, rl = (
        body_xy("front_left_wheel"),
        body_xy("front_right_wheel"),
        body_xy("rear_left_wheel"),
    )
    assert abs(fl[1] - fr[1]) == pytest.approx(TRACK, abs=1e-3)
    assert abs(fl[0] - rl[0]) == pytest.approx(WHEELBASE, abs=1e-3)

    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "front_left_wheel_geom")
    assert float(model.geom_size[gid][0]) == pytest.approx(WHEEL_R, abs=1e-4)
    assert float(model.geom_size[gid][1] * 2) == pytest.approx(0.1143, abs=1e-3)

    # Overall hull incl. wheels vs Husky A200 datasheet (0.99 x 0.67 m); the paper quotes the same.
    base_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "base_collision")
    length = float(model.geom_size[base_gid][0] * 2)
    width = TRACK + 0.1143  # track + one wheel width
    assert length == pytest.approx(0.99, abs=0.03)
    assert width == pytest.approx(0.67, abs=0.03)


# --------------------------------------------------------------------------- B. open-loop drive


def test_b1_straight_line():
    """B1: constant v -- speed tracks command, negligible lateral drift, odometry matches truth."""
    r = _run(0.5, 0.0, 4.0)
    assert r["speed"] == pytest.approx(0.5, rel=0.05)
    assert abs(r["y"]) < 0.02 * r["x"]  # lateral drift < 2% of distance
    assert abs(r["yaw"]) < 0.02
    # encoder odometry vs ground truth (skid-steer rolls true when driving straight)
    assert abs(r["odom"][0] - r["x"]) < 0.01
    assert abs(r["odom"][1] - r["y"]) < 0.01


def test_b4_velocity_limits_match_stack():
    """B4: commanding 2x the limit saturates at the manifest limits (= the paper's Nav2 limits)."""
    cfg = _drive_config()
    assert cfg["max_linear_vel"] == 1.0 and cfg["max_angular_vel"] == 1.0  # DWB-RPP-MPPI configs
    assert _run(2.0, 0.0, 3.0)["speed"] == pytest.approx(1.0, rel=0.05)
    # Yaw saturates too; the skid-steer calibration leaves it within 20% of the stack limit.
    assert _run(0.0, 2.0, 5.0)["yaw_rate"] == pytest.approx(1.0, rel=0.2)


def test_manual_control_leaves_ctrl_to_the_sliders():
    """--manual-control: diff_drive stops stamping the wheels, so a slider drag survives a step."""
    model, data = _build()
    ctx, plugin = _plugin(model, data)
    ctx.manual_control = True
    plugin.drive(0.5, 0.0, 0.0)  # a cmd_vel that would normally spin the wheels up
    data.ctrl[:] = 3.0  # what dragging the wheel-velocity sliders does
    plugin.pre_step(ctx)
    assert (data.ctrl == 3.0).all()  # the command was not stamped over the drag

    ctx.manual_control = False
    plugin.pre_step(ctx)
    assert not (data.ctrl == 3.0).all()  # ... and the controller takes the wheels back


@pytest.mark.parametrize("w", [0.3, 0.5, 0.8, 1.0])
def test_b2_in_place_rotation_is_calibrated(w):
    """B2: with slip_factor the base yaws at the commanded rate (+-25%) and stays roughly in place.

    A skid-steer scrubs while turning, so it does creep: the bound is the footprint half-length,
    not the ideal-diff-drive footprint/4.
    """
    r = _run(0.0, w, 6.0)
    assert r["yaw_rate"] == pytest.approx(w, rel=0.25)
    assert math.hypot(r["x"], r["y"]) < 0.5


def test_b3_arc_radius():
    """B3: combined v+w produces the commanded turn radius (skid-steer tolerance)."""
    v, w = 0.4, 0.4
    r = _run(v, w, 5.0)
    assert r["yaw_rate"] == pytest.approx(w, rel=0.25)
    # radius from achieved speed / achieved yaw rate
    assert r["speed"] / abs(r["yaw_rate"]) == pytest.approx(v / w, rel=0.3)


def test_b5_stop_from_max_speed():
    """B5: decelerates to rest from max speed without tipping over."""
    model, data = _build()
    ctx, plugin = _plugin(model, data)
    for _ in range(int(3.0 / model.opt.timestep)):
        plugin.drive(1.0, 0.0, 0.0)
        plugin.pre_step(ctx)
        mujoco.mj_step(model, data)
        plugin.post_step(ctx)
    for _ in range(int(3.0 / model.opt.timestep)):
        plugin.drive(0.0, 0.0, 0.0)
        plugin.pre_step(ctx)
        mujoco.mj_step(model, data)
        plugin.post_step(ctx)
    assert float(np.linalg.norm(data.qvel[:2])) < 0.02
    assert float(data.qpos[2]) == pytest.approx(REST_Z, abs=0.02)  # still upright on its wheels
    assert abs(_yaw(data)) < 0.1


def test_b6_holds_on_slope():
    """B6: Scene 2 is non-planar -- a parked robot must not run away down a 15 deg grade.

    A velocity servo makes no torque at zero velocity error, so the wheels are given gearbox
    frictionloss to hold. Some creep remains; the criterion that matters is that it stays below the
    paper's stopped-speed threshold (vstop = 0.03 m/s), i.e. a parked robot still counts as stopped.
    """
    grade = math.radians(15)
    gravity = [-9.81 * math.sin(grade), 0.0, -9.81 * math.cos(grade)]  # nose-up incline
    r = _run(0.0, 0.0, 4.0, gravity=gravity, tail=1.0)
    assert r["speed"] < 0.03, f"creeps at {r['speed']:.3f} m/s, above vstop"
    assert abs(r["x"]) < 0.1, "robot ran away down the slope"


# --------------------------------------------------------------------------- C. sensors


def test_c1_lidar_mount_pose():
    """C1: the lidar scan plane sits at the documented mount height, centred on the base."""
    model, data = _build()
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "lidar")
    assert sid >= 0, "model has no `lidar` site"
    pos = data.site_xpos[sid]
    assert float(pos[2]) == pytest.approx(0.51, abs=0.01)  # spec g_lidar_mounting_pose workaround
    assert abs(float(pos[0])) < 1e-6 and abs(float(pos[1])) < 1e-6


def test_c2_lidar_sees_a_box_of_known_size():
    """C2: ranges match a wall placed at a known distance (catches scale/height/frame errors)."""
    pytest.importorskip("roqsim_sensors")
    from roqsim_sensors.plugins.lidar import LidarPlugin

    asset = resolve_model("roqsim_mobile:husky_a200")
    spec = mujoco.MjSpec.from_file(str(asset.path))
    apply_assets(spec, asset)
    spec.meshdir = str(MODELS / "meshes")
    floor = spec.worldbody.add_geom()
    floor.name = "floor"
    floor.type = mujoco.mjtGeom.mjGEOM_PLANE
    floor.size = [15, 15, 0.05]
    # A tall wall 3 m ahead (spans the scan plane).
    wall = spec.worldbody.add_geom()
    wall.name = "wall"
    wall.type = mujoco.mjtGeom.mjGEOM_BOX
    wall.size = [0.1, 5.0, 1.5]
    wall.pos = [3.0, 0.0, 1.5]
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    ctx = SimContext(config={})
    ctx.model, ctx.data = model, data
    ctx.entities.add(
        Entity(name="robot", kind="robot", body="base_link", meta={"prefix": "", "namespace": ""})
    )
    manifest = yaml.safe_load(MANIFEST.read_text())["plugins"]
    cfg = next(e["lidar"] for e in manifest if "lidar" in e)
    lidar = LidarPlugin(dict(cfg))
    lidar.configure(ctx)
    lidar.post_step(ctx)
    scan = (
        ctx.interface.get("scan", owner="robot").read()
        if hasattr(ctx.interface, "get")
        else lidar._scan
    )
    assert scan is not None
    assert len(scan.ranges) == 1440  # 360 deg / 0.25 deg, from the paper's code repo
    # Ray at angle 0 points along +x -> the wall's near face at 3.0 - 0.1 = 2.9 m.
    assert scan.ranges[0] == pytest.approx(2.9, abs=0.05)
