"""Clearpath Jackal drive-test battery (port verification).

Mirrors the robot-porting verification battery and `test_husky_scene.py`: static sanity (A),
open-loop drive tests (B) and sensor checks (C). Everything is driven through the real
``diff_drive`` plugin with the shipped manifest, so the model, its calibration and the controller
are verified together.

Reference dimensions come from Clearpath's ``jackal_description`` (see clearpath_jackal_LICENSE):
chassis box 0.420 x 0.310 x 0.184 m, mass 16.523 kg; wheel r=0.098 m, w=0.040 m, m=0.477 kg;
track 0.37559 m, wheelbase 0.262 m, base_link rest height 0.0635 m.

Note on skid-steer: turning is scrubbed, not rolled, so the tolerances below are deliberately
looser than for an ideal diff-drive (rotation drift, arc odometry). See ``slip_factor`` in the
manifest and the port log.
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
MODEL_DIR = MODELS / "clearpath_jackal"
MANIFEST = MODEL_DIR / "clearpath_jackal.manifest.yaml"

WHEEL_R = 0.098
WHEEL_W = 0.040
TRACK = 0.37559
WHEELBASE = 0.262
REST_Z = 0.0635
TOTAL_MASS = 16.523 + 4 * 0.477  # 18.431 kg
# Fender pair span = the datasheet hull, and the fenders collide in this model (see the MJCF header).
HULL_L = 0.5106
HULL_W = 0.430
# VLP-16 laser plane, from the vendor mount chain: chassis top 0.184 + tower 0.1 + laser 0.0377.
LIDAR_Z_BASE = 0.3217
LIDAR_Z_GROUND = LIDAR_Z_BASE + REST_Z  # 0.3852


def _manifest_plugin(kind: str) -> dict:
    """The plugin config the model actually ships with (the manifest is the source of truth)."""
    for entry in yaml.safe_load(MANIFEST.read_text())["components"]:
        if kind in entry:
            return dict(entry[kind])
    raise AssertionError(f"clearpath_jackal manifest has no {kind} plugin")


def _build(gravity=None):
    """Compose the robot with a ground plane named `floor` (the wheel contact pairs reference it)."""
    asset = resolve_model("roqsim_mobile:clearpath_jackal")
    spec = mujoco.MjSpec.from_file(str(asset.path))
    apply_assets(spec, asset)
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
    plugin = DiffDrivePlugin({**_manifest_plugin("diff_drive"), **overrides})
    plugin.configure(ctx)
    plugin.on_reset(ctx)
    return ctx, plugin


def _yaw(data) -> float:
    w, x, y, z = data.qpos[3:7]
    return float(math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def _run(v, w, seconds, gravity=None):
    """Drive (v, w) for `seconds`; return ground truth + odometry.

    Yaw is ACCUMULATED (unwrapped) rather than sampled as an instantaneous rate: a scrubbing
    skid-steer's instantaneous yaw rate is noisy enough that a tail average of it hid a genuine
    servo instability during this port.
    """
    model, data = _build(gravity)
    ctx, plugin = _plugin(model, data)
    yaw_acc, prev = 0.0, 0.0
    rates, zs = [], []
    for _ in range(int(seconds / model.opt.timestep)):
        plugin.drive(v, 0.0, w)
        plugin.pre_step(ctx)
        mujoco.mj_step(model, data)
        plugin.post_step(ctx)
        assert np.all(np.isfinite(data.qpos)), f"simulation diverged at t={data.time:.3f}"
        cur = _yaw(data)
        yaw_acc += (cur - prev + math.pi) % (2 * math.pi) - math.pi
        prev = cur
        rates.append(float(data.qvel[5]))
        zs.append(float(data.qpos[2]))
    ox, oy, oyaw, *_ = plugin.read_odom()
    return dict(
        model=model,
        data=data,
        x=float(data.qpos[0]),
        y=float(data.qpos[1]),
        yaw=yaw_acc,
        yaw_rate=yaw_acc / seconds,
        yaw_rate_std=float(np.std(rates)),
        z_range=float(np.ptp(zs)),
        speed=float(np.linalg.norm(data.qvel[:2])),
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
    """A2: total and per-body masses match jackal_description; no near-zero inertials."""
    model, _ = _build()
    base = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    assert float(model.body_subtreemass[base]) == pytest.approx(TOTAL_MASS, rel=0.01)
    assert float(model.body_mass[base]) == pytest.approx(16.523, rel=0.01)
    for side in ("front_left", "front_right", "rear_left", "rear_right"):
        wid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_wheel")
        assert float(model.body_mass[wid]) == pytest.approx(0.477, rel=0.01)
        assert np.all(model.body_inertia[wid] > 1e-4), "near-zero wheel inertia"


def test_a3_rest_stability():
    """A3: settles onto its four wheels at the URDF rest height and stays there."""
    model, data = _build()
    for _ in range(int(2.0 / model.opt.timestep)):
        mujoco.mj_step(model, data)
    assert float(data.qpos[2]) == pytest.approx(REST_Z, abs=0.005)
    assert np.linalg.norm(data.qvel[:3]) < 1e-3  # < 1 mm/s drift
    assert np.linalg.norm(data.qpos[:2]) < 1e-3


def test_a4_wheel_servo_is_stable_at_dt():
    """A4: the velocity servo's time constant must not fall below the timestep.

    This is the check that would have caught the first version of this model. With the bare wheel
    inertia (0.0024 kg m^2) and no armature, a kv stiff enough to overcome scrub gives
    kv*dt/I >> 1, and the wheels ring at an order of magnitude past their command while the robot
    hops. The armature (reflected drivetrain inertia) is what makes the servo integrable here.
    """
    model, _ = _build()
    dt = model.opt.timestep
    for i in range(model.nu):
        jid = model.actuator_trnid[i][0]
        dof = model.jnt_dofadr[jid]
        inertia = float(model.dof_M0[dof])  # includes armature
        kv = float(model.actuator_gainprm[i][0])
        assert kv * dt / inertia < 2.0, (
            f"actuator {i}: kv*dt/I = {kv * dt / inertia:.2f}; the servo is stiffer than the "
            f"integrator can follow (I={inertia:.5f} incl. armature, dt={dt})"
        )


def test_a5_scale_and_geometry():
    """A5: wheel radius, track, wheelbase and the collision hull match the vendor description."""
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
    assert float(model.geom_size[gid][1] * 2) == pytest.approx(WHEEL_W, abs=1e-3)

    # Collision hull incl. fenders vs the Jackal datasheet (0.508 x 0.430 m). Cross-checked against
    # the fender MESH bounds, which are authored independently of these numbers.
    for name, want_l, want_w in (
        ("front_fender_collision", HULL_L / 2, HULL_W),
        ("rear_fender_collision", HULL_L / 2, HULL_W),
    ):
        fid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        assert float(model.geom_size[fid][0] * 2) == pytest.approx(want_l, abs=2e-3)
        assert float(model.geom_size[fid][1] * 2) == pytest.approx(want_w, abs=2e-3)
    assert HULL_L == pytest.approx(0.508, abs=0.005)
    assert HULL_W == pytest.approx(0.430, abs=0.005)


def test_a6_fenders_collide_but_do_not_jam_the_wheels():
    """A6: the fender slabs are collision geoms, and they produce no self-contact with the wheels.

    They overlap the wheel cylinders geometrically; the port relies on MuJoCo filtering
    parent<->child body pairs. Assert that rather than trusting it.
    """
    model, data = _build()
    for name in ("front_fender_collision", "rear_fender_collision", "base_collision"):
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        assert model.geom_contype[gid] != 0 and model.geom_conaffinity[gid] != 0

    wheel_gids = {
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{s}_wheel_geom")
        for s in ("front_left", "front_right", "rear_left", "rear_right")
    }
    body_gids = {
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n)
        for n in ("front_fender_collision", "rear_fender_collision", "base_collision")
    }
    ctx, plugin = _plugin(model, data)
    for _ in range(int(3.0 / model.opt.timestep)):
        plugin.drive(0.0, 0.0, 0.8)
        plugin.pre_step(ctx)
        mujoco.mj_step(model, data)
        plugin.post_step(ctx)
        for c in data.contact[: data.ncon]:
            pair = {int(c.geom1), int(c.geom2)}
            assert not (pair & wheel_gids and pair & body_gids), f"self-contact: {pair}"


# --------------------------------------------------------------------------- B. open-loop drive


def test_b1_straight_line():
    """B1: commanded speed is achieved, no lateral drift, odometry tracks ground truth."""
    r = _run(0.5, 0.0, 5.0)
    assert r["speed"] == pytest.approx(0.5, abs=0.02)
    assert abs(r["y"]) < 0.005
    assert abs(r["odom"][0] - r["x"]) < 0.01  # < 1 cm over ~2.4 m


def test_b2_in_place_rotation():
    """B2: the calibrated slip_factor delivers commanded yaw across the operating range.

    This asserts the calibration in the manifest. Measured at chi=1.7:
    0.92 / 0.98 / 1.02 / 1.02 at w = 0.3 / 0.5 / 0.8 / 1.0 rad/s.
    """
    for w in (0.3, 0.5, 0.8, 1.0):
        r = _run(0.0, w, 6.0)
        ratio = r["yaw_rate"] / w
        assert 0.85 <= ratio <= 1.15, f"w={w}: achieved/commanded yaw = {ratio:.2f}"


def test_b3_rotation_is_smooth_not_stick_slip():
    """B3: rotation must be a steady turn, not a lurch.

    The failed first attempt at this model passed a mean-yaw check while oscillating with a
    yaw-rate std of ~1.0 rad/s against a 0.5 rad/s command, hopping off the floor for two thirds
    of the run. Both symptoms are asserted away here.
    """
    r = _run(0.0, 0.5, 6.0)
    assert r["yaw_rate_std"] < 0.25, f"yaw rate std {r['yaw_rate_std']:.2f} rad/s -- stick-slip"
    assert r["z_range"] < 0.005, f"base height varies by {r['z_range'] * 1e3:.1f} mm -- hopping"


def test_b4_limit_enforcement():
    """B4: model limits equal the limits Nav2 is given (1.0 m/s, 1.0 rad/s)."""
    assert _run(2.0, 0.0, 4.0)["speed"] == pytest.approx(1.0, abs=0.02)
    assert _run(0.0, 2.0, 5.0)["yaw_rate"] == pytest.approx(1.0, abs=0.1)


def test_b5_stop_from_max_speed():
    """B5: comes to rest from top speed and stays upright."""
    model, data = _build()
    ctx, plugin = _plugin(model, data)
    for v, secs in ((1.0, 3.0), (0.0, 3.0)):
        for _ in range(int(secs / model.opt.timestep)):
            plugin.drive(v, 0.0, 0.0)
            plugin.pre_step(ctx)
            mujoco.mj_step(model, data)
            plugin.post_step(ctx)
    assert float(np.linalg.norm(data.qvel[:2])) < 0.02
    assert float(data.qpos[2]) == pytest.approx(REST_Z, abs=0.01)


def test_b6_arc():
    """B6: an arc curves the right way; odom drift is bounded but real (skid-steer, expected)."""
    r = _run(0.4, 0.5, 5.0)
    assert r["yaw"] > 1.5  # turned left by more than ~86 deg
    assert r["y"] > 0.5  # and moved left
    drift = math.hypot(r["odom"][0] - r["x"], r["odom"][1] - r["y"])
    assert drift < 0.30, f"odom drift {drift:.2f} m over a 5 s arc"


# --------------------------------------------------------------------------- C. sensors


def test_c1_lidar_mount_height():
    """C1: the scan plane sits where the vendor mount chain puts it (0.3852 m above ground)."""
    model, data = _build()
    for _ in range(int(2.0 / model.opt.timestep)):
        mujoco.mj_step(model, data)
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "lidar")
    assert float(data.site_xpos[sid][2]) == pytest.approx(LIDAR_Z_GROUND, abs=0.005)
    assert abs(float(data.site_xpos[sid][0])) < 0.01
    assert abs(float(data.site_xpos[sid][1])) < 0.01


def test_c2_lidar_manifest_matches_datasheet():
    """C2: the shipped lidar config is the VLP-16 datasheet, not a leftover from another robot."""
    cfg = _manifest_plugin("lidar")
    assert cfg["rays"] == 1800  # 360 deg / 0.2 deg at 10 Hz
    assert cfg["rate_hz"] == pytest.approx(10.0)
    assert cfg["angle_max"] == pytest.approx(2 * math.pi, abs=1e-3)
    assert cfg["max_range"] >= 100.0
    assert cfg["site"] == "lidar"


def test_c3_lidar_sees_a_wall_at_the_right_range():
    """C3: a ray cast from the lidar site measures a known wall distance."""
    from roqsim_sensors.plugins.lidar import LidarPlugin

    asset = resolve_model("roqsim_mobile:clearpath_jackal")
    spec = mujoco.MjSpec.from_file(str(asset.path))
    apply_assets(spec, asset)
    floor = spec.worldbody.add_geom()
    floor.name = "floor"
    floor.type = mujoco.mjtGeom.mjGEOM_PLANE
    floor.size = [15, 15, 0.05]
    wall = spec.worldbody.add_geom()
    wall.name = "wall"
    wall.type = mujoco.mjtGeom.mjGEOM_BOX
    wall.size = [0.1, 4.0, 1.0]
    wall.pos = [3.0, 0.0, 1.0]
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    ctx = SimContext(config={})
    ctx.model, ctx.data = model, data
    ctx.entities.add(
        Entity(name="robot", kind="robot", body="base_link", meta={"prefix": "", "namespace": ""})
    )
    lidar = LidarPlugin(_manifest_plugin("lidar"), entity="robot")
    lidar.configure(ctx)
    mujoco.mj_step(model, data)
    lidar.post_step(ctx)
    # Read through the declared endpoint, so the transport-facing path is what gets verified.
    endpoint = next(e for e in ctx.interface.all() if e.name == "scan" and e.owner == "robot")
    scan = endpoint.read()
    assert scan is not None
    ranges = np.asarray(scan.ranges)
    assert len(ranges) == 1800
    # Ray index 0 points along +x; the wall's near face is at x = 2.9.
    assert ranges[0] == pytest.approx(2.9, abs=0.05)
