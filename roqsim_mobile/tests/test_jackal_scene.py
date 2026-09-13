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
from mobile_scene_utils import named
from scan_mount_utils import (
    chain,
    endpoint,
    forward_range,
    lidar,
    pose_in_base,
    recast,
    robot_hits,
    spawn,
    static_tf,
)

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


# The VLP-16 is a `velodyne_vlp16` device on the vendor's tower, checked in a closed room.

#: jackal_description @ 4ddf9b5: `mid_mount` on base_link at the chassis top (urdf/jackal.urdf.xacro),
#: vlp16_mount's plate joint at the tower height 0.1 with the VLP-16 at its origin
#: (accessories/vlp16_mount.urdf.xacro); velodyne_description VLP-16.urdf.xacro: the scan frame
#: `velodyne` 0.0377 above the housing base.
MID_MOUNT = ((0.0, 0.0, 0.184), (0.0, 0.0, 0.0))
TOWER = ((0.0, 0.0, 0.1), (0.0, 0.0, 0.0))
LASER = ((0.0, 0.0, 0.0377), (0.0, 0.0, 0.0))
SCAN_FRAME = "velodyne"
LABEL = "velodyne"
NAMESPACE = "j100_0000"
#: The VLP-16 device's housing: 0.83 kg, velodyne_description's inertial and the data sheet's weight.
VLP16_MASS = 0.83


@pytest.fixture(scope="module")
def scan():
    engine = spawn("clearpath_jackal", {LABEL: None}, owner="jk", prefix="jk_", namespace=NAMESPACE)
    yield engine
    engine.shutdown()


def test_c1_the_scan_frame_is_the_vendor_chain(scan):
    """C1: the scan origin is the vendor chain, 0.3217 m in base_link and 0.3852 m above the ground."""
    want_pos, want_rot = chain(MID_MOUNT, TOWER, LASER)
    assert np.allclose(want_pos, (0.0, 0.0, LIDAR_Z_BASE), atol=1e-9)
    assert LIDAR_Z_BASE + REST_Z == pytest.approx(LIDAR_Z_GROUND)
    for site in (f"jk_{LABEL}_scan", f"jk_{LABEL}_{SCAN_FRAME}"):
        pos, rot = pose_in_base(scan, site, "jk_")
        assert np.allclose(pos, want_pos, atol=1e-6), f"{site} at {pos}"
        assert np.allclose(rot, want_rot, atol=1e-6), f"{site} rotation {rot}"


def test_c2_the_forward_ray_reads_the_wall(scan):
    published, true = forward_range(scan, lidar(scan, f"jk.{LABEL}"))
    assert published == pytest.approx(true, abs=1e-3), f"reads {published:.4f} m, wall at {true:.4f} m"


def test_c3_the_scan_skips_its_own_mount_and_nothing_else(scan):
    model = scan.ctx.model
    scanner = lidar(scan, f"jk.{LABEL}")
    mount = named(model, mujoco.mjtObj.mjOBJ_BODY, f"jk_{LABEL}_mount")
    assert scanner._bodyexclude == mount, "the scanner excludes something other than its housing"
    _, hits = recast(scan, scanner)
    assert mount not in set(model.geom_bodyid[hits.geomid[hits.geomid >= 0]].tolist())


def test_c4_no_ray_starts_inside_robot_geometry(scan):
    """The embedded site started every ray inside the housing cylinder modelled on base_link."""
    scanner = lidar(scan, f"jk.{LABEL}")
    inside, _ = robot_hits(scan, scanner, "jk_")
    assert not inside, {body: len(d) for body, d in inside.items()}
    assert np.asarray(scanner.latest.ranges).min() > scanner.range_min, "a ray is clamped"


def test_c5_the_scan_sees_no_part_of_the_robot(scan):
    """At 0.3217 m the full turn clears the tower below it: no robot returns."""
    _, outside = robot_hits(scan, lidar(scan, f"jk.{LABEL}"), "jk_")
    assert outside == {}, sorted(outside)


def test_c6_the_tf_chain_and_topic(scan):
    address = f"jk.{LABEL}"
    tf = static_tf(scan, address, NAMESPACE)
    assert [(t["parent"], t["child"]) for t in tf] == [("mid_mount", SCAN_FRAME)], tf
    want_pos, _ = chain(TOWER, LASER)
    assert np.allclose(tf[0]["translation"], want_pos, atol=1e-6)
    assert np.allclose(tf[0]["rotation"], (1.0, 0.0, 0.0, 0.0), atol=1e-9)
    robot = static_tf(scan, "jk", NAMESPACE)
    (mid,) = [t for t in robot if t["child"] == "mid_mount"]
    assert mid["parent"] == "base_link" and np.allclose(mid["translation"], MID_MOUNT[0])
    hints = endpoint(scan, "scan", address).backend["ros2"]
    assert (hints["frame_id"], hints["topic"]) == (SCAN_FRAME, "scan")
    assert "static_tf" not in hints, "the mount owns the chain; the scan publishes none"


def test_c7_the_scan_keeps_the_values_the_experiments_were_run_with():
    """C7: VLP-16 data sheet ray pattern, with the two values the manifest overrides kept as they were.

    The device carries the hardware's range_min 0.9 m and 0.03 m range noise; the manifest keeps
    0.4 m and no noise, the values this model's scan had when experiments were built on it.
    """
    engine = spawn("clearpath_jackal", {LABEL: None}, owner="jk", prefix="jk_", namespace=NAMESPACE)
    try:
        scanner = lidar(engine, f"jk.{LABEL}")
        assert scanner.num_rays == 1800  # 360 deg / 0.2 deg at 10 Hz
        assert (scanner.angle_min, scanner.angle_max) == pytest.approx((0.0, 2 * math.pi))
        assert (scanner.range_min, scanner.range_max) == pytest.approx((0.4, 100.0))
        assert scanner.rate_hz == pytest.approx(10.0)
        assert scanner.config["range_stddev"] == pytest.approx(0.0)
        model = engine.ctx.model
        mount = named(model, mujoco.mjtObj.mjOBJ_BODY, f"jk_{LABEL}_mount")
        assert float(model.body_mass[mount]) == pytest.approx(VLP16_MASS)
    finally:
        engine.shutdown()
