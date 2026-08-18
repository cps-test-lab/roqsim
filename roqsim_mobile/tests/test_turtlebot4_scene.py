"""TurtleBot 4 drive-test battery (port verification).

Mirrors the robot-porting verification battery and its siblings `test_husky_scene.py` /
`test_jackal_scene.py` / `test_turtlebot3_scene.py`: static sanity (A), open-loop drive tests (B) and
sensor checks (C). Everything is driven through the real ``diff_drive`` plugin -- whose *defaults* are
this platform's geometry, which is why `turtlebot4.manifest.yaml` declares a bare ``diff_drive: {}``.

Reference dimensions come from `nav2_minimal_tb4_description` (see `turtlebot4_LICENSE`): body radius
0.164 m and length 0.06 m, body mass 2.300 kg with COM 0.0228 m forward, wheel r=0.03575 m w=0.015 m
m=0.2 kg, ``wheel_separation`` 0.233 m, caster r=0.01 m, OAK-D stereo baseline 0.075 m and
``horizontal_fov`` 1.25 rad.

**This battery was written after the model shipped, and it found two defects on its first run** --
which is the argument for writing one per port rather than trusting a model that looks fine in a
viewer:

* the wheel servos had no ``armature``, so ``kv*dt/I`` was 31 at the 2 ms step: the model diverged
  (NaN in QACC at t = 0.036 s) and a 0.2 m/s command threw the robot across the floor at 4.5 m/s. A1
  and A4 are the two tests that catch it;
* the caster was made frictionless by an explicit ``<pair geom2="floor">``, which MuJoCo silently
  drops in a world whose ground geom has another name -- compiles clean, ``npair`` 0, and the robot
  then drives on a high-friction ball. B7 asserts the ``priority``/``condim`` mechanism that replaced
  it, and asserts the pair is gone.

Like the TurtleBot3 and unlike the two skid-steers, this is a true differential drive: two driven
wheels and one passive caster, so turning rolls instead of scrubbing. There is no ``slip_factor`` and
the tolerances are correspondingly tight -- a loose one passing here would hide a real defect.
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
MODEL_DIR = MODELS / "turtlebot4"
MANIFEST = MODEL_DIR / "turtlebot4.manifest.yaml"

WHEEL_R = 0.03575
WHEEL_W = 0.015
TRACK = 0.233
BODY_R = 0.164
TOTAL_MASS = 2.3 + 2 * 0.2  # 2.7 kg: create3 body + two wheels (the caster is massless geometry)
# base_link is the URDF root frame, and the wheel bodies hang 0.0402 m above it against a 0.03575 m
# radius -- so at rest the frame itself sits 4.45 mm BELOW the ground plane, plus ~0.8 mm of soft
# contact sink. Measured -0.0053. Negative is correct here and is not a sign convention slip.
REST_Z = -0.0053
LIDAR_Z = 0.192915 + REST_Z  # RPLIDAR scan plane above ground
# diff_drive's own defaults, which ARE the Create 3's rated limits (see the plugin docstring).
MAX_V = 0.31
MAX_W = 1.90


def _manifest_plugin(kind: str) -> dict:
    """The plugin config the model actually ships with (the manifest is the source of truth)."""
    for entry in yaml.safe_load(MANIFEST.read_text())["plugins"]:
        if kind in entry:
            return dict(entry[kind] or {})
    raise AssertionError(f"turtlebot4 manifest has no {kind} plugin")


def _build(settle: float = 2.0):
    """Compose the robot with a ground plane (high friction, as a real scene floor is), then settle.

    The settle matters: the MJCF starts base_link 0.03 m up and lets it drop onto wheels + caster, so
    a measurement taken from t=0 reads the fall, not the robot.
    """
    asset = resolve_model("roqsim_mobile:turtlebot4")
    spec = mujoco.MjSpec.from_file(str(asset.path))
    apply_assets(spec, asset)
    floor = spec.worldbody.add_geom()
    floor.name = "floor"
    floor.type = mujoco.mjtGeom.mjGEOM_PLANE
    floor.size = [15, 15, 0.05]
    floor.friction = [2.0, 0.005, 0.0001]
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    for _ in range(int(settle / model.opt.timestep)):
        mujoco.mj_step(model, data)
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


def _run(v, w, seconds):
    """Drive (v, w) for `seconds` from a settled start; return ground truth + encoder odometry."""
    model, data = _build()
    ctx, plugin = _plugin(model, data)
    x0, y0 = float(data.qpos[0]), float(data.qpos[1])
    yaw_acc, prev = 0.0, _yaw(data)
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
        x=float(data.qpos[0]) - x0,
        y=float(data.qpos[1]) - y0,
        yaw=yaw_acc,
        yaw_rate=yaw_acc / seconds,
        yaw_rate_std=float(np.std(rates)),
        z_range=float(np.ptp(zs)),
        speed=float(np.linalg.norm(data.qvel[:2])),
        odom=(ox, oy, oyaw),
    )


# --------------------------------------------------------------------------- A. static sanity


def test_a1_loads_and_steps_without_warnings():
    """A1: 10 s of stepping at the campaign timestep — no divergence, no MuJoCo warnings.

    This is the test the model failed before it had an `armature`: mjWARN_BADQACC at t = 0.036 s.
    """
    model, data = _build(settle=0.0)
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
    """A2: total and per-body masses match the vendor description; no near-zero wheel inertia."""
    model, _ = _build(settle=0.0)
    base = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    assert float(model.body_subtreemass[base]) == pytest.approx(TOTAL_MASS, rel=0.01)
    assert float(model.body_mass[base]) == pytest.approx(2.3, rel=0.01)  # create3 body_mass
    for side in ("left", "right"):
        wid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_wheel")
        assert float(model.body_mass[wid]) == pytest.approx(0.2, rel=0.01)
        assert np.all(model.body_inertia[wid] > 1e-6), "near-zero wheel inertia"


def test_a3_rest_stability():
    """A3: settles onto two wheels + the caster at the geometric rest height and stays there."""
    model, data = _build()
    assert float(data.qpos[2]) == pytest.approx(REST_Z, abs=0.003)
    assert np.linalg.norm(data.qvel[:3]) < 1e-3  # < 1 mm/s drift
    assert np.linalg.norm(data.qpos[:2]) < 1e-3


def test_a4_wheel_servo_is_stable_at_dt():
    """A4: the velocity servo's time constant must not fall below the timestep.

    The bare wheel spin inertia is 1.28e-4 kg*m^2, so without the wheel joints' `armature` (reflected
    drivetrain inertia) this ratio is 31 and the model diverges outright — the failure this battery
    was written to catch, and the same one the Jackal port hit.
    """
    model, _ = _build(settle=0.0)
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
    """A5: wheel radius/width, track and body radius match the vendor description."""
    model, data = _build()

    def body_xy(name):
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        return data.xpos[bid][:2]

    left, right = body_xy("left_wheel"), body_xy("right_wheel")
    assert abs(left[1] - right[1]) == pytest.approx(TRACK, abs=1e-3)
    assert abs(left[0] - right[0]) < 1e-6, "wheels must share one axle (x)"
    # The track the plugin integrates odometry with has to BE the modelled track, or encoder odometry
    # is wrong by a constant the controller cannot see.
    assert DiffDrivePlugin({}).L == pytest.approx(TRACK, abs=1e-4)
    assert DiffDrivePlugin({}).r == pytest.approx(WHEEL_R, abs=1e-5)

    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "body_collision")
    assert float(model.geom_size[gid][0]) == pytest.approx(BODY_R, abs=1e-3)
    assert float(model.geom_size[gid][1] * 2) == pytest.approx(
        0.06, abs=1e-3
    )  # create3 body_length


def test_a6_visual_meshes_are_visual_only():
    """A6: the Collada-derived meshes carry no contact, so the primitives own all collision.

    Also guards the `density="0"` trap the MJCF documents: a zero-density mesh geom renders as its
    bounding sphere.
    """
    model, _ = _build(settle=0.0)
    n_mesh = 0
    for gid in range(model.ngeom):
        if model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_MESH:
            n_mesh += 1
            assert model.geom_contype[gid] == 0 and model.geom_conaffinity[gid] == 0
    assert n_mesh >= 9, f"only {n_mesh} mesh geoms — meshes failed to resolve?"
    for name in ("shell", "body_visual", "bumper_visual", "rplidar"):
        mid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MESH, name)
        assert mid >= 0, f"mesh {name} missing"
        assert int(model.mesh_vertnum[mid]) > 100, f"{name}: decimated to nothing?"


# --------------------------------------------------------------------------- B. open-loop drive


def test_b1_straight_line():
    """B1: commanded speed is achieved, no lateral drift, encoder odometry over-counts slightly."""
    r = _run(0.2, 0.0, 5.0)
    assert r["speed"] == pytest.approx(0.2, abs=0.01)  # measured 0.1977
    assert abs(r["y"]) < 0.005
    # Encoder odometry OVER-counts: the wheels turn ~1.2% further than the robot travels (soft-contact
    # slip). Assert magnitude AND sign — odom running BEHIND ground truth would mean the wheels are
    # being dragged rather than slipping, which is a different (and worse) defect.
    over = r["odom"][0] - r["x"]
    assert 0.002 < over < 0.03, f"odometry over-count {over:+.4f} m over {r['x']:.3f} m"


def test_b2_in_place_rotation():
    """B2: a true diff-drive rolls its yaw, so achieved/commanded is within 5% with no compensation.

    Asserts the absence of a `slip_factor` too: this platform must not acquire one, because a yaw
    deficit here would mean a dragging caster (see B7), not scrub.
    """
    assert "slip_factor" not in _manifest_plugin("diff_drive")
    for w in (0.3, 0.5, 0.8, 1.2):
        r = _run(0.0, w, 5.0)
        ratio = r["yaw_rate"] / w
        assert 0.95 <= ratio <= 1.05, f"w={w}: achieved/commanded yaw = {ratio:.3f}"


def test_b3_rotation_is_smooth_not_stick_slip():
    """B3: rotation is a steady turn, not a lurch, and the base does not hop."""
    r = _run(0.0, 0.5, 5.0)
    assert r["yaw_rate_std"] < 0.05, f"yaw rate std {r['yaw_rate_std']:.3f} rad/s — stick-slip"
    assert r["z_range"] < 0.003, f"base height varies by {r['z_range'] * 1e3:.1f} mm — hopping"


def test_b4_limit_enforcement():
    """B4: the model saturates at the Create 3's rated limits, which are the plugin's defaults.

    Over-shooting them would be the real failure (the controller would be given a ceiling the platform
    does not have), so the upper bound is tight and the lower one allows the measured slip deficit.
    """
    v = _run(1.0, 0.0, 4.0)["speed"]
    assert MAX_V * 0.95 <= v <= MAX_V + 0.005, f"top speed {v:.4f} vs rated {MAX_V}"
    w = _run(0.0, 3.0, 4.0)["yaw_rate"]
    assert MAX_W * 0.90 <= w <= MAX_W + 0.05, f"top yaw rate {w:.4f} vs rated {MAX_W}"


def test_b5_stop_from_max_speed():
    """B5: comes to rest from top speed and stays upright at rest height."""
    model, data = _build()
    ctx, plugin = _plugin(model, data)
    for v, secs in ((MAX_V, 3.0), (0.0, 3.0)):
        for _ in range(int(secs / model.opt.timestep)):
            plugin.drive(v, 0.0, 0.0)
            plugin.pre_step(ctx)
            mujoco.mj_step(model, data)
            plugin.post_step(ctx)
    assert float(np.linalg.norm(data.qvel[:2])) < 0.01
    assert float(data.qpos[2]) == pytest.approx(REST_Z, abs=0.005)


def test_b6_arc():
    """B6: an arc curves the right way and encoder odometry stays close to ground truth.

    An order of magnitude tighter than the skid-steer siblings (they drift ~0.3 m over 5 s): rolling
    wheels mean the encoders actually describe the motion. Measured 0.013 m.
    """
    r = _run(0.15, 0.5, 5.0)
    assert r["yaw"] > 2.0  # turned left by more than ~115 deg
    assert r["y"] > 0.1  # and moved left
    drift = math.hypot(r["odom"][0] - r["x"], r["odom"][1] - r["y"])
    assert drift < 0.03, f"odom drift {drift:.3f} m over a 5 s arc"


def test_b7_caster_does_not_drag():
    """B7: the caster is frictionless by contact PRIORITY, in any world.

    MuJoCo combines two geoms' contact parameters by taking max(condim) and max(friction), so
    `condim="1"` alone loses to an ordinary floor and the caster drags. `priority="1"` makes the
    caster's own parameters win outright. The model used to get this from an explicit
    `<pair geom1="caster" geom2="floor">` instead, which MuJoCo silently drops in a world that names
    its ground anything else — so the mechanism is asserted here, not just its outcome, and the pair
    is asserted GONE.
    """
    model, _ = _build(settle=0.0)
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "caster")
    assert gid >= 0, "caster geom missing"
    assert int(model.geom_condim[gid]) == 1, f"caster condim {model.geom_condim[gid]} != 1"
    assert int(model.geom_priority[gid]) > 0, "caster must win contact params by priority"
    assert float(model.geom_friction[gid][0]) < 0.01
    assert model.npair == 0, "an explicit contact pair is world-name-dependent; use priority"
    # And it must actually collide -- the old model switched the caster's collision off entirely.
    assert model.geom_contype[gid] != 0 and model.geom_conaffinity[gid] != 0


def test_b8_drives_in_a_world_whose_floor_is_not_called_floor():
    """B8: the robot behaves identically on a ground geom with a different name.

    The regression test for the dropped-`<pair>` defect: before the fix this same run rolled on a
    friction-2.0 caster, which is worth ~48% of yaw rate by the TurtleBot3 port's measurement.
    """
    asset = resolve_model("roqsim_mobile:turtlebot4")
    spec = mujoco.MjSpec.from_file(str(asset.path))
    apply_assets(spec, asset)
    ground = spec.worldbody.add_geom()
    ground.name = "warehouse_ground"  # deliberately not "floor"
    ground.type = mujoco.mjtGeom.mjGEOM_PLANE
    ground.size = [15, 15, 0.05]
    ground.friction = [2.0, 0.005, 0.0001]
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    for _ in range(int(2.0 / model.opt.timestep)):
        mujoco.mj_step(model, data)

    ctx, plugin = _plugin(model, data)
    yaw_acc, prev = 0.0, _yaw(data)
    for _ in range(int(5.0 / model.opt.timestep)):
        plugin.drive(0.0, 0.0, 0.5)
        plugin.pre_step(ctx)
        mujoco.mj_step(model, data)
        plugin.post_step(ctx)
        cur = _yaw(data)
        yaw_acc += (cur - prev + math.pi) % (2 * math.pi) - math.pi
        prev = cur
    ratio = (yaw_acc / 5.0) / 0.5
    assert 0.95 <= ratio <= 1.05, f"yaw ratio {ratio:.3f} on a ground geom not named 'floor'"


# --------------------------------------------------------------------------- C. sensors


def test_c1_lidar_mount_height():
    """C1: the RPLIDAR scan plane sits at the URDF's rplidar_link height, centred, 40 mm back."""
    model, data = _build()
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "lidar")
    assert float(data.site_xpos[sid][2]) == pytest.approx(LIDAR_Z, abs=0.003)
    assert float(data.site_xpos[sid][0]) == pytest.approx(-0.04, abs=0.003)
    assert abs(float(data.site_xpos[sid][1])) < 0.003


def test_c2_manifest_ships_the_platforms_own_sensors():
    """C2: the manifest brings the TurtleBot 4's stock scanner and camera, with the URDF's frame ids.

    `frame_id: rplidar_link` is load-bearing: without it the scan is stamped with the site name
    (`lidar`) and the LaserScan is not locatable in the robot's own TF tree.
    """
    lidar = _manifest_plugin("lidar")
    assert lidar["site"] == "lidar"
    assert lidar["frame_id"] == "rplidar_link"
    assert lidar["rays"] == 360
    assert lidar["max_range"] == pytest.approx(12.0)  # RPLIDAR A1
    assert _manifest_plugin("oakd_camera")["camera"] == "oakd_rgb"


def test_c3_camera_and_stereo_frames_match_the_urdf():
    """C3: the OAK-D's FOV and stereo baseline are the vendor's, not defaults."""
    model, data = _build()
    cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "oakd_rgb")
    assert cid >= 0, "oakd_rgb camera missing"
    # The URDF gives a 1.25 rad HORIZONTAL fov; MuJoCo's fovy is vertical, so at the model's 320x240
    # the two are related by the 4:3 aspect. Check the round trip rather than the stored number.
    fovy = math.radians(float(model.cam_fovy[cid]))
    w, h = model.cam_resolution[cid]
    fovx = 2 * math.atan(math.tan(fovy / 2) * (w / h))
    assert fovx == pytest.approx(1.25, abs=0.02), f"horizontal fov {fovx:.3f} rad vs URDF 1.25"

    left = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "oakd_left")
    right = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "oakd_right")
    baseline = abs(float(data.site_xpos[left][1] - data.site_xpos[right][1]))
    assert baseline == pytest.approx(0.075, abs=1e-4)


def test_c4_wheel_encoders_imu_and_bumper_exist():
    """C4: the sensors a ROS 2 bridge publishes are present and named as the siblings' are."""
    model, _ = _build(settle=0.0)
    for name in (
        "left_wheel_pos",
        "right_wheel_pos",
        "left_wheel_vel",
        "right_wheel_vel",
        "imu_gyro",
        "imu_acc",
        "base_pos",
        "base_quat",
        "bumper",
    ):
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name) >= 0, f"missing {name}"
