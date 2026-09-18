"""TurtleBot 4 drive-test battery (port verification).

Mirrors the robot-porting verification battery and its siblings `test_husky_scene.py` /
`test_jackal_scene.py` / `test_turtlebot3_scene.py`: static sanity (A), open-loop drive tests (B) and
sensor checks (C). Everything is driven through the real ``diff_drive`` plugin -- whose *defaults* are
this platform's geometry, which is why `turtlebot4.manifest.yaml` declares a bare ``diff_drive: {}``.

Reference dimensions come from `nav2_minimal_tb4_description` (see `turtlebot4_LICENSE`): body radius
0.164 m and length 0.06 m, body mass 2.300 kg with COM 0.0228 m forward, wheel r=0.03575 m w=0.015 m
m=0.2 kg, ``wheel_separation`` 0.233 m, caster r=0.01 m, OAK-D stereo baseline 0.075 m and
``horizontal_fov`` 1.25 rad.

The RPLIDAR A1 is not in the MJCF: the manifest mounts the ``rplidar_a1`` device model at the vendor
joint, and section D pins that mount against ``turtlebot4_description`` @ 7fd29fb.

**Two defects here look fine in a viewer and fail this battery** -- which is the argument for
writing one per port rather than trusting a model by eye:

* wheel servos without ``armature`` run at ``kv*dt/I`` = 31 at the 2 ms step: the model diverges
  (NaN in QACC at t = 0.036 s) and a 0.2 m/s command throws the robot across the floor at 4.5 m/s. A1
  and A4 are the two tests that catch it;
* a caster made frictionless by an explicit ``<pair geom2="floor">`` is silently dropped by MuJoCo
  in a world whose ground geom has another name -- compiles clean, ``npair`` 0, and the robot then
  drives on a high-friction ball. B7 asserts the ``priority``/``condim`` mechanism instead, and
  asserts there is no pair.

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
import scan_mount_utils as scan_mount
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
# The base carries everything rigidly on it: the create3 body (2.3) with its bumper (0.1) and caster
# (0.01), plus the TurtleBot 4 standard's shell (0.39), four weight blocks (4 x 0.061), four tower
# standoffs (4 x 0.26), sensor plate (0.332), camera bracket (0.033) and OAK-D (0.061) --
# turtlebot4_description @ jazzy urdf/standard/*.urdf.xacro. The RPLIDAR the manifest mounts carries
# its own 0.17 kg (RPLIDAR_MASS, section D), so it is not here.
BASE_MASS = 2.3 + 0.1 + 0.01 + 0.39 + 4 * 0.061 + 4 * 0.26 + 0.332 + 0.033 + 0.061  # 4.51
# ... two wheels and their two wheel-drop suspension bodies (wheel_drop.urdf.xacro: 0.05 each).
TOTAL_MASS = BASE_MASS + 2 * 0.2 + 2 * 0.05  # 5.01 kg
# base_link is the URDF root frame, and the wheel bodies hang 0.0402 m above it against a 0.03575 m
# radius -- so at rest the frame itself sits 4.45 mm BELOW the ground plane, plus ~0.8 mm of soft
# contact sink. Measured -0.0053. Negative is correct here and is not a sign convention slip.
REST_Z = -0.0053
# diff_drive's own defaults, which ARE the Create 3's rated limits (see the plugin docstring).
MAX_V = 0.31
MAX_W = 1.90

#: base_link -> shell_link, turtlebot4_description @ 7fd29fb urdf/standard/turtlebot4.urdf.xacro:43-47:
#: z = shell_z_offset 3 cm (:13) + base_link_z_offset 6.42 cm (irobot_create_description
#: urdf/create3.urdf.xacro:53 @ 1fccb76).
SHELL_LINK = ((0.0, 0.0, 0.0942), (0.0, 0.0, 0.0))
#: shell_link -> rplidar_link, the same file :31-33 (offsets) and :114-117 (rpy 0 0 pi/2).
RPLIDAR_JOINT = ((-0.04, 0.0, 0.098715), (0.0, 0.0, math.pi / 2))
#: The RPLIDAR A1's inertial, which the device model carries (rplidar.urdf.xacro:7,34 @ 7fd29fb).
RPLIDAR_MASS = 0.17
#: The Create 3's own sensor sites (create3.urdf.xacro @ 1fccb76): four cliff sensors, seven IR
#: proximity sensors, the optical-flow mouse and the omnidirectional IR receiver.
CREATE3_SITES = {
    "cliff_front_left",
    "cliff_front_right",
    "cliff_side_left",
    "cliff_side_right",
    "ir_intensity_front_center_left",
    "ir_intensity_front_center_right",
    "ir_intensity_front_left",
    "ir_intensity_front_right",
    "ir_intensity_left",
    "ir_intensity_right",
    "ir_intensity_side_left",
    "mouse",
    "ir_omni",
}


def _manifest_plugin(kind: str) -> dict:
    """The plugin config the model actually ships with (the manifest is the source of truth)."""
    for entry in yaml.safe_load(MANIFEST.read_text())["components"]:
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

    This is the test the model fails without a wheel `armature`: mjWARN_BADQACC at t = 0.036 s.
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
    assert float(model.body_mass[base]) == pytest.approx(BASE_MASS, rel=0.01)
    for side in ("left", "right"):
        wid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_wheel")
        assert float(model.body_mass[wid]) == pytest.approx(0.2, rel=0.01)
        assert np.all(model.body_inertia[wid] > 1e-6), "near-zero wheel inertia"
        did = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"wheel_drop_{side}")
        assert float(model.body_mass[did]) == pytest.approx(0.05, rel=0.01)


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
    for name in ("shell", "body_visual", "bumper_visual", "tower_standoff"):
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
    caster's own parameters win outright. An explicit `<pair geom1="caster" geom2="floor">` gets the
    same outcome only until MuJoCo silently drops it in a world that names its ground anything else —
    so the mechanism is asserted here, not just its outcome, and the pair is asserted ABSENT.
    """
    model, _ = _build(settle=0.0)
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "caster")
    assert gid >= 0, "caster geom missing"
    assert int(model.geom_condim[gid]) == 1, f"caster condim {model.geom_condim[gid]} != 1"
    assert int(model.geom_priority[gid]) > 0, "caster must win contact params by priority"
    assert float(model.geom_friction[gid][0]) < 0.01
    assert model.npair == 0, "an explicit contact pair is world-name-dependent; use priority"
    # And it must actually collide -- a caster with its collision switched off passes every check above.
    assert model.geom_contype[gid] != 0 and model.geom_conaffinity[gid] != 0


def test_b8_drives_in_a_world_whose_floor_is_not_called_floor():
    """B8: the robot behaves identically on a ground geom with a different name.

    Guards against a world-name-dependent caster: with a dropped `<pair>` this same run rolls on a
    friction-2.0 caster, which is worth ~48% of yaw rate by the TurtleBot3 measurement.
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


def test_c1_the_mjcf_carries_no_scanner():
    """C1: the RPLIDAR is the device model the manifest mounts, so the MJCF has no scan site of its own.

    A second scan origin here would be a second, unmounted scanner the moment anything named it.
    """
    model, _ = _build(settle=0.0)
    sites = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, i) for i in range(model.nsite)}
    assert sites == {"base_imu", "oakd", "oakd_left", "oakd_right", *CREATE3_SITES}
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MESH, "rplidar") < 0


def test_c2_manifest_ships_the_platforms_own_sensors():
    """C2: the manifest mounts the TurtleBot 4's stock scanner and camera, with the URDF's frame ids.

    `frame_id: rplidar_link` is load-bearing: it is the name the TurtleBot 4's URDF gives the scan
    frame, so the LaserScan is locatable in the robot's own TF tree. One lidar override, from
    Clearpath's TurtleBot 4 datasheet: 1 deg angular resolution, so 360 rays; every other scan value
    is the RPLIDAR A1 device's.
    """
    manifest = yaml.safe_load(MANIFEST.read_text())
    assert manifest["frames"] == [
        {
            "name": "shell_link",
            "parent": "base_link",
            "pos": [*SHELL_LINK[0]],
            "rpy": [*SHELL_LINK[1]],
        }
    ]
    (mount,) = [c for c in manifest["components"] if "spawn_sensor" in c]
    assert mount["name"] == "rplidar"
    assert mount["spawn_sensor"] == {
        "model": "rplidar_a1",
        "parent_frame": "shell_link",
        "pos": [*RPLIDAR_JOINT[0]],
        "rpy": [*RPLIDAR_JOINT[1]],
        "frame_id": "rplidar_link",
    }
    assert mount["components"] == [{"lidar": {"rays": 360}}]
    assert not any("lidar" in c for c in manifest["components"])
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


def test_c3b_the_oakd_lens_matches_the_standalone_sensor_model():
    """C3b: this robot's OAK-D and `roqsim_sensors:oakd` are the same device, so one lens.

    The camera element cannot literally be shared -- it sits inside this robot's body chain, while the
    standalone mount is its own MJCF -- so the two files each hold the numbers and this asserts they
    agree. Without it "one definition" is a comment in two files that nothing enforces, and a copy of
    these optics that does not derive them drifts.

    The dependency runs the way it already does: roqsim_mobile requires roqsim_sensors, never the
    reverse, so the check lives here rather than beside the sensor model.
    """
    import mujoco as mj

    from roqsim.models import resolve_model

    model, _ = _build()
    cid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_CAMERA, "oakd_rgb")

    standalone = mj.MjSpec.from_file(str(resolve_model("roqsim_sensors:oakd").path))
    cam = next(c for c in standalone.cameras if c.name == "oakd_rgb")

    assert float(model.cam_fovy[cid]) == pytest.approx(float(cam.fovy)), (
        "turtlebot4.xml and roqsim_sensors:oakd disagree on the OAK-D's fovy; they are one device"
    )
    assert list(model.cam_resolution[cid]) == [int(v) for v in cam.resolution]


def test_c4_wheel_encoders_and_imu_exist():
    """C4: the sensors a ROS 2 bridge publishes are present and named as the siblings' are.

    No touch sensor: the bumper is the manifest's `bumper` plugin over the body collision geom,
    zoned by bearing as the Create 3's simulator zones it (C5).
    """
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
    ):
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name) >= 0, f"missing {name}"
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "bumper") < 0


# --------------------------------------------------------------------------- D. the mounted scanner


@pytest.fixture(scope="module")
def mounted():
    """The robot spawned as a world spawns it, prefixed and namespaced, its RPLIDAR cast once.

    The OAK-D is switched off: it renders, and nothing in this section reads it.
    """
    engine = scan_mount.spawn("turtlebot4", ["rplidar"], disabled=("robot.oakd_camera",))
    lidar = scan_mount.lidar(engine, "robot.rplidar")
    yield engine, lidar
    engine.shutdown()


def test_d1_scan_frame_is_the_vendor_chain(mounted):
    """D1: rays are cast from base_link -> shell_link -> rplidar_link exactly as the URDF chains it,
    and the frame the mount publishes is that same pose."""
    engine, _ = mounted
    want_pos, want_rot = scan_mount.chain(SHELL_LINK, RPLIDAR_JOINT)
    np.testing.assert_allclose(want_pos, [-0.04, 0.0, 0.192915], atol=1e-12)
    for site in ("r_rplidar_scan", "r_rplidar_rplidar_link"):
        pos, rot = scan_mount.pose_in_base(engine, site)
        np.testing.assert_allclose(pos, want_pos, atol=1e-9, err_msg=site)
        np.testing.assert_allclose(rot, want_rot, atol=1e-9, err_msg=site)


def test_d2_the_forward_ray_reads_the_true_wall_distance(mounted):
    """D2: bearing 0 is rplidar_link +x, which the pi/2 yaw turns onto base_link +y (the robot's left)."""
    engine, lidar = mounted
    origin, dirs, bearings = scan_mount.world_rays(engine, lidar)
    fwd = scan_mount.forward_index(bearings)
    np.testing.assert_allclose(dirs[fwd], [0.0, 1.0, 0.0], atol=1e-3)
    assert lidar.latest.ranges[fwd] == pytest.approx(
        scan_mount.wall_distance(origin, dirs[fwd]), abs=1e-6
    )


def test_d3_no_ray_starts_inside_the_robot_or_returns_from_its_own_mount(mounted):
    """D3: every ray meets its first surface from outside, and the housing is the only exclusion."""
    engine, lidar = mounted
    mount_body = scan_mount.mount_body(engine, "rplidar")
    dirs, hits = scan_mount.recast(engine, lidar, bodyexclude=mount_body)
    assert lidar._bodyexclude == mount_body
    hit = hits.geomid >= 0
    assert hit.all(), "a closed room leaves no ray without a return"
    facing = np.einsum("ij,ij->i", hits.normal[hit], dirs[hit])
    assert not np.any(facing > 0), f"{int((facing > 0).sum())} ray(s) start inside robot geometry"
    own = lidar._hits.geomid
    assert not np.any(engine.ctx.model.geom_bodyid[own[own >= 0]] == mount_body)


def test_d4_what_the_scan_sees_of_the_robot_is_pinned(mounted):
    """D4: 29 of 360 rays return the four tower standoffs and the camera bracket, from outside.

    360 rays because Clearpath's TurtleBot 4 datasheet gives this robot's lidar a 1 deg angular
    resolution, which the manifest overrides onto the device. Real returns: the scan plane passes
    through the tower, as on the robot. All of them lie nearer than the A1's 0.15 m minimum range, so
    the published scan carries them as too close (``-inf``), never as a measured distance.
    """
    engine, lidar = mounted
    assert lidar.num_rays == 360
    _, hits = scan_mount.recast(engine, lidar)
    bodies, meshes = scan_mount.robot_returns(engine, hits)
    assert bodies == {"r_base_link"}
    assert meshes == {"r_tower_standoff", "r_camera_bracket"}
    robot = scan_mount.robot_rays(engine, hits)
    assert int(robot.sum()) == 29
    assert float(hits.dist[robot].max()) < lidar.detection_min
    assert np.all(np.asarray(lidar.latest.ranges)[robot] == -np.inf)


def test_d5_the_static_tf_chain_is_published(mounted):
    """D5: base_link -> shell_link from the robot, shell_link -> rplidar_link from the mount."""
    engine, _ = mounted
    (shell,) = scan_mount.static_tf(engine, "robot")
    (scan,) = scan_mount.static_tf(engine, "robot.rplidar")
    assert (shell["parent"], shell["child"]) == ("base_link", "shell_link")
    assert (scan["parent"], scan["child"]) == ("shell_link", "rplidar_link")
    for tf, joint in ((shell, SHELL_LINK), (scan, RPLIDAR_JOINT)):
        np.testing.assert_allclose(tf["translation"], joint[0], atol=1e-9)
        assert scan_mount.same_rotation(tf["rotation"], scan_mount.chain(joint)[1])


def test_d6_the_scan_topic_is_the_robots(mounted):
    """D6: `scan`, relative, in the robot's namespace, stamped rplidar_link; the mount owns the TF."""
    engine, _ = mounted
    scan = scan_mount.scan_endpoint(engine)
    assert scan.owner == "robot.rplidar" and scan.namespace == scan_mount.NAMESPACE
    assert scan.backend["ros2"]["topic"] == "scan"
    assert scan.backend["ros2"]["frame_id"] == "rplidar_link"
    assert "static_tf" not in scan.backend["ros2"]


def test_d7_the_scanner_mass_is_the_devices(mounted):
    """D7: the spawned robot is the base, wheels and suspension plus the A1's own 0.17 kg, which
    the MJCF does not carry (the device model does)."""
    engine, _ = mounted
    m = engine.ctx.model
    base = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "r_base_link")
    assert float(m.body_subtreemass[base]) == pytest.approx(TOTAL_MASS + RPLIDAR_MASS, abs=1e-6)


# --------------------------------------------------------------------------- E. the Create 3's own surface


def _manifest_entries(kind: str) -> list[dict]:
    return [e for e in yaml.safe_load(MANIFEST.read_text())["components"] if kind in e]


def test_e1_the_manifest_declares_the_create3_surface():
    """E1: what the reference base carries, declared on the model so a world says nothing.

    The topic names are the Create 3 simulator's raw-stream names, so its adapter nodes read them
    with their shipped parameter files; the bumper zones are its sector table (bumpers.hpp); every
    endpoint only that stack reads is lazy.
    """
    bumper = _manifest_plugin("bumper")
    assert bumper["geoms"] == ["body_collision"]
    assert list(bumper["zones"]) == [
        "bump_right",
        "bump_front_right",
        "bump_front_center",
        "bump_front_left",
        "bump_left",
    ]
    assert bumper["zones"]["bump_front_center"] == pytest.approx([-math.pi / 10, math.pi / 10])
    assert bumper["zones"]["bump_left"] == pytest.approx([3 * math.pi / 10, math.pi / 2])

    sensors = {e["name"]: e["range_sensor"] for e in _manifest_entries("range_sensor")}
    assert set(sensors) == CREATE3_SITES - {"mouse", "ir_omni"}
    for name, cfg in sensors.items():
        assert cfg["site"] == name
        assert cfg["topics"] == {"range": f"_internal/{name}/scan"}
        assert cfg["lazy"] is True and cfg["exclude_body"] == "base_link"
        assert cfg["rate_hz"] == 62
        if name.startswith("cliff"):
            assert (cfg["range_min"], cfg["max_range"]) == (0.0001, 0.15)
            assert "h_rays" not in cfg
        else:
            assert (cfg["h_rays"], cfg["v_rays"]) == (5, 5)
            assert cfg["h_fov"] == pytest.approx(math.radians(10), abs=1e-6)
            assert (cfg["range_min"], cfg["max_range"]) == (0.025, 0.2)

    poses = {e["name"]: e["ground_truth_pose"] for e in _manifest_entries("ground_truth_pose")}
    assert poses["gt_base"]["child_frame"] == "turtlebot4"
    assert poses["gt_mouse"] == {
        "site": "mouse",
        "relative_to": "base",
        "rate_hz": 62,
        "lazy": True,
        "topics": {"pose": "_internal/sim_ground_truth_pose"},
    }
    assert poses["gt_ir_omni"]["site"] == "ir_omni"

    assert _manifest_plugin("imu")["topic"] == "imu"
    assert _manifest_plugin("imu")["pos"] == pytest.approx([0.050613, 0.043673, 0.0844])
    assert _manifest_plugin("diff_drive") == {"publish_joint_states": False}
    assert _manifest_plugin("joint_state_publisher") == {"rate_hz": 62}


@pytest.fixture(scope="module")
def create3():
    """The robot spawned as a world spawns it, in the room, camera off, settled on the floor."""
    engine = scan_mount.spawn("turtlebot4", ["rplidar"], disabled=("robot.oakd_camera",))
    for _ in range(500):
        engine.step()
    yield engine
    engine.shutdown()


def _by_topic(engine, topic):
    return next(
        e for e in engine.ctx.interface.all() if e.backend.get("ros2", {}).get("topic") == topic
    )


def test_e2_the_cliff_sensors_read_the_floor_and_the_ir_sensors_read_nothing(create3):
    """E2: on a floor the four cliff rays return under the 3 cm a cliff detector compares with,
    and the seven IR grids see nothing within their 20 cm in an open room."""
    for name in ("cliff_front_left", "cliff_front_right", "cliff_side_left", "cliff_side_right"):
        (r,) = _by_topic(create3, f"_internal/{name}/scan").read().ranges
        assert 0.005 < r < 0.03, f"{name}: {r}"
    for name in (
        "ir_intensity_front_center_left",
        "ir_intensity_front_left",
        "ir_intensity_right",
        "ir_intensity_side_left",
    ):
        ranges = _by_topic(create3, f"_internal/{name}/scan").read().ranges
        assert ranges.shape == (25,)
        assert np.all(np.isposinf(ranges)), f"{name} sees something: {ranges}"


def test_e3_one_joint_states_message_carries_wheels_and_suspension(create3):
    """E3: the message a consumer derives the wheel state from also carries the suspension."""
    (js,) = [e for e in create3.ctx.interface.all() if e.name == "joint_states"]
    names, pos, vel, eff = js.read()
    assert set(names) == {
        "left_wheel_joint",
        "right_wheel_joint",
        "wheel_drop_left_joint",
        "wheel_drop_right_joint",
    }
    assert js.rate_hz == 62.0
    # On the floor the suspension rides on its stop: under the detector's 2.25 cm release.
    for side in ("left", "right"):
        assert pos[names.index(f"wheel_drop_{side}_joint")] < 0.0225


def test_e4_the_ground_truth_stream_is_the_adapters_contract(create3):
    """E4: the base under the robot's name in the world, the mouse and IR receiver relative to it."""
    poses = [e for e in create3.ctx.interface.all() if e.name == "pose"]
    by_child = {e.read()[0][0]: e for e in poses}
    assert set(by_child) == {"turtlebot4", "mouse", "ir_omni"}
    assert by_child["turtlebot4"].backend["ros2"]["frame_id"] == "map"
    _, pos, _ = by_child["mouse"].read()[0]
    assert np.allclose(pos, [0.1015, 0.087, 0.0092], atol=1e-6)
    assert by_child["mouse"].backend["ros2"]["frame_id"] == "base_link"
    _, pos, _ = by_child["ir_omni"].read()[0]
    assert np.allclose(pos, [0.153, 0.0, 0.0992], atol=1e-6)
    assert all(e.lazy for e in poses)


def test_e5_lifting_the_robot_drops_the_wheels_and_opens_the_cliffs():
    """E5: what a kidnap looks like to the stack: both suspensions past the detector's 2.85 cm
    and every cliff sensor reading no return."""
    engine = scan_mount.spawn("turtlebot4", ["rplidar"], disabled=("robot.oakd_camera",))
    try:
        for _ in range(500):
            engine.step()
        d = engine.ctx.data
        held = d.qpos[:7].copy()
        held[2] += 0.3
        for _ in range(500):
            # Held in the air, as a hand holds it: the base's pose is pinned every step.
            d.qpos[:7] = held
            d.qvel[:6] = 0.0
            engine.step()
        (js,) = [e for e in engine.ctx.interface.all() if e.name == "joint_states"]
        names, pos, *_ = js.read()
        for side in ("left", "right"):
            assert pos[names.index(f"wheel_drop_{side}_joint")] >= 0.0285
        for name in ("cliff_front_left", "cliff_side_right"):
            (r,) = _by_topic(engine, f"_internal/{name}/scan").read().ranges
            assert np.isposinf(r), f"{name} lifted 30 cm still reads {r}"
    finally:
        engine.shutdown()


def test_e6_driving_into_the_wall_presses_the_centre_bumper_zone():
    """E6: the bumper over the body shell: a head-on wall presses bump_front_center and nothing else,
    and it releases when the robot backs off."""
    engine = scan_mount.spawn("turtlebot4", ["rplidar"], disabled=("robot.oakd_camera",))
    try:
        handle = engine.ctx.blackboard.get(f"robot:{scan_mount.OWNER}")
        read = engine.ctx.blackboard.get(f"bumper:{scan_mount.OWNER}.bumper")
        handle.drive(0.3, 0.0, 0.0)
        pressed = set()
        for _ in range(int(20.0 / engine.ctx.model.opt.timestep)):
            engine.step()
            r = read()
            if r.any_pressed:
                pressed |= {z for z, p in r.pressed.items() if p}
                break
        else:
            pytest.fail("never reached the wall")
        assert pressed == {"bump_front_center"}
        assert _by_topic(engine, "bumper/bump_front_center").read() is True
        handle.drive(-0.3, 0.0, 0.0)
        for _ in range(int(1.0 / engine.ctx.model.opt.timestep)):
            engine.step()
        assert read().any_pressed is False
    finally:
        engine.shutdown()


def test_e7_the_dock_is_a_prop_with_the_emitter_frames_the_stack_ranges_by():
    """E7: the charging dock placed as the reference simulator places it -- 0.157 m ahead of the
    robot's spawn, turned to face it -- with the halo emitter's ground truth published under the
    dock's name, relative to the dock, as the receiver's is relative to the robot."""
    from roqsim.config import load_config_from_dict
    from roqsim.engine import Engine

    world = {
        "sim": {"timestep": 0.002},
        "components": [
            {"spawn_robot": {"model": "turtlebot4"}, "name": "robot"},
            {
                "spawn_model": {
                    "model": "create3_dock",
                    "pose": {
                        "position": {"x": 0.157, "y": 0.0, "z": 0.0},
                        "orientation": {"yaw": math.pi},
                    },
                },
                "name": "standard_dock",
                "components": [
                    {
                        "ground_truth_pose": {
                            "child_frame": "standard_dock",
                            "topics": {"pose": "_internal/sim_ground_truth_dock_pose"},
                        },
                        "name": "gt_dock",
                    },
                    {
                        "ground_truth_pose": {
                            "site": "halo_link",
                            "relative_to": "base",
                            "topics": {"pose": "_internal/sim_ground_truth_dock_pose"},
                        },
                        "name": "gt_halo",
                    },
                ],
            },
        ],
    }
    overrides = {"components": {"robot.oakd_camera": {"enabled": False}}}  # no GL needed
    engine = Engine(load_config_from_dict(world, base_dir=Path("."), overrides=overrides))
    engine.ctx.seed = 0
    engine.setup()
    engine.reset()
    try:
        for _ in range(200):
            engine.step()
        poses = {
            e.read()[0][0]: e
            for e in engine.ctx.interface.all()
            if e.backend.get("ros2", {}).get("topic") == "_internal/sim_ground_truth_dock_pose"
        }
        assert set(poses) == {"standard_dock", "halo_link"}
        _, dock_pos, dock_quat = poses["standard_dock"].read()[0]
        assert np.allclose(dock_pos[:2], [0.157, 0.0], atol=1e-4)
        assert abs(float(dock_quat[3])) == pytest.approx(1.0, abs=1e-3), "turned to face the robot"
        _, halo, _ = poses["halo_link"].read()[0]
        assert np.allclose(halo, [-0.06, 0.0, 0.0904], atol=1e-6)
        # A static prop: the robot did not push it while settling next to it.
        m = engine.ctx.model
        assert mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "std_dock_link") >= 0
    finally:
        engine.shutdown()
