"""TurtleBot3 Waffle drive-test battery (port verification).

Mirrors the robot-porting verification battery and its siblings `test_husky_scene.py` /
`test_jackal_scene.py`: static sanity (A), open-loop drive tests (B) and sensor checks (C).
Everything is driven through the real ``diff_drive`` plugin with the shipped manifest, so the model
and the controller are verified together.

Reference dimensions come from ROBOTIS's own MuJoCo model (robotis_tb3 @ d8344c0, see
turtlebot3_waffle_LICENSE) cross-checked against turtlebot3_description: wheel r=0.033 m,
w=0.0184 m, m=0.0285 kg; track 0.288 m; base mass 1.8 kg; base_link rest height 0.010 m;
base_scan 0.122 m above base_link (0.132 m above the floor).

**Unlike the Husky and the Jackal, this is a true differential drive** — two driven wheels and two
passive frictionless casters, so turning rolls instead of scrubbing. There is no ``slip_factor`` to
calibrate and the tolerances here are correspondingly tight: yaw ratio within 5%, arc odometry
within centimetres. A loose tolerance passing here would hide a real defect.
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
MODEL_DIR = MODELS / "turtlebot3_waffle"
MANIFEST = MODEL_DIR / "turtlebot3_waffle.manifest.yaml"

WHEEL_R = 0.033
WHEEL_W = 0.0184
TRACK = 0.288
REST_Z = 0.010  # base_link above the floor = wheel r 0.033 - offset 0.023
TOTAL_MASS = 1.8 + 2 * 0.0285  # 1.857 kg
PLATE_L, PLATE_W = 0.272, 0.276  # chassis plate (the datasheet's 0.306 m width is the wheels)
HULL_W = 2 * 0.144 + WHEEL_W  # 0.3064 m — datasheet 0.306 m
LIDAR_Z_BASE = 0.122  # turtlebot3_description base_scan
LIDAR_Z_GROUND = LIDAR_Z_BASE + REST_Z  # 0.132
# Datasheet rated limits, which the actuator ctrlrange (+/-7.88 rad/s) encodes.
MAX_V = 0.26
MAX_W = 1.82


def _manifest_plugin(kind: str) -> dict:
    """The plugin config the model actually ships with (the manifest is the source of truth)."""
    for entry in yaml.safe_load(MANIFEST.read_text())["components"]:
        if kind in entry:
            return dict(entry[kind])
    raise AssertionError(f"turtlebot3_waffle manifest has no {kind} plugin")


def _build(gravity=None):
    """Compose the robot with a ground plane (high friction, as a real scene floor is)."""
    asset = resolve_model("roqsim_mobile:turtlebot3_waffle")
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
    """Drive (v, w) for `seconds`; return ground truth + the plugin's encoder odometry."""
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
    """A1: 10 s of stepping at the campaign timestep — no divergence, no MuJoCo warnings."""
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
    """A2: total and per-body masses match the vendor model; no near-zero wheel inertia."""
    model, _ = _build()
    base = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    assert float(model.body_subtreemass[base]) == pytest.approx(TOTAL_MASS, rel=0.01)
    assert float(model.body_mass[base]) == pytest.approx(1.8, rel=0.01)
    for side in ("left", "right"):
        wid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_wheel")
        assert float(model.body_mass[wid]) == pytest.approx(0.0285, rel=0.01)
        assert np.all(model.body_inertia[wid] > 1e-6), "near-zero wheel inertia"


def test_a3_rest_stability():
    """A3: settles onto two wheels + two casters at the URDF rest height and stays there."""
    model, data = _build()
    for _ in range(int(2.0 / model.opt.timestep)):
        mujoco.mj_step(model, data)
    assert float(data.qpos[2]) == pytest.approx(REST_Z, abs=0.003)
    assert np.linalg.norm(data.qvel[:3]) < 1e-3  # < 1 mm/s drift
    assert np.linalg.norm(data.qpos[:2]) < 1e-3


def test_a4_wheel_servo_is_stable_at_dt():
    """A4: the velocity servo's time constant must not fall below the timestep.

    The bare wheel inertia here is tiny (0.0285 kg wheel), so the armature (reflected drivetrain
    inertia, 0.01) is what keeps kv*dt/I integrable — the same failure the Jackal port hit.
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
    """A5: wheel radius/width, track, plate and hull match the vendor description + datasheet."""
    model, data = _build()

    def body_xy(name):
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        return data.xpos[bid][:2]

    left, right = body_xy("left_wheel"), body_xy("right_wheel")
    assert abs(left[1] - right[1]) == pytest.approx(TRACK, abs=1e-3)
    assert abs(left[0] - right[0]) < 1e-6, "wheels must share one axle (x)"

    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "left_wheel_geom")
    assert float(model.geom_size[gid][0]) == pytest.approx(WHEEL_R, abs=1e-4)
    assert float(model.geom_size[gid][1] * 2) == pytest.approx(WHEEL_W, abs=1e-3)

    # Chassis collision box vs the plate. Cross-checked against the visual MESH bounds below, which
    # are authored independently of these numbers (they come out of the vendor STL).
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "tb3_body_collision")
    assert float(model.geom_size[bid][0] * 2) == pytest.approx(PLATE_L, abs=0.005)
    assert float(model.geom_size[bid][1] * 2) == pytest.approx(PLATE_W, abs=0.005)
    assert HULL_W == pytest.approx(0.306, abs=0.002)


def test_a6_visual_meshes_are_not_bounding_spheres():
    """A6: every visual mesh geom is a real mesh with plausible extents.

    Guards the two silent mesh traps the porting skill documents: a `density="0"` visual mesh renders
    as its bounding sphere, and a decimation/axis mistake preserves nothing but the vertex count. The
    chassis mesh's own bounds are the independent check on the collision box asserted in A5.
    """
    model, _ = _build()
    # Sorted extents, because MuJoCo re-frames `mesh_vert` into a canonical inertial frame: the axis
    # ORDER in mesh_vert is an artefact of that reframing (the chassis reads 0.120 on x, which is its
    # height), while the multiset of extents is frame-independent. Orientation is checked by the
    # geometry tests (A5, C1) and visually; scale and decimation are checked here.
    names = {
        "tb3_waffle_base": sorted([PLATE_L, PLATE_W, 0.121]),
        "tb3_left_tire": sorted([2 * WHEEL_R, 2 * WHEEL_R, WHEEL_W]),
        "tb3_right_tire": sorted([2 * WHEEL_R, 2 * WHEEL_R, WHEEL_W]),
    }
    for mesh_name, want in names.items():
        mid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MESH, mesh_name)
        assert mid >= 0, f"mesh {mesh_name} missing"
        start = int(model.mesh_vertadr[mid])
        n = int(model.mesh_vertnum[mid])
        assert n > 500, f"{mesh_name}: only {n} vertices — decimated to nothing?"
        v = model.mesh_vert[start : start + n]
        ext = sorted(float(e) for e in (v.max(axis=0) - v.min(axis=0)))
        for got, exp in zip(ext, want, strict=True):
            assert got == pytest.approx(exp, abs=0.01), f"{mesh_name} extents {ext} != {want}"

    # And the mesh geoms really are visual-only, so the primitives own all contact.
    for gid in range(model.ngeom):
        if model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_MESH:
            assert model.geom_contype[gid] == 0 and model.geom_conaffinity[gid] == 0


# --------------------------------------------------------------------------- B. open-loop drive


def test_b1_straight_line():
    """B1: commanded speed is achieved, no lateral drift, odometry tracks ground truth.

    Tight tolerance on purpose: this is the check that catches the vendor's kv=0.1, which sits ~13%
    below command against the joints' frictionloss (see the MJCF actuator comment).
    """
    r = _run(0.2, 0.0, 5.0)
    assert r["speed"] == pytest.approx(
        0.2, abs=0.01
    )  # measured 0.1955 (2.3% low: 0.3% servo + 2% slip)
    assert abs(r["y"]) < 0.005
    # Encoder odometry OVER-counts, because the wheels turn ~2% further than the robot travels
    # (soft-contact slip at 1.86 kg). Measured +0.0197 m over 0.958 m. Assert the magnitude AND the
    # sign: odom running BEHIND ground truth would mean the wheels are being dragged, not slipping.
    over = r["odom"][0] - r["x"]
    assert 0.005 < over < 0.04, f"odometry over-count {over:+.4f} m over {r['x']:.3f} m"
    assert over / r["x"] < 0.04


def test_b2_in_place_rotation():
    """B2: a true diff-drive rolls its yaw, so achieved/commanded must be within 5% with no slip
    compensation anywhere in the model or the manifest."""
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
    """B4: the model saturates at the datasheet limits Nav2 will also be given.

    Saturation is asserted against the datasheet with the measured slip deficit allowed for: at full
    command the model reaches 0.253 m/s of 0.26 (2.6% low) and 1.70 rad/s of 1.82 (6.5% low). What
    matters for a controller comparison is that the ceiling is the platform's, not the solver's —
    over-shooting the datasheet would be the real failure, so the upper bound is tight.
    """
    v = _run(1.0, 0.0, 4.0)["speed"]
    assert MAX_V * 0.95 <= v <= MAX_V + 0.005, f"top speed {v:.4f} vs datasheet {MAX_V}"
    w = _run(0.0, 3.0, 4.0)["yaw_rate"]
    assert MAX_W * 0.90 <= w <= MAX_W + 0.05, f"top yaw rate {w:.4f} vs datasheet {MAX_W}"


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

    Tighter than the skid-steer siblings by an order of magnitude (they drift ~0.3 m over 5 s):
    rolling wheels mean the encoders actually describe the motion.
    """
    r = _run(0.15, 0.5, 5.0)
    assert r["yaw"] > 2.0  # turned left by more than ~115 deg
    assert r["y"] > 0.1  # and moved left
    drift = math.hypot(r["odom"][0] - r["x"], r["odom"][1] - r["y"])
    assert drift < 0.03, f"odom drift {drift:.3f} m over a 5 s arc"


def test_b7_casters_do_not_drag():
    """B7: the frictionless casters must not resist yaw.

    condim=1 (normal force only) is what makes them ball casters; MuJoCo max-combines friction
    coefficients, so a low-friction geom on a friction-2.0 floor would otherwise still drag and the
    robot would under-rotate. Asserts the mechanism, not just the outcome.
    """
    model, _ = _build()
    for name in ("tb3_caster_left", "tb3_caster_right"):
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        assert gid >= 0, f"{name} missing"
        assert int(model.geom_condim[gid]) == 1, f"{name}: condim {model.geom_condim[gid]} != 1"


# --------------------------------------------------------------------------- C. sensors


def test_c1_lidar_mount_height():
    """C1: the scan plane sits at turtlebot3_description's base_scan (0.132 m above ground)."""
    model, data = _build()
    for _ in range(int(2.0 / model.opt.timestep)):
        mujoco.mj_step(model, data)
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "lidar")
    assert float(data.site_xpos[sid][2]) == pytest.approx(LIDAR_Z_GROUND, abs=0.003)
    assert float(data.site_xpos[sid][0]) == pytest.approx(-0.064, abs=0.003)
    assert abs(float(data.site_xpos[sid][1])) < 0.003


def test_c2_lidar_manifest_matches_the_lds01():
    """C2: the shipped lidar config is the platform's own sensor, not a placeholder.

    The paper states no lidar parameter at all (spec gap g_lidar_params), so the manifest values ARE
    the assumption of record — pin them here so a future edit is deliberate.
    """
    cfg = _manifest_plugin("lidar")
    assert cfg["rays"] == 360
    assert cfg["range_min"] == pytest.approx(0.12)
    assert cfg["max_range"] == pytest.approx(3.5)
    assert cfg["rate_hz"] == pytest.approx(5.0)
    assert cfg["angle_max"] == pytest.approx(2 * math.pi, abs=1e-6)
    assert cfg["frame_id"] == "base_scan"


def test_c3_wheel_encoders_and_imu_exist():
    """C3: the sensors a ROS 2 bridge publishes are present and named as the siblings' are."""
    model, _ = _build()
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


def test_c4_odometry_tf_points_at_the_description_root():
    """C4: ``odom ->`` targets base_footprint, the root of every published TurtleBot 3 URDF.

    A robot_state_publisher running that description already parents base_link to base_footprint,
    so an odometry TF aimed at base_link would give the frame two parents and tf2 would resolve
    neither. The default is base_link (a TurtleBot 4 description is rooted there), which is why
    this platform has to state it.
    """
    assert _manifest_plugin("diff_drive")["odom_child_frame"] == "base_footprint"

    model, data = _build()
    ctx, _ = _plugin(model, data)
    odom = next(e for e in ctx.interface.all() if e.name == "odom")
    assert odom.backend["ros2"]["frame_id"] == "odom"
    assert odom.backend["ros2"]["child_frame_id"] == "base_footprint"


def test_c5_odometry_tf_default_is_base_link():
    """C5: platforms that say nothing keep base_link, so this change moves no existing robot."""
    model, data = _build()
    cfg = {k: v for k, v in _manifest_plugin("diff_drive").items() if k != "odom_child_frame"}
    ctx = SimContext(config={})
    ctx.model, ctx.data = model, data
    ctx.entities.add(
        Entity(name="robot", kind="robot", body="base_link", meta={"prefix": "", "namespace": ""})
    )
    plugin = DiffDrivePlugin(cfg)
    plugin.configure(ctx)
    odom = next(e for e in ctx.interface.all() if e.name == "odom")
    assert odom.backend["ros2"]["child_frame_id"] == "base_link"


def test_c6_cmd_vel_type_follows_the_stack():
    """C6: the velocity command's message type is the stack's choice, not the plugin's.

    Nav2 publishes ``geometry_msgs/TwistStamped`` when its own ``enable_stamped_cmd_vel`` is set
    -- the TurtleBot 4's shipped configuration does -- and a subscription is one type, so a
    mismatch delivers nothing at all rather than something degraded. The failure has no message
    of its own: the robot simply never moves while the controller reports it cannot make progress.
    """
    model, data = _build()

    ctx, _ = _plugin(model, data)
    plain = next(e for e in ctx.interface.all() if e.name == "cmd_vel")
    assert plain.backend["ros2"]["type"] == "geometry_msgs.msg.Twist"

    ctx, _ = _plugin(model, data, stamped_cmd_vel=True)
    stamped = next(e for e in ctx.interface.all() if e.name == "cmd_vel")
    assert stamped.backend["ros2"]["type"] == "geometry_msgs.msg.TwistStamped"
    assert stamped.backend["ros2"]["topic"] == plain.backend["ros2"]["topic"]
