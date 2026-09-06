"""Frankie (Panda + Omron LD-60) verification battery — mobile-manipulator port.

Follows robot-porting's `references/mobile_manipulator.md`: the base battery (A static, B drive) plus
the arm battery (E), run through the REAL plugins so the model, its manifest config and the two
controllers are verified together. The point of the split-controller check is that `diff_drive` and
`arm_controller` own disjoint actuators — if `arm_controller` ever grabs the wheel motors, the robot
stops driving, and that failure is silent in a plain "does it compile" test.

Unlike the husky and the G2, Frankie is a TRUE two-wheel differential drive with passive casters: it
rolls rather than scrubs to turn, so B2 asserts the ideal kinematics (±10 %) instead of the loose
scrub-limited bound those two need, and no `slip_factor` calibration exists to hide behind.

Reference geometry comes from `qut_frankie_description` (robotics-toolbox-python @ 0bb96454):
base collision box 0.68 × 0.47 × 0.38 m, arm mount at xyz (0.15, 0, 0.38) with no rotation.
Wheel/caster/mass values are substrate assumptions.
"""

from __future__ import annotations

import math
from pathlib import Path

import mujoco
import numpy as np
import pytest
import yaml
from roqsim_manipulation.plugins.arm_controller import ArmControllerPlugin

from roqsim.context import Entity, SimContext
from roqsim.models import apply_assets, resolve_model
from roqsim_mobile.plugins.diff_drive import DiffDrivePlugin

MODELS = Path(__file__).resolve().parents[1] / "src" / "roqsim_mobile_manipulation" / "models"
MODEL_DIR = MODELS / "frankie"
MANIFEST = MODEL_DIR / "frankie.manifest.yaml"

# From the source URDF (authoritative)
MOUNT = np.array([0.15, 0.0, 0.38])
BOX = np.array([0.68, 0.47, 0.38])
# Substrate assumptions (port log)
WHEEL_R = 0.0625
TRACK = 0.36
ARM_REST = (0.0, -0.3, 0.0, -2.2, 0.0, 2.0, math.pi / 4)
ARM_JOINTS = [f"joint{i}" for i in range(1, 8)]
VERIFIED_TIMESTEP = 0.0005  # must match build_frankie_mjcf.VERIFIED_TIMESTEP and the world YAML
# With explicit `joints:` ownership arm_controller namespaces its handles by controller name, so that
# two controllers on one entity cannot silently overwrite each other (see its configure()).
ARM_KEY = "arm:robot:arm_controller"
GRIP_KEY = "gripper:robot:gripper_controller"


def _manifest(plugin: str) -> dict:
    """The config the model actually ships with — the manifest is the source of truth."""
    for entry in yaml.safe_load(MANIFEST.read_text())["components"]:
        if plugin in entry:
            return dict(entry[plugin])
    raise AssertionError(f"frankie manifest has no {plugin} plugin")


def _build(wall_x: float | None = None):
    """Compose Frankie through the real substrate path (resolve_model + apply_assets), plus a floor.

    Going through `resolve_model` rather than pointing MjSpec at the XML is deliberate: it exercises
    the manifest's `assets: [roqsim_manipulation_assets]` key, i.e. that the borrowed Panda meshes resolve from
    another package. A bare load cannot find them at all.
    """
    asset = resolve_model("frankie")
    spec = mujoco.MjSpec.from_file(str(asset.path))
    apply_assets(spec, asset)
    floor = spec.worldbody.add_geom()
    floor.name = "floor"
    floor.type = mujoco.mjtGeom.mjGEOM_PLANE
    floor.size = [15, 15, 0.05]
    floor.friction = [1.0, 0.005, 0.0001]
    if wall_x is not None:
        # A target for the laser. The scan tests need something to measure; without it every ray
        # returns `inf` and "the scan is not blocked" would pass on an empty world just as well as
        # on a working sensor.
        wall = spec.worldbody.add_geom()
        wall.name = "wall"
        wall.type = mujoco.mjtGeom.mjGEOM_BOX
        wall.size = [0.05, 6.0, 1.0]
        wall.pos = [wall_x, 0.0, 1.0]
    model = spec.compile()
    # The timestep the model is verified at, not the substrate default. At dt=2 ms the stock panda's
    # PD servos produce EE acceleration noise of mean 0.94 m/s^2 with the arm at rest, which would
    # swamp any acceleration measured on it; the drive tests also need a settled arm. See
    model.opt.timestep = VERIFIED_TIMESTEP
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def _ctx(model, data) -> SimContext:
    ctx = SimContext(config={})
    ctx.model, ctx.data = model, data
    ctx.entities.add(
        Entity(name="robot", kind="robot", body="base_link", meta={"prefix": "", "namespace": ""})
    )
    return ctx


def _drive(ctx, **overrides):
    p = DiffDrivePlugin({**_manifest("diff_drive"), **overrides}, entity="robot")
    p.configure(ctx)
    p.on_reset(ctx)
    return p


def _arm(ctx, **overrides):
    p = ArmControllerPlugin({**_manifest("arm_controller"), **overrides}, entity="robot")
    p.configure(ctx)
    p.on_reset(ctx)
    return p


def _mesh_verts(name: str) -> np.ndarray:
    """Vertices of one of the base's OBJ meshes, in the base frame.

    Read from the mesh FILE rather than the compiled model: these tests assert that the site agrees
    with the CAD, and a compiled model that had lost the mesh would let that assertion pass against
    whatever the site itself says.
    """
    path = MODEL_DIR / "meshes" / f"{name}.obj"
    verts = [
        [float(v) for v in line.split()[1:4]]
        for line in path.read_text().splitlines()
        if line.startswith("v ")
    ]
    assert verts, f"{path} has no vertices"
    return np.array(verts)


def _inside_chassis(model, pos) -> bool:
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "chassis")
    half, centre = model.geom_size[gid], model.geom_pos[gid]
    return bool(np.all(np.abs(np.asarray(pos) - centre) <= half))


def _scan(plugin, ctx, model, data) -> np.ndarray:
    """One full sweep from a configured lidar plugin, as an array of ranges."""
    plugin.configure(ctx)
    plugin.on_reset(ctx)
    mujoco.mj_forward(model, data)
    plugin.post_step(ctx)
    return np.asarray(plugin.latest.ranges, dtype=float)


def _yaw(data) -> float:
    w, x, y, z = data.qpos[3:7]
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def _settle(model, data, arm, drive=None, seconds=2.0):
    for _ in range(int(seconds / model.opt.timestep)):
        arm.pre_step(_CTX_CACHE[id(model)])
        if drive is not None:
            drive.pre_step(_CTX_CACHE[id(model)])
        mujoco.mj_step(model, data)


_CTX_CACHE: dict[int, SimContext] = {}


@pytest.fixture
def rig():
    model, data = _build()
    ctx = _ctx(model, data)
    _CTX_CACHE[id(model)] = ctx
    arm = _arm(ctx)
    drive = _drive(ctx)
    # Let it settle onto its wheels with the arm held at rest.
    _settle(model, data, arm, drive=None, seconds=1.5)
    return model, data, ctx, arm, drive


# --------------------------------------------------------------------------------------- A: static
def test_a1_rest_height_and_level(rig):
    """Settles on its wheels at the wheel radius, level, without sinking or tipping."""
    model, data, *_ = rig
    assert data.qpos[2] == pytest.approx(0.0, abs=0.005), "base_link should rest at z~0"
    w, x, y, z = data.qpos[3:7]
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
    assert abs(math.degrees(roll)) < 1.0, f"rolled {math.degrees(roll):.2f} deg"
    assert abs(math.degrees(pitch)) < 1.5, f"pitched {math.degrees(pitch):.2f} deg"


def test_a2_drive_wheels_carry_the_load(rig):
    """The drive wheels must carry most of the weight — never the chassis box on the floor.

    Measured as a TIME-AVERAGED normal force, not a contact snapshot. With four near-coplanar support
    geoms a single frame reports whichever happens to be penetrating that tick: an early version of
    this model was found "resting on caster_front alone with both drive wheels off the ground" by a
    snapshot, while the averaged forces showed all four sharing load. Snapshots lie here.
    """
    model, data, ctx, arm, _ = rig
    floor = model.geom("floor").id
    force: dict[str, float] = {}
    N = 400
    for _ in range(N):
        arm.pre_step(ctx)
        mujoco.mj_step(model, data)
        for i, c in enumerate(data.contact[: data.ncon]):
            other = c.geom2 if c.geom1 == floor else (c.geom1 if c.geom2 == floor else None)
            if other is None:
                continue
            ft = np.zeros(6)
            mujoco.mj_contactForce(model, data, i, ft)
            nm = model.geom(other).name
            force[nm] = force.get(nm, 0.0) + abs(float(ft[0])) / N
    assert force, "nothing is touching the floor"
    assert "chassis" not in force, f"chassis box is resting on the floor: {sorted(force)}"
    total = sum(force.values())
    weight = model.body_subtreemass[model.body("base_link").id] * 9.81
    assert total == pytest.approx(weight, rel=0.10), (
        f"support {total:.0f} N vs weight {weight:.0f} N"
    )
    wheels = force.get("left_wheel", 0.0) + force.get("right_wheel", 0.0)
    assert wheels / total > 0.30, f"drive wheels carry only {wheels / total:.1%} of the load"


def test_a3_mount_transform_matches_source_urdf(rig):
    """panda_link0 sits exactly at the URDF's fixed-joint mount frame. Cross-checks the whole port.

    Compared in the BASE frame, not the world frame. The settled base can carry a small pitch, and a
    world-frame delta then mixes the mount offset with that rotation: an earlier version of this test
    failed by 4.3 mm in x, which is exactly 0.38 m * sin(0.65 deg) — the pitch, not a mount error.
    """
    model, data, *_ = rig
    base = data.xpos[model.body("base_link").id]
    R = data.xmat[model.body("base_link").id].reshape(3, 3)
    link0 = data.xpos[model.body("link0").id]
    np.testing.assert_allclose(R.T @ (link0 - base), MOUNT, atol=1e-4)


def test_a4_footprint_matches_declared_box(rig):
    """The simulated chassis footprint equals the URDF's declared collision box in x/y.

    Planner-facing and simulated footprints must agree (robot-porting Step 5). Height deliberately
    differs: the box is raised clear of the wheels — see the port log.
    """
    model, _, *_ = rig
    half = model.geom_size[model.geom("chassis").id]
    assert half[0] * 2 == pytest.approx(BOX[0], abs=1e-6)
    assert half[1] * 2 == pytest.approx(BOX[1], abs=1e-6)


def test_a5_mass_is_plausible(rig):
    """Total mass in the LD-60 + Panda range. Guards against an inertia-less import."""
    model, _, *_ = rig
    total = model.body_subtreemass[model.body("base_link").id]
    assert 70.0 < total < 95.0, f"total mass {total:.1f} kg outside the plausible band"


def test_a6_no_self_collision_at_rest(rig):
    """No robot-internal contact while the arm is held at its rest stance.

    Regression guard for the pose trap: at ctrl=0 (all joints zero) the stock Panda's link5 and hand
    collision geoms overlap by 0.030 m, so an arm that is not commanded finds a self-colliding pose.
    """
    model, data, *_ = rig
    floor = model.geom("floor").id
    internal = [
        (model.body(model.geom_bodyid[c.geom1]).name, model.body(model.geom_bodyid[c.geom2]).name)
        for c in data.contact[: data.ncon]
        if floor not in (c.geom1, c.geom2)
    ]
    assert not internal, f"self-collision at rest: {internal}"


# ---------------------------------------------------------------------------------------- B: drive
def test_b1_straight_line(rig):
    """Commanded 0.3 m/s for 4 s travels ~1.2 m with little lateral drift or yaw error."""
    model, data, ctx, arm, _ = rig
    drive2 = _drive(ctx, test_cmd=[0.3, 0.0])
    x0, y0 = data.qpos[0], data.qpos[1]
    for _ in range(int(4.0 / model.opt.timestep)):
        arm.pre_step(ctx)
        drive2.pre_step(ctx)
        mujoco.mj_step(model, data)
        drive2.post_step(ctx)
    dx, dy = data.qpos[0] - x0, data.qpos[1] - y0
    assert dx == pytest.approx(1.2, rel=0.12), f"travelled {dx:.3f} m, expected ~1.2 m"
    assert abs(dy) < 0.06, f"lateral drift {dy:.3f} m"
    assert abs(math.degrees(_yaw(data))) < 6.0, f"yaw drift {math.degrees(_yaw(data)):.1f} deg"


def test_b2_in_place_rotation_is_rolled_not_scrubbed(rig):
    """A true diff-drive achieves its commanded yaw rate within 10 %, and stays on the spot.

    This is the test the husky and the G2 cannot pass (their fixed four-wheel bases scrub, achieving
    ~15-30 % of commanded yaw and needing a slip_factor). Frankie has two wheels and passive casters,
    so if this fails the wheel/caster friction or the actuator authority is wrong — not the kinematics.
    """
    model, data, ctx, arm, _ = rig
    w_cmd = 0.6
    drive = _drive(ctx, test_cmd=[0.0, w_cmd])
    x0, y0, yaw0 = data.qpos[0], data.qpos[1], _yaw(data)
    T = 3.0
    for _ in range(int(T / model.opt.timestep)):
        arm.pre_step(ctx)
        drive.pre_step(ctx)
        mujoco.mj_step(model, data)
        drive.post_step(ctx)
    turned = (_yaw(data) - yaw0 + math.pi) % (2 * math.pi) - math.pi
    achieved = turned / T
    assert achieved / w_cmd == pytest.approx(1.0, rel=0.10), (
        f"achieved {achieved:.3f} rad/s of commanded {w_cmd} ({achieved / w_cmd:.2%})"
    )
    drift = math.hypot(data.qpos[0] - x0, data.qpos[1] - y0)
    assert drift < 0.10, f"drifted {drift:.3f} m while turning in place"


def test_b3_velocity_limit_is_enforced(rig):
    """A command above max_linear_vel is clamped by the plugin, not silently obeyed."""
    model, data, ctx, arm, _ = rig
    cfg = _manifest("diff_drive")
    drive = _drive(ctx, test_cmd=[10.0, 0.0])  # far above max_linear_vel
    x0 = data.qpos[0]
    T = 2.0
    for _ in range(int(T / model.opt.timestep)):
        arm.pre_step(ctx)
        drive.pre_step(ctx)
        mujoco.mj_step(model, data)
    speed = (data.qpos[0] - x0) / T
    assert speed <= cfg["max_linear_vel"] * 1.15, (
        f"reached {speed:.3f} m/s despite max_linear_vel {cfg['max_linear_vel']}"
    )


def test_b4_wheel_and_arm_actuators_are_disjoint(rig):
    """The two controllers must not fight over any actuator.

    The mobile-manipulator failure mode: `arm_controller`'s prefix scan grabs every actuator under the
    robot prefix — including the wheel velocity servos — and position-holds them, so `diff_drive`'s
    command is overwritten every tick and the robot will not move. The manifest prevents it with an
    explicit `joints:` list; this asserts the split rather than trusting it.
    """
    model, data, ctx, arm, drive = rig
    arm_ids = {aid for aid, _ in arm._joint_acts} | set(arm._aux_acts)
    wheel_ids = {model.actuator(n).id for n in ("left_wheel_motor", "right_wheel_motor")}
    assert not (arm_ids & wheel_ids), "arm_controller claimed a wheel actuator"
    assert len(arm_ids) == 8, f"expected 7 arm joints + 1 gripper tendon, got {len(arm_ids)}"


# ------------------------------------------------------------------------------- E: arm / velocity
def test_e1_arm_holds_rest_stance_under_gravity(rig):
    """The stance is reached and held — a mobile manipulator's arm must not sag away from it."""
    model, data, ctx, arm, _ = rig
    err = max(
        abs(data.qpos[model.joint(n).qposadr[0]] - v)
        for n, v in zip(ARM_JOINTS, ARM_REST, strict=True)
    )
    assert err < 0.03, f"worst joint deviates {err:.4f} rad from the rest stance"


def test_e2_velocity_command_moves_the_joint_at_the_commanded_rate(rig):
    """Joint-velocity input, integrated into the held target.

    The capability this port added to `arm_controller`: a controller that resolves to joint rates
    (a QP redundancy resolver, a teleop jog) had no path into the substrate before it.
    """
    model, data, ctx, arm, _ = rig
    assert arm.velocity_commands, "frankie manifest must enable velocity_commands"
    handle = ctx.blackboard.get(ARM_KEY)
    assert handle.set_velocities is not None, "ArmHandle must advertise the velocity capability"

    jid = model.joint("joint1").qposadr[0]
    q0 = data.qpos[jid]
    qd, T = 0.20, 1.0
    for _ in range(int(T / model.opt.timestep)):
        handle.set_velocities(["joint1"], [qd])  # refresh, as a real controller streams it
        arm.pre_step(ctx)
        mujoco.mj_step(model, data)
    moved = data.qpos[jid] - q0
    assert moved == pytest.approx(qd * T, rel=0.10), f"moved {moved:.4f} rad, expected {qd * T:.4f}"


def test_e3_velocity_watchdog_stops_a_stale_stream(rig):
    """A command that stops being refreshed must stop the arm, not integrate forever."""
    model, data, ctx, arm, _ = rig
    handle = ctx.blackboard.get(ARM_KEY)
    jid = model.joint("joint1").qposadr[0]
    handle.set_velocities(["joint1"], [0.3])
    # Run well past velocity_timeout_s without ever refreshing the command.
    for _ in range(int((arm.velocity_timeout_s + 1.0) / model.opt.timestep)):
        arm.pre_step(ctx)
        mujoco.mj_step(model, data)
    q_after_timeout = data.qpos[jid]
    for _ in range(int(1.0 / model.opt.timestep)):
        arm.pre_step(ctx)
        mujoco.mj_step(model, data)
    assert data.qpos[jid] == pytest.approx(q_after_timeout, abs=2e-3), (
        "joint kept moving after the velocity watchdog should have expired"
    )


def test_e4_position_command_supersedes_velocity(rig):
    """Mixing the two paths must not leave the integrator walking away from a commanded pose."""
    model, data, ctx, arm, _ = rig
    handle = ctx.blackboard.get(ARM_KEY)
    handle.set_velocities(["joint1"], [0.5])
    for _ in range(50):
        arm.pre_step(ctx)
        mujoco.mj_step(model, data)
    handle.set_targets(["joint1"], [0.0])
    for _ in range(int(2.0 / model.opt.timestep)):
        arm.pre_step(ctx)
        mujoco.mj_step(model, data)
    assert data.qpos[model.joint("joint1").qposadr[0]] == pytest.approx(0.0, abs=0.02)


def test_e5_velocity_integration_respects_joint_limits(rig):
    """Integrating a sustained velocity must clamp at the joint limit, not run past it."""
    model, data, ctx, arm, _ = rig
    handle = ctx.blackboard.get(ARM_KEY)
    lo, hi = model.jnt_range[model.joint("joint1").id]
    for _ in range(int(20.0 / model.opt.timestep)):  # long enough to reach the limit
        handle.set_velocities(["joint1"], [3.0])
        arm.pre_step(ctx)
        mujoco.mj_step(model, data)
    assert arm._target["joint1"] <= hi + 1e-9, f"target {arm._target['joint1']} exceeded limit {hi}"


def test_e6_gripper_closes(rig):
    """The Franka Hand is commandable — the 30 mm cube grasp depends on it."""
    model, data, ctx, arm, _ = rig
    reader = ctx.blackboard.get(GRIP_KEY)
    open_pos = reader()[0]
    arm.set_gripper(0.0)  # fully closed
    for _ in range(int(2.0 / model.opt.timestep)):
        arm.pre_step(ctx)
        mujoco.mj_step(model, data)
    assert reader()[0] < open_pos - 0.005, "gripper did not close"


# ------------------------------------------------------------------------------------ C: the laser
# The base carries a safety scanner and, until it had a mount site, nothing on this robot could
# navigate: `lidar` looks for a site of that name and this was the one mobile base in the substrate
# without one. These check the three ways the mount can be right on paper and useless in practice.


def test_c1_lidar_site_sits_at_the_scanner_the_cad_models(rig):
    """C1: the mount is the CAD's own sensor, not a plausible spot on the shell.

    The Omron mesh carries a 5 mm window slot (omron__m7) and a housing behind it (omron__m8). The
    site must be at the housing's centre and in the slot's plane -- that pairing is what makes the
    window subtend its 250 deg symmetrically, which C2 then relies on.
    """
    model = rig[0]
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "lidar")
    assert sid >= 0, "frankie has no `lidar` site, so no lidar can be mounted on it"
    pos = model.site_pos[sid]

    verts = _mesh_verts("omron__m8")
    assert pos[0] == pytest.approx((verts[:, 0].min() + verts[:, 0].max()) / 2, abs=1e-3), (
        "the mount is not at the sensor housing's centre"
    )
    window = _mesh_verts("omron__m7")
    assert pos[2] == pytest.approx((window[:, 2].min() + window[:, 2].max()) / 2, abs=1e-3), (
        "the mount is not in the window slot's plane"
    )
    assert pos[1] == pytest.approx(0.0, abs=1e-6), "the scanner is on the base's centreline"


def test_c2_the_fan_matches_the_window_the_rays_leave_through(rig):
    """C2: the configured fan is the aperture the CAD provides, not a wider one.

    A scan wider than the window is a scan through the robot's own shell. The manifest's +-125 deg
    is asserted against the slot's measured angular extent about the mount, so a later change to
    either the geometry or the config that breaks their agreement fails here.
    """
    cfg = _manifest("lidar")
    window = _mesh_verts("omron__m7")
    model = rig[0]
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "lidar")
    sx = model.site_pos[sid][0]
    bearings = np.degrees(np.arctan2(window[:, 1], window[:, 0] - sx))

    assert math.degrees(cfg["angle_min"]) == pytest.approx(bearings.min(), abs=0.5)
    assert math.degrees(cfg["angle_max"]) == pytest.approx(bearings.max(), abs=0.5)
    span = math.degrees(cfg["angle_max"] - cfg["angle_min"])
    assert span == pytest.approx(250.0, abs=1.0), f"fan spans {span:.0f} deg, window is 250"


def test_c3_the_scan_is_not_swallowed_by_the_chassis():
    """C3: the mount is inside the chassis collision box, so `exclude_body` decides everything.

    This is the failure the site alone does not prevent and the one that looks like a working
    sensor: every ray returns a fraction of a metre, a costmap fills with the robot itself, and the
    planner reports it is boxed in. Both halves are run against a wall at a known distance --
    without the manifest's `exclude_body` the scan MUST be blocked, and with it the wall must come
    back where it actually is.
    """
    wall_x = 6.0
    model, data = _build(wall_x=wall_x)
    ctx = _ctx(model, data)
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "lidar")
    assert _inside_chassis(model, model.site_pos[sid]), (
        "the mount left the chassis box -- this test no longer proves what it claims"
    )

    from roqsim_sensors.plugins.lidar import LidarPlugin

    cfg = _manifest("lidar")
    assert cfg.get("exclude_body") == "base_link", (
        "the manifest must exclude the chassis or its default scan is unusable"
    )

    blocked = _scan(LidarPlugin({**cfg, "exclude_body": None}, entity="robot"), ctx, model, data)
    clear = _scan(LidarPlugin(cfg, entity="robot"), ctx, model, data)

    # Blinded, not merely degraded: the chassis surrounds the mount, so every return is clamped to
    # range_min and the scan carries no information about the world at all.
    assert blocked.max() < 0.5, (
        f"without exclude_body the chassis should swallow the whole scan, "
        f"farthest return was {blocked.max():.2f} m"
    )
    # The forward ray, against the wall's true distance from the mount.
    sx = float(model.site_pos[sid][0])
    expected = wall_x - 0.05 - sx
    forward = clear[len(clear) // 2]
    assert forward == pytest.approx(expected, abs=0.05), (
        f"the forward ray reads {forward:.2f} m, the wall is at {expected:.2f} m from the mount"
    )
