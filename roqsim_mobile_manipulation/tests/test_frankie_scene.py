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

from roqsim import raycast
from roqsim.config import load_config_from_dict
from roqsim.context import Entity, SimContext
from roqsim.engine import Engine
from roqsim.models import apply_assets, resolve_model
from roqsim.plugin import Plugin
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
    geoms a single frame reports whichever happens to be penetrating that tick: a snapshot can show
    the model "resting on caster_front alone with both drive wheels off the ground" while the
    averaged forces show all four sharing load. Snapshots lie here.
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
    world-frame delta then mixes the mount offset with that rotation: compared in the world frame it
    is off by 4.3 mm in x, which is exactly 0.38 m * sin(0.65 deg) — the pitch, not a mount error.
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

    `arm_controller`'s path into the substrate for a controller that resolves to joint rates (a QP
    redundancy resolver, a teleop jog).
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
# The base's Safety Laser Scanner is the `omron_os32c` device the manifest mounts. Fixtures are Omron's
# numbers and the CAD's, not the model's:
#   * Omron LD-60/90 Platform User's Manual I611-E-09 ("I611") p. 1-6 and 2-13: a 240 deg field, the
#     plane 190 mm above the floor;
#   * Omron OS32C data sheet Z298-E2-05-X ("Z298") p. 5: the plane 67 mm above the scanner's base,
#     0.4 deg resolution;
#   * the CAD's scanner space (qut_frankie_description, the centre of its placeholder housing): x 0.265;
#   * the LD skin's channel as the CAD carries it (I611 p. 1-19; build_frankie_mjcf.py CHANNEL_Z):
#     open from z 0.1815 to 0.2235 over +-125 deg about the scan origin.
LABEL = "lidar"
SCAN_FRAME = "laser"  # the OS32C ROS 1 driver's default frame_id
OWNER, PREFIX, NAMESPACE = "fk", "fk_", "frankie1"
SCAN_ORIGIN = np.array([0.265, 0.0, 0.190])
DEVICE_SCAN_HEIGHT = 0.067
FIELD_HALF_DEG = 120.0
RESOLUTION_DEG = 0.4
CHANNEL_Z = (0.1815, 0.2235)
CHANNEL_HALF_DEG = 125.0
#: Inner wall faces at x, y = +-HALF around the spawn origin.
HALF = 3.0


class _Room(Plugin):
    def build(self, spec: mujoco.MjSpec, ctx: SimContext) -> None:
        t = 0.05
        for axis in (0, 1):
            for sign in (-1.0, 1.0):
                pos = [0.0, 0.0, 0.5]
                pos[axis] = sign * (HALF + t)
                size = [HALF + 2 * t, HALF + 2 * t, 0.5]
                size[axis] = t
                spec.worldbody.add_geom(
                    name=f"room_wall_{axis}_{int(sign)}",
                    type=mujoco.mjtGeom.mjGEOM_BOX,
                    pos=pos,
                    size=size,
                )


@pytest.fixture(scope="module")
def scan():
    """Frankie spawned with a prefix and namespace in a room of known walls, its scanner cast once."""
    world = {
        "sim": {"timestep": VERIFIED_TIMESTEP},
        "components": [
            {f"{__name__}:_Room": {}},
            {
                "spawn_robot": {"model": "frankie", "prefix": PREFIX, "namespace": NAMESPACE},
                "name": OWNER,
            },
        ],
    }
    engine = Engine(load_config_from_dict(world, base_dir=Path(".")))
    engine.ctx.seed = 0  # a test driving an Engine is the driver, and the seed is driver-owned
    engine.setup()
    engine.reset()
    engine.step()  # the rate gate starts open, so the first step casts
    yield engine
    engine.shutdown()


def _scanner(engine: Engine):
    (scanner,) = [
        p
        for p in engine.plugins
        if type(p).__name__ == "LidarPlugin" and p.entity == f"{OWNER}.{LABEL}"
    ]
    return scanner


def _pose_in_base(engine: Engine, site: str) -> tuple[np.ndarray, np.ndarray]:
    m, d = engine.ctx.model, engine.ctx.data
    base = m.body(PREFIX + "base_link").id
    sid = m.site(site).id
    rot = d.xmat[base].reshape(3, 3)
    return rot.T @ (d.site_xpos[sid] - d.xpos[base]), rot.T @ d.site_xmat[sid].reshape(3, 3)


def _cast(engine: Engine, bearings_rad: np.ndarray):
    """``(world directions, hits)`` of horizontal rays from the scan site, only the housing skipped."""
    m, d = engine.ctx.model, engine.ctx.data
    scanner = _scanner(engine)
    local = np.stack(
        [np.cos(bearings_rad), np.sin(bearings_rad), np.zeros_like(bearings_rad)], axis=1
    )
    dirs = local @ d.site_xmat[scanner._site_id].reshape(3, 3).T
    mount = m.body(f"{PREFIX}{LABEL}_mount").id
    hits = raycast.cast(
        m,
        d,
        d.site_xpos[scanner._site_id].copy(),
        dirs,
        cutoff=scanner.range_max,
        bodyexclude=mount,
        out=raycast.buffers(len(dirs), normals=True),
    )
    return dirs, hits


def _robot_bodies(engine: Engine, hits) -> list[str]:
    m = engine.ctx.model
    return [
        mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[g]))
        for g in hits.geomid
        if g >= 0 and m.geom_bodyid[g] != 0
    ]


def test_c1_the_scan_frame_is_at_omrons_plane(scan):
    """C1: the scan frame at base_link (0.265, 0, 0.190) -- 190 mm above the floor base_link rests on
    (A1) -- as the device's 67 mm scan height over its mounting face (0.265, 0, 0.123)."""
    mount = next(
        p.config
        for p in scan.plugins
        if type(p).__name__ == "SpawnSensorPlugin" and p.address == f"{OWNER}.{LABEL}"
    )
    assert mount["model"] == "omron_os32c" and mount["parent_frame"] == "base_link"
    assert np.allclose(mount["pos"], SCAN_ORIGIN - (0.0, 0.0, DEVICE_SCAN_HEIGHT), atol=1e-12)
    for site in (f"{PREFIX}{LABEL}_scan", f"{PREFIX}{LABEL}_{SCAN_FRAME}"):
        pos, rot = _pose_in_base(scan, site)
        assert np.allclose(pos, SCAN_ORIGIN, atol=1e-6), f"{site} at {pos}"
        assert np.allclose(rot, np.eye(3), atol=1e-6), f"{site} rotation {rot}"
    model = scan.ctx.model
    meshes = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, i) for i in range(model.nmesh)}
    assert PREFIX + "omron__m8" not in meshes, "the CAD's placeholder housing is back"


def test_c2_the_field_is_omrons_setting_for_the_ld(scan):
    """C2: 240 deg at the OS32C's 0.4 deg: 601 beams over +-120 deg (I611 p. 1-6, 2-13; Z298 p. 5),
    narrower than the device's own 270 deg and inside the skin's +-125 deg channel."""
    scanner = _scanner(scan)
    assert scanner.num_rays == round(2 * FIELD_HALF_DEG / RESOLUTION_DEG) + 1 == 601
    assert scanner.angle_min == pytest.approx(-math.radians(FIELD_HALF_DEG))
    assert scanner.angle_max == pytest.approx(math.radians(FIELD_HALF_DEG))
    assert scanner.latest.angle_increment == pytest.approx(math.radians(RESOLUTION_DEG))
    assert FIELD_HALF_DEG < CHANNEL_HALF_DEG
    # The device's own scan values are not overridden: range, rate and the driver's conventions.
    assert (scanner.range_min, scanner.range_max) == pytest.approx((0.002, 50.0))
    assert (scanner.detection_min, scanner.detection_max) == pytest.approx((0.002, 15.0))
    assert scanner.rate_hz == pytest.approx(25.0)


def test_c3_the_scan_leaves_through_the_channel(scan):
    """C3: with only the scanner's own housing skipped, no ray starts inside robot geometry and none
    meets the robot: the 240 deg field leaves through the channel. The channel is not wider than the
    CAD's: just inside +-125 deg the rays leave, just outside they meet the skin."""
    m = scan.ctx.model
    scanner = _scanner(scan)
    assert scanner._bodyexclude == m.body(f"{PREFIX}{LABEL}_mount").id, (
        "it skips more than its housing"
    )
    bearings = scanner.latest.angle_min + scanner.latest.angle_increment * np.arange(
        scanner.num_rays
    )
    dirs, hits = _cast(scan, bearings)
    on_robot = (hits.geomid >= 0) & (m.geom_bodyid[np.maximum(hits.geomid, 0)] != 0)
    inside = on_robot & (np.einsum("ij,ij->i", hits.normal, dirs) > 0)
    assert not inside.any(), f"rays start inside robot geometry: {_robot_bodies(scan, hits)}"
    assert not on_robot.any(), (
        f"the field meets the robot: {sorted(set(_robot_bodies(scan, hits)))}"
    )
    np.testing.assert_array_equal(hits.geomid, scanner._hits.geomid)

    edge = np.radians([CHANNEL_HALF_DEG - 0.5, -(CHANNEL_HALF_DEG - 0.5)])
    _, hits = _cast(scan, edge)
    assert not _robot_bodies(scan, hits), "the channel is closed inside its +-125 deg"
    beyond = np.radians([CHANNEL_HALF_DEG + 1.5, -(CHANNEL_HALF_DEG + 1.5)])
    _, hits = _cast(scan, beyond)
    assert _robot_bodies(scan, hits) == [PREFIX + "base_link"] * 2, (
        "the skin does not close the channel"
    )


def test_c4_the_forward_ray_reads_the_wall(scan):
    """C4: the forward ray reads the wall ahead at its true distance from the scan origin."""
    scanner = _scanner(scan)
    ranges = np.asarray(scanner.latest.ranges)
    fwd = scanner.num_rays // 2
    assert scanner.latest.angle_min + fwd * scanner.latest.angle_increment == pytest.approx(
        0.0, abs=1e-9
    )
    d = scan.ctx.data
    origin = d.site_xpos[scanner._site_id]
    direction = d.site_xmat[scanner._site_id].reshape(3, 3)[:, 0]
    true = (HALF - origin[0]) / direction[0]
    assert ranges[fwd] == pytest.approx(true, abs=1e-3), (
        f"reads {ranges[fwd]:.4f} m, wall at {true:.4f} m"
    )


def test_c5_the_tf_chain_and_topic(scan):
    """C5: the mount publishes base_link -> laser at the scan origin; the scan is stamped in `laser`."""
    address = f"{OWNER}.{LABEL}"
    (frames,) = [e for e in scan.ctx.interface.all() if e.name == "frames" and e.owner == address]
    assert frames.namespace == NAMESPACE
    tf = frames.backend["ros2"]["static_tf"]
    assert [(t["parent"], t["child"]) for t in tf] == [("base_link", SCAN_FRAME)], tf
    assert np.allclose(tf[0]["translation"], SCAN_ORIGIN, atol=1e-6)
    (endpoint,) = [e for e in scan.ctx.interface.all() if e.name == "scan" and e.owner == address]
    assert endpoint.backend["ros2"]["frame_id"] == SCAN_FRAME
    assert "static_tf" not in endpoint.backend["ros2"]


def test_c6_the_collision_geometry_is_cut_at_the_channel(rig):
    """C6: the URDF's collision box keeps its footprint (A4) and loses exactly the channel: below it,
    above it, and behind the plane where a +-125 deg ray leaves the box's side face."""
    model = rig[0]

    def span(name):
        gid = model.geom(name).id
        return model.geom_pos[gid] - model.geom_size[gid], model.geom_pos[gid] + model.geom_size[
            gid
        ]

    rear_x = SCAN_ORIGIN[0] + (BOX[1] / 2) / math.tan(math.radians(CHANNEL_HALF_DEG))
    for name, lo, hi in (
        ("chassis", (-0.34, -0.235, CHANNEL_Z[1]), (0.34, 0.235, BOX[2])),
        ("chassis_below_channel", (-0.34, -0.235, 0.125), (0.34, 0.235, CHANNEL_Z[0])),
        ("chassis_behind_channel", (-0.34, -0.235, CHANNEL_Z[0]), (rear_x, 0.235, CHANNEL_Z[1])),
    ):
        got_lo, got_hi = span(name)
        assert np.allclose(got_lo, lo, atol=2e-4) and np.allclose(got_hi, hi, atol=2e-4), (
            f"{name} spans {got_lo}..{got_hi}"
        )
