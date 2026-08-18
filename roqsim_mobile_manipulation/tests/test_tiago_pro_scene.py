"""PAL TIAGo Pro verification battery — dual-arm holonomic mobile-manipulator port.

Follows robot-porting's `references/mobile_manipulator.md`: the base battery (A static, B drive,
C sensors) plus the arm battery (E), run through the REAL plugins with the REAL manifest config, so
the model, its four controllers and their limits are verified together.

Two things make this battery different from the husky/frankie/G2 ones:

- **The base is holonomic.** B does not stop at "drives straight and turns"; it asserts that the base
  strafes (`vy`), translates diagonally while holding heading, and rotates independently of
  translation. B1c (drive at a non-zero heading) is the test that catches the frame bug this drive
  is most prone to: a planar actuator's `gear` acts in the WORLD frame, so a controller that forgets
  to rotate the body-frame command by the base yaw still passes every straight-line test taken from
  a standing start.
- **Four controllers share one entity** over disjoint actuators. If any `arm_controller` ever claims
  the base or the other arm's actuators, the symptom is a robot that silently stops driving or an arm
  that fights another controller — so E0 asserts the ownership split explicitly.

Reference values are PAL's own (see the package `THIRD_PARTY.md` for the provenance table); the
few substrate assumptions are named as such in the assertions they drive.
"""

from __future__ import annotations

import math
from pathlib import Path

import mujoco
import numpy as np
import pytest
import yaml
from roqsim_manipulation.plugins.arm_controller import ArmControllerPlugin
from roqsim_sensors.plugins.lidar import LidarPlugin

from roqsim.context import Entity, SimContext
from roqsim.models import apply_assets, resolve_model
from roqsim_mobile.plugins.omni_drive import OmniDrivePlugin

MODELS = Path(__file__).resolve().parents[1] / "src" / "roqsim_mobile_manipulation" / "models"
MODEL_DIR = MODELS / "tiago_pro"
MANIFEST = MODEL_DIR / "tiago_pro.manifest.yaml"

# --- authoritative source values (PAL) ---------------------------------------------------------
URDF_MASS = 62.45  # sum of the expanded URDF's <mass> values
WHEEL_R = 0.0762  # base.urdf.xacro / mobile_base_controller.yaml
TRACK = 0.44715  # wheel_separation (lateral)
WHEELBASE = 0.488  # axis_separation (longitudinal)
MAX_VX = MAX_VY = 1.0  # mobile_base_controller linear.{x,y} max_velocity
MAX_COMBINED_V = 0.7  # mobile_base_controller space.xy max_velocity (resultant cap)
MAX_WZ = 2.09  # mobile_base_controller angular.z max_velocity
LASER_Z = 0.13244 + WHEEL_R  # laser_height above base_link + base_link above the floor
TORSO_RANGE = (0.0, 0.35)
ARM_L = [f"arm_left_{i}_joint" for i in range(1, 8)]
ARM_R = [f"arm_right_{i}_joint" for i in range(1, 8)]
# PAL `home` motion, final waypoint (tiago_pro_motions_general_spherical-wrist.yaml)
HOME_L = [0.36, -1.83, 0.47, -2.35, 0.0, -1.2, 0.0]
HOME_R = [-0.36, -1.83, -0.47, -2.35, 0.0, -1.2, 0.0]

# Measured on the model, documented in the port log: gripper travel -> fingertip pad gap.
GRIP_OPEN_TRAVEL = 0.07
GRIP_MAX_GAP_MM = 41.2
GRIP_TOUCH_TRAVEL = 0.013


# --- harness -----------------------------------------------------------------------------------
def _manifest_entries(plugin: str) -> list[dict]:
    """Every manifest entry for *plugin*, in order. The manifest is the source of truth for config."""
    out = [dict(e[plugin]) for e in yaml.safe_load(MANIFEST.read_text())["plugins"] if plugin in e]
    if not out:
        raise AssertionError(f"tiago_pro manifest has no {plugin} plugin")
    return out


def _entry_by(plugin: str, key: str, value) -> dict:
    for cfg in _manifest_entries(plugin):
        if cfg.get(key) == value:
            return cfg
    raise AssertionError(f"no {plugin} manifest entry with {key}={value!r}")


@pytest.fixture(scope="module")
def built():
    """Compose the robot through the real substrate path (resolve_model + apply_assets) + a floor."""
    asset = resolve_model("tiago_pro")
    spec = mujoco.MjSpec.from_file(str(asset.path))
    apply_assets(spec, asset)
    floor = spec.worldbody.add_geom()
    floor.name = "floor"
    floor.type = mujoco.mjtGeom.mjGEOM_PLANE
    floor.size = [40, 40, 0.05]
    floor.friction = [1.0, 0.005, 0.0001]
    model = spec.compile()
    return model


@pytest.fixture
def rig(built):
    """Fresh data + the four real controllers, configured and reset, at the home stance."""
    model = built
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    ctx = SimContext(config={})
    ctx.model, ctx.data = model, data
    ctx.entities.add(
        Entity(name="robot", kind="robot", body="base_link", meta={"prefix": "", "namespace": ""})
    )

    drive = OmniDrivePlugin({**_manifest_entries("omni_drive")[0], "robot": "robot"})
    arms = {}
    for name in ("torso_head_controller", "arm_left_controller", "arm_right_controller"):
        cfg = _entry_by("arm_controller", "controller_name", name)
        arms[name] = ArmControllerPlugin({**cfg, "robot": "robot"})
    for p in (drive, *arms.values()):
        p.configure(ctx)
        p.on_reset(ctx)
    return _Rig(model, data, ctx, drive, arms)


class _Rig:
    def __init__(self, model, data, ctx, drive, arms):
        self.model, self.data, self.ctx, self.drive, self.arms = model, data, ctx, drive, arms

    def step(self, seconds: float):
        for _ in range(int(round(seconds / self.model.opt.timestep))):
            self.drive.pre_step(self.ctx)
            for a in self.arms.values():
                a.pre_step(self.ctx)
            mujoco.mj_step(self.model, self.data)
            self.drive.post_step(self.ctx)

    def settle(self, seconds: float = 2.0):
        self.drive.drive(0.0, 0.0, 0.0)
        self.step(seconds)

    @property
    def yaw(self) -> float:
        w, x, y, z = self.data.qpos[3:7]
        return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

    @property
    def xy(self) -> np.ndarray:
        return self.data.qpos[:2].copy()

    def body_vel(self) -> np.ndarray:
        """Achieved body-frame [vx, vy] (free-joint linear DOFs are world-frame)."""
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        vx, vy = self.data.qvel[0], self.data.qvel[1]
        return np.array([c * vx + s * vy, -s * vx + c * vy])

    def jid(self, name: str) -> int:
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)

    def q(self, name: str) -> float:
        return float(self.data.qpos[self.model.jnt_qposadr[self.jid(name)]])

    def set_yaw(self, yaw: float) -> None:
        self.data.qpos[3:7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
        mujoco.mj_forward(self.model, self.data)


# --- A: static ---------------------------------------------------------------------------------
def test_a1_loads_and_steps_without_warnings(rig):
    """A1: 10 s at the substrate timestep produces no MuJoCo warnings and no divergence."""
    before = rig.data.warning.number.copy()
    rig.settle(10.0)
    fired = {
        mujoco.mjtWarning(i).name: int(v)
        for i, v in enumerate(rig.data.warning.number - before)
        if v
    }
    assert not fired, f"MuJoCo warnings during a 10 s rest: {fired}"
    assert np.all(np.isfinite(rig.data.qpos)), "non-finite qpos"


def test_a2_mass_audit(rig):
    """A2: total mass matches the source URDF, and no link has an implausible near-zero mass.

    The two 1e-9 kg links are the grippers' `grasping_link` TCP frames, which the source deliberately
    gives no mass; everything else must be a real link mass.
    """
    m = rig.model
    assert m.body_mass.sum() == pytest.approx(URDF_MASS, abs=0.01)
    tcp = {"gripper_left_grasping_link", "gripper_right_grasping_link"}
    for b in range(1, m.nbody):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b)
        mass = m.body_mass[b]
        if name in tcp or mass == 0.0:
            continue  # virtual frames: source has no <inertial>, or an explicit 1e-9 TCP
        assert mass > 1e-4, f"{name} has implausible mass {mass}"


def test_a3_rest_is_stable(rig):
    """A3: from the home stance the robot settles and stays put (the planar drive holds x/y/yaw)."""
    rig.settle(5.0)
    assert np.linalg.norm(rig.data.qvel[:3]) < 1e-3, "base drifts at rest"
    assert abs(rig.data.qvel[5]) < 1e-3, "base yaws at rest"
    assert abs(rig.xy).max() < 5e-3, f"base translated at rest: {rig.xy}"


def test_a4_scale_and_geometry(rig):
    """A4: wheel radius, track and wheelbase read off the model match the source description."""
    m, d = rig.model, rig.data
    base = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    rb = d.xmat[base].reshape(3, 3)
    pb = d.xpos[base]
    pos = {}
    for w in ("front_left", "front_right", "rear_left", "rear_right"):
        j = rig.jid(f"wheel_{w}_joint")
        pos[w] = rb.T @ (d.xanchor[j] - pb)
        g = next(
            g
            for g in range(m.ngeom)
            if m.geom_bodyid[g] == m.jnt_bodyid[j]
            and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_CYLINDER
        )
        assert m.geom_size[g, 0] == pytest.approx(WHEEL_R, abs=1e-4)
    assert pos["front_left"][1] - pos["front_right"][1] == pytest.approx(TRACK, abs=2e-3)
    assert pos["front_left"][0] - pos["rear_left"][0] == pytest.approx(WHEELBASE, abs=1e-3)


def test_a5_footprint_and_height(rig):
    """A5: the visual hull matches the OMNI base footprint and TIAGo Pro's standing height."""
    m, d = rig.model, rig.data
    verts = []
    for g in range(m.ngeom):
        if m.geom_group[g] != 2 or m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        mi = m.geom_dataid[g]
        a, n = m.mesh_vertadr[mi], m.mesh_vertnum[mi]
        v = m.mesh_vert[a : a + n].astype(float)
        verts.append(v @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g])
    v = np.vstack(verts)
    # base_link's own collision boxes give the chassis footprint: 0.717 x 0.497 m.
    chassis = [
        g
        for g in range(m.ngeom)
        if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX
        and mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]) == "base_link"
    ]
    widest = max(chassis, key=lambda g: m.geom_size[g, 0])
    assert 2 * m.geom_size[widest, 0] == pytest.approx(0.717, abs=1e-3)
    assert 2 * m.geom_size[widest, 1] == pytest.approx(0.497, abs=1e-3)
    assert v[:, 2].min() >= -1e-3, "geometry below the floor at the home stance"
    assert 1.1 < v[:, 2].max() < 1.6, f"standing height {v[:, 2].max():.3f} m out of range"


# --- B: drive (holonomic) ----------------------------------------------------------------------
@pytest.mark.parametrize("vx", [0.3, 0.6])
def test_b1_straight_line(rig, vx):
    """B1: commanded vx is achieved, with no lateral drift or yaw error, and odometry matches truth."""
    rig.settle(1.0)
    p0 = rig.xy
    rig.drive.drive(vx, 0.0, 0.0)
    rig.step(4.0)
    ach = rig.body_vel()
    assert ach[0] == pytest.approx(vx, rel=0.05), f"achieved vx {ach[0]:.3f} vs {vx}"
    assert abs(ach[1]) < 0.01, f"lateral drift {ach[1]:.4f} m/s on a pure-vx command"
    assert abs(rig.yaw) < 0.01, f"yaw drift {rig.yaw:.4f} rad"
    travelled = rig.xy - p0
    assert travelled[0] > 0.9 * vx * 3.5, "did not cover the commanded distance"
    odom = np.array(rig.drive.read_odom()[:2])
    assert np.linalg.norm(odom - travelled) < 0.02, f"odom {odom} vs truth {travelled}"


@pytest.mark.parametrize("vy", [0.4, -0.4])
def test_b1b_pure_strafe(rig, vy):
    """B1b: the base translates SIDEWAYS at a locked heading — the capability a diff-drive lacks.

    This is the test that distinguishes `omni_drive` from `diff_drive`, whose `drive()` drops vy.
    """
    rig.settle(1.0)
    p0 = rig.xy
    rig.drive.drive(0.0, vy, 0.0)
    rig.step(3.0)
    ach = rig.body_vel()
    assert ach[1] == pytest.approx(vy, rel=0.05), f"achieved vy {ach[1]:.3f} vs {vy}"
    assert abs(ach[0]) < 0.01, f"forward creep {ach[0]:.4f} m/s on a pure-vy command"
    assert abs(rig.yaw) < 0.01, f"heading not held during strafe: {rig.yaw:.4f} rad"
    moved = rig.xy - p0
    assert math.copysign(1, moved[1]) == math.copysign(1, vy), "strafed the wrong way"
    assert abs(moved[1]) > 0.9 * abs(vy) * 2.5, f"strafe distance {moved[1]:.3f} m too small"
    assert abs(moved[0]) < 0.02, f"strafe was not lateral: dx={moved[0]:.3f} m"


def test_b1c_diagonal_at_nonzero_heading(rig):
    """B1c: a body-frame command must be honoured in the BODY frame at any heading.

    The planar actuators' `gear` acts in the world frame, so a controller that skips the yaw rotation
    drives along world +x here instead of the robot's own +x. From a standing start at yaw 0 that bug
    is invisible; at yaw 90 deg it sends the robot sideways. Nothing else in this battery catches it.
    """
    rig.set_yaw(math.pi / 2)
    rig.settle(1.0)
    p0 = rig.xy
    rig.drive.drive(0.4, 0.0, 0.0)  # forward in the BODY frame == world +y at yaw 90 deg
    rig.step(3.0)
    moved = rig.xy - p0
    assert moved[1] > 0.9 * 0.4 * 2.5, f"did not drive along body +x (world +y): {moved}"
    assert abs(moved[0]) < 0.05, f"drove in the world frame instead of the body frame: {moved}"
    assert abs(rig.yaw - math.pi / 2) < 0.02, "heading changed"


@pytest.mark.parametrize("wz", [0.5, 1.0, -1.5])
def test_b2_in_place_rotation(rig, wz):
    """B2: in-place rotation achieves the commanded rate and stays on the spot.

    A holonomic base is NOT scrub-limited, so unlike the husky/G2 this asserts the ideal rate to
    within 5 % rather than a loose direction-only bound — there is no `slip_factor` here.
    """
    rig.settle(1.0)
    p0 = rig.xy
    rig.drive.drive(0.0, 0.0, wz)
    rig.step(2.0)
    ach = float(rig.data.qvel[5])
    assert ach == pytest.approx(wz, rel=0.05), f"achieved wz {ach:.3f} vs {wz}"
    assert np.linalg.norm(rig.xy - p0) < 0.05, f"drifted {np.linalg.norm(rig.xy - p0):.3f} m"


def test_b3_translation_and_rotation_are_independent(rig):
    """B3: strafing while rotating — both components are tracked at once (true holonomy)."""
    rig.settle(1.0)
    rig.drive.drive(0.3, 0.3, 0.5)
    rig.step(3.0)
    ach = rig.body_vel()
    assert ach[0] == pytest.approx(0.3, rel=0.08), f"vx {ach[0]:.3f}"
    assert ach[1] == pytest.approx(0.3, rel=0.08), f"vy {ach[1]:.3f}"
    assert float(rig.data.qvel[5]) == pytest.approx(0.5, rel=0.08)


@pytest.mark.parametrize("axis,limit", [(0, MAX_VX), (1, MAX_VY)])
def test_b4_axis_limits_are_enforced(rig, axis, limit):
    """B4: an over-range single-axis command saturates AT the limit — reached, and not exceeded.

    Each axis is commanded on its own. Asking for max translation and max rotation at once is a
    tracking test, not a limit test: at wz = 2.09 rad/s the world-frame velocity target sweeps a full
    turn every 3 s and the achieved body-frame speed necessarily lags it (see B4c).
    """
    rig.settle(1.0)
    cmd = [0.0, 0.0, 0.0]
    cmd[axis] = 5.0  # far beyond the limit
    rig.drive.drive(*cmd)
    rig.step(4.0)
    ach = rig.body_vel()[axis]
    assert ach <= limit * 1.02, f"achieved {ach:.3f} exceeds the {limit} m/s limit"
    assert ach > 0.95 * limit, f"achieved {ach:.3f} does not reach its own {limit} m/s limit"


def test_b4b_angular_limit_is_enforced(rig):
    """B4b: an over-range yaw command saturates at PAL's 2.09 rad/s (120 deg/s)."""
    rig.settle(1.0)
    rig.drive.drive(0.0, 0.0, 9.0)
    rig.step(4.0)
    ach = abs(float(rig.data.qvel[5]))
    assert ach <= MAX_WZ * 1.02, f"achieved {ach:.3f} exceeds the {MAX_WZ} rad/s limit"
    assert ach > 0.95 * MAX_WZ, f"achieved {ach:.3f} does not reach the {MAX_WZ} rad/s limit"


def test_b4c_diagonal_is_limited_per_axis_only(rig):
    """B4c: a diagonal saturates at the PER-AXIS limits, so its resultant exceeds PAL's `space: xy`.

    This pins a known, deliberate divergence rather than a desired behaviour. PAL's
    mobile_base_controller.yaml also caps combined motion (`space: xy: 0.7` m/s), but that block's
    semantics cannot be resolved -- pal-robotics/omni_drive_controller is not public -- so the port
    enforces only the unambiguous per-axis limits (see the manifest and port log A6). If the combined
    cap is ever pinned down, set `max_combined_linear_vel` and invert this test.
    """
    assert "max_combined_linear_vel" not in _manifest_entries("omni_drive")[0], (
        "the manifest now sets a combined cap -- update this test to assert it"
    )
    rig.settle(1.0)
    rig.drive.drive(5.0, 5.0, 0.0)  # equal, both over-range -> a 45 deg diagonal
    rig.step(5.0)
    ach = rig.body_vel()
    assert ach[0] <= MAX_VX * 1.02 and ach[1] <= MAX_VY * 1.02, "an axis limit was exceeded"
    speed = float(np.hypot(*ach))
    assert speed > MAX_COMBINED_V, (
        f"resultant {speed:.3f} m/s is at or below PAL's combined cap -- if that is now enforced, "
        f"this test is the stale one"
    )
    assert ach[0] == pytest.approx(ach[1], rel=0.1), f"diagonal is not at 45 deg: {ach}"


def test_b5_acceleration_is_ramped(rig):
    """B5: a step command is ramped at the configured accel limit rather than applied instantly."""
    rig.settle(1.0)
    rig.drive.drive(MAX_VX, 0.0, 0.0)
    rig.step(0.25)  # accel_limit 1.0 m/s^2 -> ~0.25 m/s after 0.25 s, nowhere near 1.0
    assert rig.body_vel()[0] < 0.45, "command was not acceleration-limited"
    rig.step(3.0)
    assert rig.body_vel()[0] == pytest.approx(MAX_VX, rel=0.05), "never reached the target"


def test_b6_stop_from_full_speed(rig):
    """B6: from max speed a zero command brings the robot to rest, upright and stable."""
    rig.settle(1.0)
    rig.drive.drive(MAX_VX, MAX_VY, 0.0)
    rig.step(4.0)
    rig.drive.drive(0.0, 0.0, 0.0)
    rig.step(3.0)
    assert np.linalg.norm(rig.data.qvel[:3]) < 0.02, "still moving after a stop"
    assert abs(rig.data.qpos[2]) < 0.02, "base height changed (tipped or sank)"
    # roll/pitch: the planar drive applies force at the free joint, so it should not wheelie
    quat = rig.data.qpos[3:7]
    rot = np.zeros(9)
    mujoco.mju_quat2Mat(rot, quat)
    assert rot.reshape(3, 3)[2, 2] > 0.999, "base is not level after stopping"


def test_b7_wheels_spin_and_differ_between_drive_and_strafe(rig):
    """B7: the observational mecanum wheel servos turn, and their PATTERN distinguishes the modes.

    The wheels are not the motive force (see the port log), but a mecanum base's wheels must not all
    spin identically when strafing — if they did, the wheel kinematics would be a plain rolling model
    and `joint_states` would misreport what the base is doing.
    """
    names = [f"wheel_{w}_joint" for w in ("front_left", "front_right", "rear_left", "rear_right")]
    rig.settle(1.0)
    rig.drive.drive(0.5, 0.0, 0.0)
    rig.step(2.0)
    fwd = np.array([rig.data.qvel[rig.model.jnt_dofadr[rig.jid(n)]] for n in names])
    rig.drive.drive(0.0, 0.5, 0.0)
    rig.step(2.0)
    side = np.array([rig.data.qvel[rig.model.jnt_dofadr[rig.jid(n)]] for n in names])
    assert np.abs(fwd).min() > 1.0, f"wheels barely turn when driving: {fwd.round(2)}"
    # driving forward: all four the same sign; strafing: diagonal pairs oppose
    assert len(set(np.sign(fwd))) == 1, (
        f"forward drive should spin all wheels alike: {fwd.round(2)}"
    )
    assert len(set(np.sign(side))) == 2, f"strafe should oppose diagonal pairs: {side.round(2)}"


def test_b8_odometry_tracks_a_curved_path(rig):
    """B8: odometry integrated from the achieved twist follows ground truth on a curved path.

    With no wheel slip modelled these coincide by construction (see the plugin docstring) — the test
    guards the integration, not a slip model.
    """
    rig.settle(1.0)
    p0 = rig.xy
    rig.drive.drive(0.4, 0.2, 0.4)
    rig.step(4.0)
    odom = np.array(rig.drive.read_odom()[:2])
    assert np.linalg.norm(odom - (rig.xy - p0)) < 0.03
    assert rig.drive.read_odom()[2] == pytest.approx(rig.yaw, abs=0.02)


# --- C: sensors --------------------------------------------------------------------------------
def test_c1_lidar_mount_poses(rig):
    """C1: both scan planes sit at the source's laser height, on the diagonal base corners."""
    m, d = rig.model, rig.data
    for name, sign in (("lidar_front", +1), ("lidar_rear", -1)):
        sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, name)
        assert sid >= 0, f"site {name} missing"
        pos = d.site_xpos[sid]
        assert pos[2] == pytest.approx(LASER_Z, abs=1e-3), f"{name} scan height {pos[2]:.4f}"
        assert math.copysign(1, pos[0]) == sign, f"{name} on the wrong end of the base"
        assert abs(pos[0]) == pytest.approx(0.2751, abs=1e-3)
        assert abs(pos[1]) == pytest.approx(0.183, abs=1e-3)
        # the fan must be horizontal: the site's local z is world z
        assert d.site_xmat[sid].reshape(3, 3)[2, 2] == pytest.approx(1.0, abs=1e-6)


def test_c2_lidar_reads_a_known_wall(built):
    """C2: each lidar returns the true distance to a wall, with the source's ray count and range."""
    spec = mujoco.MjSpec.from_file(str(resolve_model("tiago_pro").path))
    apply_assets(spec, resolve_model("tiago_pro"))
    floor = spec.worldbody.add_geom()
    floor.name, floor.type, floor.size = "floor", mujoco.mjtGeom.mjGEOM_PLANE, [40, 40, 0.05]
    wall = spec.worldbody.add_geom()
    wall.name, wall.type = "wall", mujoco.mjtGeom.mjGEOM_BOX
    wall.size, wall.pos = [0.05, 6.0, 1.5], [4.0, 0.0, 1.5]
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    ctx = SimContext(config={})
    ctx.model, ctx.data = model, data
    ctx.entities.add(
        Entity(name="robot", kind="robot", body="base_link", meta={"prefix": "", "namespace": ""})
    )
    # `ctx.sim_time` is read-only (it reads data.time); advance the clock past the 10 Hz scan gate.
    data.time = 1.0
    for cfg in _manifest_entries("lidar"):
        # range_stddev is PAL's own 0.01 m; zero it so this asserts geometry, not noise.
        p = LidarPlugin({**cfg, "robot": "robot", "range_stddev": 0.0})
        p.configure(ctx)
        p.on_reset(ctx)
        p.post_step(ctx)
        scan = p._scan
        assert scan is not None, f"{cfg['site']} produced no scan"
        assert len(scan.ranges) == 815, "ray count does not match the TIM571's 818/270 deg"
        assert scan.range_max == 25.0 and scan.range_min == 0.05
        finite = scan.ranges[np.isfinite(scan.ranges)]
        assert len(finite) > 0, f"{cfg['site']} saw nothing at all"
        # The wall's inner face is at x=3.95; the front laser sits at x=0.2751.
        expected = 3.95 - 0.2751 if cfg["site"] == "lidar_front" else 3.95 + 0.2751
        assert finite.min() == pytest.approx(expected, abs=0.05), (
            f"{cfg['site']} nearest return {finite.min():.3f} m, expected ~{expected:.3f} m"
        )


def test_c3_lidar_is_not_blocked_by_the_chassis(rig):
    """C3: `exclude_body: base_link` is doing its job — base_link's top box spans the scan plane."""
    m = rig.model
    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "lidar_front")
    z = rig.data.site_xpos[sid][2]
    spanning = [
        g
        for g in range(m.ngeom)
        if mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]) == "base_link"
        and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX
        and abs(rig.data.geom_xpos[g][2] - z) < m.geom_size[g, 2]
    ]
    assert spanning, "no base_link box spans the scan plane — is exclude_body still needed?"
    for cfg in _manifest_entries("lidar"):
        assert cfg["exclude_body"] == "base_link"


def test_c4_head_camera_and_imu_exist(rig):
    """C4: the head RGB-D camera and the base IMU frame are addressable at the source's links."""
    m = rig.model
    cam = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "head_cam")
    assert cam >= 0, "head_cam missing"
    body = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.cam_bodyid[cam])
    assert body == "head_front_camera_link", f"head_cam is on {body}"
    assert m.cam_fovy[cam] == pytest.approx(42.5, abs=0.1), "D435 colour vertical FOV"
    imu = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "base_imu")
    assert (
        imu >= 0
        and mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.site_bodyid[imu]) == "base_imu_link"
    )
    for s in ("imu_gyro", "imu_acc", "base_pos", "base_quat"):
        assert mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, s) >= 0, f"sensor {s} missing"


# --- E: arms, torso, grippers ------------------------------------------------------------------
def test_e0_controllers_own_disjoint_actuators(rig):
    """E0: the four controllers partition the actuators — none claims another's.

    A silent overlap is the characteristic mobile-manipulator failure: `arm_controller`'s prefix scan
    would grab the base's velocity servos and position-hold them, and the robot would stop driving
    with nothing in the log.
    """
    owned: dict[int, str] = {}
    claims = {"omni_drive": [*rig.drive._aid, *rig.drive._waid]}
    for name, ctrl in rig.arms.items():
        claims[name] = [aid for aid, _ in ctrl._joint_acts] + list(ctrl._aux_acts)
    for who, aids in claims.items():
        for aid in aids:
            assert aid not in owned, (
                f"actuator {mujoco.mj_id2name(rig.model, mujoco.mjtObj.mjOBJ_ACTUATOR, aid)} "
                f"claimed by both {owned[aid]} and {who}"
            )
            owned[aid] = who
    assert len(owned) == rig.model.nu, (
        f"{rig.model.nu - len(owned)} actuator(s) unowned: "
        f"{[mujoco.mj_id2name(rig.model, mujoco.mjtObj.mjOBJ_ACTUATOR, a) for a in range(rig.model.nu) if a not in owned]}"
    )


def test_e1_arms_hold_the_home_stance_against_gravity(rig):
    """E1: both arms and the torso lift hold PAL's home stance under gravity."""
    rig.settle(3.0)
    for joints, home in ((ARM_L, HOME_L), (ARM_R, HOME_R)):
        for jn, target in zip(joints, home, strict=True):
            assert rig.q(jn) == pytest.approx(target, abs=0.02), f"{jn} sagged"
    # The lift carries ~28 kg on a slide; a few mm of sag is the servo's steady-state error.
    assert rig.q("torso_lift_joint") == pytest.approx(0.1, abs=0.005)


def test_e2_torso_lift_reaches_both_stops(rig):
    """E2: the torso lift traverses its full source range and holds the top under load."""
    ctrl = rig.arms["torso_head_controller"]
    rig.settle(1.0)
    ctrl.set_targets(["torso_lift_joint"], [TORSO_RANGE[1]])
    rig.step(15.0)  # source velocity limit is 0.035 m/s -> 0.25 m of travel needs ~7 s
    assert rig.q("torso_lift_joint") == pytest.approx(TORSO_RANGE[1], abs=0.01), "did not reach top"
    ctrl.set_targets(["torso_lift_joint"], [TORSO_RANGE[0]])
    rig.step(15.0)
    assert rig.q("torso_lift_joint") == pytest.approx(TORSO_RANGE[0], abs=0.01), (
        "did not reach bottom"
    )


def test_e3_arm_tracks_a_commanded_pose(rig):
    """E3: a reach command is tracked to within a few mrad on every joint (PAL's `offer` pose)."""
    offer_r = [0.25843, -0.57522, -0.50314, -2.0337, 0.0, 1.0543, -1.5708]
    rig.arms["arm_right_controller"].set_targets(ARM_R, offer_r)
    rig.step(6.0)
    for jn, target in zip(ARM_R, offer_r, strict=True):
        assert rig.q(jn) == pytest.approx(target, abs=0.03), f"{jn} did not track"


def test_e4_joint_limits_are_the_sources(rig):
    """E4: every actuated joint's range and actuator force ceiling come from the source URDF."""
    m = rig.model
    expected_effort = {
        "torso_lift_joint": 2000.0,
        "head_1_joint": 5.197,
        "head_2_joint": 2.77,
        **{j: 43.0 for j in ARM_L[:2] + ARM_R[:2]},
        **{j: 26.0 for j in ARM_L[2:] + ARM_R[2:]},
        "gripper_left_finger_joint": 10.0,
        "gripper_right_finger_joint": 10.0,
    }
    for jn, effort in expected_effort.items():
        jid = rig.jid(jn)
        aid = next(
            a
            for a in range(m.nu)
            if m.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT and m.actuator_trnid[a, 0] == jid
        )
        assert m.actuator_forcerange[aid, 1] == pytest.approx(effort, rel=1e-3), f"{jn} effort"
        # inheritrange=1: the servo cannot command outside the joint's own limits
        assert m.actuator_ctrlrange[aid] == pytest.approx(m.jnt_range[jid], abs=1e-6), f"{jn} range"
    assert m.jnt_range[rig.jid("torso_lift_joint")] == pytest.approx(TORSO_RANGE, abs=1e-6)


def test_e5_gripper_linkage_is_a_closed_four_bar(rig):
    """E5: the PRO gripper's four-bar is CLOSED, and driving the master moves the fingers.

    The URDF is a tree, so PAL breaks each finger's four-bar and fakes it with `<mimic>` multipliers
    (inner/outer -8.28, fingertip +8.28 off the prismatic master). Those are a linearisation about one
    configuration; reproducing them literally leaves the pads splayed 19-49 deg, so the gripper closes
    on an object but cannot carry it. This model instead drives the open branch by mimic and closes the
    loop with a `connect` equality at the measured fingertip/outer_finger pivot.

    So the shape of the constraint set is itself the thing under test: 6 joint equalities (the driven
    branch, 3 per gripper) plus 4 `connect`s (one per finger).
    """
    m = rig.model
    kinds = [int(m.eq_type[i]) for i in range(m.neq)]
    joints = kinds.count(int(mujoco.mjtEq.mjEQ_JOINT))
    connects = kinds.count(int(mujoco.mjtEq.mjEQ_CONNECT))
    assert (joints, connects) == (6, 4), (
        f"expected 6 joint equalities + 4 loop closures, found {joints} + {connects}"
    )
    assert all(m.eq_active0[i] for i in range(m.neq)), "an equality is inactive at reset"

    # And it actually drives: commanding the master must move the distal links, not just the master.
    # Command CLOSED, not open: the manifest holds the jaws open at rest (`gripper_ctrl: 0.07`), so
    # commanding open is a no-op and would pass this test with a completely broken linkage.
    rig.settle(1.0)
    before = [
        rig.q(jn)
        for jn in (
            "gripper_left_inner_finger_left_joint",
            "gripper_left_fingertip_left_joint",
            "gripper_left_outer_finger_left_joint",
        )
    ]
    rig.arms["arm_left_controller"].set_gripper(0.0)
    rig.step(2.5)
    after = [
        rig.q(jn)
        for jn in (
            "gripper_left_inner_finger_left_joint",
            "gripper_left_fingertip_left_joint",
            "gripper_left_outer_finger_left_joint",
        )
    ]
    assert rig.q("gripper_left_finger_joint") < 0.01, "master did not close"
    for name, b0, a0 in zip(
        ("inner_finger", "fingertip", "outer_finger"), before, after, strict=True
    ):
        assert abs(a0 - b0) > 0.05, f"{name} did not follow the master (loop closure not carrying?)"


def test_e5b_jaws_are_more_parallel_than_the_urdf_linearisation(rig):
    """E5b: closing the loop must keep the pads closer to parallel than the mimic fit did.

    The measured baseline is the URDF's own linearisation: 34.8 deg of splay at the grasp opening. The
    loop closure roughly halves it. This is a regression guard on the modelling choice, not a claim
    that the jaws are perfectly parallel -- they are not, and the port log says so.
    """
    m, d = rig.model, rig.data
    tcp = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "gripper_left_grasping_link")
    rig.arms["arm_left_controller"].set_gripper(0.015)
    rig.step(3.0)
    angles = []
    for lr in ("left", "right"):
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"gripper_left_inner_finger_{lr}_link")
        pad_normal = d.xmat[bid].reshape(3, 3) @ np.array([0.0, 1.0, 0.0])
        toward = d.xpos[tcp] - d.xpos[bid]
        toward /= np.linalg.norm(toward)
        angles.append(math.degrees(math.acos(min(1.0, abs(float(pad_normal @ toward))))))
    splay = sum(angles) / len(angles)
    assert splay < 25.0, (
        f"pad splay {splay:.1f} deg at the grasp opening (mimic-only baseline: 34.8)"
    )


def test_e6_gripper_stroke_matches_the_measured_range(rig):
    """E6: commanding open/closed produces the fingertip gap the port log documents (max 41 mm)."""
    m, d = rig.model, rig.data
    ctrl = rig.arms["arm_left_controller"]

    def pad_gap() -> float:
        out = {}
        for side in ("left", "right"):
            bid = mujoco.mj_name2id(
                m, mujoco.mjtObj.mjOBJ_BODY, f"gripper_left_fingertip_{side}_link"
            )
            pts = []
            for g in range(m.ngeom):
                if m.geom_bodyid[g] != bid or m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
                    continue
                mi = m.geom_dataid[g]
                a, n = m.mesh_vertadr[mi], m.mesh_vertnum[mi]
                v = m.mesh_vert[a : a + n].astype(float)
                pts.append(v @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g])
            out[side] = np.vstack(pts)
        # the gripper hangs at an angle in the home stance, so measure along the fingers' own axis
        return float(np.linalg.norm(out["left"].mean(0) - out["right"].mean(0)))

    rig.settle(1.0)
    ctrl.set_gripper(0.0)
    rig.step(2.0)
    closed = pad_gap()
    ctrl.set_gripper(GRIP_OPEN_TRAVEL)
    rig.step(2.0)
    opened = pad_gap()
    assert opened > closed, f"gripper did not open (closed {closed:.4f}, open {opened:.4f})"
    # Centroid separation, not the pad gap: the fingertip meshes are wedges, so this reads larger
    # than the 41.2 mm pad gap the port log records. What it pins is the STROKE.
    assert (opened - closed) * 1000 == pytest.approx(55.0, abs=10.0), (
        f"stroke {(opened - closed) * 1000:.1f} mm does not match the documented range"
    )


def test_e7_arms_do_not_disturb_the_base(rig):
    """E7: a full-speed dual-arm reach does not move the base — the planar drive holds it."""
    rig.settle(2.0)
    p0, y0 = rig.xy, rig.yaw
    rig.arms["arm_left_controller"].set_targets(
        ARM_L, [-0.25843, -0.57522, 0.50314, -2.0337, 0.0, 1.0543, 1.5708]
    )
    rig.arms["arm_right_controller"].set_targets(
        ARM_R, [0.25843, -0.57522, -0.50314, -2.0337, 0.0, 1.0543, -1.5708]
    )
    rig.arms["torso_head_controller"].set_targets(["torso_lift_joint"], [0.33])
    rig.step(8.0)
    assert np.linalg.norm(rig.xy - p0) < 0.02, f"base pushed {np.linalg.norm(rig.xy - p0):.3f} m"
    assert abs(rig.yaw - y0) < 0.02, "base yawed while the arms moved"
    rot = np.zeros(9)
    mujoco.mju_quat2Mat(rot, rig.data.qpos[3:7])
    assert rot.reshape(3, 3)[2, 2] > 0.995, "base tipped under the arms' motion"
