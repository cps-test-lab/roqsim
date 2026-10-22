"""The UR30 against Universal Robots' own numbers, and the payload it is rated to carry.

The model is generated (``external/convert/build_ur30_mjcf.py``) from UR's description data, so these
tests check what a generator could get wrong while still producing a model that compiles and moves:
a link length or a lateral offset misplaced in the frame mapping, a mass or an effort limit lost, an
inertial transformed into the wrong frame. Every vendor figure is a fixture with its source beside it.

Sources:
  [TS] UR30 technical specification (tech sheet), Universal Robots.
  [UM] UR30 user manual, PolyScope 5.19, "Maximum Payload".
  [UD] Universal_Robots_ROS2_Description config/ur30 (the generator's pinned input).

The payload test is the one the arm is here for: 25 kg held by a 3.6 kg bore gripper whose centre of
gravity lies on the tool axis, everywhere a cell beside a pallet cage reaches with the tool pointing
down, within the joint torque limits -- at rest and at the accelerations such a cell moves at.
"""

from __future__ import annotations

import math
from pathlib import Path

import mujoco
import numpy as np
import pytest

from roqsim.config import load_config_from_dict
from roqsim.engine import Engine
from roqsim.models import apply_assets, resolve_model

# [TS] arm mass without cable, reach.
VENDOR_MASS_KG = 63.5
VENDOR_REACH_M = 1.300
# [UD] default_kinematics.yaml.
DH = {"d1": 0.2363, "a2": 0.637, "a3": 0.5037, "d4": 0.201, "d5": 0.1593, "d6": 0.1543}
# [UD] joint_limits.yaml (which cites the UR20 manual for the efforts).
EFFORT_NM = {"shoulder_pan": 738.0, "shoulder_lift": 738.0, "elbow": 433.0,
             "wrist_1": 107.0, "wrist_2": 107.0, "wrist_3": 107.0}
RANGE_RAD = {"shoulder_pan_joint": 2 * math.pi, "shoulder_lift_joint": 2 * math.pi,
             "elbow_joint": math.pi, "wrist_1_joint": 2 * math.pi, "wrist_2_joint": 2 * math.pi,
             "wrist_3_joint": 2 * math.pi}
# [TS] rated payload. [UM] the payload curve limits the HORIZONTAL offset of the centre of gravity
# from the flange centre; an offset along the tool's Z with the tool vertical is unrestricted.
# "Full performance" curve read off the manual's chart at 0 and 100 mm horizontal offset.
RATED_PAYLOAD_KG = 30.0
FULL_PERFORMANCE_CURVE = {0.0: 30.0, 0.1: 30.0}

# What a heavy-part cell hangs on the flange: a 3.6 kg bore gripper and a 25 kg part, their common
# centre of gravity on the tool axis 0.12 m out.
PAYLOAD_KG = 25.0 + 3.6
PAYLOAD_COG_Z = 0.12
# The cell's accelerations: 3 m/s^2 along each axis (see the bore gripper's carry test).
ACCEL = 3.0

#: Where such a cell reaches, tool down: horizontal distance from the base axis and flange height
#: above the base, from the bottom layer of a pallet cage beside the arm up to a station top. The
#: tool-down workspace narrows as the flange rises (the wrist sits d5 + d6 above it), so the grid is
#: what the arm reaches, not a rectangle.
REACH_GRID = [(0.5, 0.3), (0.8, 0.3), (1.05, 0.3), (0.5, 0.6), (0.8, 0.6), (1.0, 0.45), (0.5, 0.9)]


def _ur30():
    asset = resolve_model("roqsim_manipulation_assets:ur30")
    spec = mujoco.MjSpec.from_file(str(asset.path))
    apply_assets(spec, asset)
    return spec


@pytest.fixture(scope="module")
def ur30():
    m = _ur30().compile()
    return m, mujoco.MjData(m)


def _anchor(m, d, joint):
    return np.array(d.xanchor[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, joint)])


def test_ur30_mass_matches_the_datasheet(ur30):
    m, _ = ur30
    assert float(np.sum(m.body_mass)) == pytest.approx(VENDOR_MASS_KG, rel=0.02)


def test_ur30_link_lengths_are_the_descriptions(ur30):
    """Joint anchors at the zero pose (arm stretched along +x) reproduce d1..d6 to 0.1 mm."""
    m, d = ur30
    d.qpos[:] = 0
    mujoco.mj_forward(m, d)
    lift, elbow, w1, w2, w3 = (_anchor(m, d, j) for j in (
        "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"))
    flange = d.site("tool_site").xpos
    assert lift[2] == pytest.approx(DH["d1"], abs=1e-4)
    assert elbow[0] - lift[0] == pytest.approx(DH["a2"], abs=1e-4)
    assert w1[0] - elbow[0] == pytest.approx(DH["a3"], abs=1e-4)
    assert w2[1] - 0.0 == pytest.approx(DH["d4"], abs=1e-4)
    assert np.linalg.norm(w3 - w2) == pytest.approx(DH["d5"], abs=1e-4)
    assert np.linalg.norm(flange - w3) == pytest.approx(DH["d6"], abs=1e-4)


def test_ur30_reach_matches_the_datasheet(ur30):
    """The farthest the flange gets from the shoulder axis: a2 + a3 + d5 when stretched."""
    m, d = ur30
    best = 0.0
    for w1 in np.linspace(-math.pi, math.pi, 73):
        d.qpos[:] = 0
        d.qpos[3] = w1
        mujoco.mj_forward(m, d)
        lift = _anchor(m, d, "shoulder_lift_joint")
        v = d.site("tool_site").xpos - lift
        best = max(best, math.hypot(v[0], v[2]))
    assert best == pytest.approx(VENDOR_REACH_M, abs=0.01)


def test_ur30_effort_limits_and_ranges_are_the_descriptions(ur30):
    m, _ = ur30
    for name, effort in EFFORT_NM.items():
        aid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        assert m.actuator_forcerange[aid].tolist() == pytest.approx([-effort, effort])
    for joint, limit in RANGE_RAD.items():
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, joint)
        assert m.jnt_range[jid].tolist() == pytest.approx([-limit, limit], abs=1e-3)


def test_ur30_carries_the_ft_cut_at_its_flange(ur30):
    """The UR30 measures at its flange, so fts_site and tool_site are both on the flange face."""
    m, d = ur30
    mujoco.mj_forward(m, d)
    fts, tool, flange = (d.site(s).xpos for s in ("fts_site", "tool_site", "attachment_site"))
    assert np.allclose(fts, flange, atol=1e-9) and np.allclose(tool, flange, atol=1e-9)


def test_cell_payload_is_inside_the_vendor_payload_curve():
    """Mass under the rated payload, centre of gravity on the tool axis (zero horizontal offset)."""
    assert PAYLOAD_KG <= FULL_PERFORMANCE_CURVE[0.0]
    assert PAYLOAD_KG <= RATED_PAYLOAD_KG


# --- the payload at the cell's reach --------------------------------------------------------------


def _with_payload(mass: float, cog_z: float) -> mujoco.MjModel:
    spec = _ur30()
    tool = spec.body("tool0")
    load = tool.add_body(name="payload", pos=[0, 0, cog_z])
    r = 0.19
    load.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[r, 0.05, 0], contype=0, conaffinity=0,
                  mass=mass)
    return spec.compile()


def _ik_tool_down(m, d, r: float, z: float, q0=None) -> np.ndarray:
    """Damped least squares: flange at (r, 0.2, z), tool axis pointing down."""
    site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "tool_site")
    target = np.array([r, 0.2, z])
    d.qpos[:6] = q0 if q0 is not None else [0, -1.2, 1.6, -1.97, -1.57, 0]
    jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
    for _ in range(3000):
        mujoco.mj_forward(m, d)
        x = d.site_xpos[site]
        axis = d.site_xmat[site].reshape(3, 3)[:, 2]
        err = np.concatenate([target - x, np.cross(axis, [0, 0, -1])])
        if np.linalg.norm(err) < 1e-6:
            break
        mujoco.mj_jacSite(m, d, jp, jr, site)
        j = np.vstack([jp, jr])[:, :6]
        d.qpos[:6] += j.T @ np.linalg.solve(j @ j.T + 1e-4 * np.eye(6), err)
    assert np.linalg.norm(err) < 1e-5, f"no tool-down pose at r={r}, z={z}"
    return d.qpos[:6].copy()


def _torques(m, d, q, qacc) -> np.ndarray:
    d.qpos[:6] = q
    d.qvel[:] = 0
    d.qacc[:6] = qacc
    mujoco.mj_inverse(m, d)
    return d.qfrc_inverse[:6].copy()


def _limits(m) -> np.ndarray:
    return np.array([m.actuator_forcerange[i][1] for i in range(6)])


def test_ur30_holds_the_cell_payload_within_its_torque_limits():
    """At every pose of the grid, static and at 3 m/s^2 along each axis, every joint stays inside."""
    m = _with_payload(PAYLOAD_KG, PAYLOAD_COG_Z)
    d = mujoco.MjData(m)
    limits = _limits(m)
    site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "tool_site")
    worst = 0.0
    for r, z in REACH_GRID:
        q = _ik_tool_down(m, d, r, z)
        static = _torques(m, d, q, np.zeros(6))
        worst = max(worst, float(np.max(np.abs(static) / limits)))
        jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        mujoco.mj_jacSite(m, d, jp, jr, site)
        j = np.vstack([jp, jr])[:, :6]
        for axis in np.vstack([np.eye(3), -np.eye(3)]):
            qacc = np.linalg.lstsq(j, np.concatenate([ACCEL * axis, np.zeros(3)]), rcond=None)[0]
            tau = _torques(m, d, q, qacc)
            ratio = np.abs(tau) / limits
            assert np.all(ratio < 1.0), f"r={r} z={z} a={axis}: {np.round(tau, 1)} Nm"
            worst = max(worst, float(np.max(ratio)))
    # A margin the test reports, so a change that eats into it shows before it fails.
    assert worst < 0.9, f"worst joint load {worst:.0%} of its limit"


def test_ur30_cannot_hold_twice_its_rating_at_full_stretch():
    """The negative control: 60 kg at the flange, arm stretched out, needs more than the shoulder has."""
    m = _with_payload(60.0, 0.0)
    d = mujoco.MjData(m)
    q = np.array([0.0, 0.0, 0.0, -math.pi / 2, 0.0, 0.0])
    tau = _torques(m, d, q, np.zeros(6))
    assert abs(tau[1]) > _limits(m)[1]


def _hold_world(tmp_path: Path, mass: float) -> dict:
    payload = tmp_path / "payload.xml"
    payload.write_text(f"""<mujoco model="payload">
  <worldbody>
    <body name="payload">
      <geom type="cylinder" size="0.19 0.05" pos="0 0 {PAYLOAD_COG_Z}" mass="{mass}" contype="0" conaffinity="0"/>
    </body>
  </worldbody>
</mujoco>""")
    return {"sim": {}, "components": [{"spawn_arm": {
        "model": "ur30", "prefix": "ur30_", "end_effector": {"model": str(payload)},
        "home": [0.0, -0.9, 1.5, -2.17, -1.5708, 0.0]}, "name": "ur30"}]}


def _flange_drift_mm(tmp_path: Path, mass: float, seconds: float = 5.0) -> float:
    engine = Engine(load_config_from_dict(_hold_world(tmp_path, mass)))
    engine.setup()
    engine.reset()
    m, d = engine.ctx.model, engine.ctx.data
    for i in range(m.nu):
        jid = m.actuator_trnid[i, 0]
        if m.actuator_trntype[i] == mujoco.mjtTrn.mjTRN_JOINT:
            d.ctrl[i] = d.qpos[m.jnt_qposadr[jid]]
    site = m.site("ur30_tool_site").id
    p0 = d.site_xpos[site].copy()
    for _ in range(int(seconds / m.opt.timestep)):
        engine.step()
    drift = float(np.linalg.norm(d.site_xpos[site] - p0)) * 1e3
    engine.shutdown()
    return drift


def test_ur30_holds_the_cell_payload_still_in_simulation(tmp_path):
    """Spawned through spawn_arm with the payload welded on, the arm stands where it was put."""
    assert _flange_drift_mm(tmp_path, PAYLOAD_KG) < 1.0


def test_ur30_sags_under_a_payload_beyond_its_torque(tmp_path):
    """The same pose with 120 kg: the limited drives cannot hold it, and the flange falls."""
    assert _flange_drift_mm(tmp_path, 120.0, seconds=2.0) > 20.0


def test_ur30_self_collision_rate_matches_siblings(ur30):
    """The reshaped collision envelope stays in the band of the package's other arms.

    ur10e and ur5e report self-collision on ~17.5 % of uniform random configurations, measured as
    here; a planner refuses those poses, so an envelope that has grown shows up as a rate above it.
    """
    m, d = ur30
    rng = np.random.default_rng(1)
    trials = 1500
    colliding = 0
    for _ in range(trials):
        d.qpos[:] = rng.uniform(m.jnt_range[:, 0], m.jnt_range[:, 1])
        mujoco.mj_forward(m, d)
        colliding += any(d.contact[c].dist < -1e-4 for c in range(d.ncon))
    assert colliding / trials < 0.22, f"self-collision on {colliding / trials:.1%} of random poses"
