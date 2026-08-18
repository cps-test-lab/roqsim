"""The OpenMANIPULATOR-X model: geometry from the vendor URDF, and it does not jam itself.

The self-collision test is the one that earns its place. MuJoCo does not filter contacts between chain
neighbours and these meshes overlap ~21 mm at every joint, so before the SRDF-derived
``<contact><exclude>`` block existed, link1-vs-link2 contact held joint1 at its torque limit and the arm
would not move at all -- while loading, compiling and reporting healthy.
"""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from roqsim.config import load_config_from_dict
from roqsim.engine import Engine

#: From the vendor URDF (ros-jazzy-open-manipulator-description). If a regenerated MJCF ever disagrees
#: with these, the upstream arm changed -- which is a decision to make, not a test to update quietly.
TOOL_AT_ZERO = np.array([0.286, 0.0, 0.1875])
TOTAL_MASS = 0.5921
JOINT_RANGES = {
    "joint1": (-3.14159265, 3.14159265),
    "joint2": (-1.5, 1.5),
    "joint3": (-1.5, 1.4),
    "joint4": (-1.7, 1.97),
}


def _engine(tmp_path, **arm_extra):
    plugins = [
        {
            "spawn_arm": {
                "model": "open_manipulator_x",
                "name": "omx",
                "prefix": "omx_",
                **arm_extra,
            }
        },
        {"arm_controller": {"arm": "omx"}},
    ]
    engine = Engine(load_config_from_dict({"sim": {}, "plugins": plugins}, base_dir=tmp_path))
    engine.setup()
    engine.reset()
    return engine


def _qadr(m):
    return [
        m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"omx_{j}")]
        for j in JOINT_RANGES
    ]


def test_geometry_matches_the_vendor_urdf(tmp_path):
    engine = _engine(tmp_path)
    m, d = engine.ctx.model, engine.ctx.data
    qadr = _qadr(m)
    d.qpos[qadr] = 0.0
    mujoco.mj_kinematics(m, d)
    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "omx_end_effector_site")
    np.testing.assert_allclose(d.site_xpos[sid], TOOL_AT_ZERO, atol=1e-6)
    assert float(m.body_mass.sum()) == pytest.approx(TOTAL_MASS, abs=1e-3)
    for joint, (lo, hi) in JOINT_RANGES.items():
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"omx_{joint}")
        np.testing.assert_allclose(m.jnt_range[jid], (lo, hi), atol=1e-6)


def test_four_dof_and_no_gripper_actuator(tmp_path):
    """4 joints, 4 actuators, and no gripper: the fingers are welded, so this arm cannot grasp."""
    engine = _engine(tmp_path)
    m = engine.ctx.model
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(m.njnt)]
    assert sorted(n for n in names if n) == ["omx_joint1", "omx_joint2", "omx_joint3", "omx_joint4"]
    assert m.nu == 4
    # The finger geometry IS present -- it collides, and MoveIt plans against it from the same URDF.
    for side in ("left", "right"):
        gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"omx_gripper_{side}_link_col")
        assert gid >= 0 and m.geom_contype[gid] != 0


#: The pairs the vendor SRDF disables (`disable_collisions`), with the two gripper links folded into
#: link5 because the MJCF welds the fingers there. The simulated arm must exclude EXACTLY these: a pair
#: MoveIt ignores that the sim collides jams the arm (this is what held joint1 at its torque limit
#: before the exclusions existed), and a pair MoveIt checks that the sim permits lets the sim run
#: through a self-collision the planner refused to plan.
SRDF_EXCLUDED = {
    ("omx_link1", "omx_link2"),
    ("omx_link1", "omx_link3"),
    ("omx_link2", "omx_link3"),
    ("omx_link3", "omx_link4"),
    ("omx_link4", "omx_link5"),
}


def test_excluded_pairs_match_the_srdf(tmp_path):
    """Chain neighbours never contact; everything else is left for MoveIt and the sim to agree on.

    Note what this deliberately does NOT assert: that the arm is self-collision-free everywhere in its
    joint ranges. It is not, and it should not be -- folded fully back, link5 really does reach link1,
    and the SRDF checks that pair too. A test demanding a collision-free workspace would have to be
    "fixed" by excluding a pair the planner checks, which is the dangerous direction.
    """
    engine = _engine(tmp_path)
    m, d = engine.ctx.model, engine.ctx.data
    qadr = _qadr(m)
    lo = np.array([r[0] for r in JOINT_RANGES.values()])
    hi = np.array([r[1] for r in JOINT_RANGES.values()])
    rng = np.random.default_rng(0)

    def body(gid):
        return mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[gid]) or ""

    seen_pairs = set()
    for _ in range(400):
        d.qpos[qadr] = rng.uniform(lo, hi)
        d.qvel[:] = 0
        mujoco.mj_forward(m, d)
        for c in range(d.ncon):
            pair = tuple(sorted((body(d.contact[c].geom1), body(d.contact[c].geom2))))
            if pair[0].startswith("omx_") and pair[1].startswith("omx_"):
                seen_pairs.add(pair)

    leaked = seen_pairs & SRDF_EXCLUDED
    assert not leaked, (
        f"contact reported for SRDF-excluded pair(s) {sorted(leaked)} -- the generated "
        "<contact><exclude> block is missing or incomplete, and the arm will jam"
    )


def test_tracks_a_target_without_saturating(tmp_path):
    """Servo gains hold the arm to its target, and inside the XM430's real 4.1 N-m stall torque."""
    engine = _engine(tmp_path)
    m, d = engine.ctx.model, engine.ctx.data
    qadr = _qadr(m)
    target = np.array([0.6, 0.3, -0.4, 0.9])
    handle = engine.ctx.blackboard.get("arm:omx")
    handle.set_targets(list(JOINT_RANGES), list(target))
    for _ in range(2500):  # 5 s at the default timestep
        engine.step()
    err = np.abs(d.qpos[qadr] - target).max()
    assert err < 0.02, f"settled {err:.4f} rad from target -- the servo is not holding"
    assert np.abs(d.actuator_force[:4]).max() < 4.1, "actuator saturated: gains are too high"


def test_ships_the_eye_in_hand_camera_mount(tmp_path):
    """The model carries the vendor's RealSense mount; a world decides whether to render from it."""
    engine = _engine(tmp_path)
    m = engine.ctx.model
    cid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "omx_d435_color")
    assert cid >= 0, "the d435_color camera is what realsense_d435 renders from"
    # ... but NOT the plugin: whether a run renders is the experiment's choice, per the manifest.
    assert not any(type(p).__name__ == "RealsenseD435Plugin" for p in engine.plugins)
