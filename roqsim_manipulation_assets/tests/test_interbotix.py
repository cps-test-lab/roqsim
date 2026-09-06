"""The Interbotix/Trossen X-Series arms: upstream's numbers, and a gripper that actually holds.

Both arms are transformations of MuJoCo Menagerie's ``trossen_vx300s`` / ``trossen_wx250s``, which
Trossen derive from the same ``interbotix_xsarm_descriptions`` xacro the ROS 2 package ships. So the
first job is asserting nothing drifted in transit.

The second is the gripper, and it is why this file exists rather than reusing the arm battery.
``references/gripper.md`` is blunt about it: *"A grasp that is only checked at the instant it reaches
height will report picks that did not happen."* This port proved that twice over. Two earlier
attempts at ``test_holds_a_payload`` reported a "grasp" that was nothing of the kind -- once the box
had simply free-fallen to the floor before the fingers closed, once the arm swept through it on the
way to the pose and knocked it aside. Both looked plausible in a summary and both would have shipped
a gripper that cannot pick anything up. The test now spawns the arm *already at* the grasp stance
(``rest:``), so there is no approach sweep, and asserts the payload leaves the floor **and** does not
creep afterwards.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import pytest

from roqsim.config import load_config_from_dict
from roqsim.engine import Engine
from roqsim.models import resolve_model

ARMS = ("vx300s", "wx250s")
JOINTS = ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"]
#: From MuJoCo Menagerie @ da76818e, which took them from Trossen's own URDF.
UPSTREAM_MASS = {"vx300s": 3.7419, "wx250s": 2.1335}
#: The finger slide's ctrlrange, metres, PER ARM -- they differ, and using one arm's values on the
#: other commands outside ctrlrange, so the gripper silently does not move. Read from the model in
#: the tests rather than hardcoded, except here where the values are the thing being asserted.
FINGER_TRAVEL = {"vx300s": (0.021, 0.057), "wx250s": (0.015, 0.037)}
#: A pose that puts the jaws around a box resting on the floor, found by sweeping the model's
#: kinematics rather than guessed: the fingers close at x = 0.320, not at the `pinch` site (0.351).
GRASP = dict(zip(JOINTS, [0.0, 0.575, 1.100, 0.0, -1.600, 0.0]))
LIFT = dict(zip(JOINTS, [0.0, 0.100, 0.700, 0.0, -1.200, 0.0]))
GRASP_X = 0.320


def _engine(model, *, rest=None, gripper_ctrl=None, box=None, grasping=True):
    gripper_ctrl = FINGER_TRAVEL[model][1] if gripper_ctrl is None else gripper_ctrl
    sim = {"cone": "elliptic", "impratio": 10, "noslip_iterations": 10} if grasping else {}
    arm = {"spawn_arm": {"model": model, "prefix": "a_"}, "name": "a"}
    if rest is not None:
        arm["components"] = [{"arm_controller": {"rest": rest, "gripper_ctrl": gripper_ctrl}}]
    components = [arm]
    if box is not None:
        components.append(
            {
                "spawn_model": {
                    "model": "graspable_box",
                    "free": True,
                    "pose": {"position": {"x": box[0], "y": box[1], "z": box[2]}},
                },
                "name": "box",
            }
        )
    engine = Engine(
        load_config_from_dict({"sim": sim, "components": components}, base_dir=Path("."))
    )
    engine.setup()
    engine.reset()
    controller = next(p for p in engine.plugins if type(p).__name__ == "ArmControllerPlugin")
    return engine, controller


def _qadr(model, joint):
    return model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"a_{joint}")]


@pytest.mark.parametrize("model", ARMS)
def test_mass_matches_upstream(model):
    path = resolve_model(f"roqsim_manipulation_assets:{model}").path
    assert mujoco.MjModel.from_xml_path(str(path)).body_mass.sum() == pytest.approx(
        UPSTREAM_MASS[model], abs=1e-3
    )


@pytest.mark.parametrize("model", ARMS)
def test_no_option_block(model):
    # Upstream pins cone="elliptic" impratio="10" -- the contact settings a grasping arm needs, but
    # world-scoped: carried in the model they would reconfigure the solver for every other robot in
    # the scene. The demo world sets them instead. Parse rather than grep: the header says "<option>".
    import xml.etree.ElementTree as ET

    path = resolve_model(f"roqsim_manipulation_assets:{model}").path
    assert ET.parse(path).getroot().find("option") is None


@pytest.mark.parametrize("model", ARMS)
def test_gripper_is_wired_as_an_aux_actuator(model):
    """The gripper is a JOINT actuator, so it must be named, not inferred.

    ``arm_controller`` infers a gripper from a non-joint (tendon) actuator. This one drives the
    ``left_finger`` slide, so left to the prefix scan it would be claimed as a seventh arm joint and
    driven by trajectories -- an arm whose gripper snaps shut whenever a trajectory is executed.
    """
    engine, controller = _engine(model)
    try:
        assert len(controller._joint_acts) == 6, "the six arm joints must be the controlled set"
        assert len(controller._aux_acts) == 1, "the gripper must be held as an aux actuator"
    finally:
        engine.shutdown()


@pytest.mark.parametrize("model", ARMS)
def test_aperture_curve_and_jaw_parallelism(model):
    """E-G1/E-G2: the jaws track the command, and stay parallel across the whole travel."""
    engine, controller = _engine(model)
    try:
        m, d = engine.ctx.model, engine.ctx.data
        left, right = _qadr(m, "left_finger"), _qadr(m, "right_finger")
        lo, hi = FINGER_TRAVEL[model]
        for commanded in np.linspace(hi, lo, 4):
            controller._gripper_ctrl_target = commanded
            for _ in range(1200):
                engine.step()
            assert float(d.qpos[left]) == pytest.approx(commanded, abs=1e-3)
            # The <equality polycoef="0 -1 0 0 0"> mirrors the fingers; splayed jaws are the failure
            # references/gripper.md warns a literal <mimic> reproduction produces.
            assert abs(float(d.qpos[left]) + float(d.qpos[right])) < 1e-4, "jaws are not parallel"
    finally:
        engine.shutdown()


@pytest.mark.parametrize("model", ARMS)
def test_gravity_hold(model):
    engine, controller = _engine(model, rest=dict(zip(JOINTS, [0.0, -0.5, 0.6, 0.0, -0.3, 0.0])))
    try:
        m, d = engine.ctx.model, engine.ctx.data
        adr = [_qadr(m, j) for j in JOINTS]
        for _ in range(1000):
            engine.step()
        start = np.array([float(d.qpos[a]) for a in adr])
        for _ in range(3000):
            engine.step()
        drift = np.degrees(np.abs(np.array([float(d.qpos[a]) for a in adr]) - start)).max()
        assert drift < 1.0, f"sagged {drift:.2f} deg under gravity"
    finally:
        engine.shutdown()


def test_holds_a_payload():
    """E-G held load: pick a 0.5 kg box off the floor and keep it there.

    The arm spawns already at the grasp stance, because commanding it there from home sweeps the
    gripper through the box and knocks it away -- which an earlier version of this test scored as a
    successful grasp.
    """
    engine, controller = _engine("vx300s", rest=GRASP, box=[GRASP_X, 0.0, 0.025])
    try:
        m, d = engine.ctx.model, engine.ctx.data
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "box")
        for _ in range(1000):
            engine.step()
        assert float(d.xpos[bid][2]) == pytest.approx(0.025, abs=0.01), (
            "the box moved before the grasp -- the arm is sweeping into it"
        )
        controller._gripper_ctrl_target = FINGER_TRAVEL["vx300s"][0]
        for _ in range(1500):
            engine.step()
        controller._target = LIFT
        for _ in range(3000):
            engine.step()
        lifted = float(d.xpos[bid][2])
        assert lifted - 0.025 > 0.03, (
            f"box only reached z={lifted:.4f}; it was not picked up, merely nudged"
        )
        # Creep is the check that separates a grasp from a box briefly resting on the jaws.
        held = [lifted]
        for _ in range(5):
            for _ in range(1000):
                engine.step()
            held.append(float(d.xpos[bid][2]))
        creep = (held[0] - held[-1]) / (5000 * engine.ctx.dt)
        assert abs(creep) < 0.005, f"payload crept {creep * 1000:.2f} mm/s out of the jaws"
    finally:
        engine.shutdown()
