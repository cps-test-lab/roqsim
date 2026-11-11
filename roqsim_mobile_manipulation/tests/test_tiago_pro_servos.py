"""TIAGo Pro's position servos are critically damped against the joint inertia they meet.

Each servo's ``kv`` is ``2 * sqrt(kp * M_jj)`` at qpos0, ``M_jj`` the diagonal of the joint-space
inertia (``dof_M0``), as ``external/convert/build_tiago_pro_mjcf.py`` bakes it. ``dampratio`` would
leave the damping to MuJoCo, which since 3.15 derives it from the operational-space inertia
``1 / (M^-1)_jj``: on this arm that is a quarter of ``M_jj`` at the shoulder, and the shoulder
overshot a step by 41 %.
"""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from roqsim.models import resolve_model

#: Servos whose step response is pinned: (actuator, step in the joint's unit).
STEPS = [("arm_left_2_pos", 0.3), ("arm_left_4_pos", 0.3), ("torso_lift_pos", 0.05)]


@pytest.fixture(scope="module")
def model():
    m = mujoco.MjModel.from_xml_path(str(resolve_model("tiago_pro").path))
    m.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST  # what `sim.integrator: auto` runs
    return m


def test_every_position_servo_is_critically_damped_against_its_joint_inertia(model):
    servos = 0
    for a in range(model.nu):
        position = model.actuator_biastype[a] == mujoco.mjtBias.mjBIAS_AFFINE
        if not (position and model.actuator_biasprm[a, 1] < 0):  # a position servo: bias -kp q
            continue
        joint = int(model.actuator_trnid[a, 0])
        inertia = float(model.dof_M0[model.jnt_dofadr[joint]])
        kp, kv = float(model.actuator_gainprm[a, 0]), -float(model.actuator_biasprm[a, 2])
        assert kv == pytest.approx(2.0 * np.sqrt(kp * inertia), rel=1e-5), model.actuator(a).name
        servos += 1
    assert servos == 19


@pytest.mark.parametrize(("actuator", "step"), STEPS)
def test_a_step_settles_without_overshoot(model, actuator, step):
    """Every servo held where it stands, then one stepped: it reaches the target and barely passes it."""
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    for a in range(model.nu):
        if model.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT:
            data.ctrl[a] = data.qpos[model.jnt_qposadr[model.actuator_trnid[a, 0]]]
    for _ in range(2000):
        mujoco.mj_step(model, data)
    act = model.actuator(actuator)
    qadr = int(model.jnt_qposadr[model.actuator_trnid[act.id, 0]])
    start = float(data.qpos[qadr])
    data.ctrl[act.id] = start + step
    moved = []
    for _ in range(4000):
        mujoco.mj_step(model, data)
        moved.append(float(data.qpos[qadr]) - start)
    assert moved[-1] == pytest.approx(step, rel=1e-3)
    assert max(moved) <= step * 1.01, f"{actuator} overshoots by {max(moved) / step - 1:.1%}"
