# Copyright (C) 2026 Frederik Pasch
# SPDX-License-Identifier: Apache-2.0

"""A mobile manipulator's arm is held up by its motors, and its motors stand on the base.

So the floor carries the whole robot, arm included, while the arm holds its pose. Compensated as
MuJoCo's external ``body_gravcomp`` alone, the arm's weight was cancelled at the free joint as well:
the wheels carried the base only, and the arm's reach no longer moved load between them.
"""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from roqsim.config import load_config_from_dict
from roqsim.engine import Engine

ROBOTS = ["frankie", "tiago_pro"]


def _held(model: str) -> Engine:
    """The robot at rest on the floor, every position servo commanded to the pose it stands in."""
    cfg = load_config_from_dict(
        {"sim": {}, "components": [{"spawn_robot": {"model": model}, "name": "robot"}]}
    )
    engine = Engine(cfg)
    engine.ctx.seed = 0  # a test driving an Engine is the driver, and the seed is driver-owned
    engine.setup()
    engine.reset()
    m, d = engine.ctx.model, engine.ctx.data
    for i in range(m.nu):
        jid = m.actuator_trnid[i, 0]
        servo = m.actuator_biastype[i] == mujoco.mjtBias.mjBIAS_AFFINE and m.actuator_biasprm[i, 1]
        if m.actuator_trntype[i] == mujoco.mjtTrn.mjTRN_JOINT and m.jnt_type[jid] > 1 and servo:
            d.ctrl[i] = d.qpos[m.jnt_qposadr[jid]]
    return engine


def _base_free_joint(m) -> int:
    return next(j for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE)


@pytest.mark.parametrize("model", ROBOTS)
def test_the_floor_carries_the_whole_robot(model):
    engine = _held(model)
    m, d = engine.ctx.model, engine.ctx.data
    for _ in range(int(3.0 / m.opt.timestep)):
        engine.step()

    floor = m.geom("floor").id
    force = np.zeros(6)
    carried = 0.0
    for i in range(d.ncon):
        contact = d.contact[i]
        if floor in (contact.geom1, contact.geom2):
            mujoco.mj_contactForce(m, d, i, force)
            carried += abs(float(force[0] * contact.frame[2]))
    free = _base_free_joint(m)
    weight = float(m.body_subtreemass[m.jnt_bodyid[free]]) * float(-m.opt.gravity[2])
    lift = float(d.qfrc_gravcomp[m.jnt_dofadr[free] + 2])
    engine.shutdown()

    assert carried == pytest.approx(weight, rel=5e-3), (
        f"the floor carries {carried:.1f} of {weight:.1f} N"
    )
    assert abs(lift) < 1e-6 * weight, f"gravity compensation lifts the base by {lift:.1f} N"


@pytest.mark.parametrize("model", ROBOTS)
def test_the_arm_holds_the_pose_it_is_commanded(model):
    """Unchanged by where the reaction goes: the arm's joints get the same holding torque."""
    engine = _held(model)
    m, d = engine.ctx.model, engine.ctx.data
    held = [
        int(m.actuator_trnid[i, 0])
        for i in range(m.nu)
        if m.actuator_trntype[i] == mujoco.mjtTrn.mjTRN_JOINT
        and m.jnt_actgravcomp[m.actuator_trnid[i, 0]]
        # The fingers follow the gripper command arm_controller keeps writing, not this one.
        and not {"gripper", "finger"} & set(m.joint(m.actuator_trnid[i, 0]).name.split("_"))
    ]
    assert held, "no joint's drive carries its gravity term"
    want = {j: float(d.qpos[m.jnt_qposadr[j]]) for j in held}
    for _ in range(int(4.0 / m.opt.timestep)):
        engine.step()
    drift = {m.joint(j).name: abs(float(d.qpos[m.jnt_qposadr[j]]) - want[j]) for j in held}
    engine.shutdown()

    worst = max(drift, key=drift.get)
    assert drift[worst] < 1e-3, f"{worst} drifted {drift[worst]:.4f} from the pose it was commanded"
