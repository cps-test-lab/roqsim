# Copyright (C) 2026 Frederik Pasch
# SPDX-License-Identifier: Apache-2.0

"""A mobile manipulator's arm is held up by its motors, and its motors stand on the base.

So the floor carries the whole robot, arm included, while the arm holds its pose. MuJoCo's external
``body_gravcomp`` alone would cancel the arm's weight at the free joint as well: the wheels would
carry the base only, and the arm's reach would not move load between them.
"""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from roqsim.config import load_config_from_dict
from roqsim.engine import Engine
from roqsim.plugin import PluginError

ROBOTS = ["frankie", "tiago_pro"]


def _engine(model: str, sim: dict | None = None, **spawn) -> Engine:
    cfg = load_config_from_dict(
        {
            "sim": sim or {},
            "components": [{"spawn_robot": {"model": model, **spawn}, "name": "robot"}],
        }
    )
    return Engine(cfg)


def _held(model: str) -> Engine:
    """The robot at rest on the floor, every position servo commanded to the pose it stands in."""
    engine = _engine(model)
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


def test_gravity_compensation_false_leaves_every_body_to_its_servo():
    """The world-level opt-out for a robot whose drives supply no gravity term."""
    engine = _engine("frankie", gravity_compensation=False)
    engine.ctx.seed = 0
    engine.setup()
    m = engine.ctx.model
    assert not np.any(m.body_gravcomp[1:]), "a body is still compensated"  # 0 is presence's marker
    assert not np.any(m.jnt_actgravcomp), "a drive still supplies a gravity term"
    engine.shutdown()


def test_rk4_is_refused_naming_the_robot_and_the_fix():
    engine = _engine("frankie", sim={"integrator": "rk4"})
    engine.ctx.seed = 0
    with pytest.raises(PluginError) as refused:
        engine.setup()
    message = str(refused.value)
    assert "sim.integrator: rk4" in message
    assert "rooted at ['base_link']" in message, f"the robot is not named: {message}"
    assert "implicitfast" in message and "gravity_compensation: false" in message


def test_rk4_steps_a_robot_that_opted_out():
    engine = _engine("frankie", sim={"integrator": "rk4"}, gravity_compensation=False)
    engine.ctx.seed = 0
    engine.setup()
    engine.reset()
    engine.step()
    engine.shutdown()
