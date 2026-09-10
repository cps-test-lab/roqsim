# Copyright (C) 2026 Frederik Pasch
# SPDX-License-Identifier: Apache-2.0

"""A robot that is both an arm and a base gets the arm's physics and the base's, separately.

`spawn_arm` and `spawn_robot` had different answers to who holds a joint up, so the same Panda
arm carried its own weight on a bench and not on a base. It does now, and the base does not --
which is the half a wheeled machine needs, because a compensated base presses on the floor with
less than it weighs and stays upright while doing it.
"""

from __future__ import annotations

from roqsim.config import load_config_from_dict
from roqsim.engine import Engine


def _built():
    cfg = load_config_from_dict({"sim": {}, "components": [
        {"spawn_robot": {"model": "frankie"}, "name": "frankie"}]})
    engine = Engine(cfg)
    engine.setup()
    engine.reset()
    return engine


def test_the_arm_links_carry_their_own_weight_and_the_base_does_not():
    """Both halves in one assertion, because either alone is satisfied by the wrong answer.

    Compensating nothing passes the second; compensating everything passes the first.
    """
    engine = _built()
    model = engine.ctx.model
    compensated = {model.body(b).name for b in range(model.nbody) if model.body_gravcomp[b] > 0}
    engine.shutdown()

    assert {"link1", "link7", "hand"} <= compensated, "the arm hangs off its own motors"
    assert compensated.isdisjoint({"base_link", "left_wheel", "right_wheel"}), (
        "the base and its wheels carry the robot; cancelling their weight unloads the floor"
    )


def test_the_arm_holds_the_pose_it_is_commanded():
    """The behaviour the marking is for, measured rather than inferred from the flags."""
    engine = _built()
    model, data = engine.ctx.model, engine.ctx.data
    for i in range(model.nu):
        if model.actuator_trntype[i] == 0:
            jid = model.actuator_trnid[i, 0]
            if model.jnt_type[jid] in (2, 3):
                data.ctrl[i] = data.qpos[model.jnt_qposadr[jid]]

    arm = [f"joint{i}" for i in range(1, 8)]
    want = {j: float(data.qpos[model.jnt_qposadr[model.joint(j).id]]) for j in arm}
    for _ in range(int(4.0 / model.opt.timestep)):
        engine.step()
    drift = max(abs(float(data.qpos[model.jnt_qposadr[model.joint(j).id]]) - want[j]) for j in arm)
    engine.shutdown()

    assert drift < 1e-3, f"the arm drifted {drift:.4f} rad from the pose it was commanded"
