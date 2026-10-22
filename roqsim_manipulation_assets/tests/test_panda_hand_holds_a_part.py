"""The Franka Hand holds what it closes on: a thin part, grasped by friction alone, stays in the
pads under its own weight and through a lift.

The fingers are a position servo on the hand's split tendon, so the grip on a part narrower than
the commanded closing is the servo's stiffness times the remaining travel. Too soft a servo lets the
part slide down the pads without any grasp failing loudly: the arm carries the part lower than it
took it.
"""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from roqsim.config import load_config_from_dict
from roqsim.engine import Engine

PART = """<mujoco model="thin_part">
  <worldbody>
    <body name="thin_part">
      <geom name="bar" type="box" size="0.006 0.006 0.04" pos="0 0 0.04" mass="0.2"
            friction="1.0 0.005 0.0001" condim="4" solimp="0.9 0.95 0.001" solref="0.005 1"/>
    </body>
  </worldbody>
</mujoco>
"""
DT = 0.001


def _world(tmp_path):
    part = tmp_path / "thin_part.xml"
    part.write_text(PART)
    return load_config_from_dict(
        {
            "sim": {"timestep": DT, "gravity": [0.0, 0.0, 0.0], "cone": "elliptic"},
            "components": [
                {
                    "spawn_arm": {
                        "model": "panda",
                        "prefix": "panda_",
                        "pose": {"position": {"x": 0.0, "y": 0.0, "z": 0.5}},
                        "pedestal": True,
                    },
                    "name": "panda",
                    "components": [
                        {
                            "cartesian_admittance": {
                                "controller_type": "cartesian_motion_controller",
                                "site": "pinch",
                                "max_linear_vel": 0.2,
                            },
                            "name": "cartesian",
                        }
                    ],
                },
                {
                    "spawn_model": {
                        "model": str(part),
                        "motion": "physics",
                        "pose": {"position": {"x": 1.0, "y": 0.0, "z": 0.0}},
                    },
                    "name": "part",
                },
            ],
        },
        base_dir=tmp_path,
    )


def test_panda_hand_holds_a_thin_part_under_gravity_and_through_a_lift(tmp_path):
    eng = Engine(_world(tmp_path))
    eng.ctx.seed = 1
    eng.setup()
    eng.reset()
    m, d = eng.ctx.model, eng.ctx.data
    cart = eng.ctx.blackboard.require("cartesian:panda")
    grip = next(
        e for e in eng.ctx.interface.all() if e.owner == "panda" and e.name == "gripper_cmd"
    )
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "thin_part")
    jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "free")
    q, v = int(m.jnt_qposadr[jid]), int(m.jnt_dofadr[jid])
    # The part standing between the open fingers, its top 8 mm above the pinch site.
    pinch, _ = cart.read_pose()
    d.qpos[q : q + 3] = (pinch[0], pinch[1], pinch[2] - 0.072)
    d.qpos[q + 3 : q + 7] = (1.0, 0.0, 0.0, 0.0)
    d.qvel[v : v + 6] = 0.0
    mujoco.mj_forward(m, d)
    grip.write({"position": 0.0})
    for _ in range(int(0.8 / DT)):
        eng.step()
    held = d.xpos[bid][2] - np.asarray(cart.read_pose()[0])[2]
    # Weight comes on: the part must not slide in the pads.
    m.opt.gravity[:] = (0.0, 0.0, -9.81)
    for _ in range(int(2.0 / DT)):
        eng.step()
    assert d.xpos[bid][2] - np.asarray(cart.read_pose()[0])[2] == pytest.approx(held, abs=2e-3)
    # And through a lift of 0.1 m.
    pos0 = np.asarray(cart.read_pose()[0])
    cart.set_goal(pos0 + (0.0, 0.0, 0.1), None)
    for _ in range(int(3.0 / DT)):
        eng.step()
    pos = np.asarray(cart.read_pose()[0])
    assert pos[2] - pos0[2] > 0.09
    assert d.xpos[bid][2] - pos[2] == pytest.approx(held, abs=2e-3)
