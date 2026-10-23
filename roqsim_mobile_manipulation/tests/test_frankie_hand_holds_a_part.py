"""Frankie's Franka Hand holds what it closes on: a thin part, grasped by friction alone, stays in
the pads when its weight comes on.

The fingers are a position servo on the hand's split tendon, so the grip on a part narrower than
the commanded closing is the servo's stiffness times the remaining travel. Too soft a servo lets
the part slide down the pads without any grasp failing loudly.
"""

from __future__ import annotations

import mujoco
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


def test_frankie_hand_holds_a_thin_part_under_gravity(tmp_path):
    part = tmp_path / "thin_part.xml"
    part.write_text(PART)
    cfg = load_config_from_dict(
        {
            "sim": {"timestep": DT, "gravity": [0.0, 0.0, 0.0], "cone": "elliptic"},
            "components": [
                {"spawn_robot": {"model": "frankie"}, "name": "frankie"},
                {
                    "spawn_model": {
                        "model": str(part),
                        "motion": "physics",
                        "pose": {"position": {"x": 2.0, "y": 0.0, "z": 0.0}},
                    },
                    "name": "part",
                },
            ],
        },
        base_dir=tmp_path,
    )
    eng = Engine(cfg)
    eng.ctx.seed = 1
    eng.setup()
    eng.reset()
    m, d = eng.ctx.model, eng.ctx.data
    grip = next(e for e in eng.ctx.interface.all() if e.name == "gripper_cmd")
    pinch = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "pinch")
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "thin_part")
    jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "free")
    q, v = int(m.jnt_qposadr[jid]), int(m.jnt_dofadr[jid])
    # The part standing upright between the open fingers, its top 8 mm above the pinch site.
    mujoco.mj_forward(m, d)
    p = d.site_xpos[pinch]
    d.qpos[q : q + 3] = (p[0], p[1], p[2] - 0.072)
    d.qpos[q + 3 : q + 7] = (1.0, 0.0, 0.0, 0.0)
    d.qvel[v : v + 6] = 0.0
    mujoco.mj_forward(m, d)
    grip.write({"position": 0.0})
    for _ in range(int(0.8 / DT)):
        eng.step()
    held = d.xpos[bid][2] - d.site_xpos[pinch][2]
    # Weight comes on: the part must not slide in the pads.
    m.opt.gravity[:] = (0.0, 0.0, -9.81)
    for _ in range(int(2.0 / DT)):
        eng.step()
    assert d.xpos[bid][2] - d.site_xpos[pinch][2] == pytest.approx(held, abs=2e-3)
    eng.shutdown()
