"""A weld's pose written at run time survives ``mj_setConst``."""

from __future__ import annotations

import mujoco
import numpy as np

from roqsim.weld import set_relpose

MJCF = """<mujoco><worldbody>
  <body name="hand" pos="0 0 1"><freejoint/><geom size=".05"/></body>
  <body name="box" pos="1 0 0"><freejoint/><geom type="box" size=".1 .1 .1"/></body>
</worldbody></mujoco>"""


def _welded():
    spec = mujoco.MjSpec.from_string(MJCF)
    eq = spec.add_equality()
    eq.type = mujoco.mjtEq.mjEQ_WELD
    eq.objtype = mujoco.mjtObj.mjOBJ_BODY
    eq.name1, eq.name2 = "box", "hand"
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def test_mj_set_const_recomputes_a_pose_written_into_the_row():
    """Why the helper exists: MuJoCo moves a weld compiled without a relpose back to qpos0's pose."""
    model, data = _welded()
    model.eq_data[0][3:6] = [0.1, 0.2, 0.3]
    mujoco.mj_setConst(model, data)
    np.testing.assert_allclose(model.eq_data[0][3:6], [-1.0, 1.0, 1.0])


def test_set_relpose_survives_mj_set_const():
    model, data = _welded()
    set_relpose(model, 0, [0.0, 0.0, 0.01], [0.1, 0.2, 0.3], [0.0, 1.0, 0.0, 0.0])
    mujoco.mj_setConst(model, data)
    np.testing.assert_allclose(model.eq_data[0][0:10], [0, 0, 0.01, 0.1, 0.2, 0.3, 0, 1, 0, 0])
