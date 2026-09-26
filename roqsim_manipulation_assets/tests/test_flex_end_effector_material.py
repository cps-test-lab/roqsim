"""A flex in an arm's end effector takes its material from the world, like any other flex.

The end effector is attached inside ``spawn_arm``'s build, under the arm's prefix, so the
``flex_material`` component is declared after the arm: plugins build in YAML order. The
measured quantity is the beam's own bending, in the frame of the body it is pinned to, so the arm's
pose does not enter it.
"""

from __future__ import annotations

import numpy as np
import pytest

from roqsim.config import load_config_from_dict
from roqsim.engine import Engine

_TOOL = """
<mujoco>
  <worldbody>
    <body name="beam_holder">
      <flexcomp name="beam" type="grid" count="6 2 2" spacing=".02 .02 .02" dim="3" mass=".05"
                radius=".001">
        <elasticity young="1e5" poisson="0.3" damping="0.01"/>
        <contact contype="0" conaffinity="0" selfcollide="none"/>
        <pin gridrange="0 0 0 0 1 1"/>
      </flexcomp>
    </body>
  </worldbody>
</mujoco>
"""


def _bending(tmp_path, material: list) -> float:
    tool = tmp_path / "beam_tool.xml"
    tool.write_text(_TOOL, encoding="utf-8")
    cfg = load_config_from_dict(
        {
            "sim": {"timestep": 0.001},
            "components": [
                {
                    "spawn_arm": {
                        "model": "ur5e",
                        "prefix": "ur5e_",
                        "end_effector": {"model": str(tool)},
                    },
                    "name": "ur5e",
                },
                *material,
            ],
        },
        base_dir=tmp_path,
    )
    engine = Engine(cfg)
    engine.ctx.seed = 0
    try:
        engine.setup()
        engine.reset()
        model, data = engine.ctx.model, engine.ctx.data
        holder = model.body("ur5e_beam_holder").id

        def local() -> np.ndarray:
            rot = data.xmat[holder].reshape(3, 3)
            return (np.array(data.flexvert_xpos) - data.xpos[holder]) @ rot

        rest = local()
        for _ in range(1000):
            engine.step()
        return float(np.linalg.norm(local() - rest, axis=1).max())
    finally:
        engine.shutdown()


def test_young_reaches_a_flex_in_an_end_effector(tmp_path):
    nominal = _bending(tmp_path, [])
    stiffer = _bending(tmp_path, [{"flex_material": {"flex": "ur5e_beam", "young": 3.0e5}}])
    assert nominal > 0.002, f"the tool's beam should bend visibly, got {nominal * 1e3:.2f} mm"
    assert stiffer / nominal == pytest.approx(1 / 3, rel=0.15), (nominal, stiffer)
