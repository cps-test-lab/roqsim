"""A model whose contact pairs name the world's ``floor`` refuses a world without one.

MuJoCo compiles the robot without a ``<pair>`` whose geom does not exist, and says nothing: the
wheels then run on default contact parameters, a different robot from the one the model describes.
"""

from __future__ import annotations

# `roqsim` selects MuJoCo's GL backend on import, so it comes first (see test_wheels_roll.py).
import roqsim  # noqa: F401, I001
from pathlib import Path  # noqa: E402

import pytest  # noqa: E402

from roqsim.config import load_config_from_dict  # noqa: E402
from roqsim.engine import Engine  # noqa: E402

_GROUND = (
    '<mujoco><worldbody><geom name="{name}" type="plane" size="10 10 .1"/></worldbody></mujoco>'
)


def _setup(tmp_path, ground):
    world = tmp_path / "ground.xml"
    world.write_text(_GROUND.format(name=ground))
    cfg = load_config_from_dict(
        {
            "sim": {"world": str(world)},
            "components": [{"spawn_robot": {"model": "rosbot"}, "name": "r"}],
        },
        base_dir=Path(tmp_path),
    )
    engine = Engine(cfg, preview=True)
    engine.setup()
    return engine


def test_a_world_whose_ground_has_another_name_is_refused_naming_it(tmp_path):
    with pytest.raises(RuntimeError, match="'floor'"):
        _setup(tmp_path, "ground")


def test_a_ground_called_floor_keeps_the_models_pairs(tmp_path):
    engine = _setup(tmp_path, "floor")
    try:
        assert engine.ctx.model.npair > 0
    finally:
        engine.shutdown()
