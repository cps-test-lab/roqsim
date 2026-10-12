"""An arm's entity names its model's root body, the body every consumer of the entity looks up."""

from __future__ import annotations

import mujoco
import pytest
from roqsim_manipulation_assets.models import MODELS_DIR

from roqsim.config import load_config_from_dict
from roqsim.engine import Engine

MODELS = sorted(
    p.name for p in MODELS_DIR.iterdir() if p.is_dir() and (p / f"{p.name}.xml").is_file()
)


def _engine(tmp_path, **spawn):
    world = {
        "sim": {},
        "components": [
            {"spawn_arm": {"prefix": "a_", "default_plugins": False, **spawn}, "name": "arm"}
        ],
    }
    return Engine(load_config_from_dict(world, base_dir=tmp_path))


@pytest.mark.parametrize("model", MODELS)
def test_the_entity_body_is_the_models_root_body(tmp_path, model):
    engine = _engine(tmp_path, model=model)
    engine.setup()
    m = engine.ctx.model
    body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, engine.ctx.entities.get("arm").body)
    assert body > 0, f"{model}: the entity names a body the compiled model does not have"
    assert m.body_parentid[body] == 0


def test_a_base_body_the_model_lacks_is_refused(tmp_path):
    with pytest.raises(RuntimeError, match=r"base_body 'base' is not a body of arm model 'xarm7'"):
        _engine(tmp_path, model="xarm7", base_body="base").setup()


def test_a_stated_base_body_is_registered(tmp_path):
    engine = _engine(tmp_path, model="ur10e", base_body="wrist_3_link")
    engine.setup()
    assert engine.ctx.entities.get("arm").body == "a_wrist_3_link"
