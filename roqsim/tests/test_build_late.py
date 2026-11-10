"""``build_late`` runs after every plugin's ``build``, so a plugin can link to a model listed after it."""

from __future__ import annotations

import mujoco

from roqsim.config import load_config_from_dict
from roqsim.engine import Engine
from roqsim.plugin import Plugin

CALLS: list[str] = []


class Linker(Plugin):
    """Welds its own body to the body named ``to``, which another plugin builds."""

    def validate_config(self, config):
        return []

    def build(self, spec, ctx):
        CALLS.append(f"build {self.address}")
        body = spec.worldbody.add_body(name=f"{self.address}/hand", pos=[0, 0, 1], mocap=True)
        body.add_site(name=f"{self.address}/grip")

    def build_late(self, spec, ctx):
        CALLS.append(f"build_late {self.address}")
        eq = spec.add_equality()
        eq.type = mujoco.mjtEq.mjEQ_WELD
        eq.objtype = mujoco.mjtObj.mjOBJ_SITE
        eq.name1 = f"{self.address}/grip"
        eq.name2 = f"{self.config['to']}/grip"
        eq.active = False


class Thing(Plugin):
    def validate_config(self, config):
        return []

    def build(self, spec, ctx):
        CALLS.append(f"build {self.address}")
        body = spec.worldbody.add_body(name=f"{self.address}/body", pos=[1, 0, 0.5])
        body.add_freejoint()
        body.add_geom(size=[0.1, 0, 0])
        body.add_site(name=f"{self.address}/grip")


def test_build_late_sees_a_model_listed_after_it():
    CALLS.clear()
    cfg = load_config_from_dict(
        {
            "sim": {"pacing": "asap"},
            "components": [
                {"test_build_late:Linker": {"to": "thing"}, "name": "hand"},
                {"test_build_late:Thing": {}, "name": "thing"},
            ],
        }
    )
    engine = Engine(cfg)
    engine.ctx.seed = 1
    engine.setup()
    try:
        assert CALLS == ["build hand", "build thing", "build_late hand"]
        m = engine.ctx.model
        assert m.neq == 1 and m.eq_type[0] == mujoco.mjtEq.mjEQ_WELD
        assert not m.eq_active0[0]
    finally:
        engine.shutdown()
