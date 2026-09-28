"""The walker's ``body_poses``: declared on its method, owned by the walker it registers."""

from __future__ import annotations

from pathlib import Path

from roqsim.config import load_config_from_dict
from roqsim.engine import Engine
from roqsim.introspection import get_plugin_details


def test_body_poses_describes_each_bone_and_belongs_to_the_walker():
    (row,) = get_plugin_details("walker")["endpoints"]
    assert row["name"] == "body_poses" and row["rate_hz"] == 30.0
    (bone,) = row["result"]["items"]
    assert [i.get("unit") for i in bone["items"]] == [None, "m", None]

    world = {
        "sim": {},
        "components": [
            {"walker": {"walker": "MaleVisitorWalk", "namespace": "peds"}, "name": "ped"}
        ],
    }
    engine = Engine(load_config_from_dict(world, base_dir=Path(".")))
    engine.setup()
    try:
        ep = engine.ctx.interface.find("ped", "body_poses")
        assert ep.namespace == "peds" and len(ep.read()) == 17
    finally:
        engine.shutdown()
