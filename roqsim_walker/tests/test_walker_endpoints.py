"""The walker's ``body_poses``: a ``Transforms`` of its bones, owned by the walker it registers."""

from __future__ import annotations

from pathlib import Path

from roqsim.config import load_config_from_dict
from roqsim.engine import Engine
from roqsim.introspection import get_plugin_details
from roqsim.types import Transforms


def test_body_poses_carries_each_bone_and_belongs_to_the_walker():
    (row,) = get_plugin_details("walker")["endpoints"]
    assert row["name"] == "body_poses" and row["rate_hz"] == 30.0
    assert row["payload"] == "Transforms"

    world = {
        "sim": {},
        "components": [
            {"walker": {"walker": "MaleVisitorWalk", "namespace": "peds"}, "name": "ped"}
        ],
    }
    engine = Engine(load_config_from_dict(world, base_dir=Path(".")))
    engine.setup()
    engine.reset()
    try:
        ep = engine.ctx.interface.find("ped", "body_poses")
        assert ep.namespace == "peds"
        poses = ep.read()
        assert isinstance(poses, Transforms) and len(poses.transforms) == 17
        assert all(t.parent == "" and t.child.startswith("ped/") for t in poses.transforms)
    finally:
        engine.shutdown()
