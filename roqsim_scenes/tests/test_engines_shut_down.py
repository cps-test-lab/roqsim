"""The scenes tools that build an engine shut it down when they fail after setup.

The ``dummy`` plugin counts its own shutdowns.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from roqsim_scenes.cli import scene_to_map, world_describe

from roqsim.engine import Engine

WORLD = "sim:\n  timestep: 0.005\ncomponents:\n  - dummy: {}\n    name: d0\n"


def _raise(*args, **kwargs):
    raise RuntimeError("injected after setup")


ENTRY_POINTS = {
    "scene-to-map": lambda w, out: scene_to_map.main(
        ["--world", w, "--out", str(out / "map"), "--scan-height", "0.1"]
    ),
    "describe": lambda w, out: world_describe.main([w, "--entities"]),
}


@pytest.mark.parametrize("name", sorted(ENTRY_POINTS))
def test_a_failure_after_setup_leaves_no_engine_running(name, monkeypatch, tmp_path: Path):
    seen: list[Engine] = []
    setup = Engine.setup

    def recording_setup(self):
        seen.append(self)
        setup(self)

    monkeypatch.setattr(Engine, "setup", recording_setup)
    monkeypatch.setattr(Engine, "reset", _raise)
    world = tmp_path / "w.yaml"
    world.write_text(WORLD)
    try:
        status = ENTRY_POINTS[name](str(world), tmp_path)
    except (Exception, SystemExit):
        pass
    else:
        assert status != 0, f"{name} succeeded, so nothing failed after its setup"
    assert seen, f"{name} set up no engine"
    for engine in seen:
        assert engine.ctx.blackboard.get("dummy_counts::d0")["shutdown"] == 1, name
