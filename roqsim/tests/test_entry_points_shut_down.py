"""Every roqsim tool that builds an engine shuts it down when it fails after setup.

The failure is injected after setup succeeded, so the plugins hold what ``configure`` opened. The
``dummy`` plugin counts its own shutdowns.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from roqsim import check, export_capture, export_gltf, export_moveit, export_web, render, runner
from roqsim.engine import Engine

WORLD = "sim:\n  timestep: 0.005\ncomponents:\n  - dummy: {}\n    name: d0\n"


@pytest.fixture
def engines(monkeypatch):
    """Every engine set up while the test runs."""
    seen: list[Engine] = []
    setup = Engine.setup

    def recording_setup(self):
        seen.append(self)
        setup(self)

    monkeypatch.setattr(Engine, "setup", recording_setup)
    return seen


def _raise(*args, **kwargs):
    raise RuntimeError("injected after setup")


def _recorded(world: str, out: Path) -> str:
    path = out / "run.npz"
    runner.run(world, headless=True, max_steps=2, pacing="asap", record=str(path))
    return str(path)


def _sim(world, out, monkeypatch):
    monkeypatch.setattr(Engine, "reset", _raise)
    return runner.main([world, "--headless", "--steps", "1", "--pacing", "asap"])


def _render(world, out, monkeypatch):
    monkeypatch.setattr(Engine, "reset", _raise)
    return render.main([world, "--out", str(out / "x.png")])


def _render_state(world, out, monkeypatch):
    state = _recorded(world, out)
    monkeypatch.setattr("roqsim.recording.Recording.describe", _raise)
    return render.main(["--state", state, "--check", "--out", str(out / "x.png")])


def _check(world, out, monkeypatch):
    monkeypatch.setattr(Engine, "reset", _raise)
    return check.main([world])


def _export_web(world, out, monkeypatch):
    monkeypatch.setattr(Engine, "reset", _raise)
    return export_web.main(["--world", world, "--out", str(out)])


def _export_gltf(world, out, monkeypatch):
    monkeypatch.setattr(Engine, "reset", _raise)
    return export_gltf.main(["--world", world, "--out", str(out / "x.glb")])


def _export_moveit(world, out, monkeypatch):
    # No injection: the dummy world has no arm, which the export refuses after setup.
    return export_moveit.main(["--world", world, "--out", str(out)])


def _export_capture(world, out, monkeypatch):
    state = _recorded(world, out)
    monkeypatch.setattr("roqsim.export_capture.write_capture", _raise)
    return export_capture.main(["--state", state, "--out", str(out / "capture")])


ENTRY_POINTS = {
    "sim": _sim,
    "render": _render,
    "render --state": _render_state,
    "check": _check,
    "export web": _export_web,
    "export gltf": _export_gltf,
    "export moveit": _export_moveit,
    "export capture": _export_capture,
}


@pytest.mark.parametrize("name", sorted(ENTRY_POINTS))
def test_a_failure_after_setup_leaves_no_engine_running(name, engines, monkeypatch, tmp_path: Path):
    world = tmp_path / "w.yaml"
    world.write_text(WORLD)
    try:
        status = ENTRY_POINTS[name](str(world), tmp_path, monkeypatch)
    except (Exception, SystemExit):
        pass
    else:
        assert status != 0, f"{name} succeeded, so nothing failed after its setup"
    assert engines, f"{name} set up no engine"
    for engine in engines:
        assert engine.ctx.blackboard.get("dummy_counts::d0")["shutdown"] == 1, name
