"""A window that cannot open says why on stderr and exits 2 -- no display, or a scene that does not
load -- so the MCP tool, which relays the subprocess's stderr, hands the caller the reason."""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("MUJOCO_GL", "egl")


@pytest.fixture
def _no_display(monkeypatch):
    monkeypatch.setattr("roqsim.viewer.has_display", lambda: False)
    monkeypatch.delenv("DISPLAY", raising=False)


def test_the_review_window_names_the_missing_display_on_stderr(_no_display, capsys):
    from roqsim_scene_builder.scene_window import run_window

    assert run_window("roqsim_scenes:depot") == 2
    out, err = capsys.readouterr()
    assert out == "" and "no DISPLAY" in err


def test_the_sketch_window_names_the_missing_display_on_stderr(_no_display, capsys):
    from roqsim_scene_builder.floorplan_window import run_window

    assert run_window() == 2
    out, err = capsys.readouterr()
    assert out == "" and "no DISPLAY" in err


def test_a_scene_that_does_not_load_is_exit_2_with_a_sentence(monkeypatch, capsys, tmp_path):
    from roqsim_scene_builder.scene_window import run_window

    pytest.importorskip("tkinter")
    monkeypatch.setattr("roqsim.viewer.has_display", lambda: True)
    missing = tmp_path / "missing.yaml"
    assert run_window(str(missing)) == 2
    out, err = capsys.readouterr()
    assert out == ""
    assert "cannot load" in err and str(missing) in err and "Traceback" not in err
