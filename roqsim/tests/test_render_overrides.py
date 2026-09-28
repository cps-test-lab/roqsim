# Copyright (C) 2026 Frederik Pasch
#
# SPDX-License-Identifier: Apache-2.0

"""``roqsim render --set/--override``: ``roqsim sim``'s options, so the picture is the run's world."""

from __future__ import annotations

from roqsim import render


def _capture(monkeypatch) -> dict:
    seen: dict = {}

    def fake(*args, **kwargs):
        seen.update(kwargs)
        return {"path": "/tmp/x.png", "rendered": True}

    monkeypatch.setattr(render, "render_target", fake)
    return seen


def test_an_override_file_reaches_the_render_under_set(monkeypatch, tmp_path):
    """Files first, --set over them -- the merge `roqsim sim` applies."""
    overrides = tmp_path / "run.overrides.yaml"
    overrides.write_text(
        "sim: {pacing: realtime}\ncomponents:\n  obstacle:\n    instances: []\n", encoding="utf-8"
    )
    seen = _capture(monkeypatch)
    argv = ["w.yaml", "--override", str(overrides), "--set", "sim.pacing=asap"]
    assert render.main(argv) == 0
    assert seen["overrides"] == {
        "sim": {"pacing": "asap"},
        "components": {"obstacle": {"instances": []}},
    }


def test_an_override_file_that_cannot_be_read_is_bad_input(monkeypatch, tmp_path, capsys):
    _capture(monkeypatch)
    assert render.main(["w.yaml", "--override", str(tmp_path / "missing.yaml")]) == 2
    captured = capsys.readouterr()
    assert captured.out == "" and "missing.yaml" in captured.err
