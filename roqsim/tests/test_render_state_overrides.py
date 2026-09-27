# Copyright (C) 2026 Frederik Pasch
#
# SPDX-License-Identifier: Apache-2.0

"""``roqsim render --state`` refuses an override it cannot apply, rather than ignoring it.

A recording rebuilds its world from its own provenance, so only ``sim.view`` -- the camera -- can
be changed over it. Anything else would be accepted and have no effect on the picture.
"""

from __future__ import annotations

import contextlib

import pytest

from roqsim import recording, render


def test_a_component_override_with_state_is_refused_by_name(tmp_path):
    with pytest.raises(render.RenderError, match=r"components\.obstacle\.instances, sim\.pacing"):
        render.render_target(
            None,
            str(tmp_path / "x.png"),
            state=str(tmp_path / "run.npz"),
            overrides={"sim": {"pacing": "asap"}, "components": {"obstacle": {"instances": []}}},
        )


def test_the_command_exits_2_naming_the_ignored_key(tmp_path, capsys):
    argv = ["--state", str(tmp_path / "run.npz"), "--set", "components.table.pose.position.x=1"]
    assert render.main(argv) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "components.table.pose.position.x" in captured.err and "sim.view" in captured.err


def test_a_view_override_with_state_is_not_refused(monkeypatch, tmp_path):
    """sim.view is the one key a recording's render applies; it goes on to the recording."""
    seen = {}

    def fake(state, target, out, size, merged, *rest, **kwargs):
        seen["merged"] = merged
        return {"rendered": False}

    # render_target opens the recording itself, so the file it names need not exist here.
    monkeypatch.setattr(recording, "open_recording", contextlib.nullcontext)
    monkeypatch.setattr(render, "_render_recording", fake)
    render.render_target(
        None,
        str(tmp_path / "x.png"),
        state=str(tmp_path / "run.npz"),
        overrides={"sim": {"view": {"azimuth": 90}}},
    )
    assert seen["merged"] == {"sim": {"view": {"azimuth": 90}}}
