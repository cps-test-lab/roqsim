# Copyright (C) 2026 Frederik Pasch
#
# SPDX-License-Identifier: Apache-2.0

"""`roqsim scenes inputs`: which files a world is made of, for a caller that is not a roqsim
process.

A campaign runner staging a world into a container asks this and copies what comes back. Two
answers are therefore not interchangeable: what resolved, and everything. `--require-complete`
is which one was asked for -- without it a short list reads exactly like a whole one, and the
run fails later on a file that never travelled.
"""

from __future__ import annotations

import json

from roqsim_scenes.cli import world_inputs


def _world(tmp_path, body: str, name: str = "w.yaml"):
    path = tmp_path / name
    path.write_text(body)
    return path


def test_a_world_that_fully_resolves_is_listed(capsys, tmp_path):
    world = _world(tmp_path, "sim: {timestep: 0.01}\ncomponents: []\n")

    assert world_inputs.main([str(world)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["packaged"] is False
    assert str(world.resolve()) in payload["inputs"]


def test_the_same_world_is_complete_when_completeness_is_required(capsys, tmp_path):
    """The flag must not make an ordinary world fail: it reports a failure to *look*, and an
    absent optional file is not one."""
    world = _world(tmp_path, "sim: {timestep: 0.01}\ncomponents: []\n")

    assert world_inputs.main([str(world), "--require-complete"]) == 0
    assert json.loads(capsys.readouterr().out)["inputs"]


def test_an_unresolvable_parent_is_a_partial_answer_by_default(capsys, tmp_path):
    """Best-effort stays the default: a caller about to report its own error is not pre-empted."""
    world = _world(tmp_path, "extends: ./no_such_parent.yaml\ncomponents: []\n")

    assert world_inputs.main([str(world)]) == 0
    assert json.loads(capsys.readouterr().out)["inputs"] == [str(world.resolve())]


def test_an_unresolvable_parent_fails_when_completeness_is_required(capsys, tmp_path):
    """And it says which parent, because that is what the caller has to go and fix.

    The parent carries plugins, the MJCF the chain settles on and that MJCF's meshes. Staging
    the leaf alone puts none of them in the container, and every check that reads the workspace
    still passes.
    """
    world = _world(tmp_path, "extends: ./no_such_parent.yaml\ncomponents: []\n")

    assert world_inputs.main([str(world), "--require-complete"]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "no_such_parent.yaml" in captured.err
