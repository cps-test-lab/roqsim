# Copyright (C) 2026 Frederik Pasch
#
# SPDX-License-Identifier: Apache-2.0
"""A world condition a plugin raises at build reaches every record of a result: the compiled model,
a recording's provenance, ``roqsim render``'s JSON record, ``roqsim check``'s warnings and the run's
output directory (``roqsim.conditions``)."""

from __future__ import annotations

import json
import textwrap

import pytest

from roqsim import conditions
from roqsim.config import load_config

PLUGIN = '''
from roqsim.conditions import raise_condition
from roqsim.plugin import Plugin


class Weaker(Plugin):
    def build_late(self, spec, ctx):
        raise_condition(ctx, source=self.address, code="drawn_weaker", message="drawn weaker",
                        fix="roqsim fix-it world.yaml", details={"why": "a test"})
'''


@pytest.fixture
def world(tmp_path):
    (tmp_path / "weaker.py").write_text(PLUGIN)
    path = tmp_path / "world.yaml"
    path.write_text(
        textwrap.dedent(
            """
            sim: {world: empty_room}
            components:
              - "weaker.py:Weaker": {}
                name: weaker
            """
        ).lstrip()
    )
    return path


def _engine(world):
    from roqsim.engine import Engine

    engine = Engine(load_config(str(world)), preview=True)
    engine.setup()
    return engine


def test_a_condition_is_kept_on_the_context_and_in_the_model(world):
    engine = _engine(world)
    try:
        (c,) = engine.ctx.conditions
        assert c["code"] == "drawn_weaker" and c["level"] == "warning"
        assert c["fix"] == "roqsim fix-it world.yaml" and c["details"] == {"why": "a test"}
        assert conditions.of_model(engine.ctx.model) == [c]
    finally:
        engine.shutdown()


def test_a_world_without_one_carries_none():
    from roqsim.config import load_config_from_dict
    from roqsim.engine import Engine

    engine = Engine(load_config_from_dict({"sim": {"world": "empty_room"}}), preview=True)
    engine.setup()
    try:
        assert engine.ctx.conditions == [] and conditions.of_model(engine.ctx.model) == []
    finally:
        engine.shutdown()


def test_the_recording_provenance_carries_it(world, tmp_path):
    from roqsim.capture import StateRecorder, snap_fps

    engine = _engine(world)
    try:
        recorder = StateRecorder(
            engine.ctx, tmp_path / "run.mcap", snap_fps(30, 0.002), config=engine.config, world="w"
        )
        assert [c["code"] for c in recorder._provenance["conditions"]] == ["drawn_weaker"]
    finally:
        engine.shutdown()


def test_roqsim_check_reports_it_as_a_warning(world):
    from roqsim.check import check_world

    report = check_world(str(world))
    (warning,) = [w for w in report["warnings"] if w["check"] == "condition"]
    assert warning["code"] == "drawn_weaker" and warning["hint"] == "roqsim fix-it world.yaml"
    assert "drawn weaker" in warning["message"]


def test_roqsim_renders_record_carries_it(world, tmp_path):
    from roqsim.render import render_target

    record = render_target(str(world), tmp_path / "out.png", check=True)
    assert [c["code"] for c in record["conditions"]] == ["drawn_weaker"]


def test_the_run_file_lands_in_the_runs_output_dir(tmp_path, monkeypatch):
    monkeypatch.delenv("OUTPUT_DIR", raising=False)
    monkeypatch.setenv("RUN_OUTPUT_DIR", str(tmp_path / "run"))
    c = conditions.make(source="x", code="y", message="z")
    path = conditions.write_run_file([c])
    assert path == tmp_path / "run" / conditions.RUN_FILE
    assert json.loads(path.read_text()) == {"conditions": [c]}


def test_no_output_dir_no_file(monkeypatch):
    for var in conditions.OUTPUT_DIR_VARS:
        monkeypatch.delenv(var, raising=False)
    assert conditions.write_run_file([]) is None


def test_roqsim_sim_writes_the_run_file(world, tmp_path, monkeypatch):
    from roqsim.runner import run

    monkeypatch.delenv("OUTPUT_DIR", raising=False)
    monkeypatch.setenv("RUN_OUTPUT_DIR", str(tmp_path / "run"))
    run(str(world), headless=True, max_steps=2, seed=1)
    data = json.loads((tmp_path / "run" / conditions.RUN_FILE).read_text())
    assert [c["code"] for c in data["conditions"]] == ["drawn_weaker"]


def test_a_condition_must_be_plain_data():
    with pytest.raises(TypeError):
        conditions.make(source="x", code="y", message="z", details={"obj": object()})
    with pytest.raises(ValueError):
        conditions.make(source="x", code="y", message="z", level="fatal")
