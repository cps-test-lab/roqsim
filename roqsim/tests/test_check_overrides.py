# Copyright (C) 2026 Frederik Pasch
#
# SPDX-License-Identifier: Apache-2.0

"""``roqsim check --set/--override``: the world checked is the one ``roqsim sim`` would build.

A campaign's overrides add obstacles and change plugin config. A check of the base file alone passes
a world the run never loads, and misses what the overrides break -- an address the world does not
have, an obstacle placed into a table.
"""

from __future__ import annotations

import json
import textwrap

import pytest

from roqsim.check import check_world, main

TABLE = """
<mujoco><worldbody><body name="table">
  <geom name="table_top" type="box" size=".4 .4 .2" pos="0 0 .2"/>
</body></worldbody></mujoco>
"""

# A table, and an obstacle population that is empty until a campaign's overrides fill it.
WORLD = """
sim: {}
components:
  - spawn_model: {model: table.xml, motion: static}
    name: table
  - boxes: {instances: []}
    name: obstacle
"""


def _write(tmp_path, name: str, body: str):
    path = tmp_path / name
    path.write_text(textwrap.dedent(body).lstrip(), encoding="utf-8")
    return path


@pytest.fixture
def world(tmp_path):
    pytest.importorskip("roqsim_assets", reason="the boxes plugin lives in roqsim_assets")
    _write(tmp_path, "table.xml", TABLE)
    return _write(tmp_path, "world.yaml", WORLD)


def _obstacle_file(tmp_path, x: float):
    return _write(
        tmp_path,
        "run.overrides.yaml",
        f"""
        components:
          obstacle:
            instances:
            - {{pose: {{position: {{x: {x}, y: 0.0}}}}, size: [0.2, 0.2, 0.2]}}
        """,
    )


def _check(capsys, *argv) -> tuple[int, dict]:
    code = main([*map(str, argv), "--json"])
    return code, json.loads(capsys.readouterr().out)


def test_an_obstacle_an_override_places_into_the_table_is_reported(capsys, tmp_path, world):
    code, plain = _check(capsys, world)
    assert code == 0 and plain["warnings"] == [], "the base world has nothing in the table"
    assert plain["overrides"] == {}

    overrides = _obstacle_file(tmp_path, 0.0)
    code, report = _check(capsys, world, "--override", overrides)
    assert code == 0, "an overlap is a warning, as it is without overrides"
    (warning,) = report["warnings"]
    assert warning["check"] == "interpenetration"
    assert "'table_top' (entity 'table')" in warning["message"]
    assert "(entity 'obstacle_0')" in warning["message"]
    # The overrides the check applied are in its report, and the world it reports on is theirs.
    assert report["overrides"] == {
        "components": {
            "obstacle": {
                "instances": [{"pose": {"position": {"x": 0.0, "y": 0.0}}, "size": [0.2, 0.2, 0.2]}]
            }
        }
    }
    assert "obstacle_0" in {e["name"] for e in report["world"]["entities"]}


def test_set_wins_over_the_file_as_it_does_for_roqsim_sim(capsys, tmp_path, world):
    """The file places the obstacle clear of the table; the --set moves it back in."""
    overrides = _obstacle_file(tmp_path, 2.0)
    code, clear = _check(capsys, world, "--override", overrides)
    assert code == 0 and clear["warnings"] == []

    moved = (
        "components.obstacle.instances="
        "[{pose: {position: {x: 0.0, y: 0.0}}, size: [0.2, 0.2, 0.2]}]"
    )
    code, report = _check(capsys, world, "--override", overrides, "--set", moved)
    assert [w["check"] for w in report["warnings"]] == ["interpenetration"]
    assert report["overrides"]["components"]["obstacle"]["instances"][0]["pose"] == {
        "position": {"x": 0.0, "y": 0.0}
    }


def test_an_override_addressing_nothing_is_a_config_problem(capsys, world):
    """Refused as a run refuses it -- not silently dropped, which would check the base world."""
    code, report = _check(capsys, world, "--set", "components.obstacel.instances=[]")
    assert code == 1
    assert report["ok"] is False and report["reached"] == "resolve"
    assert [p["stage"] for p in report["problems"]] == ["config"]
    assert "obstacel" in report["problems"][0]["message"]
    assert report["overrides"] == {"components": {"obstacel": {"instances": []}}}


def test_an_override_giving_a_plugin_a_bad_value_is_a_config_problem(tmp_path, world):
    report = check_world(
        str(world), {"components": {"obstacle": {"instances": [{"size": [0.2, -1, 0.2]}]}}}
    )
    assert report["ok"] is False
    assert [p["stage"] for p in report["problems"]] == ["config"]
    assert "'size' must be positive" in report["problems"][0]["message"]


def test_the_text_report_names_the_overrides(capsys, world):
    assert main([str(world), "--set", "sim.timestep=0.001"]) == 0
    out = capsys.readouterr().out
    assert 'overrides: {"sim": {"timestep": 0.001}}' in out
    assert "timestep 0.001s" in out, "and the world it describes is the overridden one"


def test_an_override_file_that_cannot_be_read_is_not_a_verdict(capsys, tmp_path, world):
    """Exit 2 and a line on stderr: 0 and 1 mean a report is on stdout."""
    assert main([str(world), "--override", str(tmp_path / "missing.yaml"), "--json"]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "missing.yaml" in captured.err
