# Copyright (C) 2026 Frederik Pasch
#
# SPDX-License-Identifier: Apache-2.0

"""``roqsim check``: every problem at once, and the right one blamed.

Each stage gets a world that fails only there. The one that matters most is the last: a plugin whose
config is perfectly valid and whose model compiles, but which cannot find the site it mounts on --
the failure a syntax check cannot see and the one that otherwise shows up as a dead run.
"""

from __future__ import annotations

import json
import textwrap

import pytest

from roqsim.check import STAGES, check_world, main


def _world(tmp_path, body: str, name: str = "world.yaml"):
    path = tmp_path / name
    path.write_text(textwrap.dedent(body).lstrip(), encoding="utf-8")
    return path


GOOD = """
    # One robot in the default room: the smallest world with entities and endpoints in it.
    sim: {}
    components:
      - spawn_robot: {model: turtlebot4}
        name: robot
"""


def test_a_working_world_reaches_the_last_stage_and_says_what_it_is(tmp_path):
    pytest.importorskip("roqsim_mobile", reason="the turtlebot4 model lives in roqsim_mobile")
    report = check_world(str(_world(tmp_path, GOOD)))
    assert report["ok"] is True and report["problems"] == []
    assert report["reached"] == STAGES[-1]

    world = report["world"]
    assert world["model"]["nbody"] > 1
    assert world["integrator"] == "implicitfast"  # the spelling a world writes, not the enum's
    assert {e["name"] for e in world["entities"]} == {"robot"}
    # The inventory is the half that is not about failure: what to write the next thing against.
    assert {"cmd_vel", "odom", "scan"} <= {e["topic"] for e in world["endpoints"]}
    assert all(c["address"] for c in world["components"])
    assert report["inputs"], "the files this world is defined by"


def test_an_unresolvable_target_is_reported_before_anything_is_loaded():
    report = check_world("nope_xyz:missing")
    assert report["ok"] is False
    assert report["reached"] is None
    assert report["problems"][0]["stage"] == "resolve"
    assert "roqsim catalog worlds" in report["problems"][0]["hint"]


def test_a_bad_key_is_a_config_problem_and_every_bad_key_is_reported(tmp_path):
    """The aggregation is the point: a world with three mistakes should need one run to find them."""
    pytest.importorskip("roqsim_sensors")
    report = check_world(
        str(
            _world(
                tmp_path,
                """
                sim: {}
                components:
                  - lidar: {rays: -5, max_range: -1, rate_hz: 0}
                """,
            )
        )
    )
    assert report["reached"] == "resolve"
    assert [p["stage"] for p in report["problems"]] == ["config"]
    message = report["problems"][0]["message"]
    assert "'rays' must be > 0" in message
    assert "'max_range' must be > 0" in message
    assert "'rate_hz' must be > 0" in message


def test_a_name_the_compiled_model_does_not_have_is_a_configure_problem(tmp_path):
    """Valid config, a model that compiles, and a mount that does not exist -- the dead-run case."""
    pytest.importorskip("roqsim_sensors")
    report = check_world(
        str(_world(tmp_path, "sim: {}\ncomponents:\n  - lidar: {site: nowhere}\n"))
    )
    assert report["reached"] == "config", "the config was fine; the model is what refused"
    assert report["problems"][0]["stage"] == "configure"
    assert "nowhere" in report["problems"][0]["message"]
    assert "catalog model" in report["problems"][0]["hint"]


def test_an_unknown_plugin_is_named_rather_than_traced(tmp_path):
    report = check_world(str(_world(tmp_path, "sim: {}\ncomponents:\n  - not_a_plugin_xyz: {}\n")))
    assert report["ok"] is False
    assert report["problems"][0]["stage"] == "config"
    assert "not_a_plugin_xyz" in report["problems"][0]["message"]


def test_a_package_ref_is_accepted_the_way_roqsim_sim_takes_one():
    pytest.importorskip("roqsim_mobile")
    report = check_world("roqsim_mobile:turtlebot4_demo")
    assert report["ok"] is True
    assert {e["name"] for e in report["world"]["entities"]} == {"robot"}
    topics = {e["topic"] for e in report["world"]["endpoints"]}
    assert {"cmd_vel", "odom", "scan"} <= topics


# -- the command ------------------------------------------------------------------------------


def test_the_exit_code_is_the_verdict(tmp_path, capsys):
    pytest.importorskip("roqsim_sensors")
    assert main([str(_world(tmp_path, GOOD))]) == 0
    assert "ok" in capsys.readouterr().out
    assert main(["nope_xyz:missing"]) == 1
    assert "FAIL" in capsys.readouterr().out


def test_json_is_the_same_report(tmp_path, capsys):
    pytest.importorskip("roqsim_sensors")
    assert main([str(_world(tmp_path, GOOD)), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True and payload["reached"] == "configure"


def test_the_text_report_names_the_stages_it_did_not_reach(tmp_path, capsys):
    main(["nope_xyz:missing"])
    out = capsys.readouterr().out
    assert "reached: nothing" in out
    assert " -> ".join(STAGES) in out
