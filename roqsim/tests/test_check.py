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
    # The robot, and the scanner its manifest mounts: a mounted device is an entity of its own.
    assert {e["name"] for e in world["entities"]} == {"robot", "robot.rplidar"}
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
    assert {e["name"] for e in report["world"]["entities"]} == {"robot", "robot.rplidar"}
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
    assert payload["ok"] is True and payload["reached"] == "reset"
    assert payload["warnings"] == []


def test_the_text_report_names_the_stages_it_did_not_reach(tmp_path, capsys):
    main(["nope_xyz:missing"])
    out = capsys.readouterr().out
    assert "reached: nothing" in out
    assert " -> ".join(STAGES) in out


# -- `actuators:`: which stage a bad block is blamed on ------------------------------------------
#
# The stage is a contract, not a detail. An external validator checks a world before spending compute
# on it by compiling one -- `roqsim scenes describe --entities` is that call -- so a build-stage
# refusal reaches its author rather than a trial, and a config-stage one reaches them without
# compiling at all. Blaming the wrong stage sends a reader to the wrong file, so each is asserted.

ARM = """
    sim: {{}}
    components:
      - spawn_arm: {{model: ur5e, actuators: {block}}}
        name: arm
"""


def _arm_world(tmp_path, block: str, name: str):
    return _world(tmp_path, ARM.format(block=block), name=name)


def test_a_bad_gain_key_is_a_config_problem(tmp_path):
    """A key typo must not blame the model -- nothing has been compiled when it is found."""
    pytest.importorskip("roqsim_manipulation", reason="spawn_arm lives in roqsim_manipulation")
    report = check_world(str(_arm_world(tmp_path, "{control: impedance, kp: 2.0}", "key.yaml")))
    assert report["ok"] is False
    stages = {problem["stage"] for problem in report["problems"]}
    assert stages == {"config"}
    assert "'p'" in report["problems"][0]["message"]


def test_an_unknown_actuator_is_a_build_problem(tmp_path):
    """Whether the model HAS the actuator needs the model, so it is found where the model is read."""
    pytest.importorskip("roqsim_manipulation", reason="spawn_arm lives in roqsim_manipulation")
    report = check_world(
        str(_arm_world(tmp_path, "{each: {no_such_actuator: {p: 1.0}}}", "name.yaml"))
    )
    assert report["ok"] is False
    assert [problem["stage"] for problem in report["problems"]] == ["build"]
    assert "roqsim catalog model ur5e" in report["problems"][0]["message"]


def test_a_joint_name_is_a_build_problem_naming_its_actuator(tmp_path):
    pytest.importorskip("roqsim_manipulation", reason="spawn_arm lives in roqsim_manipulation")
    report = check_world(
        str(_arm_world(tmp_path, "{each: {wrist_3_joint: {p: 1.0}}}", "joint.yaml"))
    )
    assert [problem["stage"] for problem in report["problems"]] == ["build"]
    assert "'wrist_3'" in report["problems"][0]["message"]


def test_a_world_that_declares_gains_the_model_accepts_is_ok(tmp_path):
    pytest.importorskip("roqsim_manipulation", reason="spawn_arm lives in roqsim_manipulation")
    report = check_world(
        str(_arm_world(tmp_path, "{control: impedance, stiffness: 2.0, damping: 0.02}", "ok.yaml"))
    )
    assert report["ok"] is True and report["problems"] == []


# -- reset: the state a trial starts from ---------------------------------------------------------
#
# A world whose reset state puts one body inside another loads, compiles and resolves, and then
# the contact solver flings the bodies apart on the first steps. `check` resets the world and names
# the pair: a warning, since the world does start and an overlap can be deliberate.

# A solid block rather than a thin top: a capsule that passes all the way through a thin box gets
# no contact from MuJoCo at all, so the solver never acts on it and there is nothing to report.
TABLE = """
<mujoco><worldbody><body name="table">
  <geom name="table_top" type="box" size=".4 .4 .2" pos="0 0 .2"/>
</body></worldbody></mujoco>
"""
CRATE = """
<mujoco><worldbody><body name="crate">
  <geom name="crate" type="box" size=".05 .05 .05"/>
</body></worldbody></mujoco>
"""


def _props_world(tmp_path, crate_z: float):
    (tmp_path / "table.xml").write_text(TABLE, encoding="utf-8")
    (tmp_path / "crate.xml").write_text(CRATE, encoding="utf-8")
    return _world(
        tmp_path,
        f"""
        sim: {{}}
        components:
          - spawn_model: {{model: table.xml, motion: static}}
            name: table
          - spawn_model: {{model: crate.xml, pose: {{position: {{x: 0, y: 0, z: {crate_z}}}}}}}
            name: crate
        """,
    )


def test_a_box_placed_into_a_table_is_a_warning_naming_both_and_the_depth(tmp_path):
    report = check_world(str(_props_world(tmp_path, 0.42)))
    assert report["ok"] is True, "the world starts; the overlap is a warning, not a problem"
    assert report["reached"] == "reset"
    (warning,) = report["warnings"]
    assert set(warning) == {"check", "message", "hint"}
    assert warning["check"] == "interpenetration"
    assert "'table_top' (entity 'table')" in warning["message"]
    assert "'crate' (entity 'crate')" in warning["message"]
    assert "30.0 mm" in warning["message"]
    assert "spawn pose" in warning["hint"]


def test_a_box_resting_on_a_table_is_not_a_warning(tmp_path):
    report = check_world(str(_props_world(tmp_path, 0.45)))
    assert report["ok"] is True and report["warnings"] == []


def test_an_arm_home_that_buries_the_arm_in_the_table_is_named(tmp_path, capsys):
    """The arm stands on the table; its `home` folds the upper arm down through the top."""
    pytest.importorskip("roqsim_manipulation_assets", reason="the ur5e model lives there")
    (tmp_path / "table.xml").write_text(TABLE, encoding="utf-8")
    world = _world(
        tmp_path,
        """
        sim: {}
        components:
          - spawn_model: {model: table.xml, motion: static}
            name: table
          - spawn_arm: {model: ur5e, pos: [0, 0, 0.4], home: [0, 0.9, 0, 0, 0, 0]}
            name: arm
        """,
    )
    report = check_world(str(world))
    assert report["ok"] is True and report["warnings"]
    messages = [w["message"] for w in report["warnings"]]
    into_table = [m for m in messages if "geom 'table_top' (entity 'table')" in m]
    assert into_table, "the table is one side"
    assert all("(entity 'arm')" in m and " mm at reset" in m for m in into_table)
    assert all("(entity 'arm')" in m for m in messages), "every overlap here is the arm's"
    assert "`home` (spawn_arm)" in report["warnings"][0]["hint"]

    assert main([str(world)]) == 0, "a warning does not fail the check"
    out = capsys.readouterr().out
    assert "WARN  [interpenetration]" in out and "table_top" in out


def test_a_plugin_whose_reset_fails_is_a_reset_problem(tmp_path):
    (tmp_path / "boom.py").write_text(
        textwrap.dedent(
            """
            from roqsim.plugin import Plugin

            class Boom(Plugin):
                def on_reset(self, ctx):
                    raise RuntimeError("cannot re-home")
            """
        ),
        encoding="utf-8",
    )
    report = check_world(str(_world(tmp_path, "sim: {}\ncomponents:\n  - boom.py:Boom: {}\n")))
    assert report["ok"] is False
    assert report["reached"] == "configure"
    assert [p["stage"] for p in report["problems"]] == ["reset"]
    assert "cannot re-home" in report["problems"][0]["message"]
