# Copyright (C) 2026 Frederik Pasch
#
# SPDX-License-Identifier: Apache-2.0

"""A plugin reports the values it runs with where they bound a run, before the run.

A drive clips a command at its cap and a diff drive's odometry carries no error unless the world
gives it one; ``roqsim check`` and ``roqsim scenes describe`` list both, with whether the world or
the plugin's default set each value.
"""

from __future__ import annotations

import textwrap

import pytest

from roqsim.check import check_world
from roqsim.plugin import Plugin


def _world(tmp_path, body: str):
    path = tmp_path / "world.yaml"
    path.write_text(textwrap.dedent(body).lstrip(), encoding="utf-8")
    return str(path)


def test_a_drive_reports_its_caps_and_exact_odometry(tmp_path):
    pytest.importorskip("roqsim_mobile", reason="the turtlebot4 model lives in roqsim_mobile")
    report = check_world(
        _world(
            tmp_path,
            """
        sim: {}
        components:
          - spawn_robot: {model: turtlebot4}
            name: robot
    """,
        )
    )
    drive = next(r for r in report["world"]["realised"] if r["plugin"] == "DiffDrivePlugin")
    values = {v["key"]: v for v in drive["values"]}
    assert values["max_linear_vel"]["unit"] == "m/s"
    assert values["max_linear_vel"]["value"] > 0
    assert "clipped" in values["max_linear_vel"]["effect"]
    assert values["odom_noise"]["value"] is None and values["odom_noise"]["source"] == "default"


def test_a_value_the_world_sets_is_reported_as_set_by_it():
    class Capped(Plugin):
        REALISED = (("max_speed", "cap", "m/s", "a faster command is clipped"),)

        def __init__(self, config=None, **kwargs):
            super().__init__(config, **kwargs)
            self.cap = float(self.config.get("max_speed", 1.0))

    assert Capped({"max_speed": 0.3}, name="c").realised()[0] == {
        "key": "max_speed",
        "value": 0.3,
        "unit": "m/s",
        "source": "config",
        "effect": "a faster command is clipped",
    }
    assert Capped({}, name="c").realised()[0]["source"] == "default"


def test_a_plugin_that_declares_nothing_reports_nothing():
    assert Plugin({}, name="p").realised() == []
