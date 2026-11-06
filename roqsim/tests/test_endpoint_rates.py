# Copyright (C) 2026 Frederik Pasch
#
# SPDX-License-Identifier: Apache-2.0

"""Every endpoint a world offers, with the rate each output is published at.

A publish lands on a physics step, so an output asking for a rate the step does not divide is
published at the nearest one the step can hold. ``roqsim check`` and ``roqsim scenes describe`` list
both rates for every output, and warn where the two differ by more than the notable band.
"""

from __future__ import annotations

import textwrap

import pytest

from roqsim.check import check_world
from roqsim.endpoint import rate_snap_warnings


def _world(tmp_path, body: str):
    path = tmp_path / "world.yaml"
    path.write_text(textwrap.dedent(body).lstrip(), encoding="utf-8")
    return str(path)


def test_every_output_is_listed_with_the_rate_a_run_publishes_it_at(tmp_path):
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
    outputs = [e for e in report["world"]["endpoints"] if e["requested_hz"]]
    assert outputs, "a robot publishes something at a rate"
    for row in outputs:
        assert row["realised_hz"] is not None and row["every_steps"] >= 1


def test_a_rate_the_step_moves_far_is_a_warning_and_a_near_one_is_not():
    rows = [
        {
            "name": "scan",
            "owner": "robot.lidar",
            "topic": "scan",
            "requested_hz": 30.0,
            "realised_hz": 31.25,
            "every_steps": 16,
        },
        {
            "name": "odom",
            "topic": "odom",
            "requested_hz": 50.0,
            "realised_hz": 50.0,
            "every_steps": 10,
        },
        {
            "name": "cmd_vel",
            "topic": "cmd_vel",
            "requested_hz": None,
            "realised_hz": None,
            "every_steps": None,
        },
    ]
    warnings = rate_snap_warnings(rows)
    assert [w["endpoint"] for w in warnings] == ["scan"]
    assert "31.25 Hz" in warnings[0]["message"] and "16" in warnings[0]["message"]
    assert "output robot.lidar.scan (scan)" in warnings[0]["message"]
