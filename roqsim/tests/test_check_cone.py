# Copyright (C) 2026 Frederik Pasch
#
# SPDX-License-Identifier: Apache-2.0
"""``roqsim check`` warns when a world that grasps, pushes or carries leaves the friction cone
pyramidal: the pyramid quantises the tangential force's direction, and such a world's outcome rides
on it. A world with nothing to grasp or carry, and a world that chose its cone, are left alone."""

from __future__ import annotations

import textwrap

import pytest

from roqsim.check import check_world


def _world(tmp_path, body: str):
    path = tmp_path / "world.yaml"
    path.write_text(textwrap.dedent(body).lstrip(), encoding="utf-8")
    return path


def _cone_warnings(report: dict) -> list[dict]:
    assert report["ok"], report["problems"]
    return [w for w in report["warnings"] if w["check"] == "friction-cone"]


def test_an_object_under_the_default_cone_is_reported(tmp_path):
    # `dummy` registers an `object` entity: the kind a carried part or a package registers as.
    world = _world(
        tmp_path,
        """
        sim: {}
        components:
          - dummy: {size: 0.05}
            name: part
        """,
    )
    (warning,) = _cone_warnings(check_world(str(world)))
    assert "pyramidal" in warning["message"] and "(part)" in warning["message"]
    assert "cone: elliptic" in warning["hint"]


def test_a_world_that_chose_its_cone_is_not(tmp_path):
    world = _world(
        tmp_path,
        """
        sim: {cone: elliptic}
        components:
          - dummy: {size: 0.05}
            name: part
        """,
    )
    assert _cone_warnings(check_world(str(world))) == []


@pytest.mark.parametrize("cone", ["pyramidal", "elliptic"])
def test_a_world_with_nothing_to_carry_is_not(tmp_path, cone):
    world = _world(
        tmp_path,
        f"""
        sim: {{cone: {cone}}}
        components: []
        """,
    )
    assert _cone_warnings(check_world(str(world))) == []
