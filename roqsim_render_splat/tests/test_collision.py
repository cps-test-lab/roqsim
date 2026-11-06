# Copyright (C) 2026 Frederik Pasch
# SPDX-License-Identifier: Apache-2.0
"""``roqsim splat collision`` fits what the robot meets to what the camera sees."""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import pytest

import roqsim_render_splat
from roqsim.pose import rpy_to_quat
from roqsim_render_splat import collision, synthetic
from roqsim_render_splat.ply import write_splat

EXAMPLE = Path(roqsim_render_splat.WORLDS_DIR) / "splat_room_example.yaml"


def _dense_room(tmp_path) -> Path:
    """The example room captured as densely as a real capture is (Gaussians 2 cm apart)."""
    pose = synthetic.EXAMPLE_POSE
    splat = synthetic.room_splat(
        synthetic.example_room(), rpy_to_quat(*pose["rpy"]), pose["position"], pose["scale"], 0.02
    )
    write_splat(tmp_path / "room.ply", splat)
    world = tmp_path / "room.yaml"
    text = EXAMPLE.read_text().replace(
        "roqsim_render_splat:splat_room_example/room.ply", "room.ply"
    )
    world.write_text(text)
    return world


def test_the_example_room_s_walls_and_table_are_found(tmp_path):
    world = _dense_room(tmp_path)
    out = tmp_path / "collision.xml"
    assert collision.main([str(world), "--out", str(out)]) == 0
    model = mujoco.MjModel.from_xml_path(str(out))  # it compiles: floor, 4 walls, the table
    assert model.ngeom == 6
    points, _ = collision.world_points(world)
    result = collision.fit(points)
    (lo, hi) = result["footprint"]
    assert lo == pytest.approx([-2.0, -1.5], abs=0.06) and hi == pytest.approx([2.0, 1.5], abs=0.06)
    # The table (0.45..1.35 x 0.2..0.8 m) is one box, to within a grid cell or two.
    (table,) = result["boxes"]
    x, y, _ = table["centre"]
    hx, hy, _ = table["half"]
    assert [x - hx, x + hx, y - hy, y + hy] == pytest.approx([0.45, 1.35, 0.2, 0.8], abs=0.06)


def test_a_world_that_inherits_its_splat_is_fitted_as_the_renderer_draws_it(tmp_path):
    child = tmp_path / "mine" / "room.yaml"
    child.parent.mkdir()
    child.write_text("extends: roqsim_render_splat:splat_room_example\n")
    points, block = collision.world_points(child)
    example, _ = collision.world_points(EXAMPLE)
    assert block["file"] == "roqsim_render_splat:splat_room_example/room.ply"
    np.testing.assert_array_equal(points, example)


def test_a_world_that_places_no_splat_is_said(tmp_path, capsys):
    world = tmp_path / "w.yaml"
    world.write_text("sim: {}\ncomponents: []\n")
    assert collision.main([str(world), "--out", str(tmp_path / "c.xml")]) == 2
    assert "places no splat" in capsys.readouterr().err
