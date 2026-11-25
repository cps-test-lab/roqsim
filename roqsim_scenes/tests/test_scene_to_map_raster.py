"""A map's cells hold the wall faces that touch them, the same on both sides of a room."""

from __future__ import annotations

import numpy as np

from roqsim_scenes.cli.scene_to_map import _OCC, _cell, _raster

# The origin main() computes for empty_room: bounds -5.1 less a 0.3 m pad, -5.3999999999999995.
ORIGIN = np.array([-5.1, -5.1]) - 0.3
CELLS = 216


def _occupied_columns(x: float) -> set[int]:
    seg = (np.array([x, -1.0]), np.array([x, 1.0]))
    grid = _raster([seg], ORIGIN, 0.05, CELLS, CELLS)
    return set(np.nonzero((grid == _OCC).any(axis=0))[0].tolist())


def test_a_face_on_a_cell_edge_occupies_both_cells_beside_it():
    # -5.0 m is the edge between cells 7 and 8; from this origin it is 7.999999999999989 cells.
    assert _occupied_columns(-5.0) == {7, 8}
    assert _occupied_columns(5.0) == {207, 208}


def test_the_two_faces_of_a_symmetric_room_are_drawn_symmetrically():
    left, right = _occupied_columns(-5.0), _occupied_columns(5.0)
    assert {CELLS - 1 - c for c in left} == right


def test_a_face_inside_a_cell_occupies_that_cell():
    assert _occupied_columns(4.97) == {207}


def test_a_point_left_of_the_origin_is_outside_the_map():
    assert _occupied_columns(-5.42) == set()
    assert _cell(-5.42, ORIGIN[0], 0.05) == -1
