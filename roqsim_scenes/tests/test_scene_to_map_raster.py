"""A map's cells hold the points they cover, at their edges as well as inside them."""

from __future__ import annotations

import numpy as np

from roqsim_scenes.cli.scene_to_map import _OCC, _raster


def _occupied_columns(x: float) -> set[int]:
    seg = (np.array([x, -1.0]), np.array([x, 1.0]))
    # The origin as main() computes it for empty_room: bounds -5.1 less a 0.3 m pad, -5.3999999999999995.
    grid = _raster([seg], np.array([-5.1, -5.1]) - 0.3, 0.05, 216, 216)
    return set(np.nonzero((grid == _OCC).any(axis=0))[0].tolist())


def test_a_wall_face_on_a_cell_edge_lands_in_the_cell_it_starts():
    # 5.0 m is where cell 208 starts; from that origin, 5.0 m is 207.99999999999997 cells.
    assert _occupied_columns(5.0) == {208}


def test_the_two_faces_of_a_symmetric_room_are_drawn_symmetrically():
    left, right = _occupied_columns(-5.0), _occupied_columns(5.0)
    assert left == {8} and right == {208}
    assert min(left) + max(right) == 216


def test_a_point_left_of_the_origin_is_outside_the_map():
    assert _occupied_columns(-5.42) == set()
