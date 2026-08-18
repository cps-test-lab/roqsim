"""`grid_to_floorplan`: the shared grid -> wall-segments -> floorplan core.

What is worth testing here is the FRAME and the schema, not the tracing quality: the segmentation is
a heuristic whose output a human checks, but the row-down -> y-up flip is exact, and getting it wrong
produces a plan that is the right shape in the wrong place.
"""

from __future__ import annotations

import numpy as np
import pytest

from roqsim_scenes import grid_to_floorplan as g2f

CELL = 0.1


# -- frame convention ---------------------------------------------------------------------------
def test_grid_row_zero_becomes_the_highest_y():
    rows = 10
    top = g2f.to_floorplan([("h", 0, 0, 4)], rows, CELL)["lines"][0]
    bottom = g2f.to_floorplan([("h", rows - 1, 0, 4)], rows, CELL)["lines"][0]
    assert top["y0_m"] > bottom["y0_m"]
    assert bottom["y0_m"] == pytest.approx(CELL / 2)


def test_y_depends_on_the_grid_height_not_on_the_grid_width():
    """A wide plan must not be pushed up the y axis by its own width.

    `fixed` is a row for an 'h' segment but a COLUMN for a 'v' one, so deriving the row count from
    the segments makes every y a function of the aspect ratio. On a 10 x 30 m room that put the whole
    floorplan at y in [20, 30] -- correct shape, wrong frame, and the tracer's debug overlay (which
    uses the grid's real height) then drew the walls off the image entirely.
    """
    rows, cols = 100, 300  # 10 m x 30 m at CELL
    segs = [
        ("h", 0, 0, cols - 1),  # top wall, spans the full width
        ("h", rows - 1, 0, cols - 1),  # bottom wall
        ("v", 0, 0, rows - 1),  # left wall: fixed=0 is a COLUMN
        ("v", cols - 1, 0, rows - 1),  # right wall
    ]
    ys = [v for ln in g2f.to_floorplan(segs, rows, CELL)["lines"] for v in (ln["y0_m"], ln["y1_m"])]
    assert min(ys) == pytest.approx(CELL / 2)
    assert max(ys) == pytest.approx((rows - 0.5) * CELL)


def test_origin_places_the_grids_bottom_left_corner():
    """Same ROS convention as gridmap_to_world, so a floorplan and a map from one grid share a frame."""
    rows = 10
    plan = g2f.to_floorplan([("h", rows - 1, 0, 4)], rows, CELL, origin=(-4.5, 2.0))
    line = plan["lines"][0]
    assert line["x0_m"] == pytest.approx(-4.5)
    assert line["y0_m"] == pytest.approx(2.0 + CELL / 2)


def test_emitted_keys_are_the_sketch_windows_keys():
    """floorplan_to_world.py must not be able to tell a generated plan from a drawn one."""
    plan = g2f.to_floorplan([("h", 0, 0, 4)], 10, CELL)
    assert set(plan) == {"lines", "doors", "rooms", "markers"}
    assert set(plan["lines"][0]) == {"id", "x0_m", "y0_m", "x1_m", "y1_m"}


# -- segmentation -------------------------------------------------------------------------------
def test_segments_claim_the_longest_wall_first():
    grid = np.zeros((6, 12), dtype=bool)
    grid[2, :] = True  # a 12-cell horizontal wall
    grid[:, 9] = True  # a 6-cell vertical one crossing it
    assert g2f.segments(grid.copy(), min_cells=3, gap=0)[0][0] == "h"


def test_a_crossing_wall_survives_a_junction_only_if_gap_covers_the_consumed_band():
    """Claiming a wall also clears one cell to each side, biting 3 cells out of anything crossing it.

    That is why `gap` is not a free parameter: it must be >= 3 cells for the remainder of a crossing
    wall to re-join across the bite. Below it, T-junction walls fragment or vanish -- silently.
    """
    grid = np.zeros((6, 12), dtype=bool)
    grid[2, :] = True
    grid[:, 9] = True
    assert [s[0] for s in g2f.segments(grid.copy(), min_cells=3, gap=2)] == ["h"]
    assert [s[0] for s in g2f.segments(grid.copy(), min_cells=3, gap=3)] == ["h", "v"]


def test_runs_bridge_holes_but_drop_speckle():
    line = np.array([1, 1, 0, 1, 1, 0, 0, 0, 1], dtype=bool)
    assert g2f._runs(line, min_cells=3, gap=1) == [(0, 4)]  # hole of 1 bridged; lone tail dropped
    assert g2f._runs(line, min_cells=3, gap=0) == []  # no bridging: nothing reaches 3


def test_a_crisp_four_wall_room_comes_out_as_four_lines():
    grid = np.zeros((20, 20), dtype=bool)
    grid[0, :] = grid[-1, :] = grid[:, 0] = grid[:, -1] = True
    segs = g2f.segments(grid, min_cells=3, gap=3)
    assert len(segs) == 4
    assert sorted(s[0] for s in segs) == ["h", "h", "v", "v"]
