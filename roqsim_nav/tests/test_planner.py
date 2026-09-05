"""The global planner: A* over the occupancy grid, and the grid's own inflation.

These moved here with the code they test, out of ``roqsim_walker``. Nothing about them is
pedestrian-specific -- the planner is the same one a navigated prop or an opponent robot searches
with -- which is why the code left that package in the first place.
"""

from __future__ import annotations

import pytest

from roqsim_nav.occupancy import OccupancyGrid
from roqsim_nav.planner import GridPlanner

# -- planner -----------------------------------------------------------------------------------
_BOUNDS = (-2.0, -2.0, 2.0, 2.0)  # OccupancyGrid.from_polygons pads this by 1 m on every side


def _wall_grid(top: float = 1.0):
    """A room bisected by a wall at x=0 running from below the padded floor up to ``top``, so the
    only way across is over the gap at the top."""
    wall = [(-0.1, -3.5), (0.1, -3.5), (0.1, top), (-0.1, top)]
    return OccupancyGrid.from_polygons([wall], resolution=0.05, bounds=_BOUNDS)


def test_planner_routes_around_a_wall():
    planner = GridPlanner(_wall_grid(), inflation_radius=0.2)
    path = planner.plan((-1.0, -1.0), (1.0, -1.0))
    assert path, "goal should be reachable via the gap above the wall"
    assert path[-1] == pytest.approx((1.0, -1.0))
    # The straight line would cross x=0 at y=-1 (solid); the plan must detour over the wall's top.
    assert max(y for _, y in path) > 1.0


def test_planner_returns_none_when_the_goal_is_walled_off():
    # A wall spanning the whole padded grid seals the two halves apart.
    sealed = _wall_grid(top=3.5)
    planner = GridPlanner(sealed, inflation_radius=0.0)
    assert planner.plan((-1.0, 0.0), (1.0, 0.0)) is None


def test_planner_simplifies_a_clear_run_to_a_straight_shot():
    grid = OccupancyGrid.from_polygons([[(9.0, 9.0), (9.1, 9.0), (9.1, 9.1)]], bounds=_BOUNDS)
    planner = GridPlanner(grid, inflation_radius=0.0)
    path = planner.plan((-1.5, -1.5), (1.5, 1.5))
    assert path is not None
    assert len(path) <= 2, f"line-of-sight string-pulling should collapse the path, got {path}"


def test_occupancy_inflation_grows_obstacles():
    grid = _wall_grid()
    base = grid.inflate(0.0).sum()
    grown = grid.inflate(0.3).sum()
    assert grown > base


# -- blueprint ---------------------------------------------------------------------------------
