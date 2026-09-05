"""Remembered blockages: the planner's only knowledge its grid does not have.

The grid is static, so `avoidance.reroute` on its own would compute the identical path and drive
into the same obstacle again. What makes it work is that a blockage is *remembered*, and what keeps
it honest is that the memory *expires* -- a mover must not accumulate a private map of everything
that ever got in its way.

These test the planner directly. The navigator's half -- who reports a blockage, and when -- is in
``test_caution.py`` and ``test_navigator_replan.py``.
"""

from __future__ import annotations

import math

import pytest

from roqsim_nav.occupancy import OccupancyGrid
from roqsim_nav.planner import GridPlanner

_BOUNDS = (-3.0, -3.0, 3.0, 3.0)
_RES = 0.05
#: A plan clears a mark by its radius less one cell: A* keeps every *cell* out of the disc, and the
#: string-pulled polyline joins cell centres, so a chord may cut half a cell inside.
_CLEARS = 0.5 - _RES


def _open_room():
    """A room with one wall stub, leaving the middle wide open in both directions."""
    stub = [(-0.1, 2.5), (0.1, 2.5), (0.1, 4.0), (-0.1, 4.0)]
    return OccupancyGrid.from_polygons([stub], resolution=_RES, bounds=_BOUNDS)


def _clearance(path, point=(0.0, 0.0)):
    """How close the planned polyline comes to ``point``, sampled along its legs.

    The property a detour has and a straight line does not, stated as a distance rather than as a
    box: the plan is judged against the radius of the blockage it was supposed to avoid.
    """
    pts = [(0.0, -2.0), *[(float(a), float(b)) for a, b in path]]
    best = float("inf")
    for (x0, y0), (x1, y1) in zip(pts, pts[1:], strict=False):
        for t in [i / 50.0 for i in range(51)]:
            px, py = x0 + (x1 - x0) * t, y0 + (y1 - y0) * t
            best = min(best, math.hypot(px - point[0], py - point[1]))
    return best


def test_a_blockage_pushes_the_path_off_the_straight_line():
    planner = GridPlanner(_open_room(), inflation_radius=0.1)
    straight = planner.plan((0.0, -2.0), (0.0, 2.0), now=0.0)
    assert _clearance(straight) < 0.1, "with nothing in the way the plan goes up the middle"

    planner.add_blockage((0.0, 0.0), 0.5, expires_at=5.0)
    detour = planner.plan((0.0, -2.0), (0.0, 2.0), now=1.0)
    assert detour, "the goal is still reachable around the blockage"
    assert _clearance(detour) >= _CLEARS, "the plan must clear what stopped the mover, not cross it"


def test_a_blockage_expires():
    planner = GridPlanner(_open_room(), inflation_radius=0.1)
    planner.add_blockage((0.0, 0.0), 0.5, expires_at=5.0)
    assert _clearance(planner.plan((0.0, -2.0), (0.0, 2.0), now=1.0)) >= _CLEARS
    # Past its expiry the obstacle is forgotten: an obstacle that moved on is not a wall.
    assert _clearance(planner.plan((0.0, -2.0), (0.0, 2.0), now=6.0)) < 0.1
    assert planner.live_marks(6.0) == []


def test_the_same_obstacle_is_remembered_once():
    """A mover held against one obstacle for seconds must not accumulate a mark per tick.

    The expiries RISE across the calls, because that is what the navigator does: every blocked tick
    asks for ``now + forget_after``, and ``now`` has moved on. A dedup that only recognised an
    identical expiry would let a few seconds of being stuck become hundreds of discs, each one
    stamped onto the raster on every later plan.
    """
    planner = GridPlanner(_open_room(), inflation_radius=0.1)
    assert planner.add_blockage((0.0, 0.0), 0.5, expires_at=5.0) is True
    for tick in range(1, 200):
        assert planner.add_blockage((0.05, 0.05), 0.5, expires_at=5.0 + tick * 0.05) is False
    assert planner.add_blockage((2.0, 0.0), 0.5, expires_at=5.0) is True
    assert len(planner.live_marks(0.0)) == 2


def test_being_blocked_again_keeps_the_memory_alive():
    """Still blocked there is a reason to keep remembering it, not to remember it twice."""
    planner = GridPlanner(_open_room(), inflation_radius=0.1)
    planner.add_blockage((0.0, 0.0), 0.5, expires_at=5.0)
    planner.add_blockage((0.0, 0.0), 0.5, expires_at=20.0)
    assert len(planner.live_marks(10.0)) == 1, "the mark expired while the obstacle was still there"


def test_a_goal_inside_a_blockage_is_still_attempted():
    """Better to drive up to it and stop than to give up on the route entirely."""
    planner = GridPlanner(_open_room(), inflation_radius=0.1)
    planner.add_blockage((0.0, 2.0), 0.5, expires_at=5.0)
    path = planner.plan((0.0, -2.0), (0.0, 2.0), now=1.0)
    assert path, "the ladder retries without marks rather than reporting the goal unreachable"
    assert path[-1] == pytest.approx((0.0, 2.0))


def test_marks_do_not_leak_into_the_shared_raster():
    """The grid is shared by every mover in the world; one mover's experience is not another's."""
    grid = _open_room()
    a, b = GridPlanner(grid, inflation_radius=0.1), GridPlanner(grid, inflation_radius=0.1)
    a.add_blockage((0.0, 0.0), 0.5, expires_at=5.0)
    a.plan((0.0, -2.0), (0.0, 2.0), now=1.0)
    other = b.plan((0.0, -2.0), (0.0, 2.0), now=1.0)
    assert _clearance(other) < 0.1, "the other mover detoured around a mark it never made"
    assert not grid.occupied[grid.world_to_cell(0.0, 0.0)], "the raster itself was mutated"


def test_a_reset_forgets_everything():
    planner = GridPlanner(_open_room(), inflation_radius=0.1)
    planner.add_blockage((0.0, 0.0), 0.5, expires_at=1e9)
    planner.forget_blockages()
    assert _clearance(planner.plan((0.0, -2.0), (0.0, 2.0), now=1.0)) < 0.1
