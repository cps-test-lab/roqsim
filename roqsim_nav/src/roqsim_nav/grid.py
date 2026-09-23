"""Building the planner's occupancy grid from the compiled model, once per world.

The grid is derived from :func:`~roqsim_nav.obstacles.wall_polygons`, so it holds exactly the world's
**static** geometry. Everything that moves -- a robot, a walker, a navigated prop -- is deliberately
absent from it and is handled by the layers that can react to motion.

**One grid serves every agent in a world.** Building it walks every geom in the model, convex-hulls
each wall footprint and rasterizes the lot; doing that once per navigator would repeat identical work
for every agent in the scene. :func:`grid_key` is what makes sharing safe: two agents may share a
raster only when they agree about the *resolution* and about which vertical band counts as a wall, so
a short prop that can pass under a beam does not inherit a tall walker's grid. Inflation is not part
of the key -- :class:`~roqsim_nav.planner.GridPlanner` memoises that per radius on top of one shared
raster, so differently sized agents still share.
"""

from __future__ import annotations

import numpy as np

from .obstacles import wall_polygons
from .occupancy import OccupancyGrid

#: Default cell size (m). Fine enough for a doorway, coarse enough that a 20 m room is a 400x400
#: raster rather than a megapixel.
DEFAULT_RESOLUTION = 0.05


def grid_key(resolution: float, z_lo: float, z_hi: float, resting_roots=()) -> str:
    """The blackboard key of a shareable grid: agents agreeing on these share one raster.

    A string, because that is what ``Blackboard`` declares its keys to be. The rounding is what makes
    two agents that wrote the same numbers different ways land on the same raster.

    ``resting_roots`` is in the key because it changes what the raster contains. Every navigator in a
    world derives the same set -- the undriven props -- so in practice they still share one grid, and
    the key is what makes that a fact rather than an assumption.
    """
    roots = ",".join(str(int(b)) for b in sorted(resting_roots))
    return f"nav:grid:{float(resolution):.6f}:{float(z_lo):.6f}:{float(z_hi):.6f}:{roots}"


def build_grid(
    model,
    data,
    *,
    extra_points=(),
    z_lo: float = 0.1,
    z_hi: float = 1.8,
    resolution: float = DEFAULT_RESOLUTION,
    resting_roots=(),
) -> OccupancyGrid | None:
    """An occupancy grid covering the world's walls and ``extra_points``, or ``None``.

    ``None`` means there was nothing static to plan around -- a wall-less MJCF. That is not an error:
    the behaviour tree falls back to straight-line legs between goals.

    ``extra_points`` are world xy the grid must cover even if they sit outside the walls' extent --
    an agent's spawn and its goals, which would otherwise fall off a grid sized to the walls alone
    and be clamped to its edge.

    ``resting_roots`` names bodies that have DOFs but nothing driving them -- a free prop somebody
    parked. Each is rasterized as a wall while it is standing still, so a crate left in a doorway is
    routed around rather than planned through and stopped in front of. Anything under a navigator, and
    the robot under test, are deliberately NOT in that set: they are traffic, and traffic is the
    business of the layers that can react to it.
    """
    polys = wall_polygons(model, data, z_lo=z_lo, z_hi=z_hi, resting_roots=resting_roots)
    if not polys:
        return None
    pts = [p for poly in polys for p in poly]
    pts.extend(tuple(p)[:2] for p in extra_points)
    arr = np.asarray(pts, dtype=float)
    bounds = (arr[:, 0].min(), arr[:, 1].min(), arr[:, 0].max(), arr[:, 1].max())
    return OccupancyGrid.from_polygons(polys, resolution=resolution, bounds=bounds)
