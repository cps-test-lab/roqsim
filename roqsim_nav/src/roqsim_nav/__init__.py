"""2D navigation for roqsim: plan a route, follow it, and move something along it.

The global-plan / local-control split, as reusable parts:

* :mod:`~roqsim_nav.obstacles` reads the world's static wall footprints straight off the compiled
  MuJoCo model -- no map file, and the same polygons feed the planner and any avoidance model, so
  the two layers cannot disagree about where a wall is;
* :mod:`~roqsim_nav.occupancy` rasterizes those into a grid and :mod:`~roqsim_nav.grid` builds one
  per world rather than one per agent;
* :mod:`~roqsim_nav.planner` searches it with A* and string-pulls the result to sparse waypoints;
* :mod:`~roqsim_nav.behavior` follows that path, advances goals, and recovers from being stuck;
* :mod:`~roqsim_nav.state` is the eleven-attribute contract those two work against, so any
  embodiment can satisfy it.

Nothing here knows what is being moved. That is the ``navigator`` plugin's ``output``, resolved from
the ``roqsim_nav.outputs`` entry-point group -- a wheeled base's velocity command, a mocap prop's
pose, a pedestrian's animated skeleton -- and local avoidance is resolved the same way from
``roqsim_nav.avoidance``. Neither registry is a name list in this package: an out-of-tree package
adds an embodiment or a local planner without editing anything here.
"""

from pathlib import Path

from .behavior import NavCore, NavParams, build_tree
from .grid import DEFAULT_RESOLUTION, build_grid, grid_key
from .obstacles import dynamic_obstacle_bodies, wall_polygons
from .occupancy import OccupancyGrid
from .planner import GridPlanner
from .state import NavState, NavStateLike

#: Where `roqsim sim roqsim_nav:<world>` looks. Declared for the one example world; this package
#: ships no models, deliberately.
WORLDS_DIR = Path(__file__).parent / "worlds"

__all__ = [
    "WORLDS_DIR",
    "DEFAULT_RESOLUTION",
    "GridPlanner",
    "NavCore",
    "NavParams",
    "NavState",
    "NavStateLike",
    "OccupancyGrid",
    "build_grid",
    "build_tree",
    "dynamic_obstacle_bodies",
    "grid_key",
    "wall_polygons",
]
