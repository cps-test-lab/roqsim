# SPDX-License-Identifier: Apache-2.0
"""Re-export of :mod:`roqsim.floorplan_geometry`, which is where this now lives.

The arithmetic moved into core when a second package needed it: the ``floorplan`` plugin builds its
walls from segments, and core is already where the floorplan geometry MuJoCo consumes lives
(:mod:`roqsim.floorplan_collision`). Kept here because this path is the one three readers already
import, and a move is not a reason to break them.
"""

from roqsim.floorplan_geometry import (
    Opening,
    Room,
    WallPlan,
    assign_doors,
    bounds,
    cut_openings,
    label_point,
    label_spot,
    line_segments,
    plan_walls,
    point_in_polygon,
    polygon_area,
    room_polygons,
    wall_pieces,
)

__all__ = [
    "Opening",
    "Room",
    "WallPlan",
    "assign_doors",
    "bounds",
    "cut_openings",
    "label_point",
    "label_spot",
    "line_segments",
    "plan_walls",
    "point_in_polygon",
    "polygon_area",
    "room_polygons",
    "wall_pieces",
]
