# SPDX-License-Identifier: Apache-2.0
"""Re-export of :mod:`roqsim.floorplan_geometry`.

The arithmetic lives in core because a second package needs it: the ``floorplan`` plugin builds its
walls from segments, and core is where the floorplan geometry MuJoCo consumes lives
(:mod:`roqsim.floorplan_collision`). This path stays importable because three readers import it.
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
