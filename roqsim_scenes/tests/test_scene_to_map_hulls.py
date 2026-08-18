"""The convex-hull footprint pass, which exists because triangles and collisions disagree for a mesh.

MuJoCo collides a mesh geom by its CONVEX HULL but raycasts its real triangles. So a table whose scan
plane passes between its legs is invisible both to the lidar and to a triangle-sliced map, while still
being a solid block to the physics engine -- measured in the pick world, rays fired at the table pass
straight through and hit the wall 3 m beyond, and the map got four isolated cells for the whole table.
A planner then routes through it and the base collides with something it never saw.
"""

from __future__ import annotations

import numpy as np

from roqsim_scenes.cli.scene_to_map import _hull_edges


def _ring(edges):
    """The closed polygon the edges describe, as a set of rounded points."""
    return {(round(float(a[0]), 6), round(float(a[1]), 6)) for a, _ in edges}


def test_hull_of_a_square_is_its_four_corners():
    pts = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.5, 0.5]])  # + interior point
    edges = _hull_edges(pts)
    assert len(edges) == 4, "an interior point was kept on the hull"
    assert _ring(edges) == {(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)}


def test_hull_edges_form_a_closed_loop():
    """Open edges would leak the flood fill into the obstacle's interior, so it must close."""
    rng = np.random.default_rng(0)
    edges = _hull_edges(rng.normal(size=(40, 2)))
    starts = [tuple(np.round(a, 9)) for a, _ in edges]
    ends = [tuple(np.round(b, 9)) for _, b in edges]
    assert sorted(starts) == sorted(ends), "the hull is not a closed ring"


def test_a_table_shaped_leg_cloud_becomes_one_solid_footprint():
    """Four thin legs must yield the tabletop's whole outline, not four separate dots.

    This is the actual geometry: an industrial_table's legs at 0.05 m map resolution are one cell each, and
    the 1.20 x 0.66 m outline between them is what the robot cannot drive through.
    """
    legs = np.array(
        [[x, y] for x in (11.70, 12.30) for y in (3.97, 5.13) for _ in range(4)], dtype=float
    )
    edges = _hull_edges(legs)
    ring = np.array(sorted(_ring(edges)))
    assert len(ring) == 4
    assert ring[:, 0].min() == 11.70 and ring[:, 0].max() == 12.30
    assert ring[:, 1].min() == 3.97 and ring[:, 1].max() == 5.13


def test_degenerate_input_is_not_an_obstacle():
    """Fewer than three distinct points has no interior, so it must produce no edges rather than raise."""
    assert _hull_edges(np.array([[0.0, 0.0], [1.0, 1.0]])) == []
    assert _hull_edges(np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])) == []
