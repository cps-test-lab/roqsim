"""`gridmap_to_world`: occupancy grid -> world YAML + co-registered Nav2 map.

The contract that matters is **co-registration**: an obstacle placed in the world at (x, y) must land
in an occupied pixel of the emitted map, and vice versa. Everything else here is guarding the two
conventions that make that true -- row 0 is the top of the image, and the origin is the grid's
bottom-left corner in world metres -- because an off-by-one-row or a flipped axis produces a world
and a map that are each individually plausible and jointly wrong.
"""

from __future__ import annotations

import numpy as np
import pytest
import yaml

from roqsim_scenes.cli import gridmap_to_world as g2w

CELL = 0.15


def _grid(pattern: list[str]) -> np.ndarray:
    """Build a grid from rows of '.' (free) and '#' (occupied); row 0 is the TOP row."""
    return np.array([[1 if ch == "#" else 0 for ch in row] for row in pattern], dtype=np.uint8)


# -- frame convention ---------------------------------------------------------------------------
def test_row_zero_is_the_top_row_of_the_world():
    grid = _grid(["#.", ".."])  # occupied cell at (row 0, col 0)
    rows = grid.shape[0]
    x, y = g2w.cell_centre(0, 0, rows, CELL, (0.0, 0.0))
    x_bottom, y_bottom = g2w.cell_centre(rows - 1, 0, rows, CELL, (0.0, 0.0))
    assert x == pytest.approx(0.075)
    assert y > y_bottom  # row 0 is the HIGH-y (top) row


def test_origin_is_the_bottom_left_corner_in_world_metres():
    rows = 4
    x, y = g2w.cell_centre(rows - 1, 0, rows, CELL, (-4.5, 2.0))
    assert (x, y) == pytest.approx((-4.5 + CELL / 2, 2.0 + CELL / 2))


# -- co-registration ----------------------------------------------------------------------------
def test_world_props_land_in_occupied_map_pixels(tmp_path):
    grid = _grid(
        [
            "..#..",
            ".###.",
            "..#..",
            ".....",
        ]
    )
    rows, cols = grid.shape
    origin = (-1.0, 0.5)

    props = g2w.obstacle_plugins(
        grid,
        cell_size=CELL,
        origin=origin,
        obstacle="cylinder",
        radius=CELL / 2,
        height=0.5,
        color=[1, 0, 0, 1],
        shell_only=False,
        prefix="obs_",
    )
    g2w.write_map(grid, tmp_path / "m", cell_size=CELL, origin=origin)

    pgm = (tmp_path / "m.pgm").read_bytes()
    header_end = pgm.index(b"255\n") + 4
    pixels = np.frombuffer(pgm[header_end:], dtype=np.uint8).reshape(rows, cols)

    assert len(props) == int(grid.sum())
    for entry in props:
        x, y = entry["cylinder"]["pos"]
        # invert cell_centre
        c = round((x - origin[0]) / CELL - 0.5)
        r = rows - 1 - round((y - origin[1]) / CELL - 0.5)
        assert grid[r][c] == 1, f"prop at ({x}, {y}) maps to a FREE grid cell ({r}, {c})"
        assert pixels[r][c] == 0, f"prop at ({x}, {y}) maps to a FREE map pixel ({r}, {c})"

    # ... and every occupied pixel has a prop.
    assert int((pixels == 0).sum()) == int(grid.sum())


def test_map_yaml_pins_resolution_and_origin(tmp_path):
    grid = _grid(["#.", ".."])
    g2w.write_map(grid, tmp_path / "m", cell_size=CELL, origin=(-4.5, 0.0))
    meta = yaml.safe_load((tmp_path / "m.yaml").read_text())
    assert meta["resolution"] == pytest.approx(CELL)
    assert meta["origin"] == pytest.approx([-4.5, 0.0, 0.0])
    assert meta["image"] == "m.pgm"


# -- shell_only ---------------------------------------------------------------------------------
def test_shell_only_drops_enclosed_cells_but_never_the_map():
    grid = _grid(
        [
            "#####",
            "#####",
            "#####",
        ]
    )
    # Enclosed = every 8-neighbour is in-grid AND occupied. Here that is exactly row 1, cols 1..3;
    # everything else touches the grid edge, which counts as exposed.
    assert [g2w.is_shell(grid, 1, c) for c in range(5)] == [True, False, False, False, True]
    assert g2w.is_shell(grid, 0, 0) is True

    full = g2w.obstacle_plugins(
        grid,
        cell_size=CELL,
        origin=(0, 0),
        obstacle="cylinder",
        radius=CELL / 2,
        height=0.5,
        color=[1, 0, 0, 1],
        shell_only=False,
        prefix="o_",
    )
    shell = g2w.obstacle_plugins(
        grid,
        cell_size=CELL,
        origin=(0, 0),
        obstacle="cylinder",
        radius=CELL / 2,
        height=0.5,
        color=[1, 0, 0, 1],
        shell_only=True,
        prefix="o_",
    )
    assert len(full) == 15
    assert len(shell) == 12  # the three enclosed cells are dropped


def test_shell_only_leaves_the_map_complete(tmp_path):
    """The map is what the planner is given; a hollow blob in the map would let it plan inside."""
    grid = _grid(["#####", "#####", "#####"])
    g2w.write_map(grid, tmp_path / "m", cell_size=CELL, origin=(0, 0))
    pgm = (tmp_path / "m.pgm").read_bytes()
    pixels = np.frombuffer(pgm[pgm.index(b"255\n") + 4 :], dtype=np.uint8)
    assert int((pixels == 0).sum()) == 15  # every occupied cell, not just the shell


# -- prop geometry ------------------------------------------------------------------------------
def test_cylinder_radius_defaults_leave_a_diagonal_gap_box_would_seal():
    grid = _grid(["#.", ".#"])
    props = g2w.obstacle_plugins(
        grid,
        cell_size=CELL,
        origin=(0, 0),
        obstacle="cylinder",
        radius=CELL / 2,
        height=0.5,
        color=[1, 0, 0, 1],
        shell_only=False,
        prefix="o_",
    )
    (x0, y0) = props[0]["cylinder"]["pos"]
    (x1, y1) = props[1]["cylinder"]["pos"]
    centre_dist = float(np.hypot(x1 - x0, y1 - y0))
    assert centre_dist == pytest.approx(CELL * np.sqrt(2))
    assert centre_dist - CELL > 0  # tangent-cylinder gap; boxes of side CELL would touch


def test_box_obstacles_fill_their_cell():
    grid = _grid(["#"])
    props = g2w.obstacle_plugins(
        grid,
        cell_size=CELL,
        origin=(0, 0),
        obstacle="box",
        radius=0.0,
        height=0.4,
        color=[1, 0, 0, 1],
        shell_only=False,
        prefix="o_",
    )
    assert props[0]["box"]["size"] == [CELL, CELL, 0.4]


# -- world assembly -----------------------------------------------------------------------------
def test_world_yaml_round_trips_and_places_the_robot_first(tmp_path):
    grid = _grid(["#.", ".#"])
    world = g2w.build_world(
        grid,
        cell_size=CELL,
        origin=(0, 0),
        obstacle="cylinder",
        radius=CELL / 2,
        height=0.5,
        color=[1, 0, 0, 1],
        shell_only=False,
        prefix="o_",
        robot="clearpath_jackal",
        start=(1.0, 2.0),
        yaw=0.0,
        extra_plugins=[{"ros2_bridge": {}}],
    )
    g2w.write_world(world, tmp_path / "w.yaml", header="generated by a test")
    loaded = yaml.safe_load((tmp_path / "w.yaml").read_text())
    kinds = [next(iter(p)) for p in loaded["components"]]
    assert kinds[0] == "spawn_robot"
    assert kinds[-1] == "ros2_bridge"
    assert kinds.count("cylinder") == 2
    assert loaded["components"][0]["spawn_robot"]["pos"] == pytest.approx([1.0, 2.0])
    assert (tmp_path / "w.yaml").read_text().startswith("# generated by a test")


def test_load_grid_reads_npy_and_text(tmp_path):
    grid = _grid(["#.", ".#"])
    np.save(tmp_path / "g.npy", grid)
    (tmp_path / "g.txt").write_text("1 0\n0 1\n")
    assert np.array_equal(g2w.load_grid(tmp_path / "g.npy"), grid)
    assert np.array_equal(g2w.load_grid(tmp_path / "g.txt"), grid)


def test_load_grid_rejects_non_2d(tmp_path):
    np.save(tmp_path / "g.npy", np.zeros((2, 2, 2)))
    with pytest.raises(ValueError, match="2D grid"):
        g2w.load_grid(tmp_path / "g.npy")


# -- rectangle merging --------------------------------------------------------------------------
def _footprint(entries, cell_size: float, origin, shape) -> np.ndarray:
    """Rasterise the emitted box plugins back onto the source grid, to compare unions exactly."""
    rows, cols = shape
    out = np.zeros(shape, dtype=bool)
    for entry in entries:
        cfg = entry["box"]
        x, y = cfg["pos"]
        sx, sy, _ = cfg["size"]
        # cell index of the box's lower-left corner, inverting the row axis on the way back
        c0 = round((x - sx / 2 - origin[0]) / cell_size)
        r1 = rows - 1 - round((y - sy / 2 - origin[1]) / cell_size)
        h, w = round(sy / cell_size), round(sx / cell_size)
        out[r1 - h + 1 : r1 + 1, c0 : c0 + w] = True
    return out


def _boxes(grid, *, merge, origin=(0.0, 0.0)):
    return g2w.obstacle_plugins(
        grid,
        cell_size=CELL,
        origin=origin,
        obstacle="box",
        radius=0.0,
        height=0.4,
        color=[1, 0, 0, 1],
        shell_only=False,
        prefix="o_",
        merge=merge,
    )


def test_merging_covers_exactly_the_same_cells():
    """The whole licence for merging: the union is unchanged, so the map still describes the world."""
    grid = _grid(
        [
            "####.",
            "####.",
            "..#..",
            "..###",
        ]
    )
    origin = (-0.6, 0.3)
    plain = _footprint(_boxes(grid, merge=False, origin=origin), CELL, origin, grid.shape)
    merged = _footprint(_boxes(grid, merge=True, origin=origin), CELL, origin, grid.shape)
    assert np.array_equal(plain, grid.astype(bool))
    assert np.array_equal(merged, grid.astype(bool))


def test_merging_a_solid_block_yields_one_box():
    grid = _grid(["###", "###", "###"])
    entries = _boxes(grid, merge=True)
    assert len(entries) == 1
    assert entries[0]["box"]["size"] == pytest.approx([3 * CELL, 3 * CELL, 0.4])
    assert entries[0]["box"]["pos"] == pytest.approx([1.5 * CELL, 1.5 * CELL])


def test_merging_a_scattered_field_buys_nothing():
    """A checkerboard has nothing colinear to merge -- the pass must not claim a win it did not get."""
    grid = _grid(["#.#.", ".#.#", "#.#.", ".#.#"])
    assert len(_boxes(grid, merge=True)) == len(_boxes(grid, merge=False)) == int(grid.sum())


def test_merging_is_refused_for_cylinders():
    """Squaring a cylinder field into rectangles fills the gaps that are the point of cylinders."""
    grid = _grid(["##", "##"])
    with pytest.raises(ValueError, match="box-only"):
        g2w.obstacle_plugins(
            grid,
            cell_size=CELL,
            origin=(0, 0),
            obstacle="cylinder",
            radius=CELL / 2,
            height=0.4,
            color=[1, 0, 0, 1],
            shell_only=False,
            prefix="o_",
            merge=True,
        )


def test_merging_does_not_touch_the_map(tmp_path):
    """Unlike --shell-only, merging changes nothing the planner is told."""
    grid = _grid(["###", "#.#", "###"])
    g2w.write_map(grid, tmp_path / "m", cell_size=CELL, origin=(0.0, 0.0))
    pgm = (tmp_path / "m.pgm").read_bytes()
    assert pgm.count(bytes([0])) == int(grid.sum())  # every occupied cell still marked


def test_merged_and_plain_worlds_place_the_same_obstacles_for_a_wall():
    """A 1-cell-thick wall is the case merging exists for: N props collapse to one box."""
    grid = _grid(["#####", ".....", "....."])
    plain, merged = _boxes(grid, merge=False), _boxes(grid, merge=True)
    assert len(plain) == 5 and len(merged) == 1
    assert merged[0]["box"]["size"][0] == pytest.approx(5 * CELL)
