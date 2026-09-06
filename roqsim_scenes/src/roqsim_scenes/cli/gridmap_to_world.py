"""Occupancy grid → roqsim world YAML + co-registered Nav2 map (map.pgm + map.yaml).

A stage-1 front-end for worlds that are **generated rather than authored**. The package's other
front-ends (``sdf_to_scene.py``, ``usd_to_scene.py``) import a world from a source *format* and hand
``scene.json`` to ``scene_to_mjcf.py`` for baking. This one has no source file and no meshes: its
input is a 2D occupancy grid, and every occupied cell becomes one parametric prop::

    grid (rows x cols of 0/1) --[ gridmap_to_world.py ]--> world.yaml + map.pgm + map.yaml

That makes it the front-end for procedurally generated obstacle fields — cellular automata, random
fills, maze generators, downsampled real maps — which are common in navigation benchmarking and which
the mesh pipeline serves badly: there is nothing to convert, pin, or hull, and baking 200 cylinders
into an MJCF would put them in the scene, hence in any map generated from it.

**Why props in the world YAML rather than a baked scene** (the same argument ``roqsim_assets.box`` makes,
and the reason the map is emitted here instead of by ``scene_to_map.py``): a baked scene is what an
occupancy grid gets *derived from*. Here the grid is the primary artifact — it is the thing the
generator produced and the thing the planner is given — so the map is written directly from it and is
co-registered with the world **by construction**, not by projection. There is no scan height to pick
and no slicing to get wrong.

Run::

    roqsim scenes gridmap-to-world --grid grid.npy --cell-size 0.15 \\
        --obstacle cylinder --radius 0.075 --height 0.5 \\
        --out-world worlds/w0.yaml --out-map maps/w0 --origin -4.5 0.0

Grid and frame convention (ONE convention, applied everywhere -- see the note at the bottom of this
docstring for why that is worth being firm about):

* ``grid[r][c] != 0`` means occupied. Row 0 is the **top** row of the image, as in a PGM.
* The map origin is the **bottom-left** corner of the grid in world metres, ROS style.
* Cell ``(r, c)`` therefore has its centre at
  ``x = origin_x + (c + 0.5) * cell_size``, ``y = origin_y + (rows - 1 - r + 0.5) * cell_size``.

**When the cells are walls rather than obstacles, this is the wrong tool.** ``gridmap_to_floorplan.py``
takes the same grid down the floorplan route instead, where a downsampled 40x40 building map is 7 wall
lines rather than 199 props — and comes back with real wall thickness, door openings and an optional
ceiling. The rule of thumb is what the cells *are*: a procedural obstacle field belongs here, as
parametric props a campaign can vary without re-baking a scene; architecture belongs there.

``--merge`` (box only) covers the occupied cells with as few rectangles as a greedy largest-first
pass manages. It is exact — the union is unchanged, so the world's footprint and the map still agree
— which makes it strictly unlike ``--shell-only`` below. It is refused for cylinders on purpose: the
gap between two tangent posts is the point of choosing a cylinder, and rectangles would fill it.

``--shell-only`` emits a prop only for occupied cells that have at least one non-occupied (or
out-of-grid) 8-neighbour. The interior of a solid blob is unreachable, so the props there are
invisible to the robot and to its sensors while still costing geoms and contacts every step; on a
dense grid this typically removes a third of them. **The map is not affected** — it always marks
every occupied cell, so the planner's picture stays complete. Off by default, because it makes the
world and the map differ in a way that surprises anyone who renders both.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

_OCC, _FREE = 0, 254
_DEFAULT_OBSTACLE_RGBA = [0.648, 0.192, 0.192, 1.0]


def load_grid(path: Path) -> np.ndarray:
    """Load an occupancy grid from ``.npy`` or whitespace/comma-separated text."""
    if path.suffix == ".npy":
        grid = np.load(path)
    else:
        text = path.read_text().replace(",", " ")
        grid = np.array(
            [[int(v) for v in line.split()] for line in text.splitlines() if line.strip()]
        )
    grid = np.asarray(grid)
    if grid.ndim != 2:
        raise ValueError(f"{path}: expected a 2D grid, got shape {grid.shape}")
    return (grid != 0).astype(np.uint8)


def cell_centre(r: int, c: int, rows: int, cell_size: float, origin: tuple[float, float]):
    """World-metre centre of grid cell (r, c). Row 0 is the top row; origin is the bottom-left."""
    x = origin[0] + (c + 0.5) * cell_size
    y = origin[1] + (rows - 1 - r + 0.5) * cell_size
    return x, y


def is_shell(grid: np.ndarray, r: int, c: int) -> bool:
    """True if the occupied cell (r, c) touches free space or the grid edge in its 8-neighbourhood."""
    rows, cols = grid.shape
    for rr in range(r - 1, r + 2):
        for cc in range(c - 1, c + 2):
            if rr == r and cc == c:
                continue
            if not (0 <= rr < rows and 0 <= cc < cols):
                return True
            if grid[rr][cc] == 0:
                return True
    return False


def _largest_rectangle(grid: np.ndarray) -> tuple[int, int, int, int]:
    """Largest all-occupied axis-aligned rectangle in `grid`, as (r0, c0, height, width).

    Standard largest-rectangle-in-histogram, run once per row over the column heights above it, so
    the whole grid is scanned in O(rows*cols) rather than by trying every corner pair.
    """
    rows, cols = grid.shape
    heights = np.zeros(cols + 1, dtype=int)  # trailing 0 flushes the stack at the end of each row
    best = (0, 0, 0, 0, 0)  # (area, r0, c0, h, w)
    for r in range(rows):
        heights[:cols] = np.where(grid[r], heights[:cols] + 1, 0)
        stack: list[int] = []
        for c in range(cols + 1):
            while stack and heights[stack[-1]] >= heights[c]:
                h = int(heights[stack.pop()])
                left = stack[-1] + 1 if stack else 0
                w = c - left
                if h and h * w > best[0]:
                    best = (h * w, r - h + 1, left, h, w)
            stack.append(c)
    _, r0, c0, h, w = best
    return r0, c0, h, w


def merge_rectangles(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Cover the occupied cells with as few rectangles as a greedy largest-first pass manages.

    The cover is **exact**: a rectangle of h x w cells is the same union as the h*w unit boxes it
    replaces, so the world's occupied footprint is unchanged and the map needs no adjustment. That is
    the difference from ``--shell-only``, which removes geometry the map still shows.

    What it buys depends entirely on whether the grid has structure. Measured on 40x40 grids: a
    random 35%-fill field goes 539 -> 321 rectangles (a random field has almost nothing colinear to
    merge), a maze 553 -> 91, a downsampled building map 199 -> 7. The last number is the grid saying
    it is architecture, not an obstacle field -- at that point ``gridmap_to_floorplan.py`` gives real
    walls, doors and a ceiling for the same input.
    """
    remaining = np.asarray(grid, dtype=bool).copy()
    rects: list[tuple[int, int, int, int]] = []
    while remaining.any():
        r0, c0, h, w = _largest_rectangle(remaining)
        remaining[r0 : r0 + h, c0 : c0 + w] = False
        rects.append((r0, c0, h, w))
    return rects


def obstacle_plugins(
    grid: np.ndarray,
    *,
    cell_size: float,
    origin: tuple[float, float],
    obstacle: str,
    radius: float,
    height: float,
    color,
    shell_only: bool,
    prefix: str,
    merge: bool = False,
) -> list[dict]:
    """One plugin entry per occupied cell (or per shell cell, or per merged rectangle).

    ``merge`` is a **box-only** option and this raises rather than quietly ignoring it for cylinders:
    a merged run of cylinders is not the same geometry. Two diagonally adjacent 0.15 m boxes touch at
    their corners while two tangent cylinders on that pitch leave a gap, which is the whole point of
    the cylinder for a clearance benchmark -- squaring a cylinder field off into rectangles would
    change what the experiment measures while looking like an optimisation.
    """
    if merge and obstacle != "box":
        raise ValueError(
            f"--merge is box-only, not {obstacle!r}: merging cylinders into rectangles replaces the "
            f"gaps between tangent posts with solid geometry, which changes the clearance the run "
            f"measures. Use --obstacle box to merge, or drop --merge to keep the posts."
        )

    rows, _ = grid.shape
    kept = np.asarray(grid, dtype=bool).copy()
    if shell_only:
        for r in range(rows):
            for c in range(grid.shape[1]):
                if kept[r][c] and not is_shell(grid, r, c):
                    kept[r][c] = False

    if merge:
        # (r0, c0) is the rectangle's top-left cell, so its centre sits half its extent down/right of
        # that cell's centre -- the row axis inverts on the way to world y, hence the minus.
        placements = []
        for r0, c0, h, w in merge_rectangles(kept):
            x, y = cell_centre(r0, c0, rows, cell_size, origin)
            placements.append(
                (
                    x + (w - 1) * cell_size / 2.0,
                    y - (h - 1) * cell_size / 2.0,
                    w * cell_size,
                    h * cell_size,
                )
            )
    else:
        placements = [
            (*cell_centre(r, c, rows, cell_size, origin), cell_size, cell_size)
            for r in range(rows)
            for c in range(grid.shape[1])
            if kept[r][c]
        ]

    entries: list[dict] = []
    for index, (x, y, size_x, size_y) in enumerate(placements):
        name = f"{prefix}{index}"
        cfg = {"prefix": f"{name}_", "pos": [round(x, 6), round(y, 6)]}
        if obstacle == "cylinder":
            cfg |= {"radius": radius, "height": height, "color": list(color)}
        else:
            cfg |= {"size": [round(size_x, 6), round(size_y, 6), height], "color": list(color)}
        # `name` is a SIBLING of the plugin ref, not one of its config keys: the label an entry
        # answers to is a property of the entry, and a world whose obstacles all carry the plugin's
        # default label is refused for duplicate labels. `prefix` stays in the config, because that
        # one really is the plugin's -- it is what keeps the generated MJCF names distinct.
        entries.append({obstacle: cfg, "name": name})
    return entries


def write_map(
    grid: np.ndarray, out_stem: Path, *, cell_size: float, origin: tuple[float, float]
) -> None:
    """Write ``<stem>.pgm`` + ``<stem>.yaml`` for ROS ``map_server``, co-registered with the world.

    Every occupied cell is marked, including any the world skipped under ``--shell-only``: the map is
    what the planner is given, and a planner that believes a solid blob is hollow will plan into it.
    """
    rows, cols = grid.shape
    out_stem.parent.mkdir(parents=True, exist_ok=True)

    pgm = out_stem.with_suffix(".pgm")
    pixels = np.where(grid != 0, _OCC, _FREE).astype(np.uint8)  # row 0 = top, as PGM expects
    with pgm.open("wb") as fh:
        fh.write(f"P5\n{cols} {rows}\n255\n".encode("ascii"))
        fh.write(pixels.tobytes())

    meta = {
        "image": pgm.name,
        "resolution": float(cell_size),
        "origin": [float(origin[0]), float(origin[1]), 0.0],
        "negate": 0,
        "occupied_thresh": 0.65,
        "free_thresh": 0.196,
    }
    out_stem.with_suffix(".yaml").write_text(yaml.safe_dump(meta, sort_keys=False))


def build_world(
    grid: np.ndarray,
    *,
    cell_size: float,
    origin: tuple[float, float],
    obstacle: str,
    radius: float,
    height: float,
    color,
    shell_only: bool,
    prefix: str,
    robot: str | None,
    start: tuple[float, float] | None,
    yaw: float,
    extra_plugins: list | None,
    merge: bool = False,
) -> dict:
    plugins: list[dict] = []
    if robot:
        position = (
            {"x": round(start[0], 6), "y": round(start[1], 6)}
            if start is not None
            else {"x": 0.0, "y": 0.0}
        )
        spawn = {
            "model": robot,
            "name": "robot",
            "pose": {"position": position, "orientation": {"yaw": yaw}},
        }
        plugins.append({"spawn_robot": spawn})
    plugins.extend(
        obstacle_plugins(
            grid,
            cell_size=cell_size,
            origin=origin,
            obstacle=obstacle,
            radius=radius,
            height=height,
            color=color,
            shell_only=shell_only,
            prefix=prefix,
            merge=merge,
        )
    )
    if extra_plugins:
        plugins.extend(extra_plugins)

    # No sim.world: the engine's built-in empty room supplies ground and lighting, which is all a
    # generated obstacle field needs -- there is no scene to bake and nothing to include.
    return {"sim": {"pacing": "realtime"}, "components": plugins}


def write_world(world: dict, path: Path, header: str | None = None) -> None:
    """Write the world YAML, with `header` as a leading comment block (YAML carries no comments)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = yaml.safe_dump(world, sort_keys=False, default_flow_style=None, width=120)
    if header:
        text = "".join(f"# {line}\n" for line in header.splitlines()) + "\n" + text
    path.write_text(text)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument(
        "--grid", type=Path, required=True, help=".npy or text occupancy grid (1 = occupied)"
    )
    p.add_argument("--cell-size", type=float, required=True, help="metres per cell")
    p.add_argument(
        "--origin",
        type=float,
        nargs=2,
        default=(0.0, 0.0),
        help="world metres of the grid's BOTTOM-LEFT corner (ROS map convention)",
    )
    p.add_argument("--obstacle", choices=("cylinder", "box"), default="cylinder")
    p.add_argument(
        "--radius", type=float, default=None, help="cylinder radius (default: half a cell)"
    )
    p.add_argument("--height", type=float, default=0.5, help="obstacle height in metres")
    p.add_argument("--color", type=float, nargs="+", default=_DEFAULT_OBSTACLE_RGBA)
    p.add_argument(
        "--shell-only",
        action="store_true",
        help="emit props only for occupied cells touching free space (map is unaffected)",
    )
    p.add_argument(
        "--merge",
        action="store_true",
        help="BOX ONLY: cover the occupied cells with as few rectangular boxes as "
        "possible instead of one box per cell. Exact -- same footprint, same map. "
        "Pays off on structured grids (a maze 553 cells -> 91 boxes) and barely at "
        "all on a random field (539 -> 321). Rejected for cylinders, whose gaps are "
        "the point.",
    )
    p.add_argument("--prefix", default="obs_", help="entity/MJCF name prefix for the obstacles")
    p.add_argument(
        "--robot", default=None, help="roqsim robot model to spawn, e.g. clearpath_jackal"
    )
    p.add_argument(
        "--start", type=float, nargs=2, default=None, help="robot start x y in world metres"
    )
    p.add_argument("--yaw", type=float, default=0.0, help="robot start yaw in radians")
    p.add_argument("--out-world", type=Path, required=True)
    p.add_argument("--out-map", type=Path, default=None, help="stem for <stem>.pgm + <stem>.yaml")
    p.add_argument(
        "--header", default=None, help="comment block written at the top of the world YAML"
    )
    args = p.parse_args(argv)

    grid = load_grid(args.grid)
    radius = args.radius if args.radius is not None else args.cell_size / 2.0

    try:
        world = build_world(
            grid,
            cell_size=args.cell_size,
            origin=tuple(args.origin),
            obstacle=args.obstacle,
            radius=radius,
            height=args.height,
            color=args.color,
            shell_only=args.shell_only,
            prefix=args.prefix,
            robot=args.robot,
            start=tuple(args.start) if args.start else None,
            yaw=args.yaw,
            extra_plugins=None,
            merge=args.merge,
        )
    except ValueError as exc:  # a rejected option combination: say what, not where
        raise SystemExit(str(exc)) from exc
    write_world(world, args.out_world, header=args.header)

    n_props = sum(1 for entry in world["components"] if {"cylinder", "box"} & set(entry))
    n_occ = int(grid.sum())
    if args.out_map:
        write_map(grid, args.out_map, cell_size=args.cell_size, origin=tuple(args.origin))

    rows, cols = grid.shape
    notes = [n for n, on in (("shell-only", args.shell_only), ("merged", args.merge)) if on]
    print(
        f"GRID_OK {args.grid.name}: {rows}x{cols} cells @ {args.cell_size} m "
        f"({rows * args.cell_size:.2f} x {cols * args.cell_size:.2f} m), "
        f"{n_occ} occupied -> {n_props} props" + (f" ({', '.join(notes)})" if notes else "")
    )
    # A grid that collapses this far is architecture, not an obstacle field, and the floorplan route
    # expresses it properly (walls with thickness, doors, a ceiling) instead of as a box heap.
    if args.merge and n_occ >= 50 and n_props <= n_occ / 8:
        print(
            f"  note: {n_occ} cells reduced to {n_props} boxes -- this grid looks like walls rather "
            f"than obstacles. gridmap_to_floorplan.py turns it into a floorplan instead."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
