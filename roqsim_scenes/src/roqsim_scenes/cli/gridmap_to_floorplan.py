"""Occupancy grid → floorplan JSON (axis-aligned wall segments in metres).

The **architecture** route for a grid, and the sibling of ``gridmap_to_world.py``. Same input, and
the choice between them is not cosmetic -- it is a question about what the grid *is*::

    grid --[ gridmap_to_world.py ]-----> world YAML (one prop per cell) + map.pgm/map.yaml
         --[ gridmap_to_floorplan.py ]-> floorplan.json --[ floorplan_to_world.py ]-> world + scene

Take the **world** route when the occupied cells are *obstacles*: a procedurally generated field,
scattered posts, a random fill. They stay parametric props declared in the world YAML, out of the
baked scene, which is what lets the map be written directly from the grid and what lets a campaign
vary the field as a factor without re-baking anything.

Take **this** route when the occupied cells are *architecture*: a downsampled building map, a maze,
anything whose walls are walls. One prop per cell is then the wrong representation no matter how few
props it merges into -- a 40x40 building map is 199 occupied cells and **7 wall lines**. What you get
for going through the floorplan is everything the cell field cannot express: walls with real
thickness, door openings with lintels, an optional ceiling, markers resolved to props, and a scene
that round-trips back to a floorplan through ``scene_to_floorplan.py``. The Nav2 map comes from
``scene_to_map.py`` afterwards, projected out of the baked scene rather than written from the grid.

Run::

    roqsim scenes gridmap-to-floorplan --grid map.npy --cell-size 0.15 \\
        --origin -4.5 0.0 --out floorplan.json

Grid and frame convention, identical to ``gridmap_to_world.py`` so a floorplan and a map built from
one grid land in one frame:

* ``grid[r][c] != 0`` means occupied. Row 0 is the **top** row, as in a PGM.
* ``--origin`` is the world-metre position of the grid's **bottom-left** corner, ROS style.

Two knobs, both defaulting to 3 cells and for different reasons:

* ``--min-length`` is the shortest wall to keep. Anything below it is dropped as speckle -- real
  features included, so keep it near the smallest wall you actually believe in.
* ``--gap`` bridges holes within one wall. It is **not** free to lower: claiming a wall also consumes
  one cell to each side across it, which bites 3 cells out of any wall crossing it, so a crossing
  wall needs a gap of 3 cells to re-join across that bite. Below that, the walls at every T-junction
  come out fragmented or not at all.

Reducing cells to lines is lossy in **both** directions, so the tool reports both and refuses on
either. *Kept* is the share of occupied cells that ended up on a wall. *Invented* is the share of the
emitted wall standing over cells that were free — the one that actually decides whether this was the
right tool, and the one a coverage number on its own hides: handed a field of scattered obstacles, a
greedy segmenter draws long plausible lines straight through it, scoring well on coverage while
walling off space the robot could drive through. A wall that was never there does not weaken the
world, it blocks it. High *invented* means the grid wanted ``gridmap_to_world.py``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from roqsim_scenes.grid_to_floorplan import segments, to_floorplan

from .gridmap_to_world import load_grid


def wall_cells(
    segs: list[tuple[str, int, int, int]], shape: tuple[int, int], spread: int = 0
) -> np.ndarray:
    """Mask of the cells the emitted walls occupy, optionally dilated by `spread` across each wall."""
    rows, cols = shape
    mask = np.zeros(shape, dtype=bool)
    for kind, fixed, lo, hi in segs:
        for d in range(-spread, spread + 1):
            f = fixed + d
            if kind == "h" and 0 <= f < rows:
                mask[f, lo : hi + 1] = True
            elif kind == "v" and 0 <= f < cols:
                mask[lo : hi + 1, f] = True
    return mask


def agreement(grid: np.ndarray, segs: list[tuple[str, int, int, int]]) -> tuple[float, float]:
    """How well the emitted walls reproduce the grid, as (kept, invented) fractions.

    Two numbers, because the reduction fails in two directions and only one of them is obvious:

    * **kept** -- occupied cells lying on a wall, within one cell across it (a source wall may be
      drawn 2-3 cells thick where the floorplan's is one line). Low means the grid was thrown away.
    * **invented** -- emitted wall cells standing on cells that were FREE, with no such tolerance.
      A line only ever gets there by bridging a hole, so on real walls this stays near zero (a door
      wider than ``--gap`` is not bridged at all, it splits the wall into two lines), while a field
      of scattered obstacles turns it into the dominant term.

    Measured on 40x40 grids, which is where the 15% default threshold comes from -- the gap between
    the two populations is wide and empty:

    ======================  ======  =========
    grid                      kept   invented
    ======================  ======  =========
    building map              100%         0%
    building map with doors   100%         0%
    recursive-division maze    99%         9%
    random fill, 15%           85%        51%
    random fill, 35%           95%        47%
    random fill, 50%          100%        42%
    ======================  ======  =========

    Note the 50%-fill row: it keeps *every* occupied cell and is still nonsense. That is why a
    coverage number on its own cannot make this call.
    """
    occupied = grid != 0
    emitted = wall_cells(segs, grid.shape)
    kept = (wall_cells(segs, grid.shape, spread=1) & occupied).sum() / max(1, occupied.sum())
    invented = (emitted & ~occupied).sum() / max(1, emitted.sum())
    return float(kept), float(invented)


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
    p.add_argument(
        "--min-length",
        type=float,
        default=None,
        help="shortest wall to keep, metres (default: 3 cells; below this is speckle)",
    )
    p.add_argument(
        "--gap",
        type=float,
        default=None,
        help="bridge holes up to this long within one wall, metres (default: 3 cells, the "
        "width a claimed wall bites out of anything crossing it -- lower it and "
        "T-junction walls fragment or vanish)",
    )
    p.add_argument(
        "--min-kept",
        type=float,
        default=0.5,
        help="fail if the walls account for less than this fraction of the occupied cells",
    )
    p.add_argument(
        "--max-invented",
        type=float,
        default=0.15,
        help="fail if more than this fraction of the emitted wall cells stand over free "
        "space: the grid holds obstacles rather than walls, and it wants "
        "gridmap_to_world.py",
    )
    p.add_argument("--out", type=Path, required=True, help="floorplan JSON to write")
    args = p.parse_args(argv)

    grid = load_grid(args.grid)
    rows, cols = grid.shape
    min_len = args.min_length if args.min_length is not None else 3 * args.cell_size
    min_cells = max(1, round(min_len / args.cell_size))
    gap_cells = max(
        0, round((args.gap if args.gap is not None else 3 * args.cell_size) / args.cell_size)
    )

    segs = segments(grid.astype(bool), min_cells, gap_cells)
    if not segs:
        raise SystemExit(
            f"no wall segments reached --min-length ({min_cells} cells): either the grid holds no "
            f"walls -- in which case this is a job for gridmap_to_world.py, which places a prop per "
            f"cell -- or --min-length is too long for it"
        )
    plan = to_floorplan(segs, rows, args.cell_size, origin=tuple(args.origin))

    occupied = int((grid != 0).sum())
    kept, invented = agreement(grid, segs)

    total = sum(
        abs(line["x1_m"] - line["x0_m"]) + abs(line["y1_m"] - line["y0_m"])
        for line in plan["lines"]
    )
    print(
        f"FLOORPLAN_OK {args.grid.name}: {rows}x{cols} cells @ {args.cell_size} m "
        f"({rows * args.cell_size:.2f} x {cols * args.cell_size:.2f} m), "
        f"{occupied} occupied -> {len(segs)} wall lines, {total:.1f} m of wall"
    )
    print(f"agreement: {kept:.0%} of the occupied cells kept, {invented:.0%} of the wall invented")
    # Checked BEFORE writing: a floorplan that misrepresents the grid must not exist on disk for a
    # later step to pick up as though it were the world.
    if invented > args.max_invented:
        raise SystemExit(
            f"{invented:.0%} of the emitted wall stands over free space (> --max-invented "
            f"{args.max_invented:.0%}). This grid does not reduce to walls -- a greedy segmenter "
            f"draws long lines through a field of scattered obstacles and walls off space the robot "
            f"could drive through, which blocks the world rather than merely thinning it. Use "
            f"gridmap_to_world.py: one prop per cell, nothing added and nothing lost."
        )
    if kept < args.min_kept:
        raise SystemExit(
            f"only {kept:.0%} of the occupied cells came through as walls (< --min-kept "
            f"{args.min_kept:.0%}); {occupied} occupied cells would mostly vanish from the world. "
            f"Either lower --min-length, or use gridmap_to_world.py."
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(plan, indent=2) + "\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
