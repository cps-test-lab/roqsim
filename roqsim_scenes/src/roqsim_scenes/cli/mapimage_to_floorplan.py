"""Map screenshot → floorplan JSON (axis-aligned wall segments in metres).

A stage-0 front-end for the world nobody published. Papers routinely show their environment only as
a *picture* — an RViz occupancy grid, a Gazebo top view, a hand-drawn plan — and ship no SDF, no
`.pgm`, and no dimensions. This tool turns such a picture into the same `floorplan.json` that
``roqsim_scene_builder``'s human sketch window returns, so the deterministic generator downstream is
unchanged::

    map screenshot --[ mapimage_to_floorplan.py ]--> floorplan.json --[ floorplan_to_world.py ]--> world + scene
                                                                    --[ scene_to_map.py ]-------> Nav2 map.pgm/yaml

What it is NOT: a substitute for a published world. Everything it produces is a *measurement of a
figure*, and the measurement's accuracy is bounded by the figure's resolution — which is why the
scale is a required argument (there is no way to guess it) and why it always writes a debug overlay:
the trace has to be checked against the source by eye before anything is built on it.

Run::

    roqsim scenes mapimage-to-floorplan --image fig3.png --scale 29 \\
        --crop 21 48 482 535 --out floorplan.json --overlay trace.png

``--scale`` is **pixels per metre**, and getting it right is the whole ball game. Ways to obtain it,
best first:

1. A grid in the screenshot with a known cell size (RViz's default grid is 1 m; Gazebo's ground plane
   grid is 1 m). Measure its period in pixels — ``--report-grid`` prints the dominant period so the
   number is measured, not eyeballed.
2. A published map ``resolution`` (m/cell) plus the screenshot's zoom, if both are known.
3. A metric quantity visible in the same figure — a robot of known width, a corridor of stated width.

Cross-check the result against something independent of the image before trusting it (e.g. a path
length or arena size the paper states in text). ``--report-grid`` and the printed extent exist for
exactly that arithmetic.

How the trace works, and where it can lie to you:

* **Threshold.** ``--thresh`` selects *dark* pixels as walls. On an RViz costmap screenshot the walls
  are the near-black occupied cells; the cyan/pink halos are the inflation layer and must NOT be
  traced (they are a costmap parameter, not geometry). Check the overlay: inflation caught by too high
  a threshold shows up as walls that are far too thick.
* **De-rotation.** A screenshot is rarely axis-aligned to the world. The angle that maximises the
  sharpness of the row/column projections is found by search (``--max-rotation``) and applied before
  segmentation, so the emitted walls are axis-aligned. A world that genuinely has diagonal walls is
  the wrong input for this tool.
* **Greedy segmentation.** The longest run of occupied cells in any row or column becomes a wall; its
  cells are consumed; repeat until nothing reaches ``--min-length``. Long walls therefore win over
  short ones at junctions, which is what makes an L-shape come out as two segments instead of four
  fragments. Speckle below ``--min-length`` is dropped — SLAM noise, and also any real feature
  smaller than that, so keep it near the smallest wall you actually believe in.
* **``--gap`` is coupled to ``--cell``, not free.** Claiming a wall also consumes one cell to each
  side across it (a traced wall is 2-3 cells thick, and leaving the neighbours would re-emit the same
  wall shifted by one), which bites 3 cells out of any wall crossing it. ``--gap`` must therefore
  cover 3 cells for a crossing wall to re-join across that bite; the defaults sit exactly on that
  boundary (0.15 m at a 0.05 m cell). Coarsen the cell or shrink the gap and the walls at every
  T-junction do not come out shortened — they do not come out at all.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

try:
    from PIL import Image
except ImportError as exc:  # pragma: no cover - environment problem, not logic
    raise SystemExit("mapimage_to_floorplan needs Pillow: pip install pillow") from exc

# Everything from the rasterised grid onwards is shared with gridmap_to_floorplan: this module is
# the part that gets a picture as far as a grid, and nothing else.
from roqsim_scenes.grid_to_floorplan import segments, to_floorplan


def load_mask(
    path: Path,
    crop: tuple[int, int, int, int] | None,
    thresh: int,
    drops: list[tuple[int, int, int, int]],
) -> np.ndarray:
    """Boolean wall mask (True = wall) from the image's dark pixels."""
    rgb = np.asarray(Image.open(path).convert("RGB")).astype(int)
    if crop:
        x0, y0, x1, y1 = crop
        rgb = rgb[y0:y1, x0:x1]
    for dx0, dy0, dx1, dy1 in drops:
        rgb[dy0:dy1, dx0:dx1] = 255
    return rgb.mean(axis=2) < thresh


def report_grid(path: Path, crop, thresh_lo: int = 170, thresh_hi: int = 235) -> None:
    """Print the dominant period of the image's *grey grid lines*, in pixels.

    A grid line is grey (r≈g≈b) and mid-bright, which separates it from both the dark walls and the
    white free space. The printed period is the candidate `--scale` when the grid is 1 m.
    """
    rgb = np.asarray(Image.open(path).convert("RGB")).astype(int)
    if crop:
        x0, y0, x1, y1 = crop
        rgb = rgb[y0:y1, x0:x1]
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    bright = rgb.mean(axis=2)
    grey = (abs(r - g) < 8) & (abs(g - b) < 8) & (bright > thresh_lo) & (bright < thresh_hi)
    for axis, label in ((0, "columns"), (1, "rows")):
        prof = grey.sum(axis=axis).astype(float)
        idx = np.where(prof > prof.max() * 0.45)[0]
        groups: list[list[int]] = []
        for i in idx:
            if groups and i - groups[-1][-1] <= 2:
                groups[-1].append(int(i))
            else:
                groups.append([int(i)])
        centres = [float(np.mean(gr)) for gr in groups]
        gaps = np.diff(centres) if len(centres) > 1 else np.array([])
        # Lines are often missed, so a gap may be an integer multiple of the true period: fold each
        # gap down by its nearest multiple of the smallest gap before averaging.
        period = None
        if gaps.size:
            base = float(np.min(gaps))
            folded = [g / max(1, round(g / base)) for g in gaps]
            period = float(np.median(folded))
        print(
            f"grid {label}: {len(centres)} lines, gaps {list(np.round(gaps, 1))}, period ~{period}"
        )


def derotate(mask: np.ndarray, max_deg: float, step_deg: float = 0.1) -> tuple[np.ndarray, float]:
    """Rotate the wall points so walls are axis-aligned; return (points, angle_deg).

    Score = sum of squared projection counts on both axes. Aligned walls concentrate into few rows
    and columns, which maximises that sum; a rotated wall smears across many.
    """
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        raise SystemExit("no wall pixels found — check --thresh and --crop")
    pts = np.stack([xs, ys]).astype(float)
    cx, cy = pts[0].mean(), pts[1].mean()
    pts -= np.array([[cx], [cy]])
    best, best_a = -1.0, 0.0
    for a in np.arange(-max_deg, max_deg + 1e-9, step_deg):
        t = math.radians(a)
        rot = np.array([[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]]) @ pts
        score = 0.0
        for axis in (0, 1):
            hist = np.bincount(np.round(rot[axis] - rot[axis].min()).astype(int))
            score += float((hist.astype(float) ** 2).sum())
        if score > best:
            best, best_a = score, float(a)
    t = math.radians(best_a)
    rot = np.array([[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]]) @ pts
    return rot, best_a


def rasterize(pts: np.ndarray, scale_px_m: float, cell_m: float) -> np.ndarray:
    """Bin rotated wall points into an occupancy grid of `cell_m` cells."""
    px_per_cell = scale_px_m * cell_m
    cols = np.floor((pts[0] - pts[0].min()) / px_per_cell).astype(int)
    rows = np.floor((pts[1] - pts[1].min()) / px_per_cell).astype(int)
    grid = np.zeros((rows.max() + 1, cols.max() + 1), dtype=bool)
    grid[rows, cols] = True
    return grid


def write_overlay(plan: dict, grid: np.ndarray, cell_m: float, out: Path) -> None:
    """Render the traced segments over the rasterised mask so a human can check the trace."""
    h, w = grid.shape
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[grid] = (90, 90, 90)  # what was traced FROM, in grey
    rows = h
    for line in plan["lines"]:
        x0 = int(line["x0_m"] / cell_m)
        x1 = int(line["x1_m"] / cell_m)
        y0 = rows - 1 - int(line["y0_m"] / cell_m)
        y1 = rows - 1 - int(line["y1_m"] / cell_m)
        if y0 == y1:
            img[max(0, min(y0, h - 1)), max(0, x0) : min(w, x1), :] = (255, 60, 60)
        else:
            lo, hi = sorted((y0, y1))
            img[max(0, lo) : min(h, hi + 1), max(0, min(x0, w - 1)), :] = (60, 160, 255)
    Image.fromarray(img).resize((w * 3, h * 3), Image.NEAREST).save(out)


def main(argv: list | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--image", type=Path, required=True, help="map screenshot")
    ap.add_argument("--scale", type=float, help="PIXELS PER METRE (required unless --report-grid)")
    ap.add_argument("--crop", type=int, nargs=4, metavar=("X0", "Y0", "X1", "Y1"))
    ap.add_argument(
        "--drop",
        type=int,
        nargs=4,
        action="append",
        default=[],
        metavar=("X0", "Y0", "X1", "Y1"),
        help="blank this rectangle before tracing (robot marker, legend, laser speckle)",
    )
    ap.add_argument("--thresh", type=int, default=110, help="wall = pixel brightness below this")
    ap.add_argument("--cell", type=float, default=0.05, help="grid cell size in metres")
    ap.add_argument(
        "--min-length",
        type=float,
        default=0.4,
        help="shortest wall to keep, metres (below this is treated as speckle)",
    )
    ap.add_argument(
        "--gap",
        type=float,
        default=0.15,
        help="bridge holes up to this long within one wall, metres",
    )
    ap.add_argument(
        "--max-rotation",
        type=float,
        default=5.0,
        help="search +/- this many degrees for the de-rotation angle",
    )
    ap.add_argument(
        "--report-grid",
        action="store_true",
        help="print the grey grid-line period (candidate --scale) and exit",
    )
    ap.add_argument("--out", type=Path, help="floorplan JSON to write")
    ap.add_argument("--overlay", type=Path, help="debug overlay PNG (strongly recommended)")
    args = ap.parse_args(argv)

    crop = tuple(args.crop) if args.crop else None
    if args.report_grid:
        report_grid(args.image, crop)
        return 0
    if args.scale is None or args.scale <= 0:
        raise SystemExit("--scale (pixels per metre) is required; use --report-grid to measure it")
    if not args.out:
        raise SystemExit("--out is required")

    mask = load_mask(args.image, crop, args.thresh, [tuple(d) for d in args.drop])
    pts, angle = derotate(mask, args.max_rotation)
    grid = rasterize(pts, args.scale, args.cell)
    segs = segments(
        grid, max(1, round(args.min_length / args.cell)), max(0, round(args.gap / args.cell))
    )
    if not segs:
        raise SystemExit("no wall segments survived --min-length; loosen it or check --scale")
    plan = to_floorplan(segs, grid.shape[0], args.cell)

    xs = [v for line in plan["lines"] for v in (line["x0_m"], line["x1_m"])]
    ys = [v for line in plan["lines"] for v in (line["y0_m"], line["y1_m"])]
    print(f"de-rotated by {angle:+.1f} deg")
    print(
        f"{mask.sum()} wall px -> {grid.sum()} cells @ {args.cell} m -> {len(segs)} wall segments"
    )
    print(f"extent: {max(xs) - min(xs):.2f} x {max(ys) - min(ys):.2f} m")
    total = sum(
        abs(line["x1_m"] - line["x0_m"]) + abs(line["y1_m"] - line["y0_m"])
        for line in plan["lines"]
    )
    print(f"total wall length: {total:.1f} m")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(plan, indent=2) + "\n")
    print(f"wrote {args.out}")
    if args.overlay:
        write_overlay(plan, grid, args.cell, args.overlay)
        print(f"wrote {args.overlay} — CHECK IT against the source figure before building on it")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
