# Copyright (C) 2026 Frederik Pasch
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""Fit a captured room's collision model from its splat: what the robot meets is what the camera sees.

A splat has no collision geometry of its own. This reads the world that places it (its
``sim.renderers.splat.splat`` block: the file, the pose, the scale), takes the opaque Gaussians in
the world's frame, and writes an MJCF of simple shapes:

- **the floor**, a plane at z = 0 (where ``roqsim splat frame`` put it);
- **four walls**, boxes just outside the footprint the room's Gaussians cover (1st to 99th
  percentile along x and y, which the frame tool laid along the walls);
- **the furniture the robot meets**: the Gaussians in a band of heights above the floor
  (``--band``, by default 0.08-0.40 m: above a wheeled robot's ground clearance, below a desk top
  it passes under) rasterised onto a grid, holes filled, each connected patch split until boxes
  cover it at least ``--fill`` -- and each box stands from the floor to ``--box-top``.

The boxes are what the robot collides with and what MuJoCo's renderer draws; the splat renderer
draws the capture instead, so ``roqsim render <world> --renderer mujoco`` beside the splat's picture
shows whether they agree.

    roqsim splat collision my_room.yaml --out my_room/collision.xml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from roqsim.config import load_config
from roqsim.plugin import PluginError
from roqsim.world import resolve_file_ref

from .ply import SplatFileError, read_splat
from .renderer import splat_placement


def world_points(world: str | Path, min_opacity: float = 0.5) -> tuple[np.ndarray, dict]:
    """The opaque Gaussians' centres of *world*'s splat, in the world's frame and metres.

    The world is loaded as a run loads it, so a splat it inherits through ``extends`` is the one
    the renderer draws, its file resolved beside *world*.
    """
    world = Path(world)
    config = load_config(world)
    block = config.renderers.get("splat", {}).get("splat")
    if not block:
        raise SplatFileError(f"{world} places no splat (sim.renderers.splat.splat)")
    path = resolve_file_ref(block["file"], config.base_dir, "splat file")
    splat = read_splat(path)
    opacity = 1.0 / (1.0 + np.exp(-splat.opacity_logits.astype(np.float64)))
    rot, trans, scale = splat_placement(block)
    pts = splat.means[opacity >= min_opacity].astype(np.float64)
    return pts @ rot.T * scale + trans, block


def _rectangles(mask: np.ndarray, fill: float, min_cells: int) -> list[tuple[int, int, int, int]]:
    """Axis-aligned rectangles (row0, row1, col0, col1; exclusive ends) that cover *mask*'s cells,
    each at least *fill* full, splitting a patch along its longer side until they are."""
    from scipy import ndimage  # noqa: PLC0415 -- only this tool needs it

    out = []
    stack = [mask]
    offsets = [(0, 0)]
    while stack:
        part, (r0, c0) = stack.pop(), offsets.pop()
        labels, n = ndimage.label(part, structure=np.ones((3, 3), bool))
        for index, sl in enumerate(ndimage.find_objects(labels), start=1):
            cells = labels[sl] == index
            if cells.sum() < min_cells:
                continue
            rows, cols = sl
            if cells.mean() >= fill or min(cells.shape) <= 2:
                out.append((r0 + rows.start, r0 + rows.stop, c0 + cols.start, c0 + cols.stop))
                continue
            if cells.shape[0] >= cells.shape[1]:
                cut = cells.shape[0] // 2
                halves = (
                    (cells[:cut], (rows.start, cols.start)),
                    (cells[cut:], (rows.start + cut, cols.start)),
                )
            else:
                cut = cells.shape[1] // 2
                halves = (
                    (cells[:, :cut], (rows.start, cols.start)),
                    (cells[:, cut:], (rows.start, cols.start + cut)),
                )
            for half, (hr, hc) in halves:
                stack.append(half)
                offsets.append((r0 + hr, c0 + hc))
    return sorted(out)


def fit(
    points: np.ndarray,
    band: tuple[float, float] = (0.08, 0.40),
    cell: float = 0.025,
    min_points: int = 3,
    fill: float = 0.55,
    min_area: float = 0.01,
    box_top: float = 0.45,
    wall_height: float = 2.5,
    wall_margin: float = 0.12,
) -> dict:
    """Floor, walls and furniture boxes for the room whose surfaces are *points* (world frame, m)."""
    from scipy import ndimage  # noqa: PLC0415

    upright = points[(points[:, 2] > 0.1) & (points[:, 2] < 2.0)]
    lo = np.percentile(upright[:, :2], 1, axis=0)
    hi = np.percentile(upright[:, :2], 99, axis=0)
    inner_lo, inner_hi = lo + wall_margin, hi - wall_margin
    sel = (
        (points[:, 2] >= band[0])
        & (points[:, 2] <= band[1])
        & np.all(points[:, :2] >= inner_lo, axis=1)
        & np.all(points[:, :2] <= inner_hi, axis=1)
    )
    pts = points[sel]
    shape = np.ceil((inner_hi - inner_lo) / cell).astype(int) + 1
    ix = np.clip(((pts[:, :2] - inner_lo) / cell).astype(int), 0, shape - 1)
    counts = np.zeros(shape, int)
    np.add.at(counts, (ix[:, 0], ix[:, 1]), 1)
    occupied = ndimage.binary_closing(counts >= min_points, iterations=1)
    occupied = ndimage.binary_fill_holes(occupied)
    min_cells = max(1, int(round(min_area / cell**2)))
    boxes = []
    for r0, r1, c0, c1 in _rectangles(occupied, fill, min_cells):
        x0, y0 = inner_lo + np.array([r0, c0]) * cell
        x1, y1 = inner_lo + np.array([r1, c1]) * cell
        boxes.append(
            {
                "centre": [(x0 + x1) / 2, (y0 + y1) / 2, box_top / 2],
                "half": [(x1 - x0) / 2, (y1 - y0) / 2, box_top / 2],
            }
        )
    t = 0.05
    cx, cy = (lo + hi) / 2
    hx, hy = (hi - lo) / 2
    walls = [
        {"centre": [cx, hi[1] + t, wall_height / 2], "half": [hx + 2 * t, t, wall_height / 2]},
        {"centre": [cx, lo[1] - t, wall_height / 2], "half": [hx + 2 * t, t, wall_height / 2]},
        {"centre": [hi[0] + t, cy, wall_height / 2], "half": [t, hy, wall_height / 2]},
        {"centre": [lo[0] - t, cy, wall_height / 2], "half": [t, hy, wall_height / 2]},
    ]
    return {
        "footprint": [lo.tolist(), hi.tolist()],
        "walls": walls,
        "boxes": boxes,
        "band": list(band),
    }


def mjcf(result: dict, name: str) -> str:
    """The fitted room as an MJCF of static geoms: what the robot collides with."""

    def fmt(v) -> str:
        return " ".join(f"{float(x):.4f}" for x in v)

    lines = [
        f'<mujoco model="{name}">',
        "  <!-- Fitted from the splat by `roqsim splat collision`: the robot collides with these;",
        "       the splat renderer draws the capture instead, MuJoCo's renderer draws them. -->",
        "  <worldbody>",
        '    <light pos="0 0 3" dir="0 0 -1" diffuse=".7 .7 .7"/>',
        '    <geom name="floor" type="plane" size="0 0 0.05" rgba=".5 .5 .5 1"/>',
    ]
    for i, wall in enumerate(result["walls"]):
        lines.append(
            f'    <geom name="wall_{i}" type="box" pos="{fmt(wall["centre"])}" size="{fmt(wall["half"])}" '
            'rgba=".8 .8 .75 1"/>'
        )
    for i, box in enumerate(result["boxes"]):
        lines.append(
            f'    <geom name="furniture_{i}" type="box" pos="{fmt(box["centre"])}" size="{fmt(box["half"])}" '
            'rgba=".55 .4 .25 1"/>'
        )
    lines += ["  </worldbody>", "</mujoco>", ""]
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="roqsim splat collision",
        description="Fit floor, walls and furniture boxes to the splat a world places, and write "
        "them as the MJCF the world collides with.",
    )
    parser.add_argument(
        "world", help="the world YAML whose sim.renderers.splat.splat places the capture"
    )
    parser.add_argument("--out", required=True, help="the MJCF to write (the world's sim.world)")
    parser.add_argument(
        "--band",
        nargs=2,
        type=float,
        default=(0.08, 0.40),
        metavar=("LOW", "HIGH"),
        help="the heights above the floor whose Gaussians are furniture the robot meets, in m",
    )
    parser.add_argument(
        "--box-top", type=float, default=0.45, help="how high a furniture box stands, in m"
    )
    parser.add_argument("--cell", type=float, default=0.025, help="the grid's cell, in m")
    parser.add_argument(
        "--fill", type=float, default=0.55, help="how full a box must be before it is kept"
    )
    parser.add_argument("--wall-height", type=float, default=2.5, help="the walls' height, in m")
    args = parser.parse_args(argv)
    try:
        points, _block = world_points(args.world)
    except (FileNotFoundError, SplatFileError, PluginError) as err:
        print(f"roqsim splat collision: {err}", file=sys.stderr)
        return 2
    result = fit(
        points,
        tuple(args.band),
        args.cell,
        fill=args.fill,
        box_top=args.box_top,
        wall_height=args.wall_height,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(mjcf(result, out.stem))
    (lo, hi) = result["footprint"]
    print(
        f"{out}: floor, 4 walls around {hi[0] - lo[0]:.2f} x {hi[1] - lo[1]:.2f} m, "
        f"{len(result['boxes'])} furniture box(es) from {len(points):,} opaque Gaussians"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
