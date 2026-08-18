"""`mapimage_to_floorplan`: map screenshot -> floorplan JSON.

The grid -> floorplan half is tested in ``test_grid_to_floorplan``; what is left here is the part
that gets a picture as far as a grid, plus the debug overlay -- which is the tool's only correctness
check, so an overlay that silently disagrees with the plan it draws is the worst failure it has.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from roqsim_scenes.cli import mapimage_to_floorplan as m2f

SCALE = 20.0  # px per metre
CELL = 0.1


def _room_png(path: Path, h_px: int, w_px: int, wall_px: int = 4) -> None:
    """A white image with a black rectangular room border."""
    img = np.full((h_px, w_px, 3), 255, np.uint8)
    img[:wall_px, :] = img[-wall_px:, :] = img[:, :wall_px] = img[:, -wall_px:] = 0
    Image.fromarray(img).save(path)


def _trace(path: Path):
    mask = m2f.load_mask(path, None, thresh=110, drops=[])
    pts, _ = m2f.derotate(mask, max_deg=1.0)
    grid = m2f.rasterize(pts, SCALE, CELL)
    segs = m2f.segments(grid, min_cells=4, gap=3)
    return grid, m2f.to_floorplan(segs, grid.shape[0], CELL)


def test_load_mask_selects_dark_pixels_and_honours_drops(tmp_path):
    png = tmp_path / "r.png"
    _room_png(png, 40, 60)
    assert m2f.load_mask(png, None, 110, []).sum() > 0
    blanked = m2f.load_mask(png, None, 110, [(0, 0, 60, 40)])  # blank the whole image
    assert blanked.sum() == 0


def test_a_wide_room_traces_to_its_true_metric_extent(tmp_path):
    """10 x 30 m room: the plan must come out 10 m tall, not pushed up the y axis by its width."""
    png = tmp_path / "wide.png"
    _room_png(png, 200, 600)  # 10 m x 30 m at SCALE
    _, plan = _trace(png)
    xs = [v for ln in plan["lines"] for v in (ln["x0_m"], ln["x1_m"])]
    ys = [v for ln in plan["lines"] for v in (ln["y0_m"], ln["y1_m"])]
    assert max(xs) - min(xs) == pytest.approx(30.0, abs=0.3)
    assert max(ys) - min(ys) == pytest.approx(10.0, abs=0.3)
    assert min(ys) < 0.5  # the plan starts at the grid's bottom, not somewhere up the axis


def test_the_overlay_draws_every_wall_the_plan_holds(tmp_path):
    """The overlay is the mandated human check, so it must share the plan's frame.

    It renders in the rasterised grid's own height; when `to_floorplan` derived its row count from
    the segments instead, a non-square image put the horizontal walls on row 0 and clipped the
    vertical ones out of the picture entirely -- an overlay that looked plausible and showed a
    different world from the one being written.
    """
    png = tmp_path / "wide.png"
    _room_png(png, 200, 600)
    grid, plan = _trace(png)
    out = tmp_path / "overlay.png"
    m2f.write_overlay(plan, grid, CELL, out)
    img = np.asarray(Image.open(out))
    horizontal = int((img[..., 0] > 200).sum())  # red
    vertical = int((img[..., 2] > 200).sum())  # blue
    assert horizontal > 0 and vertical > 0
    drawn = horizontal + vertical
    expected = (
        sum(abs(ln["x1_m"] - ln["x0_m"]) + abs(ln["y1_m"] - ln["y0_m"]) for ln in plan["lines"])
        / CELL
        * 9
    )  # the overlay is upscaled 3x in both axes
    assert drawn == pytest.approx(expected, rel=0.15)


def test_a_blank_image_fails_loudly_rather_than_emitting_an_empty_plan(tmp_path):
    png = tmp_path / "blank.png"
    Image.fromarray(np.full((40, 40, 3), 255, np.uint8)).save(png)
    with pytest.raises(SystemExit, match="no wall pixels"):
        m2f.derotate(m2f.load_mask(png, None, 110, []), max_deg=1.0)
