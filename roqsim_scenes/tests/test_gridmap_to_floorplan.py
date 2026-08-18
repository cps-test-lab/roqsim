"""`gridmap_to_floorplan`: occupancy grid -> floorplan JSON, the architecture route for a grid.

The segmentation itself is tested in ``test_grid_to_floorplan``. What this tool adds is the *frame*
(shared with ``gridmap_to_world`` so one grid gives one coordinate system) and the *refusal* -- the
judgement about whether the grid held walls at all. That refusal is the reason the tool can be safe
to point at any grid: a greedy segmenter never declines on its own, it just draws lines.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from roqsim_scenes.cli import gridmap_to_floorplan as gmf

CELL = 0.15


def _room(n: int = 20) -> np.ndarray:
    grid = np.zeros((n, n), dtype=np.uint8)
    grid[0, :] = grid[-1, :] = grid[:, 0] = grid[:, -1] = 1
    return grid


def _scattered(n: int = 40, fill: float = 0.35) -> np.ndarray:
    return (np.random.default_rng(0).random((n, n)) < fill).astype(np.uint8)


def _run(grid, tmp_path, *extra):
    np.save(tmp_path / "g.npy", grid)
    out = tmp_path / "fp.json"
    gmf.main(
        ["--grid", str(tmp_path / "g.npy"), "--cell-size", str(CELL), "--out", str(out), *extra]
    )
    return json.loads(out.read_text())


# -- the frame ------------------------------------------------------------------------------------
def test_origin_matches_gridmap_to_world_so_one_grid_gives_one_frame(tmp_path):
    """--origin is the grid's bottom-left corner in world metres, the ROS map convention."""
    grid = _room()
    plan = _run(grid, tmp_path, "--origin", "-1.5", "2.0")
    xs = [v for ln in plan["lines"] for v in (ln["x0_m"], ln["x1_m"])]
    ys = [v for ln in plan["lines"] for v in (ln["y0_m"], ln["y1_m"])]
    assert min(xs) == pytest.approx(-1.5, abs=CELL)
    assert min(ys) == pytest.approx(2.0, abs=CELL)
    assert max(xs) - min(xs) == pytest.approx(grid.shape[1] * CELL, abs=CELL)


def test_a_four_wall_room_becomes_four_lines(tmp_path):
    plan = _run(_room(), tmp_path)
    assert len(plan["lines"]) == 4
    assert plan["doors"] == [] and plan["rooms"] == [] and plan["markers"] == []


# -- the refusal ----------------------------------------------------------------------------------
def test_agreement_separates_walls_from_an_obstacle_field():
    """The number that decides which of the two grid tools the input wanted."""
    from roqsim_scenes import grid_to_floorplan as g2f

    walls = _room()
    field = _scattered()
    _, invented_walls = gmf.agreement(walls, g2f.segments(walls.astype(bool), 3, 3))
    _, invented_field = gmf.agreement(field, g2f.segments(field.astype(bool), 3, 3))
    assert invented_walls < 0.05
    assert invented_field > 0.3  # long plausible lines drawn straight through scattered posts


def test_a_scattered_field_is_refused_and_writes_nothing(tmp_path):
    """A wall that was never there blocks the world; better no floorplan than a fictional one."""
    np.save(tmp_path / "g.npy", _scattered())
    out = tmp_path / "fp.json"
    with pytest.raises(SystemExit, match="stands over free space"):
        gmf.main(["--grid", str(tmp_path / "g.npy"), "--cell-size", str(CELL), "--out", str(out)])
    assert not out.exists()


def test_a_field_of_scattered_posts_keeps_a_high_coverage_score_anyway():
    """Why coverage alone cannot make the call: a dense field keeps nearly every cell and is still
    the wrong answer, because the lines that 'cover' it also wall off the space between."""
    from roqsim_scenes import grid_to_floorplan as g2f

    field = (np.random.default_rng(0).random((40, 40)) < 0.5).astype(np.uint8)
    kept, invented = gmf.agreement(field, g2f.segments(field.astype(bool), 3, 3))
    assert kept > 0.9
    assert invented > 0.3


def test_an_empty_grid_fails_loudly(tmp_path):
    np.save(tmp_path / "g.npy", np.zeros((10, 10), dtype=np.uint8))
    with pytest.raises(SystemExit, match="no wall segments"):
        gmf.main(
            [
                "--grid",
                str(tmp_path / "g.npy"),
                "--cell-size",
                str(CELL),
                "--out",
                str(tmp_path / "fp.json"),
            ]
        )


def test_the_refusal_can_be_overridden_deliberately(tmp_path):
    """--max-invented is a knob, not a wall: a caller who knows what the grid is may proceed."""
    plan = _run(_scattered(), tmp_path, "--max-invented", "1.0")
    assert plan["lines"]
