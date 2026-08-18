"""Floorplan sketch core, all without a display: independent wall lines with stable ids, snapping /
splitting / crossings that keep walls sharing points, freehand -> immediate lines, doors attached to
lines, and the result shape."""

from __future__ import annotations

import json

import pytest
from roqsim_scene_builder.floorplan_window import (
    SketchModel,
    axis_snap,
    clamp_to_room,
    detect_rooms,
    door_fits,
    door_geom,
    door_interval,
    erase_line_at,
    hit_line_end,
    lineify,
    load_sketch,
    m_to_px,
    place_door,
    px_to_m,
    rooms_with_names,
    segment_intersection,
    write_result,
)


def _rect(m: SketchModel, x0, y0, x1, y1) -> None:
    """Add the four walls of an axis-aligned rectangle, sharing corner points."""
    for a, b in (
        ((x0, y0), (x1, y0)),
        ((x1, y0), (x1, y1)),
        ((x1, y1), (x0, y1)),
        ((x0, y1), (x0, y0)),
    ):
        m.add_line(*a, *b)


# -- ids: stable, split mints a new one, delete keeps the rest --


def test_line_ids_are_stable_and_split_mints_a_new_id():
    m = SketchModel()
    m.add_line(0, 0, 4, 0)  # id 1
    m.add_line(4, 0, 4, 3)  # id 2
    m.add_line(4, 3, 0, 3)  # id 3
    m.snap_or_split(2.0, 0.05, tol_m=0.3)  # split line 1's interior
    ids = [line.id for line in m.lines]
    assert {1, 2, 3, 4} == set(ids) and len(ids) == 4  # existing ids kept, one new id minted
    first_half = next(line for line in m.lines if line.id == 1)
    assert first_half.x1_m == 2.0  # line 1 keeps its id, now ending at the split point


def test_duplicate_wall_between_same_points_is_dropped():
    m = SketchModel()
    a = m.add_line(0, 0, 4, 0)  # id 1
    again = m.add_line(0, 0, 4, 0)  # same two points -> the same line, no new one
    assert again is a and len(m.lines) == 1
    rev = m.add_line(4, 0, 0, 0)  # reversed direction is the same wall
    assert rev is a and len(m.lines) == 1
    m.add_wall(0, 0, 4, 0, tol_m=0.3)  # the interactive path drops it too
    assert len(m.lines) == 1


def test_prune_removes_collapsed_edge():
    m = SketchModel()
    m.add_line(0, 0, 4, 0)  # id 1  A-B
    m.add_line(4, 0, 4, 3)  # id 2  B-C
    m.line_by_id(1).set_endpoint(0, 4, 0)  # drag A onto B -> line 1 is zero-length
    m.prune_lines()
    assert [ln.id for ln in m.lines] == [2]  # the collapsed wall is dropped


def test_prune_merges_reversed_duplicate_and_flips_door_t():
    m = SketchModel()
    m.add_line(0, 0, 4, 0)  # id 1  P-X
    m.add_line(4, 0, 6, 0)  # id 2  X-Q
    m.add_door(2, 0.25)
    m.line_by_id(1).set_endpoint(0, 6, 0)  # drag P onto Q -> line 1 == line 2 reversed
    m.prune_lines()
    assert [ln.id for ln in m.lines] == [1]  # earliest wall kept
    assert [(d.line_id, round(d.t, 2)) for d in m.doors] == [(1, 0.75)]  # door moved, t flipped


def test_delete_line_keeps_other_ids_and_takes_its_doors():
    m = SketchModel()
    m.add_line(0, 0, 4, 0)  # id 1
    m.add_line(4, 0, 4, 3)  # id 2
    m.add_line(4, 3, 0, 3)  # id 3
    m.add_door(1, 0.5)
    m.add_door(3, 0.5)
    m.delete_line(1)
    assert [line.id for line in m.lines] == [2, 3]  # NOT renumbered
    assert [(d.id, d.line_id) for d in m.doors] == [(1, 3)]  # door on line 1 gone; line 3's kept


# -- snap / split --


def test_snap_or_split_reuses_a_nearby_endpoint():
    m = SketchModel()
    m.add_line(0.0, 0.0, 4.0, 0.0)
    assert m.snap_or_split(4.15, 0.1, tol_m=0.3) == (4.0, 0.0)  # snap to the (4,0) endpoint
    assert len(m.lines) == 1  # no split, no new point


def test_snap_or_split_splits_a_nearby_line_interior():
    m = SketchModel()
    m.add_line(0.0, 0.0, 4.0, 0.0)
    assert m.snap_or_split(2.0, 0.1, tol_m=0.3) == (2.0, 0.0)
    assert len(m.lines) == 2
    assert m.lines[0].id == 1 and m.lines[1].id == 2  # first half keeps id 1, new half id 2


def test_snap_or_split_excludes_the_dragged_line():
    m = SketchModel()
    dragged = m.add_line(0.0, 0.0, 4.0, 0.0)
    assert m.snap_or_split(2.0, 0.05, tol_m=0.3, exclude=dragged) == (2.0, 0.05)
    assert len(m.lines) == 1


# -- crossings --


def test_segment_intersection_interior_only():
    assert segment_intersection((0, 0), (4, 0), (2, -1), (2, 1)) == pytest.approx((2.0, 0.0))
    assert segment_intersection((0, 0), (4, 0), (0, 1), (4, 1)) is None  # parallel
    assert (
        segment_intersection((0, 0), (4, 0), (4, 0), (4, 4)) is None
    )  # shared endpoint, not a cross


def test_add_wall_splits_a_crossed_line_at_a_shared_point():
    m = SketchModel()
    m.add_line(0.0, 0.0, 4.0, 0.0)  # a horizontal wall, id 1
    m.add_wall(2.0, -2.0, 2.0, 2.0, tol_m=0.2)  # a vertical wall crossing it at (2,0)
    # the horizontal wall is split at (2,0); every wall meeting there uses that exact point.
    xs = sorted(
        {round(line.x1_m, 3) for line in m.lines} | {round(line.x0_m, 3) for line in m.lines}
    )
    assert 2.0 in xs
    assert all(
        _touches(line, (2.0, 0.0)) for line in m.lines if _spans_x(line, 2.0) or _spans_y(line, 0.0)
    )


def _touches(line, p):
    return line.endpoint(0) == pytest.approx(p) or line.endpoint(1) == pytest.approx(p)


def _spans_x(line, x):
    return min(line.x0_m, line.x1_m) < x < max(line.x0_m, line.x1_m)


def _spans_y(line, y):
    return min(line.y0_m, line.y1_m) < y < max(line.y0_m, line.y1_m)


# -- freehand -> immediate lines --


def test_add_freehand_makes_straight_lines_and_forgets_the_pencil():
    m = SketchModel()
    # a wobbly L: a horizontal run then a vertical run, ~2 cm noise.
    path = [(x * 0.2, 0.02 * (-1) ** i) for i, x in enumerate(range(21))]  # 0..4 m, wobble
    path += [(4.0, y * 0.2) for y in range(1, 16)]  # up to 3 m
    made = m.add_freehand(path, tol_m=0.2, eps_m=0.15)
    assert len(made) == 2  # collapsed to two straight walls
    assert [line.id for line in m.lines] == [1, 2]
    assert m.lines[0].endpoint(1) == m.lines[1].endpoint(0)  # they share the corner point


def test_add_freehand_closes_a_loop_by_reusing_the_start():
    m = SketchModel()
    # a rectangle drawn freehand whose last point lands back near the first.
    path = (
        [(x * 0.5, 0.0) for x in range(9)]
        + [(4.0, y * 0.5) for y in range(1, 7)]
        + [(4.0 - x * 0.5, 3.0) for x in range(1, 9)]
        + [(0.0, 3.0 - y * 0.5) for y in range(1, 7)]
    )
    m.add_freehand(path, tol_m=0.3, eps_m=0.1)
    assert len(m.lines) == 4  # four walls
    # the loop is closed: the last wall's end reuses the first wall's start point exactly.
    assert m.lines[-1].endpoint(1) == pytest.approx(m.lines[0].endpoint(0))


# -- rooms (closed loops) --


def test_detect_rooms_finds_one_rectangle():
    m = SketchModel()
    _rect(m, 0, 0, 4, 3)
    rooms = detect_rooms(m.lines)
    assert len(rooms) == 1
    assert set(rooms[0]["line_ids"]) == {1, 2, 3, 4}


def test_detect_rooms_ignores_an_open_shape():
    m = SketchModel()
    m.add_line(0, 0, 4, 0)
    m.add_line(4, 0, 4, 3)  # an L, not closed
    assert detect_rooms(m.lines) == []


def test_detect_rooms_two_rooms_sharing_a_wall():
    # Draw the outer box, then a middle wall that crosses it -> two rooms.
    m = SketchModel()
    _rect(m, 0, 0, 4, 2)
    m.add_wall(2, 0, 2, 2, tol_m=0.1)  # splits top & bottom walls, makes two rooms
    rooms = detect_rooms(m.lines)
    assert len(rooms) == 2


def test_rooms_with_names_uses_stored_name():
    m = SketchModel()
    _rect(m, 0, 0, 4, 3)
    m.room_names[frozenset({1, 2, 3, 4})] = "Kitchen"
    rooms = rooms_with_names(m.lines, m.room_names)
    assert rooms[0]["id"] == 1 and rooms[0]["name"] == "Kitchen"


# -- geometry helpers --


def test_px_metre_roundtrip():
    for px, py in ((0, 0), (400, 300), (799, 599)):
        x_m, y_m = px_to_m(px, py, 800, 600, 8.0, 6.0)
        assert m_to_px(x_m, y_m, 800, 600, 8.0, 6.0) == pytest.approx((px, py))


def test_clamp_to_room_keeps_dot_inside():
    assert clamp_to_room(-1.0, 3.0, 8.0, 6.0) == (0.0, 3.0)
    assert clamp_to_room(9.5, 7.2, 8.0, 6.0) == (8.0, 6.0)


def test_axis_snap_levels_and_leaves_diagonals():
    assert axis_snap([(0.0, 0.0), (4.0, 0.2)], 10.0) == [(0.0, 0.0), (4.0, 0.0)]
    assert axis_snap([(0.0, 0.0), (3.0, 3.0)], 10.0) == [(0.0, 0.0), (3.0, 3.0)]


def test_lineify_collapses_noise_keeps_corners():
    pts = [(x * 0.2, 0.02 * (-1) ** i) for i, x in enumerate(range(26))]
    assert lineify(pts, 0.1) == [pts[0], pts[-1]]
    corner = [(0.0, 0.0), (2.0, 0.0), (2.0, 2.0)]
    assert lineify(corner, 0.05) == corner


# -- doors --


def test_place_door_and_geom():
    segs = [((0.0, 0.0), (4.0, 0.0))]
    assert place_door(segs, 2.0, 0.1, tol_m=0.3, width_m=0.9) == (0, 0.5)
    assert place_door([((0.0, 0.0), (0.5, 0.0))], 0.25, 0.05, tol_m=0.3, width_m=0.9) is None
    assert door_geom(((0.0, 0.0), (4.0, 0.0)), 0.5, 0.9) == (2.0, 0.0, 0.0)
    cx, cy, _ = door_geom(((0.0, 0.0), (4.0, 0.0)), 0.0, 0.9)
    assert (cx, cy) == (0.45, 0.0)  # clamped so the opening fits


def test_door_interval_clamps_to_wall():
    assert door_interval(0.5, 4.0, 0.9) == pytest.approx((1.55, 2.45))
    assert door_interval(0.0, 4.0, 0.9) == pytest.approx((0.0, 0.9))  # clamped to the near end
    assert door_interval(1.0, 4.0, 0.9) == pytest.approx((3.1, 4.0))  # clamped to the far end


def test_door_fits_rejects_overlap_allows_abut():
    length = 6.0
    # a door centred at t=0.25 occupies [1.05, 1.95] on a 6 m wall
    assert door_fits([], length, 0.25, 0.9) is True
    assert door_fits([0.25], length, 0.30, 0.9) is False  # 0.30 -> [1.35,2.25] overlaps [1.05,1.95]
    assert door_fits([0.25], length, 0.55, 0.9) is True  # 0.55 -> [2.85,3.75] clears it
    # exactly abutting is allowed (touching edges, no overlap)
    assert door_fits([0.15], length, 0.15 + 0.9 / length, 0.9) is True


def test_doors_roundtrip_and_cascade():
    m = SketchModel()
    m.add_line(0, 0, 4, 0)
    m.add_door(1, 0.5)
    result = write_result(None, "", m)
    assert result["doors"] == [{"id": 1, "line_id": 1, "t": 0.5, "width_m": 0.9}]
    model, _ = load_sketch({"lines": result["lines"], "doors": result["doors"]})
    assert model.doors[0].line_id == 1
    model.delete_line(1)
    assert model.doors == []  # the door died with its wall


# -- erase / hit --


def test_erase_line_at():
    m = SketchModel()
    m.add_line(0.0, 0.0, 4.0, 0.0)
    m.add_line(0.0, 2.0, 4.0, 2.0)
    out = erase_line_at(m.lines, 2.0, 0.05, tol_m=0.2)
    assert [line.id for line in out] == [2]  # ids NOT renumbered on erase
    assert erase_line_at(m.lines, 2.0, 1.0, tol_m=0.2) is None


def test_hit_line_end():
    m = SketchModel()
    m.add_line(1.0, 1.0, 5.0, 5.0)
    _, end = hit_line_end(m, 1.1, 1.05, tol_m=0.3)
    assert end == 0
    assert hit_line_end(m, 8.0, 2.0, tol_m=0.3) is None


# -- seeding + result --


def test_load_sketch_preserves_line_ids_and_markers():
    initial = {
        "lines": [{"id": 7, "x0_m": 0, "y0_m": 0, "x1_m": 4, "y1_m": 0}],
        "markers": [{"id": 1, "x_m": 2.5, "y_m": 3.0, "comment": "office table"}],
    }
    model, comment = load_sketch(initial)
    assert model.lines[0].id == 7  # stable id preserved from the seed
    assert (model.markers[0].x_m, model.markers[0].comment) == (2.5, "office table")
    assert comment == ""  # marker comments load, but not the top-level one


def test_marker_yaw_roundtrips_only_when_set():
    # a dragged heading survives load->write; a headingless marker omits yaw_deg entirely
    initial = {
        "lines": [{"id": 1, "x0_m": 0, "y0_m": 0, "x1_m": 4, "y1_m": 0}],
        "markers": [
            {"id": 1, "x_m": 1.0, "y_m": 1.0, "comment": "bed", "yaw_deg": 180},
            {"id": 2, "x_m": 2.0, "y_m": 1.0, "comment": "plant"},
        ],
    }
    model, _ = load_sketch(initial)
    assert model.markers[0].yaw_deg == 180.0
    assert model.markers[1].yaw_deg is None
    result = write_result(None, "", model)
    assert result["markers"][0]["yaw_deg"] == 180.0
    assert "yaw_deg" not in result["markers"][1]  # no heading -> field absent, marker unchanged


def test_load_sketch_ignores_top_level_comment():
    # the comment box is the human's reply channel -- a seeded note must not pre-fill it
    _, comment = load_sketch({"comment": "Revised: added hallway band + storeroom"})
    assert comment == ""


def test_write_result_shape_rooms_then_lines(tmp_path):
    m = SketchModel()
    _rect(m, 0, 0, 4, 3)
    m.room_names[frozenset({1, 2, 3, 4})] = "Lab"
    mk = m.add_marker(2.0, 1.5)  # inside the 4x3 room
    mk.comment = "sofa"
    m.add_marker(9.0, 9.0)  # outside every room
    out = tmp_path / "floorplan.json"
    result = write_result(str(out), "layout", m)
    # no overall dimensions (unbounded canvas); rooms come before lines in the output.
    assert list(result) == ["comment", "rooms", "lines", "doors", "markers"]
    assert result["rooms"] == [{"id": 1, "name": "Lab", "line_ids": [1, 2, 3, 4]}]
    # each marker carries the id of the room that contains it (computed), null if outside.
    assert result["markers"][0] == {
        "id": 1,
        "x_m": 2.0,
        "y_m": 1.5,
        "comment": "sofa",
        "in_room": 1,
    }
    assert result["markers"][1]["in_room"] is None
    assert len(result["lines"]) == 4
    assert json.loads(out.read_text()) == result
