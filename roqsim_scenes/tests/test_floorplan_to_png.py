# SPDX-License-Identifier: Apache-2.0
"""floorplan_to_png.py: floorplan -> PNG plan view, and the geometry it shares with the world baker.

The load-bearing test is the first one: the plan's walls must be cut exactly where
``roqsim scenes floorplan-to-world`` cuts the baked ones, or the preview shows a building that is not the one
that gets simulated. The rest covers the scale bar (its round length, its positions, the fail-loud on a
bad one), the opening classification (what a doors-map says vs what may not be guessed without it),
the room reconstruction, and the input resolution rules including the scenes that have no floorplan."""

from __future__ import annotations

import importlib.util
import json
import struct

import pytest

from roqsim_scenes import floorplan_geometry as fg
from roqsim_scenes import floorplan_to_png as f2p
from roqsim_scenes.cli import floorplan_to_world as fw

# A 4 x 3 m room: south wall with a 0.9 m door, west wall with a window that sets its own height.
LINES = [
    {"id": 1, "x0_m": 0, "y0_m": 0, "x1_m": 4, "y1_m": 0},
    {"id": 2, "x0_m": 4, "y0_m": 0, "x1_m": 4, "y1_m": 3},
    {"id": 3, "x0_m": 4, "y0_m": 3, "x1_m": 0, "y1_m": 3},
    {"id": 4, "x0_m": 0, "y0_m": 3, "x1_m": 0, "y1_m": 0},
]
DOORS = [
    {"id": 1, "line_id": 1, "t": 0.5, "width_m": 0.9},
    {"id": 2, "line_id": 4, "t": 0.5, "width_m": 1.2, "height_m": 2.06},
]
FLOORPLAN = {
    "description": "one room",
    "rooms": [{"id": 1, "name": "office", "line_ids": [1, 2, 3, 4]}],
    "lines": LINES,
    "doors": DOORS,
    "markers": [{"id": 1, "x_m": 2.0, "y_m": 1.5, "comment": "industrial_table"}],
}


def _png_size(path) -> tuple[int, int]:
    """Width/height straight out of the PNG IHDR -- no image library needed."""
    with open(path, "rb") as fh:
        head = fh.read(24)
    assert head[:8] == b"\x89PNG\r\n\x1a\n", "not a PNG"
    return struct.unpack(">II", head[16:24])


# --- the shared geometry: the plan must cut what the baker cuts ------------------------------------


def test_plan_walls_agree_with_the_baker():
    walls = fg.plan_walls(LINES, DOORS, opening_h=2.0)
    # The baker's floor-standing pieces (z0 == 0) are the solid wall spans; lintels sit above openings.
    baked = [
        (p0, p1)
        for p0, p1, z0, _ in fw.wall_pieces(LINES, DOORS, ceiling_h=2.5, opening_h=2.0)
        if z0 == 0.0
    ]
    drawn = [(w.point_at(t0), w.point_at(t1)) for w in walls for t0, t1 in w.solid]
    assert len(drawn) == len(baked)
    for (a0, a1), (b0, b1) in zip(drawn, baked, strict=True):
        assert a0 == pytest.approx(b0)
        assert a1 == pytest.approx(b1)


def test_openings_keep_their_door_ids_and_heights():
    walls = {w.line_id: w for w in fg.plan_walls(LINES, DOORS, opening_h=2.0)}
    (door,) = walls[1].openings
    (window,) = walls[4].openings
    assert (door.door_id, door.width_m, door.height_m) == (1, 0.9, 2.0)
    assert (window.door_id, window.width_m, window.height_m) == (2, 1.2, 2.06)
    assert walls[1].openings[0].span == pytest.approx((1.55, 2.45))
    assert not walls[2].openings


def test_plan_walls_rejects_a_door_on_a_missing_line():
    with pytest.raises(ValueError, match="line 9"):
        fg.plan_walls(LINES, [{"id": 7, "line_id": 9, "t": 0.5, "width_m": 0.9}])


# --- rooms ----------------------------------------------------------------------------------------


def test_room_polygon_area_and_label_point():
    (room,) = fg.room_polygons(FLOORPLAN)
    assert room.name == "office"
    assert room.area_m2 == pytest.approx(12.0)
    x, y = fg.label_point(room.polygon)
    assert fg.point_in_polygon(room.polygon, x, y)


def test_label_point_is_centred_in_a_rectangular_room():
    """Every point down a 6 x 8 room's middle is equally clear of the walls; the label goes in the
    centre, not at an arbitrary end of that band."""
    x, y = fg.label_point([(0, 0), (6, 0), (6, 8), (0, 8)])
    assert (x, y) == (pytest.approx(3.0, abs=0.1), pytest.approx(4.0, abs=0.1))


def test_label_point_stays_inside_an_l_shape():
    """The area centroid of an L falls in the notch; the label point must not."""
    poly = [(0, 0), (6, 0), (6, 2), (2, 2), (2, 6), (0, 6)]
    x, y = fg.label_point(poly)
    assert fg.point_in_polygon(poly, x, y)


def test_room_with_too_few_walls_is_skipped_not_guessed():
    fp = {"rooms": [{"id": 1, "name": "half", "line_ids": [1, 2]}], "lines": LINES}
    assert fg.room_polygons(fp) == []


# --- scale bar ------------------------------------------------------------------------------------


def test_nice_scale_length_is_a_round_number():
    assert f2p.nice_scale_length(28.8) == 5.0
    assert f2p.nice_scale_length(4.0) == 1.0
    assert f2p.nice_scale_length(120.0) == 20.0


@pytest.mark.parametrize(
    "text,expected",
    [
        ("bottom-left", ("bottom-left", None)),
        ("TOP-RIGHT", ("top-right", None)),
        ("none", ("none", None)),
        ("12,3.5", ("xy", (12.0, 3.5))),
    ],
)
def test_parse_scale_bar_pos(text, expected):
    assert f2p.parse_scale_bar_pos(text) == expected


def test_parse_scale_bar_pos_rejects_nonsense():
    with pytest.raises(ValueError, match="bottom-left"):
        f2p.parse_scale_bar_pos("sideways")


# --- type size ------------------------------------------------------------------------------------


def test_font_scale_grows_labels_and_strokes_together():
    """Text alone at 3x next to hairline jambs reads as a broken drawing, so both scale."""
    big = f2p.Style(scale=3.0)
    assert big.pt("room") == pytest.approx(3 * f2p.Style().pt("room"))
    assert big.lw("jamb") == pytest.approx(3 * f2p.Style().lw("jamb"))


def test_font_size_sets_the_room_label_and_keeps_the_ratios():
    style = f2p.Style.from_font_size(15.0)
    assert style.pt("room") == pytest.approx(15.0)
    assert style.pt("title") / style.pt("room") == pytest.approx(
        f2p.Style().pt("title") / f2p.Style().pt("room")
    )


def test_absurd_font_scale_is_refused():
    with pytest.raises(ValueError, match="usable range"):
        f2p.Style(scale=50.0)


def _fit(name, polygon, scale=3.0, pt_per_m=30.0):
    room = fg.Room(id=1, name=name, polygon=polygon)
    return f2p._fit_label(
        room,
        areas=True,
        area_decimals=0,
        clearance_m=fg.label_spot(polygon)[2],
        pt_per_m=pt_per_m,
        style=f2p.Style(scale=scale),
    )


def test_a_label_that_fits_keeps_the_plans_type_size_on_one_line():
    hall = [(0, 0), (12, 0), (12, 9), (0, 9)]
    text, size, rotation = _fit("hall", hall)
    assert (text, size, rotation) == (
        "hall\n108 m²",
        pytest.approx(f2p.Style(scale=3.0).pt("room")),
        0,
    )


def test_a_long_name_is_wrapped_before_the_type_is_shrunk():
    """Wrapping is preferred to shrinking: two lines of full-size type beat one small line."""
    room = [(0, 0), (4.2, 0), (4.2, 3.6), (0, 3.6)]
    wrapped, wrapped_pt, rotation = _fit("Meeting Room", room)
    (one_line_pt, _) = f2p._label_sizes("Meeting Room\n15 m²", room, fg.label_spot(room)[2], 30.0)
    assert "\n" in wrapped.split("\n15 m²")[0]  # the *name* was broken, not just the area line
    assert wrapped_pt > one_line_pt and rotation == 0


def test_a_label_stands_up_only_in_a_room_too_narrow_to_wrap_into():
    corridor = [(0, 0), (1.2, 0), (1.2, 9), (0, 9)]
    text, size, rotation = _fit("corridor", corridor)
    assert rotation == 90  # nothing to wrap and no room across: turn it along the corridor
    assert size < f2p.Style(scale=3.0).pt("room")
    assert size >= f2p.Style(scale=3.0).pt("room") * f2p._MIN_LABEL_FRACTION  # never past the floor


# --- room colours and area labels -----------------------------------------------------------------


def test_parse_room_colors_takes_names_lists_and_a_highlight():
    colours = f2p.parse_room_colors(["meeting room,Server room=#d8e8d0", "4=#eee"], "office")
    assert colours == {
        "meeting room": "#d8e8d0",
        "server room": "#d8e8d0",
        "4": "#eee",
        "office": f2p._INK["highlight"],
    }


def test_parse_room_colors_rejects_a_spec_without_a_colour():
    with pytest.raises(ValueError, match="ROOM"):
        f2p.parse_room_colors(["meeting room"], None)


def test_a_room_colour_matches_by_name_or_id():
    (room,) = fg.room_polygons(FLOORPLAN)  # name "office", id 1
    assert f2p._room_fill(room, {"office": "#abc"})[0] == "#abc"
    assert f2p._room_fill(room, {"1": "#abc"})[0] == "#abc"
    assert f2p._room_fill(room, {"kitchen": "#abc"})[0] == f2p._INK["room_fill"]


def test_area_decimals_control_the_label():
    (room,) = fg.room_polygons(FLOORPLAN)
    assert f2p.room_label(room, areas=True) == "office\n12.0 m²"
    assert f2p.room_label(room, areas=True, area_decimals=0) == "office\n12 m²"
    assert f2p.room_label(room, areas=False) == "office"


def test_an_area_only_room_keeps_its_area_and_drops_its_name():
    """A room whose identity does not matter to the plan still contributes its floor area to it."""
    (room,) = fg.room_polygons(FLOORPLAN)
    assert f2p.room_label(room, areas=True, area_decimals=0, name="") == "12 m²"
    assert f2p.room_label(room, areas=False, name="") == ""  # nothing left to draw


def test_room_keys_expands_an_id_range():
    assert f2p.room_keys("hall, 4-6, Server room") == ["hall", "4", "5", "6", "server room"]
    with pytest.raises(ValueError, match="backwards"):
        f2p.room_keys("9-4")


# --- opening kinds --------------------------------------------------------------------------------


def _opening(door_id, height_m=2.0):
    return fg.Opening(door_id=door_id, t_m=2.0, width_m=0.9, height_m=height_m)


def test_doors_map_decides_what_fills_an_opening():
    doors_map = {"1": {"skip": True}, "2": {"leaf": False}, "3": {"controllable": True}}
    assert f2p.opening_kind(_opening(1), doors_map, 2.0) == "window"
    assert f2p.opening_kind(_opening(2), doors_map, 2.0) == "doorway"
    assert f2p.opening_kind(_opening(3), doors_map, 2.0) == "door"


def test_without_a_doors_map_an_odd_height_stays_unknown():
    """A floorplan alone does not say window vs tall doorway, so it must not be drawn as either."""
    assert f2p.opening_kind(_opening(1, height_m=3.5), {}, 2.0) == "opening"
    assert f2p.opening_kind(_opening(1), {}, 2.0) == "door"


def test_a_full_height_opening_never_becomes_a_swinging_door():
    """``door_placements`` leaves a full-height opening a bare gate, so no leaf may be drawn in one."""
    tall = _opening(1, height_m=3.5)
    assert f2p.opening_kind(tall, {"1": {"controllable": True}}, 2.0, ceiling_h=3.5) == "opening"
    assert f2p.opening_kind(tall, {"1": {"controllable": True}}, 3.5, ceiling_h=4.0) == "door"


# --- door leaves ----------------------------------------------------------------------------------
# The plan must swing a leaf the way the world hangs it: floorplan_to_world.door_placements' defaults
# (hinge_side left, swing 1) feeding the roqsim_assets `door` plugin.


def _south_wall_door(entry, swing_deg=90.0):
    """A 0.9 m door mid-way along a 4 m wall running +x, with the room on its +y side."""
    (wall,) = [w for w in fg.plan_walls(LINES, DOORS[:1], opening_h=2.0) if w.line_id == 1]
    (op,) = wall.openings
    return f2p._leaf_geometry(wall, op, entry, swing_deg)


def test_default_leaf_hinges_at_the_first_edge_and_opens_into_the_room():
    hinge, tip, arc = _south_wall_door({})
    assert hinge == pytest.approx((1.55, 0.0))  # opening spans 1.55..2.45 -> hinge at its -x edge
    assert tip == pytest.approx((1.55, 0.9))  # swung 90 deg to +y, one leaf width long
    assert arc[0] == pytest.approx((2.45, 0.0))  # the arc starts at the closed leaf's tip
    assert arc[-1] == pytest.approx(tip, abs=1e-9)


def test_hinge_side_right_mirrors_the_leaf():
    hinge, tip, _ = _south_wall_door({"hinge_side": "right"})
    assert hinge == pytest.approx((2.45, 0.0))
    assert tip == pytest.approx((2.45, -0.9))  # a -x leaf swinging +theta sweeps to -y


def test_swing_minus_one_opens_to_the_other_face():
    _, tip, _ = _south_wall_door({"swing": -1})
    assert tip == pytest.approx((1.55, -0.9))


def test_zero_swing_draws_the_leaf_closed_in_its_opening():
    hinge, tip, _ = _south_wall_door({}, swing_deg=0.0)
    assert hinge == pytest.approx((1.55, 0.0))
    assert tip == pytest.approx((2.45, 0.0))


# --- input resolution -----------------------------------------------------------------------------


@pytest.fixture
def scene_dir(tmp_path):
    """A floorplan-authored scene: scene.json *references* floorplan.json, with a doors map beside it."""
    scene = tmp_path / "scenes" / "office"
    scene.mkdir(parents=True)
    (scene / "floorplan.json").write_text(json.dumps(FLOORPLAN))
    (scene / "scene.json").write_text(json.dumps({"name": "office", "floorplan": "floorplan.json"}))
    (scene / "doors_map.json").write_text(json.dumps({"2": {"skip": True}}))
    return scene


def test_load_source_follows_a_scene_reference_and_finds_the_doors_map(scene_dir):
    src = f2p.load_source(str(scene_dir))
    assert src.name == "office"
    assert src.path == scene_dir / "floorplan.json"
    assert src.doors_map == {"2": {"skip": True}}


def test_load_source_can_ignore_the_doors_map(scene_dir):
    assert f2p.load_source(str(scene_dir), use_doors_map=False).doors_map == {}


def test_load_source_takes_a_bare_floorplan_json(scene_dir):
    src = f2p.load_source(str(scene_dir / "floorplan.json"))
    assert src.floorplan["rooms"][0]["name"] == "office"


def test_load_source_errors_on_a_scene_without_a_floorplan(tmp_path):
    scene = tmp_path / "imported"
    scene.mkdir()
    (scene / "scene.json").write_text(json.dumps({"name": "imported", "objects": []}))
    with pytest.raises(ValueError, match="no 'floorplan' reference"):
        f2p.load_source(str(scene))


def test_load_source_errors_on_a_world_with_no_scene(tmp_path):
    """A hand-written MJCF world has no floorplan; that is an error, not an approximate drawing."""
    worlds = tmp_path / "worlds"
    worlds.mkdir(parents=True)
    world = worlds / "handmade.yaml"
    world.write_text("sim:\n  world: handmade/handmade.xml\n")
    with pytest.raises(ValueError, match="not floorplan-authored"):
        f2p.load_source(str(world))


def test_load_source_finds_the_scene_of_a_world(scene_dir):
    worlds = scene_dir.parent.parent / "worlds"
    worlds.mkdir()
    (worlds / "office.yaml").write_text("sim:\n  world: office/office.xml\n")
    assert f2p.load_source(str(worlds / "office.yaml")).path == scene_dir / "floorplan.json"


def test_load_source_follows_extends_to_the_base_world(scene_dir):
    """A populated world carries no scene of its own -- its plan is the world it extends."""
    worlds = scene_dir.parent.parent / "worlds"
    worlds.mkdir()
    (worlds / "office.yaml").write_text("sim:\n  world: office/office.xml\n")
    (worlds / "office_populated.yaml").write_text(f"extends: {worlds / 'office.yaml'}\n")
    src = f2p.load_source(str(worlds / "office_populated.yaml"))
    assert src.path == scene_dir / "floorplan.json"


def test_load_source_errors_on_an_unknown_source():
    with pytest.raises(ValueError, match="not a file"):
        f2p.load_source("no_such_package:no_such_world")


# --- rendering ------------------------------------------------------------------------------------
# Only the drawing needs the preview extra; the geometry above is the part that must never go untested.

needs_matplotlib = pytest.mark.skipif(
    importlib.util.find_spec("matplotlib") is None, reason="roqsim_scenes[preview] not installed"
)


@needs_matplotlib
def test_renders_a_png_of_the_requested_width(scene_dir, tmp_path):
    out = tmp_path / "plan.png"
    stats = f2p.render(f2p.load_source(str(scene_dir)), out, width_px=800, dpi=100)
    width, height = _png_size(out)
    assert width == 800  # exactly as asked: no "tight" crop silently overriding it
    assert 600 < height < 800  # the plan's own aspect ratio (4.1 x 3.1 m plus the margins)
    assert (stats.rooms, stats.walls) == (1, 4)
    assert stats.width_m == pytest.approx(4.1)  # 4 m of wall + the drawn wall thickness
    assert stats.scale_bar_m == 1.0
    assert stats.kinds == {"door": 1, "window": 1}  # the doors map turns opening 2 into a window


@needs_matplotlib
def test_scale_bar_can_be_switched_off_and_placed_by_metres(scene_dir, tmp_path):
    src = f2p.load_source(str(scene_dir))
    assert f2p.render(src, tmp_path / "a.png", scale_bar="none").scale_bar_m is None
    stats = f2p.render(src, tmp_path / "b.png", scale_bar="1.0,0.5", scale_bar_length=2)
    assert stats.scale_bar_m == 2.0


def test_a_corner_scale_bar_sits_flush_with_the_plans_edge():
    """Not floating in the canvas below the drawing: the block ends on the plan's own bottom edge."""
    box = (0.0, 0.0, 20.0, 10.0)
    assert f2p.corner_anchor("bottom-left", box) == ((0.0, 0.0), "left", "bottom")
    assert f2p.corner_anchor("bottom-right", box) == ((20.0, 0.0), "right", "bottom")
    assert f2p.corner_anchor("top-left", box) == ((0.0, 10.0), "left", "top")


def test_the_bar_stays_a_slim_band_as_the_font_grows():
    """Its labels need the font's space, the ruler itself does not -- a fat black slab is not a scale."""
    thin, block_1x = f2p._scale_bar_block(5.0, 28.8, f2p.Style())
    thick, block_4x = f2p._scale_bar_block(5.0, 28.8, f2p.Style(scale=4.0))
    assert thick < thin * 4 and block_4x > block_1x * 2  # bar sub-linear, label band linear
    assert f2p._scale_bar_block(5.0, 28.8, f2p.Style(scale=4.0), height_m=0.1)[0] == 0.1


@needs_matplotlib
def test_cli_writes_a_png_and_reports_it(scene_dir, tmp_path, capsys):
    out = tmp_path / "cli.png"
    assert f2p.main([str(scene_dir), "-o", str(out), "--ids", "--grid", "1", "--axes"]) == 0
    assert out.exists()
    assert "1 rooms" in capsys.readouterr().out


@needs_matplotlib
def test_cli_renders_at_presentation_font_scale(scene_dir, tmp_path):
    out = tmp_path / "slide.png"
    assert (
        f2p.main([str(scene_dir), "-o", str(out), "--font-scale", "3", "--width-px", "1200"]) == 0
    )
    assert _png_size(out)[0] == 1200


@needs_matplotlib
def test_highlighting_a_room_that_does_not_exist_is_an_error(scene_dir, tmp_path):
    """A typo'd room name would otherwise render an unhighlighted plan that looks finished."""
    src = f2p.load_source(str(scene_dir))
    with pytest.raises(ValueError, match="match no room"):
        f2p.render(src, tmp_path / "x.png", room_colors={"kitchen": "#abc"})
    assert f2p.render(src, tmp_path / "ok.png", room_colors={"office": "#abc"}).rooms == 1


@needs_matplotlib
def test_cli_can_omit_the_legend(scene_dir, tmp_path):
    out = tmp_path / "bare.png"
    assert f2p.main([str(scene_dir), "-o", str(out), "--no-legend", "--door-swing", "0"]) == 0
    assert out.exists()


@needs_matplotlib
def test_cli_takes_the_label_selections(scene_dir, tmp_path):
    out = tmp_path / "areas.png"
    argv = [str(scene_dir), "-o", str(out), "--area-only", "1", "--no-title", "--legend", "none"]
    assert f2p.main(argv) == 0
    assert out.exists()
    assert f2p.main([*argv[:3], "--area-only", "nope"]) == 2  # a key matching no room is an error


def test_font_scale_and_font_size_are_mutually_exclusive(scene_dir, tmp_path):
    """Two ways to say the same thing; taking both would leave the winner undefined."""
    with pytest.raises(SystemExit):
        f2p.main(
            [
                str(scene_dir),
                "-o",
                str(tmp_path / "x.png"),
                "--font-scale",
                "2",
                "--font-size",
                "20",
            ]
        )


def test_cli_reports_a_bad_source_as_an_error(tmp_path, capsys):
    assert f2p.main(["nope.json", "-o", str(tmp_path / "x.png")]) == 2
    assert "error:" in capsys.readouterr().err


@needs_matplotlib
def test_grid_too_fine_is_refused(scene_dir, tmp_path):
    with pytest.raises(ValueError, match="coarser"):
        f2p.render(f2p.load_source(str(scene_dir)), tmp_path / "g.png", grid=0.001)
