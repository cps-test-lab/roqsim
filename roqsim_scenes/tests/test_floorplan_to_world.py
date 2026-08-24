"""floorplan_to_world.py: the floorplan -> world emitters, tested without baking or a display.

Covers the geometry (a wall slab spans its segment and stands on the floor), the scene.json shape
(object names match floorplan.scene.yaml's globs; the floorplan is referenced, not embedded), the
world doc (one spawn_model per marker at its metre position), and the fail-loud rule (an unmapped
marker raises)."""

from __future__ import annotations

import math

import pytest

from roqsim_scenes.cli import floorplan_to_world as fw
from roqsim_scenes.cli import scene_to_floorplan as s2f


def test_line_segments_unpacks_each_line():
    lines = [
        {"id": 1, "x0_m": 0, "y0_m": 0, "x1_m": 4, "y1_m": 0},
        {"id": 2, "x0_m": 4, "y0_m": 0, "x1_m": 4, "y1_m": 3},
    ]
    assert fw.line_segments(lines) == [((0.0, 0.0), (4.0, 0.0)), ((4.0, 0.0), (4.0, 3.0))]


def test_wall_box_spans_segment_and_stands_on_floor():
    verts, faces, uv = fw.wall_box((0.0, 0.0), (4.0, 0.0), z0=0.0, z1=3.0, thickness=0.1)
    assert verts.shape == (24, 3)  # unwrapped per face: 4 corners x 6 faces, so every face has UVs
    assert faces.shape == (12, 3)
    assert uv.shape == (24, 2)
    # spans x in [0,4] (length 4), thin in y (~0.1), floor at z=0 up to ceiling 3.
    assert verts[:, 0].min() == pytest.approx(0.0, abs=1e-6)
    assert verts[:, 0].max() == pytest.approx(4.0, abs=1e-6)
    assert (verts[:, 1].max() - verts[:, 1].min()) == pytest.approx(0.1, abs=1e-6)
    assert verts[:, 2].min() == pytest.approx(0.0, abs=1e-6)
    assert verts[:, 2].max() == pytest.approx(3.0, abs=1e-6)


def test_wall_box_lintel_floats_between_z0_and_z1():
    # a lintel over an opening: from z=2.0 up to the 2.5 ceiling, not standing on the floor.
    verts, _, _ = fw.wall_box((0.0, 0.0), (4.0, 0.0), z0=2.0, z1=2.5, thickness=0.1)
    assert verts[:, 2].min() == pytest.approx(2.0, abs=1e-6)
    assert verts[:, 2].max() == pytest.approx(2.5, abs=1e-6)


def test_wall_box_rotated_segment():
    # a vertical segment (0,0)->(0,4): now thin in x, spanning y in [0,4].
    verts, _, _ = fw.wall_box((0.0, 0.0), (0.0, 4.0), z0=0.0, z1=2.5, thickness=0.2)
    assert verts[:, 1].min() == pytest.approx(0.0, abs=1e-6)
    assert verts[:, 1].max() == pytest.approx(4.0, abs=1e-6)
    assert (verts[:, 0].max() - verts[:, 0].min()) == pytest.approx(0.2, abs=1e-6)


def test_wall_box_rejects_degenerate_segment():
    with pytest.raises(ValueError):
        fw.wall_box((1.0, 1.0), (1.0, 1.0), z0=0.0, z1=3.0, thickness=0.1)


def test_floor_box_covers_bbox_top_at_zero():
    verts, _, _ = fw.floor_box(-1.0, 2.0, 8.0, 6.0)  # arbitrary bbox (may have negative coords)
    assert verts[:, 0].min() == pytest.approx(-1.0)
    assert verts[:, 0].max() == pytest.approx(8.0)
    assert verts[:, 1].min() == pytest.approx(2.0)
    assert verts[:, 1].max() == pytest.approx(6.0)
    assert verts[:, 2].max() == pytest.approx(0.0)  # top of the floor slab is z=0


def test_bbox_of_pads_the_walls_extent():
    lines = [{"id": 1, "x0_m": 1.0, "y0_m": 1.0, "x1_m": 5.0, "y1_m": 3.0}]
    assert fw.bbox_of(lines, margin=0.2) == (0.8, 0.8, 5.2, 3.2)


def test_scene_manifest_names_match_globs():
    manifest = fw.scene_manifest("myroom", (0.0, 0.0, 8.0, 6.0), 2.5, n_walls=4)
    names = [o["name"] for o in manifest["objects"]]
    assert names == ["Floor", "Wall_01", "Wall_02", "Wall_03", "Wall_04"]
    assert manifest["bounds_max"] == [8.0, 6.0, 2.5]
    assert manifest["ground_z"] == 0.0
    wall = next(o for o in manifest["objects"] if o["name"] == "Wall_01")
    assert wall["collide"] is True
    assert "floorplan" not in manifest  # no floorplan reference given -> no key


def test_scene_manifest_adds_the_ceiling_only_when_asked():
    # Off by default: an open plan is what most generated rooms want, and every existing scene keeps
    # the object list it was baked with.
    assert all(
        o["name"] != "Ceiling"
        for o in fw.scene_manifest("r", (0.0, 0.0, 8.0, 6.0), 2.5, n_walls=1)["objects"]
    )
    roofed = fw.scene_manifest("r", (0.0, 0.0, 8.0, 6.0), 2.5, n_walls=1, ceiling=True)
    ceiling = next(o for o in roofed["objects"] if o["name"] == "Ceiling")
    assert ceiling["collide"] is False  # visual only: a convex hull would fill the building
    assert roofed["bounds_max"][2] > 2.5  # the slab stands above the walls' top edge


def test_ceiling_box_soffit_sits_at_the_wall_top():
    verts, _, uv = fw.ceiling_box(-1.0, 2.0, 8.0, 6.0, 3.5)
    assert verts[:, 2].min() == pytest.approx(3.5)  # the underside, i.e. what the room sees
    assert verts[:, 2].max() > 3.5
    assert uv.shape == (24, 2)
    # Entirely above a 2.6 m cut, which is what lets the `ceiling` plugin delete it by height.
    assert verts[:, 2].min() > 2.6


def test_bake_config_precedence(tmp_path):
    # A scene dir's own scene.yaml beats the shared look; --bake-config beats both.
    assert fw.resolve_bake_config(tmp_path, None) == fw._SHARED_CONFIG
    own = tmp_path / "scene.yaml"
    own.write_text("materials: []\n")
    assert fw.resolve_bake_config(tmp_path, None) == own
    explicit = tmp_path / "other.yaml"
    explicit.write_text("materials: []\n")
    assert fw.resolve_bake_config(tmp_path, explicit) == explicit


def test_a_ceiling_pulls_the_bake_light_under_the_soffit(tmp_path):
    import yaml

    config = tmp_path / "look.yaml"
    config.write_text(yaml.safe_dump({"light": {"height": 4.0, "cutoff": 90.0}}))
    out = fw.light_under_ceiling(config, 3.5, tmp_path)
    assert out != config  # the shared config is copied, never edited in place
    lowered = yaml.safe_load(out.read_text())
    assert lowered["light"]["height"] < 3.5
    assert lowered["light"]["cutoff"] == 90.0  # the rest of the look is carried over
    # A light already under the soffit is left alone -- and the config file itself is reused.
    fine = tmp_path / "fine.yaml"
    fine.write_text(yaml.safe_dump({"light": {"height": 3.0}}))
    assert fw.light_under_ceiling(fine, 3.5, tmp_path) == fine


def test_scene_manifest_references_floorplan_for_roundtrip():
    # scene.json carries only the relative path to the authored floorplan, never an embedded copy.
    manifest = fw.scene_manifest(
        "r", (0.0, 0.0, 4.0, 1.0), 2.5, n_walls=3, floorplan_ref="floorplan.json"
    )
    assert manifest["floorplan"] == "floorplan.json"


def test_world_doc_spawns_mapped_markers():
    markers = [{"id": 1, "x_m": 2.5, "y_m": 3.0, "comment": "table"}]
    doc = fw.world_doc("r", "r/r.xml", (0.0, 0.0, 8.0, 6.0), markers, {"1": "industrial_table"})
    assert doc["sim"]["world"] == "r/r.xml"
    assert doc["components"] == [
        {
            "spawn_model": {
                "model": "industrial_table",
                "name": "marker_1",
                "prefix": "marker_1_",
                "pos": [2.5, 3.0, 0.0],
            }
        }
    ]


def test_world_doc_errors_on_unmapped_marker():
    markers = [{"id": 1, "x_m": 2.5, "y_m": 3.0, "comment": "table"}]
    with pytest.raises(KeyError, match="marker 1"):
        fw.world_doc("r", "r/r.xml", (0.0, 0.0, 8.0, 6.0), markers, {})


def test_world_doc_yaw_from_map_and_sketch_with_map_override():
    # map value may be a bare name or {model, yaw_deg}; a map yaw wins, else the marker's own yaw
    markers = [
        {"id": 1, "x_m": 1.0, "y_m": 1.0},  # yaw from the map dict
        {"id": 2, "x_m": 2.0, "y_m": 1.0, "yaw_deg": 90},  # yaw from the sketch (bare-name map)
        {"id": 3, "x_m": 3.0, "y_m": 1.0, "yaw_deg": 90},
    ]  # map yaw_deg 0 overrides sketch 90
    mmap = {
        "1": {"model": "bed", "yaw_deg": 180},
        "2": "plant",
        "3": {"model": "desk", "yaw_deg": 0},
    }
    plugins = [
        p["spawn_model"]
        for p in fw.world_doc("r", "r/r.xml", (0.0, 0.0, 4.0, 4.0), markers, mmap)["components"]
    ]
    assert plugins[0]["model"] == "bed" and plugins[0]["rpy"] == pytest.approx(
        [0, 0, math.pi], abs=1e-4
    )
    assert plugins[1]["model"] == "plant" and plugins[1]["rpy"] == pytest.approx(
        [0, 0, math.radians(90)], abs=1e-4
    )
    assert "rpy" not in plugins[2]  # heading resolved to 0 -> axis-aligned, no rpy emitted


def test_world_doc_errors_on_map_dict_without_model():
    markers = [{"id": 1, "x_m": 0.0, "y_m": 0.0}]
    with pytest.raises(KeyError, match="no 'model'"):
        fw.world_doc("r", "r/r.xml", (0.0, 0.0, 4.0, 4.0), markers, {"1": {"yaw_deg": 10}})


def test_door_placements_defaults_to_a_passive_wooden_door():
    lines = [{"id": 1, "x0_m": 0.0, "y0_m": 0.0, "x1_m": 4.0, "y1_m": 0.0}]
    doors = [{"id": 1, "line_id": 1, "t": 0.5, "width_m": 0.9}]
    (entry,) = fw.door_placements(lines, doors, {}, ceiling_h=2.5, opening_h=2.0)
    d = entry["door"]
    assert d["name"] == "door_1" and d["model"] == "door"
    assert d["pos"] == [2.0, 0.0, 0.0]  # opening centre on the wall
    assert d["rpy"][2] == pytest.approx(0.0)  # yaw along the +x wall
    assert d["width"] == 0.9 and d["hinge_side"] == "left"
    assert d["controllable"] is False  # automatic is opt-in


def test_door_placements_applies_map_overrides():
    lines = [{"id": 1, "x0_m": 0.0, "y0_m": 0.0, "x1_m": 4.0, "y1_m": 0.0}]
    doors = [{"id": 1, "line_id": 1, "t": 0.5, "width_m": 0.9}]
    dmap = {
        "1": {
            "model": "door_glass",
            "hinge_side": "right",
            "swing": -1,
            "open": 0.3,
            "controllable": True,
            "namespace": "foyer",
        }
    }
    (d,) = [e["door"] for e in fw.door_placements(lines, doors, dmap, 2.5, 2.0)]
    assert d["model"] == "door_glass" and d["hinge_side"] == "right"
    assert d["swing"] == -1 and d["open"] == 0.3
    assert d["controllable"] is True and d["namespace"] == "foyer"


def test_door_placements_skips_full_height_openings():
    lines = [{"id": 1, "x0_m": 0.0, "y0_m": 0.0, "x1_m": 4.0, "y1_m": 0.0}]
    doors = [{"id": 1, "line_id": 1, "t": 0.5, "width_m": 3.2, "height_m": 3.5}]
    assert fw.door_placements(lines, doors, {}, ceiling_h=3.5, opening_h=2.0) == []


def test_world_doc_emits_doors_before_markers():
    markers = [{"id": 1, "x_m": 2.5, "y_m": 3.0}]
    doors = [{"door": {"name": "door_1"}}]
    plugins = fw.world_doc(
        "r", "r/r.xml", (0.0, 0.0, 8.0, 6.0), markers, {"1": "industrial_table"}, doors=doors
    )["components"]
    assert list(plugins[0]) == ["door"] and list(plugins[1]) == ["spawn_model"]


def test_cut_openings_splits_and_drops_slivers():
    # A 4 m wall with a 0.9 m door at 2 m -> two solid pieces; a door hugging the end leaves no stub.
    assert fw.cut_openings(4.0, [(2.0, 0.9)]) == [(0.0, 1.55), (2.45, 4.0)]
    assert fw.cut_openings(4.0, [(0.46, 0.9)]) == [(0.91, 4.0)]  # 1 cm stub vanishes


def test_wall_pieces_cuts_floor_and_adds_a_lintel_on_the_door_line():
    lines = [
        {"id": 1, "x0_m": 0.0, "y0_m": 0.0, "x1_m": 4.0, "y1_m": 0.0},
        {"id": 2, "x0_m": 4.0, "y0_m": 0.0, "x1_m": 4.0, "y1_m": 3.0},
    ]
    doors = [{"id": 1, "line_id": 1, "t": 0.5, "width_m": 0.9}]  # centred on line 1
    pieces = fw.wall_pieces(lines, doors, ceiling_h=2.5, opening_h=2.0)
    # line 1: two floor pieces around the opening + one lintel over it; line 2: one full-height piece.
    assert len(pieces) == 4
    floor = [p for p in pieces if p[2] == 0.0]
    lintels = [p for p in pieces if p[2] == 2.0]
    assert ((0.0, 0.0), (1.55, 0.0), 0.0, 2.5) in floor
    assert ((2.45, 0.0), (4.0, 0.0), 0.0, 2.5) in floor
    assert ((4.0, 0.0), (4.0, 3.0), 0.0, 2.5) in floor  # untouched wall, full height
    assert lintels == [
        ((1.55, 0.0), (2.45, 0.0), 2.0, 2.5)
    ]  # beam over the 0.9 m opening at z 2..2.5


def test_wall_pieces_no_lintel_when_opening_reaches_ceiling():
    lines = [{"id": 1, "x0_m": 0.0, "y0_m": 0.0, "x1_m": 4.0, "y1_m": 0.0}]
    doors = [{"id": 1, "line_id": 1, "t": 0.5, "width_m": 0.9}]
    pieces = fw.wall_pieces(lines, doors, ceiling_h=2.5, opening_h=2.5)  # opening_h == ceiling
    assert all(p[2] == 0.0 for p in pieces)  # only floor pieces, no lintel


def test_wall_pieces_per_door_height_overrides_the_global():
    lines = [{"id": 1, "x0_m": 0.0, "y0_m": 0.0, "x1_m": 4.0, "y1_m": 0.0}]
    doors = [{"id": 1, "line_id": 1, "t": 0.5, "width_m": 0.9, "height_m": 1.2}]
    pieces = fw.wall_pieces(lines, doors, ceiling_h=2.5, opening_h=2.0)
    lintels = [p for p in pieces if p[2] != 0.0]
    assert lintels == [((1.55, 0.0), (2.45, 0.0), 1.2, 2.5)]  # bottom at the door's own height


def test_assign_doors_errors_on_unknown_line():
    lines = [{"id": 1, "x0_m": 0.0, "y0_m": 0.0, "x1_m": 4.0, "y1_m": 0.0}]
    with pytest.raises(ValueError, match="line 9"):
        fw.assign_doors(lines, [{"id": 7, "line_id": 9, "t": 0.5, "width_m": 0.9}], opening_h=2.0)


def test_assign_doors_errors_when_wall_too_short():
    lines = [{"id": 1, "x0_m": 0.0, "y0_m": 0.0, "x1_m": 0.5, "y1_m": 0.0}]
    with pytest.raises(ValueError, match="too short|is 0.50 m|needs"):
        fw.assign_doors(lines, [{"id": 1, "line_id": 1, "t": 0.5, "width_m": 0.9}], opening_h=2.0)


def test_generate_rejects_a_wall_less_floorplan(tmp_path):
    # No wall lines -> nothing to build; fail loud rather than emit an empty world.
    floorplan = {"lines": []}
    with pytest.raises(ValueError, match="no wall lines"):
        fw.generate(floorplan, tmp_path / "s", "s", tmp_path / "w.yaml", {}, 2.5, 0.1, 2.0)


def test_scene_to_floorplan_follows_the_reference(tmp_path):
    import json

    floorplan = {"comment": "c", "rooms": [], "lines": [{"id": 1}], "doors": [], "markers": []}
    (tmp_path / "scene.json").write_text(json.dumps({"name": "r", "floorplan": "floorplan.json"}))
    (tmp_path / "floorplan.json").write_text(json.dumps(floorplan))
    assert s2f.floorplan_of(tmp_path) == floorplan


def test_scene_to_floorplan_errors_when_reference_missing_file(tmp_path):
    import json

    # scene.json references a floorplan file that is not there -> fail loud, no approximation.
    (tmp_path / "scene.json").write_text(json.dumps({"name": "r", "floorplan": "floorplan.json"}))
    with pytest.raises(ValueError, match="does not exist"):
        s2f.floorplan_of(tmp_path)


def test_scene_to_floorplan_errors_loudly_when_not_floorplan_authored(tmp_path):
    import json

    # An imported scene (e.g. USD) carries no floorplan reference -> not round-trippable.
    (tmp_path / "scene.json").write_text(json.dumps({"name": "r", "source": "lab.usd"}))
    with pytest.raises(ValueError, match="no 'floorplan' reference|not authored"):
        s2f.floorplan_of(tmp_path)
