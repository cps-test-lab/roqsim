"""Dot bookkeeping and result assembly: dots anchor to a 3D world point + picked target, number
1..N, delete renumbers, and colours cycle -- all without a display."""

from __future__ import annotations

from roqsim_scene_builder.scene_window import (
    DotModel,
    MoveModel,
    _walk_key,
    apply_prop_pose,
    color_for,
    move_record,
    rgba_hex,
    write_result,
)


def test_add_numbers_sequentially():
    m = DotModel()
    m.add([0, 0, 0])
    m.add([1, 1, 1])
    assert [d.id for d in m.dots] == [1, 2]


def test_delete_renumbers_and_keeps_world():
    m = DotModel()
    m.add([0.1, 0.1, 0.1])
    m.add([0.2, 0.2, 0.2])
    m.add([0.3, 0.3, 0.3])
    m.delete(2)
    assert [d.id for d in m.dots] == [1, 2]
    assert [round(d.world[0], 1) for d in m.dots] == [0.1, 0.3]


def test_label_from_target():
    m = DotModel()
    d = m.add([0, 0, 0], target={"geom": "table_top", "body": "table"})
    assert d.label == "table"  # body preferred over geom
    d2 = m.add([0, 0, 0], target={"geom": "floor", "body": None})
    assert d2.label == "floor"
    d3 = m.add([0, 0, 0], target=None)
    assert d3.label == "(point)"


def test_to_annotations_shape():
    m = DotModel()
    d = m.add([0.25, 0.75, 1.5], target={"geom": "g", "body": "shelf"})
    d.comment = "clips wall"
    assert m.to_annotations() == [
        {
            "id": 1,
            "world": [0.25, 0.75, 1.5],
            "target": {"geom": "g", "body": "shelf"},
            "comment": "clips wall",
        }
    ]


def test_colors_cycle_and_hex():
    assert color_for(1) == color_for(1 + 8)  # palette length 8
    assert rgba_hex((1.0, 0.0, 0.0, 1.0)) == "#ff0000"


def test_write_result_roundtrip(tmp_path):
    import json

    m = DotModel()
    m.add([0.5, 0.5, 0.5], target={"geom": "g", "body": "b"})
    out = tmp_path / "r.json"
    result = write_result(str(out), "fail", "off-centre", m)
    assert result["verdict"] == "fail"
    assert result["comment"] == "off-centre"
    assert len(result["annotations"]) == 1
    assert result["moves"] == []  # no props moved -> empty, never absent
    assert json.loads(out.read_text()) == result


def test_write_result_carries_moves():
    m = DotModel()
    moves = [move_record("industrial_table_1", "industrial_table", [1.234, 5.678, 0.0], 90.04)]
    result = write_result(None, "pass", "", m, moves)
    assert result["moves"] == [
        {
            "entity": "industrial_table_1",
            "model": "industrial_table",
            "pos": [1.234, 5.678, 0.0],
            "yaw_deg": 90.0,
        }
    ]


def test_apply_prop_pose_sets_pos_and_yaw():
    cfg = {"model": "industrial_table", "name": "industrial_table_1", "pos": [0.0, 0.0, 0.0]}
    apply_prop_pose(cfg, [10.911, 0.904, 0.0], 90.0)
    assert cfg["pos"] == [10.911, 0.904, 0.0]
    assert cfg["rpy"][2] == round(__import__("math").radians(90.0), 5)
    assert cfg["rpy"][0] == 0.0 and cfg["rpy"][1] == 0.0


def test_apply_prop_pose_omits_rpy_when_flat():
    cfg = {"model": "chair", "pos": [1.0, 1.0, 0.0]}
    apply_prop_pose(cfg, [2.0, 2.0, 0.0], 0.0)  # no rotation at all
    assert cfg["pos"] == [2.0, 2.0, 0.0]
    assert "rpy" not in cfg  # a flat prop's entry stays terse


def test_apply_prop_pose_preserves_roll_pitch():
    cfg = {"model": "lamp", "pos": [0.0, 0.0, 0.0], "rpy": [0.1, 0.2, 0.0]}
    apply_prop_pose(cfg, [3.0, 4.0, 0.0], 45.0)
    assert cfg["rpy"][0] == 0.1 and cfg["rpy"][1] == 0.2  # roll/pitch untouched
    assert cfg["rpy"][2] == round(__import__("math").radians(45.0), 5)


def test_movemodel_set_updates_existing_keeps_id_and_orig():
    m = MoveModel()
    a = m.set("table_1", "table", [1.0, 1.0, 0.0], 0.0, spec="SPEC", orig_config={"model": "table"})
    b = m.set("table_1", "table", [2.0, 3.0, 0.0], 90.0)  # re-drag the same prop
    assert a is b and len(m.moves) == 1  # one row, updated in place
    assert b.id == 1 and b.pos == [2.0, 3.0, 0.0] and b.yaw_deg == 90.0
    assert b.orig_config == {"model": "table"}  # first original kept, not overwritten
    assert b.spec == "SPEC"


def test_movemodel_delete_renumbers_and_payload():
    m = MoveModel()
    m.set("a", "ma", [1.0, 0.0, 0.0], 0.0)
    m.set("b", "mb", [2.0, 0.0, 0.0], 45.0)
    m.set("c", "mc", [3.0, 0.0, 0.0], 0.0)
    m.delete(2)
    assert [mv.id for mv in m.moves] == [1, 2]  # contiguous after delete
    assert [mv.entity for mv in m.moves] == ["a", "c"]
    assert m.to_payload() == [
        {"entity": "a", "model": "ma", "pos": [1.0, 0.0, 0.0], "yaw_deg": 0.0},
        {"entity": "c", "model": "mc", "pos": [3.0, 0.0, 0.0], "yaw_deg": 0.0},
    ]


def test_move_label_shows_pose_and_heading():
    m = MoveModel()
    flat = m.set("desk", "desk", [1.234, 5.678, 0.0], 0.0)
    turned = m.set("chair", "chair", [2.0, 2.0, 0.0], 90.0)
    assert flat.label == "desk → 1.23, 5.68"  # no angle when axis-aligned
    assert turned.label == "chair → 2.00, 2.00 ∠90°"


def test_enable_edit_shortcuts_binds_select_all():
    import types

    from roqsim_scene_builder.annotate_ui import enable_edit_shortcuts

    bound: dict = {}
    fake_root = types.SimpleNamespace(
        bind_class=lambda cls, seq, fn: bound.__setitem__((cls, seq), fn)
    )
    enable_edit_shortcuts(fake_root)
    # Ctrl+A (and caps variant) select-all wired for both text widget classes
    for cls in ("Entry", "Text"):
        assert (cls, "<Control-a>") in bound
        assert (cls, "<Control-A>") in bound

    # Entry handler selects the whole field and stops the default (line-start) via "break"
    calls = []
    entry = types.SimpleNamespace(
        select_range=lambda a, b: calls.append(("range", a, b)),
        icursor=lambda p: calls.append(("cursor", p)),
    )
    assert bound[("Entry", "<Control-a>")](types.SimpleNamespace(widget=entry)) == "break"
    assert ("range", 0, "end") in calls

    # Text handler tags the whole buffer as the selection
    tcalls = []
    text = types.SimpleNamespace(
        tag_add=lambda *a: tcalls.append(("tag", *a)),
        mark_set=lambda *a: tcalls.append(("mark", *a)),
    )
    assert bound[("Text", "<Control-a>")](types.SimpleNamespace(widget=text)) == "break"
    assert ("tag", "sel", "1.0", "end-1c") in tcalls


def test_dot_yaw_in_annotations_only_when_set():
    m = DotModel()
    plain = m.add([0.0, 0.0, 0.0])
    facing = m.add([1.0, 0.0, 0.0])
    facing.yaw_deg = 90.0
    anns = m.to_annotations()
    assert "yaw_deg" not in anns[0]  # headingless dot unchanged
    assert anns[1]["yaw_deg"] == 90.0
    assert facing.label.endswith("∠90°") and plain.yaw_deg is None


def test_arrow_keys_are_aliases_of_the_walk_letters():
    # Either hand flies: WASD/QE as themselves, arrows and Page Up/Down mapped onto the same six.
    assert [_walk_key(k) for k in ("Up", "Down", "Left", "Right", "Prior", "Next")] == [
        "w",
        "s",
        "a",
        "d",
        "e",
        "q",
    ]
    assert [_walk_key(k) for k in ("w", "A", "q")] == ["w", "a", "q"]
    assert _walk_key("Return") == "return"  # anything else passes through, to be ignored downstream
    assert _walk_key("") == ""
