"""A floorplan sketch states its schema, and a reader refuses a newer one or a key it does not read.

A sketch is persisted as ``floorplan.json`` beside a generated scene and is the single source of truth
for it, so it outlives the tool that drew it. Read with the keys that happen to overlap, a sketch
written to a later schema -- or with a misspelt ``width_m`` -- generates a plausible world with the
defaults in its place. Absent means schema 1, the layout every sketch had before the stamp existed.
"""

from __future__ import annotations

import pytest

from roqsim.floorplan_geometry import SKETCH_SCHEMA, check_sketch, stamp_sketch

LINE = {"id": 1, "x0_m": 0, "y0_m": 0, "x1_m": 4, "y1_m": 0}


def test_absent_means_the_first_schema():
    assert check_sketch({"lines": [LINE]}, "plan") == 1


def test_the_current_schema_reads():
    assert check_sketch(stamp_sketch({"lines": [LINE]}), "plan") == SKETCH_SCHEMA


def test_stamp_puts_the_schema_first_and_keeps_the_rest():
    stamped = stamp_sketch({"lines": [LINE], "schema": 99})
    assert list(stamped)[0] == "schema" and stamped["schema"] == SKETCH_SCHEMA
    assert stamped["lines"] == [LINE]


def test_a_newer_schema_is_refused_naming_both():
    with pytest.raises(ValueError, match=rf"schema {SKETCH_SCHEMA + 1}.*up to {SKETCH_SCHEMA}"):
        check_sketch({"schema": SKETCH_SCHEMA + 1}, "plan")


@pytest.mark.parametrize("bad", [0, "1", 1.5, True])
def test_a_schema_that_is_not_a_positive_integer_is_refused(bad):
    with pytest.raises(ValueError, match="schema"):
        check_sketch({"schema": bad}, "plan")


@pytest.mark.parametrize(
    ("sketch", "where", "hint"),
    [
        ({"line": []}, r"plan: unknown key\(s\) 'line'", "lines"),
        ({"lines": [{**LINE, "x2_m": 1}]}, r"plan: lines\[0\]: unknown key\(s\) 'x2_m'", None),
        (
            {"doors": [{"id": 1, "line_id": 1, "t": 0.5, "widht_m": 1.0}]},
            r"plan: doors\[0\]: unknown key\(s\) 'widht_m'",
            "width_m",
        ),
        ({"rooms": [{"id": 1, "line_ids": [1], "nam": "x"}]}, r"rooms\[0\]", "name"),
        ({"markers": [{"id": 1, "x_m": 0, "y_m": 0, "yaw": 90}]}, r"markers\[0\]", None),
    ],
)
def test_an_unknown_key_is_refused_where_it_sits(sketch, where, hint):
    with pytest.raises(ValueError, match=where) as err:
        check_sketch(sketch, "plan")
    if hint:
        assert f"did you mean {hint!r}" in str(err.value)


def test_every_key_a_writer_emits_is_known():
    check_sketch(
        {
            "schema": 1,
            "comment": "",
            "description": "a flat",
            "rooms": [{"id": 1, "name": "hall", "line_ids": [1], "description": "d"}],
            "lines": [LINE],
            "doors": [{"id": 1, "line_id": 1, "t": 0.5, "width_m": 0.9, "height_m": 2.0}],
            "markers": [
                {"id": 1, "x_m": 1, "y_m": 1, "comment": "chair", "in_room": 1, "yaw_deg": 90}
            ],
        },
        "plan",
    )
