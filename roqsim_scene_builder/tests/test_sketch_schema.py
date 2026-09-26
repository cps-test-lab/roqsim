"""The sketch window stamps what it returns, and refuses to seed from a sketch it cannot read.

Seeding from a newer sketch would drop the keys the window does not know and send back a sketch in
this schema, silently losing what the newer one said.
"""

from __future__ import annotations

import pytest
from roqsim_scene_builder.floorplan_window import SketchModel, load_sketch, write_result

from roqsim.floorplan_geometry import SKETCH_SCHEMA


def test_the_result_is_stamped():
    assert write_result(None, "", SketchModel())["schema"] == SKETCH_SCHEMA


def test_a_newer_seed_is_refused():
    with pytest.raises(ValueError, match=rf"schema {SKETCH_SCHEMA + 1}"):
        load_sketch({"schema": SKETCH_SCHEMA + 1, "lines": []})


def test_an_unstamped_seed_loads():
    model, _ = load_sketch({"lines": [{"id": 1, "x0_m": 0, "y0_m": 0, "x1_m": 1, "y1_m": 0}]})
    assert len(model.lines) == 1
