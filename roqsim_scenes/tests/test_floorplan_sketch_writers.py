"""Every writer of a floorplan sketch stamps its schema, and the generator checks it before building."""

from __future__ import annotations

import pytest

from roqsim.floorplan_geometry import SKETCH_SCHEMA
from roqsim_scenes import dxf_to_floorplan as d2f
from roqsim_scenes.cli import floorplan_to_world as fw
from roqsim_scenes.grid_to_floorplan import to_floorplan

# A minimal DXF in millimetres: one 1 m wall.
_DXF = (
    "0\nSECTION\n2\nHEADER\n9\n$INSUNITS\n70\n4\n0\nENDSEC\n"
    "0\nSECTION\n2\nENTITIES\n0\nLINE\n10\n0\n20\n0\n11\n1000\n21\n0\n0\nENDSEC\n0\nEOF\n"
)
_LINE = {"id": 1, "x0_m": 0, "y0_m": 0, "x1_m": 4, "y1_m": 0}


def test_the_grid_writer_stamps_the_schema():
    plan = to_floorplan([("h", 0, 0, 3)], 1, 1.0)
    assert plan["schema"] == SKETCH_SCHEMA


def test_the_dxf_writer_stamps_the_schema(tmp_path):
    path = tmp_path / "plan.dxf"
    path.write_text(_DXF, encoding="utf-8")
    sketch, _ = d2f.dxf_to_sketch(str(path))
    assert sketch["schema"] == SKETCH_SCHEMA


def _generate(tmp_path, plan):
    return fw.generate(plan, tmp_path / "scene", "s", tmp_path / "w.yaml", {}, 2.5, 0.1, 2.1)


def test_generate_refuses_a_newer_schema_before_building(tmp_path):
    plan = {"schema": SKETCH_SCHEMA + 1, "lines": [_LINE]}
    with pytest.raises(ValueError, match=rf"schema {SKETCH_SCHEMA + 1}"):
        _generate(tmp_path, plan)
    assert not (tmp_path / "scene").exists()


def test_generate_refuses_an_unknown_door_key(tmp_path):
    plan = {"lines": [_LINE], "doors": [{"id": 1, "line_id": 1, "t": 0.5, "widht_m": 1.2}]}
    with pytest.raises(ValueError, match=r"doors\[0\].*did you mean 'width_m'"):
        _generate(tmp_path, plan)
    assert not (tmp_path / "scene").exists()
