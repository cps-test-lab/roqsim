# SPDX-License-Identifier: Apache-2.0
"""dxf_to_floorplan.py: DXF drawing -> floorplan JSON.

Covers unit scaling from $INSUNITS, LINE + closed-LWPOLYLINE expansion, recentre/dedup, and the
fail-loud rules (a polyline bulge and a curved entity both raise rather than dropping a wall)."""

from __future__ import annotations

import pytest

from roqsim_scenes import dxf_to_floorplan as d2f


def _dxf(header_units: int, entities: str) -> str:
    """A minimal but valid DXF: a HEADER carrying $INSUNITS + an ENTITIES section."""
    return (
        "0\nSECTION\n2\nHEADER\n"
        f"9\n$INSUNITS\n70\n{header_units}\n"
        "0\nENDSEC\n"
        "0\nSECTION\n2\nENTITIES\n"
        f"{entities}"
        "0\nENDSEC\n0\nEOF\n"
    )


_LINE = "0\nLINE\n10\n0\n20\n0\n11\n1000\n21\n0\n"  # (0,0)->(1000,0) in mm
# A closed 3-vertex polyline -> 3 segments (2 spans + 1 closing).
_TRI = "0\nLWPOLYLINE\n90\n3\n70\n1\n10\n0\n20\n0\n10\n1000\n20\n0\n10\n0\n20\n1000\n"


def _write(tmp_path, text):
    p = tmp_path / "d.dxf"
    p.write_text(text, encoding="utf-8")
    return str(p)


def test_millimetre_units_scale_to_metres(tmp_path):
    sketch, info = d2f.dxf_to_sketch(_write(tmp_path, _dxf(4, _LINE)))
    assert info["scale_m_per_unit"] == 0.001
    assert sketch["lines"] == [{"id": 1, "x0_m": 0.0, "y0_m": 0.0, "x1_m": 1.0, "y1_m": 0.0}]


def test_closed_lwpolyline_expands_and_closes(tmp_path):
    sketch, info = d2f.dxf_to_sketch(_write(tmp_path, _dxf(4, _TRI)), recenter=False)
    assert info["final_lines"] == 3
    # ids are 1..N; empty rooms/doors/markers so the human fills them in the 2D window.
    assert [ln["id"] for ln in sketch["lines"]] == [1, 2, 3]
    assert sketch["rooms"] == [] and sketch["doors"] == [] and sketch["markers"] == []


def test_recenter_moves_bbox_min_to_origin(tmp_path):
    off = "0\nLINE\n10\n5000\n20\n5000\n11\n6000\n21\n5000\n"
    sketch, _ = d2f.dxf_to_sketch(_write(tmp_path, _dxf(4, off)), recenter=True)
    assert sketch["lines"][0] == {"id": 1, "x0_m": 0.0, "y0_m": 0.0, "x1_m": 1.0, "y1_m": 0.0}


def test_dedup_drops_coincident_overlap(tmp_path):
    # The LINE and the polyline's first span are the same wall; dedup keeps one.
    sketch, info = d2f.dxf_to_sketch(_write(tmp_path, _dxf(4, _LINE + _TRI)))
    assert info["raw_segments"] == 4  # 1 line + 3 polyline segments
    assert info["final_lines"] == 3  # one duplicate collapsed


def test_bulge_polyline_raises(tmp_path):
    bulged = "0\nLWPOLYLINE\n90\n2\n70\n0\n10\n0\n20\n0\n42\n0.5\n10\n1000\n20\n0\n"
    with pytest.raises(ValueError, match="bulge"):
        d2f.dxf_to_sketch(_write(tmp_path, _dxf(4, bulged)))


def test_curved_entity_raises(tmp_path):
    arc = "0\nARC\n10\n0\n20\n0\n40\n500\n"
    with pytest.raises(ValueError, match="ARC"):
        d2f.dxf_to_sketch(_write(tmp_path, _dxf(4, arc)))


def test_unitless_without_scale_raises(tmp_path):
    with pytest.raises(ValueError, match="INSUNITS"):
        d2f.dxf_to_sketch(_write(tmp_path, _dxf(0, _LINE)))
    # ...but an explicit --scale rescues it.
    sketch, info = d2f.dxf_to_sketch(_write(tmp_path, _dxf(0, _LINE)), scale=1.0)
    assert info["scale_m_per_unit"] == 1.0
    assert sketch["lines"][0]["x1_m"] == 1000.0
