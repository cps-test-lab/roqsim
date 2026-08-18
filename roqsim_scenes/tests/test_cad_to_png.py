# SPDX-License-Identifier: Apache-2.0
"""cad_to_png.py: DWG/DXF drawing -> PNG preview.

Covers the layer filter (the reason the tool exists -- isolating wall layers in a 40-layer
architectural drawing), the layer report, and the fail-loud rules: an unsupported extension, a
filter that selects nothing, and a DWG with no converter installed all raise rather than writing
an empty or misleading image."""

from __future__ import annotations

import struct

import pytest

from roqsim_scenes import cad_to_png

ezdxf = pytest.importorskip("ezdxf", reason="roqsim_scenes[preview] not installed")
pytest.importorskip("matplotlib", reason="roqsim_scenes[preview] not installed")


@pytest.fixture
def drawing(tmp_path):
    """A 2-layer DXF: a WALLS square plus a TEXT-layer label and an off layer."""
    doc = ezdxf.new(setup=True)
    doc.header["$INSUNITS"] = 6  # metres
    doc.layers.add("A_WALL")
    doc.layers.add("A_TEXT")
    doc.layers.add("A_HIDDEN").off()
    msp = doc.modelspace()
    for x0, y0, x1, y1 in ((0, 0, 4, 0), (4, 0, 4, 3), (4, 3, 0, 3), (0, 3, 0, 0)):
        msp.add_line((x0, y0), (x1, y1), dxfattribs={"layer": "A_WALL"})
    msp.add_text("Buero", dxfattribs={"layer": "A_TEXT"}).set_placement((1, 1))
    # Far outside the walls: proves the framing follows the *selected* entities only.
    msp.add_line((100, 100), (110, 100), dxfattribs={"layer": "A_HIDDEN"})
    path = tmp_path / "plan.dxf"
    doc.saveas(path)
    return str(path)


def _png_size(path) -> tuple[int, int]:
    """Width/height straight out of the PNG IHDR -- no image library needed."""
    with open(path, "rb") as fh:
        head = fh.read(24)
    assert head[:8] == b"\x89PNG\r\n\x1a\n", "not a PNG"
    return struct.unpack(">II", head[16:24])


def test_renders_a_png_of_the_requested_width(drawing, tmp_path):
    out = tmp_path / "plan.png"
    info = cad_to_png.render(drawing, str(out), width_px=800, dpi=100)
    assert info["entities_drawn"] == 5  # 4 walls + the label; the off layer is skipped
    assert info["layers_off_skipped"] == ["A_HIDDEN"]
    assert _png_size(out)[0] == 800


def test_layer_filter_selects_and_frames_only_those_layers(drawing, tmp_path):
    out = tmp_path / "walls.png"
    info = cad_to_png.render(drawing, str(out), layers=["a_wall*"], width_px=400, dpi=100)
    assert info["entities_drawn"] == 4
    # 4 x 3 m of wall -> the extent is the walls', not the drawing's.
    assert info["extent"] == (4.0, 3.0)
    # ...and the pixel height follows that aspect ratio.
    assert _png_size(out) == (400, 300)


def test_exclude_layers_drops_the_pattern(drawing, tmp_path):
    info = cad_to_png.render(
        drawing, str(tmp_path / "o.png"), exclude_layers=["a_text"], width_px=200, dpi=100
    )
    assert info["entities_drawn"] == 4


def test_off_layers_can_be_included(drawing, tmp_path):
    info = cad_to_png.render(
        drawing, str(tmp_path / "o.png"), include_off_layers=True, width_px=200, dpi=100
    )
    assert info["entities_drawn"] == 6
    assert info["layers_off_skipped"] == []


def test_layer_report_counts_modelspace_entities(drawing):
    rows = dict((name, count) for name, count, _ in cad_to_png.layer_report(drawing))
    assert rows["A_WALL"] == 4
    assert rows["A_TEXT"] == 1
    # A layer with no entities is still reported -- an empty wall layer is the surprise
    # worth seeing.
    assert rows["Defpoints"] == 0
    assert dict((name, off) for name, _, off in cad_to_png.layer_report(drawing))["A_HIDDEN"]


def test_empty_selection_is_an_error_not_a_blank_image(drawing, tmp_path):
    out = tmp_path / "nothing.png"
    with pytest.raises(RuntimeError, match="nothing to draw"):
        cad_to_png.render(drawing, str(out), layers=["a_nonexistent*"])
    assert not out.exists()


def test_unsupported_extension_is_rejected(tmp_path):
    src = tmp_path / "plan.svg"
    src.write_text("<svg/>", encoding="utf-8")
    with pytest.raises(ValueError, match="expected a .dwg or .dxf"):
        cad_to_png.render(str(src), str(tmp_path / "o.png"))


def test_dwg_without_a_converter_names_the_install_options(tmp_path, monkeypatch):
    """A DWG needs an external converter; missing it must not fall back to anything."""
    monkeypatch.setattr(cad_to_png, "find_dwg_backend", lambda: None)
    dwg = tmp_path / "plan.dwg"
    dwg.write_bytes(b"AC1032")
    with pytest.raises(RuntimeError, match="no DWG converter found"):
        cad_to_png.render(str(dwg), str(tmp_path / "o.png"))


def test_bad_colors_argument_is_rejected(drawing, tmp_path):
    with pytest.raises(ValueError, match="--colors"):
        cad_to_png.render(drawing, str(tmp_path / "o.png"), colors="rainbow")
