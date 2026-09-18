"""Overlays: found by name, placed by anchor, and refused when they hand back a frame ffmpeg cannot take."""

from __future__ import annotations

import json

import numpy as np
import pytest

from roqsim import overlays
from roqsim.overlays import OverlayError, Placement


def _frame(w=160, h=90):
    return np.zeros((h, w, 3), dtype=np.uint8)


# -- specs ----------------------------------------------------------------------------------------


def test_a_bare_name_has_no_options():
    assert overlays.parse_spec("clock") == ("clock", {})


def test_a_mapping_carries_options_as_dict_or_json():
    assert overlays.parse_spec({"clock": {"anchor": "top"}}) == ("clock", {"anchor": "top"})
    assert overlays.parse_spec(json.dumps({"clock": {"size": 0.1}})) == ("clock", {"size": 0.1})
    assert overlays.parse_spec({"clock": None}) == ("clock", {})


@pytest.mark.parametrize("bad", [{"a": {}, "b": {}}, {"clock": 3}, 42, "{not json"])
def test_a_malformed_spec_is_refused(bad):
    with pytest.raises(OverlayError):
        overlays.parse_spec(bad)


def test_an_unknown_overlay_lists_what_is_available(monkeypatch):
    monkeypatch.setattr(overlays, "_entry_points", lambda: ())
    with pytest.raises(OverlayError, match="Available: clock"):
        overlays.build("costmap", 160, 90)


# -- placement --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "anchor, expected",
    [
        ("top-left", (12, 12)),
        ("top-right", (160 - 40 - 12, 12)),
        ("bottom-left", (12, 90 - 20 - 12)),
        ("bottom-right", (160 - 40 - 12, 90 - 20 - 12)),
        ("top", ((160 - 40) // 2, 12)),
        ("bottom", ((160 - 40) // 2, 90 - 20 - 12)),
        ("left", (12, (90 - 20) // 2)),
        ("right", (160 - 40 - 12, (90 - 20) // 2)),
        ("center", ((160 - 40) // 2, (90 - 20) // 2)),
    ],
)
def test_anchors_place_the_inset(anchor, expected):
    assert Placement(anchor=anchor).box(160, 90, 40, 20) == expected


def test_width_is_a_fraction_or_pixels():
    assert Placement(width=0.25).pixels(400) == 100
    assert Placement(width=120).pixels(400) == 120


def test_a_bad_anchor_is_refused():
    with pytest.raises(OverlayError, match="anchor 'middle'"):
        Placement.from_spec({"anchor": "middle"}, "clock")


# -- the built-in clock -----------------------------------------------------------------------------


def test_the_clock_paints_pixels_and_keeps_the_frame_shape():
    clock = overlays.build("clock", 160, 90)
    frame = _frame()
    out = overlays.apply_all([clock], frame, 12.5)
    assert out.shape == (90, 160, 3) and out.dtype == np.uint8 and out.flags.c_contiguous
    assert out.any(), "the clock drew nothing"


def test_the_clock_refuses_an_unknown_option():
    with pytest.raises(OverlayError, match="unknown option"):
        overlays.build({"clock": {"colour": "red"}}, 160, 90)


def test_placement_keys_are_taken_out_before_the_overlay_sees_them():
    clock = overlays.build({"clock": {"anchor": "bottom-left", "format": "{t:.0f}"}}, 160, 90)
    assert clock.placement.anchor == "bottom-left" and clock.fmt == "{t:.0f}"


def test_the_clock_keeps_out_of_the_top_right_corner_by_default():
    """An inset takes the top-right unless told otherwise, so the clock's own default is elsewhere."""
    assert overlays.build("clock", 160, 90).placement.anchor == "top-left"
    assert overlays.Placement.from_spec({}, "x").anchor == "top-right"


# -- the contract ---------------------------------------------------------------------------------


class _Wrong:
    name = "wrong"

    def __init__(self, kind):
        self.kind = kind

    def draw(self, frame, t):
        if self.kind == "rgba":
            return np.zeros((*frame.shape[:2], 4), dtype=np.uint8)
        if self.kind == "float":
            return frame.astype(np.float32)
        return "a string"


@pytest.mark.parametrize("kind", ["rgba", "float", "str"])
def test_a_frame_ffmpeg_could_not_take_is_refused_by_name(kind):
    with pytest.raises(OverlayError, match="overlay 'wrong' returned"):
        overlays.apply_all([_Wrong(kind)], _frame(), 0.0)


def test_overlays_are_found_through_the_entry_point_group(monkeypatch):
    class _Dot:
        name = "dot"

        def __init__(self, placement):
            self.placement = placement

        def draw(self, frame, t):
            frame[0, 0] = 255
            return frame

    class _EP:
        name = "dot"
        value = "some_pkg.video:Dot"

        def load(self):
            return _Dot

    monkeypatch.setattr(overlays, "_entry_points", lambda: (_EP(),))
    assert overlays.available() == {"clock": "roqsim", "dot": "some_pkg.video:Dot"}
    dot = overlays.build("dot", 160, 90)
    assert overlays.apply_all([dot], _frame(), 0.0)[0, 0].tolist() == [255, 255, 255]


def test_an_overlay_whose_import_fails_says_so_by_name(monkeypatch):
    class _EP:
        name = "broken"
        value = "gone:Thing"

        def load(self):
            raise ImportError("No module named 'gone'")

    monkeypatch.setattr(overlays, "_entry_points", lambda: (_EP(),))
    with pytest.raises(OverlayError, match="'broken' is registered .* could not be imported"):
        overlays.build("broken", 160, 90)


def test_an_already_built_overlay_passes_through():
    clock = overlays.ClockOverlay(Placement())
    assert overlays.build(clock, 160, 90) is clock
