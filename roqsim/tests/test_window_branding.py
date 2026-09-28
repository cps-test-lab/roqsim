"""The X11 window branding: the title we substitute for MuJoCo's, and the ``_NET_WM_ICON``
payload built from the packaged mark. Neither test needs a display -- the property is built
before any X call, and the title is pure string work."""

from __future__ import annotations

import ctypes

import numpy as np
import pytest

from roqsim.window_branding import _ICON_SIZES, _display_title, _icon_property


@pytest.mark.parametrize(
    "name,expected",
    [
        ("depot", "Roqsim: depot"),
        ("MuJoCo Model", "Roqsim"),  # MuJoCo's default name is never shown
        ("", "Roqsim"),
    ],
)
def test_display_title(name, expected):
    assert _display_title(name) == expected


def test_icon_property_carries_every_size_in_ewmh_layout():
    payload = _icon_property()
    assert payload is not None, "the packaged icon must ship with the wheel"
    # 32-bit property data is an array of C long, and nelements counts the items, not the bytes.
    words = np.frombuffer(payload, dtype=np.uintp)
    assert len(payload) == len(words) * ctypes.sizeof(ctypes.c_ulong)

    sizes, offset = [], 0
    while offset < len(words):
        width, height = int(words[offset]), int(words[offset + 1])
        assert width == height
        sizes.append(width)
        offset += 2 + width * height
    assert offset == len(words), "trailing words: a size's pixel run is mis-counted"
    assert sizes == list(_ICON_SIZES)

    # One X request carries the whole property, so it has to fit a server without BIG-REQUESTS
    # (max request 256 KiB, four wire bytes per item).
    assert len(words) * 4 < 256 * 1024


def test_icon_property_is_opaque_argb():
    words = np.frombuffer(_icon_property(), dtype=np.uintp)
    pixels = words[2 : 2 + _ICON_SIZES[0] ** 2].astype(np.uint32)
    # The mark is a solid badge: its centre is opaque, and alpha lives in the top byte.
    assert (pixels >> 24)[len(pixels) // 2 + _ICON_SIZES[0] // 2] > 0
