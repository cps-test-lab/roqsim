# Copyright (C) 2026 Frederik Pasch
#
# SPDX-License-Identifier: Apache-2.0

"""The viewer's text overlay, shared by slot.

``set_texts`` replaces the whole overlay, so the bug this module exists to prevent is one writer
silently wiping another's text. That is what the first test here is.
"""

from unittest import mock

from roqsim import overlay

_FONT, _TOP, _BOTTOM = 150, "topright", "bottomleft"


class _Handle:
    """A stand-in viewer handle recording what it was handed. Weak-referenceable, as MuJoCo's is."""

    def __init__(self):
        self.payloads = []
        self.cleared = 0

    def set_texts(self, texts):
        self.payloads.append(texts)

    def clear_texts(self):
        self.cleared += 1


def _slots(handle):
    """The gridpos of each entry in the last payload, in order."""
    return [entry[1] for entry in handle.payloads[-1]]


def test_a_second_slot_does_not_take_the_first_ones_text_down():
    # The regression this module exists for: the mode notice and the key list are written by
    # different callers, and set_texts replaces the whole overlay.
    handle = _Handle()
    overlay.set_text(handle, overlay.SLOT_MODE, font=_FONT, gridpos=_BOTTOM, text1="camera")
    overlay.set_text(handle, overlay.SLOT_HELP, font=_FONT, gridpos=_TOP, text1="roqsim keys")
    assert _slots(handle) == [_BOTTOM, _TOP]


def test_clearing_one_slot_leaves_the_other_up():
    handle = _Handle()
    overlay.set_text(handle, overlay.SLOT_MODE, font=_FONT, gridpos=_BOTTOM, text1="camera")
    overlay.set_text(handle, overlay.SLOT_HELP, font=_FONT, gridpos=_TOP, text1="roqsim keys")
    overlay.clear_text(handle, overlay.SLOT_MODE)
    assert _slots(handle) == [_TOP]
    assert handle.cleared == 0


def test_clearing_the_last_slot_clears_the_window():
    handle = _Handle()
    overlay.set_text(handle, overlay.SLOT_HELP, font=_FONT, gridpos=_TOP, text1="roqsim keys")
    overlay.clear_text(handle, overlay.SLOT_HELP)
    assert handle.cleared == 1


def test_clearing_a_slot_that_was_never_written_says_nothing_to_the_window():
    handle = _Handle()
    overlay.clear_text(handle, overlay.SLOT_HELP)
    assert handle.payloads == [] and handle.cleared == 0


def test_both_columns_reach_the_window():
    handle = _Handle()
    overlay.set_text(handle, overlay.SLOT_HELP, font=_FONT, gridpos=_TOP, text1="F1", text2="help")
    assert handle.payloads[-1] == [(_FONT, _TOP, "F1", "help")]


def test_rewriting_a_slot_keeps_its_place():
    handle = _Handle()
    overlay.set_text(handle, overlay.SLOT_MODE, font=_FONT, gridpos=_BOTTOM, text1="mouse")
    overlay.set_text(handle, overlay.SLOT_HELP, font=_FONT, gridpos=_TOP, text1="keys")
    overlay.set_text(handle, overlay.SLOT_MODE, font=_FONT, gridpos=_BOTTOM, text1="fly")
    assert _slots(handle) == [_BOTTOM, _TOP]
    assert handle.payloads[-1][0][2] == "fly"


def test_two_windows_do_not_share_their_slots():
    one, two = _Handle(), _Handle()
    overlay.set_text(one, overlay.SLOT_HELP, font=_FONT, gridpos=_TOP, text1="keys")
    overlay.set_text(two, overlay.SLOT_MODE, font=_FONT, gridpos=_BOTTOM, text1="camera")
    assert _slots(one) == [_TOP] and _slots(two) == [_BOTTOM]


def test_a_window_without_the_overlay_api_costs_nothing_but_the_text():
    # Same stance as the loading splash: cosmetic, so an installed MuJoCo without it, or a window
    # already closing, is a debug line and not an exception into the render loop.
    handle = _Handle()
    handle.set_texts = mock.Mock(side_effect=AttributeError("no overlay here"))
    overlay.set_text(handle, overlay.SLOT_HELP, font=_FONT, gridpos=_TOP, text1="keys")
    overlay.clear_text(handle, overlay.SLOT_HELP)  # and the slot still went, so state stays sane


def test_reflush_writes_the_live_slots_again():
    # For a window that may have dropped them under it -- MuJoCo's in-place `sim.load`.
    handle = _Handle()
    overlay.set_text(handle, overlay.SLOT_HELP, font=_FONT, gridpos=_TOP, text1="keys")
    overlay.reflush(handle)
    assert len(handle.payloads) == 2 and handle.payloads[-1] == handle.payloads[-2]
