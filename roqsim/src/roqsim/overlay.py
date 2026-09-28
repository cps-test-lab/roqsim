# Copyright (C) 2026 Frederik Pasch
#
# SPDX-License-Identifier: Apache-2.0

"""Slot-owned text overlays in the MuJoCo viewer window.

``Handle.set_texts`` replaces the *whole* text overlay and ``clear_texts`` takes all of it down, so
two writers cannot use it directly: the second would silently erase the first, and neither would know
it had happened. This module is the one writer. Callers name a **slot** -- the camera-mode notice, the
key list -- and every change re-flushes every live slot, so a slot only ever removes its own text.

Text is here what images are in :mod:`roqsim.splash`, its sibling on the other half of the same API,
and it takes the same stance: cosmetic, so nothing here raises. An installed MuJoCo without the
overlay API, or a window already closing, costs the caller a debug line and nothing else.

The enums stay with the caller (``mjtFontScale`` / ``mjtGridPos``), which is what keeps this module
free of a ``mujoco`` import -- and testable against a stand-in handle.
"""

from __future__ import annotations

import logging
import weakref

log = logging.getLogger(__name__)

#: The camera-mode notice: which navigation mode a switch just entered. Transient.
SLOT_MODE = "camera-mode"
#: The key list toggled with F1. Sticky until it is toggled off.
SLOT_HELP = "help"

#: Handle -> {slot: (font, gridpos, text1, text2)}. Weakly keyed like :data:`roqsim.viewer._VIEWER_WALK`,
#: so a closed window's text goes with it and callers keep passing the handle they already hold.
#: Insertion-ordered, so the payload's order is the order slots were first written.
_TEXTS: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def set_text(handle, slot: str, *, font, gridpos, text1: str, text2: str = "") -> None:
    """Put ``text1``/``text2`` in ``slot``, leaving every other slot up."""
    slots = _TEXTS.setdefault(handle, {})
    slots[slot] = (font, gridpos, text1, text2)
    _flush(handle)


def clear_text(handle, slot: str) -> None:
    """Take ``slot`` down. Safe for a slot that was never written."""
    slots = _TEXTS.get(handle)
    if not slots or slots.pop(slot, None) is None:
        return
    _flush(handle)


def reflush(handle) -> None:
    """Write every live slot again, for a window that may have dropped them (an in-place reload)."""
    _flush(handle)


def _flush(handle) -> None:
    """Hand MuJoCo the whole overlay: every live slot, or nothing at all. Never raises."""
    slots = _TEXTS.get(handle) or {}
    try:
        if slots:
            handle.set_texts([tuple(entry) for entry in slots.values()])
        else:
            handle.clear_texts()
    except Exception as err:  # noqa: BLE001 — an overlay never breaks the run it is drawn over
        log.debug("viewer overlay text skipped: %s", err)
