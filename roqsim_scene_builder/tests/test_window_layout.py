# SPDX-License-Identifier: Apache-2.0
"""The review panel must keep its controls on screen whatever the caller writes in ``message``.

``message`` and ``title`` come from the agent calling ``review_scene_by_human``, and nothing bounds
their length. Tk's packer hands out parcels in packing order, so while the Pass/Fail row was packed
last it got whatever cavity the text above it had not already eaten -- with a few hundred words, that
was nothing, and the window opened with no visible way to answer it. These tests measure the real
mapped geometry, because that is the only place the bug existed: every model and handler was fine.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.usefixtures("virtual_display")

LONG_MESSAGE = " ".join(
    f"Sentence {i} of a review question long enough to overflow the panel on any screen."
    for i in range(60)
)


def _app(message: str, title: str = "Long title for the scene under review"):
    """A real ``_ReviewApp`` over a bundled model, mapped on the virtual display."""
    import tkinter as tk

    from roqsim_scene_builder.scene_window import _ReviewApp, load_engine

    from roqsim import FrameRenderer

    engine, _view = load_engine("roqsim_assets:industrial_table")
    width, height = 900, 420  # deliberately short, so the panel has to overflow
    fr = FrameRenderer(engine.ctx.model, width, height)
    app = _ReviewApp(tk, engine, fr, message, None, width, height, title)
    app.root.update()
    return app


def _close(app) -> None:
    app.fr.close()
    app.engine.shutdown()
    app.root.destroy()


def _find(widget, text: str):
    """The first descendant whose ``text`` is ``text`` -- the buttons are not stored on the app."""
    for child in widget.winfo_children():
        try:
            if child.cget("text") == text:
                return child
        except Exception:  # noqa: BLE001 - not every widget has a `text` option
            pass
        found = _find(child, text)
        if found is not None:
            return found
    return None


@pytest.mark.parametrize("label", ["✓ Pass", "✗ Fail"])
def test_verdict_buttons_stay_on_screen_under_a_long_message(label):
    app = _app(LONG_MESSAGE)
    try:
        button = _find(app.root, label)
        assert button is not None, f"{label} was never built"
        assert button.winfo_ismapped(), f"{label} was squeezed out of the layout entirely"
        bottom = button.winfo_rooty() + button.winfo_height()
        limit = app.root.winfo_rooty() + app.root.winfo_height()
        assert bottom <= limit, f"{label} runs {bottom - limit}px past the bottom of the window"
    finally:
        _close(app)


def test_a_long_message_does_not_inflate_the_window():
    """The panel must fit the window rather than resize it.

    This is the failure as a person meets it: the pre-fix panel had no scroll region, so its requested
    height was the height of the whole message and the window grew to match -- 2282px for the message
    below. A window manager caps that at the screen, and everything past the cap, the verdict buttons
    included, is simply off the bottom edge. Measured on a 1280x1024 virtual display, so a window that
    tall cannot hide inside the screen.
    """
    tall = " ".join(LONG_MESSAGE for _ in range(3))
    app = _app(tall)
    try:
        assert app.root.winfo_height() <= 600, (
            f"the message grew the window to {app.root.winfo_height()}px "
            "instead of scrolling inside the panel"
        )
    finally:
        _close(app)


def test_the_long_message_is_what_scrolls():
    """The overflow has to land in the scroll region -- not be clipped, and not push the footer."""
    app = _app(LONG_MESSAGE)
    try:
        canvas = _scroll_canvas(app.root)
        assert canvas is not None, "no scroll region was built"
        top, bottom = (float(v) for v in canvas.cget("scrollregion").split()[1::2])
        assert bottom - top > canvas.winfo_height(), (
            "the message did not overflow the scroll region"
        )
    finally:
        _close(app)


def test_a_short_message_shows_no_scrollbar():
    """The scrollbar is only mapped when it is needed, so a normal panel looks untouched."""
    app = _app("Short question.", title="")
    try:
        canvas = _scroll_canvas(app.root)
        assert canvas is not None
        bars = [w for w in canvas.master.winfo_children() if w.winfo_class() == "Scrollbar"]
        assert bars, "the scroll region has no scrollbar at all"
        assert not bars[0].winfo_ismapped(), "a short panel should not show a scrollbar"
    finally:
        _close(app)


def _scroll_canvas(widget):
    """The panel's scroll canvas: the one carrying a non-empty ``scrollregion``."""
    import tkinter as tk

    if isinstance(widget, tk.Canvas) and widget.cget("scrollregion"):
        return widget
    for child in widget.winfo_children():
        found = _scroll_canvas(child)
        if found is not None:
            return found
    return None
