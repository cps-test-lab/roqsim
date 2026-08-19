# SPDX-License-Identifier: Apache-2.0
"""Shared building blocks for the scene-builder's human-annotation windows.

Both windows -- the 3D scene review (:mod:`scene_window`) and the 2D floorplan sketch
(:mod:`floorplan_window`) -- present the same thing: a canvas on the left, and on the right a stack
of **numbered comment rows** plus a general comment box and a button row, over one dark theme. This
module owns those shared parts so neither window re-implements them:

* the dark palette and the per-number marker colours (:func:`color_for`, :func:`rgba_hex`),
* :func:`renumber`, the 1..N contiguous renumbering both point models use after a delete,
* :func:`build_point_rows`, :func:`build_comment_box`, :func:`build_button_row` -- the right-panel
  widgets, driven by whatever items the window holds,
* :func:`build_scrollable`, the panel's one scrolling region -- what keeps a long ``message`` from
  pushing the submit buttons out of the window.

Everything here is either pure (theme/colours/renumber, tested headless) or a thin tkinter widget
factory; the window-specific left canvas, picking, and result assembly stay in each window module.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Protocol

# Dark palette shared by every widget (tkinter has no real theming; we colour widgets by hand).
BG = "#1e1e1e"
PANEL = "#252526"
FG = "#e8e8e8"
MUTED = "#9a9a9a"
ENTRY_BG = "#2a2a2a"
BORDER = "#444444"
PASS_BG = "#1f6b2e"
FAIL_BG = "#7a1f1f"
SEND_BG = "#1f4f7a"

# Per-item marker colours (rgba, 0..1), cycled by number, used for both the canvas marker and the
# row's colour swatch so an item in the canvas is easy to match to its comment.
_PALETTE = [
    (0.95, 0.77, 0.06, 1.0),
    (0.20, 0.60, 0.86, 1.0),
    (0.18, 0.80, 0.44, 1.0),
    (0.90, 0.30, 0.24, 1.0),
    (0.61, 0.35, 0.71, 1.0),
    (0.90, 0.49, 0.13, 1.0),
    (0.10, 0.74, 0.61, 1.0),
    (0.93, 0.51, 0.93, 1.0),
]


def color_for(item_id: int) -> tuple[float, float, float, float]:
    """Marker rgba for a 1-based item number (cycles the palette)."""
    return _PALETTE[(item_id - 1) % len(_PALETTE)]


def rgba_hex(rgba: tuple[float, float, float, float]) -> str:
    """``#rrggbb`` for a 0..1 rgba tuple (for tkinter swatches)."""
    return "#{:02x}{:02x}{:02x}".format(*(int(round(c * 255)) for c in rgba[:3]))


def renumber(items: list) -> None:
    """Reassign ``.id`` 1..N in list order, so ids stay contiguous after a delete."""
    for i, item in enumerate(items, start=1):
        item.id = i


class Numbered(Protocol):
    """What :func:`build_point_rows` needs from an item: a number, a label, and a comment."""

    id: int
    comment: str

    @property
    def label(self) -> str: ...


def build_point_rows(
    tk,
    parent,
    items: Sequence[Numbered],
    on_delete: Callable[[int], None],
    color: Callable[[int], tuple[float, float, float, float]] = color_for,
    with_comment: bool = True,
) -> dict[int, object]:
    """(Re)build one numbered comment row per item under ``parent``.

    Each row is a colour swatch (``color(item.id)``), the item's ``label``, a ``✕`` delete button
    (calls ``on_delete(item.id)``) and -- unless ``with_comment`` is false (points that carry no
    text, like wall vertices) -- a per-item comment ``Entry`` that live-writes back into
    ``item.comment``. Clears ``parent`` first, so callers just call this after any change.

    Returns ``{item.id: Entry}`` for the rows that have one, so a caller can focus the field
    belonging to a point (e.g. after the point was clicked or created).
    """
    for child in parent.winfo_children():
        child.destroy()
    entries: dict[int, object] = {}
    for item in items:
        row = tk.Frame(parent, bg=PANEL)
        row.pack(fill="x", pady=(4, 0))
        head = tk.Frame(row, bg=PANEL)
        head.pack(fill="x")
        tk.Label(
            head,
            text=f" {item.id} ",
            bg=rgba_hex(color(item.id)),
            fg="#1e1e1e",
            font=("TkDefaultFont", 9, "bold"),
        ).pack(side="left")
        tk.Label(
            head,
            text=f" {item.label}",
            bg=PANEL,
            fg=FG,
            anchor="w",
            font=("TkDefaultFont", 9),
        ).pack(side="left", fill="x", expand=True)
        tk.Button(
            head,
            text="✕",
            command=lambda i=item.id: on_delete(i),
            bg=PANEL,
            fg="#ff8888",
            activebackground=PANEL,
            activeforeground="#ffaaaa",
            relief="flat",
            bd=0,
            highlightthickness=0,
            padx=2,
            pady=0,
            font=("TkDefaultFont", 9),
        ).pack(side="right")
        if not with_comment:
            continue
        entry = tk.Entry(
            row,
            bg=ENTRY_BG,
            fg=FG,
            insertbackground=FG,
            relief="flat",
            highlightthickness=1,
            highlightbackground=BORDER,
        )
        entry.insert(0, item.comment)
        entry.pack(fill="x", pady=(2, 0))
        entry.bind("<KeyRelease>", lambda e, it=item, ent=entry: setattr(it, "comment", ent.get()))
        entries[item.id] = entry
    return entries


def enable_edit_shortcuts(root) -> None:
    """Wire the usual clipboard/select-all keys in every ``Entry``/``Text`` of ``root``'s interpreter.

    Copy/paste/cut (Ctrl+C/V/X) already work through Tk's built-in ``<<Copy>>``/``<<Paste>>``/
    ``<<Cut>>`` class bindings; the gap on X11 is **Ctrl+A**, which defaults to line-start rather than
    select-all. We replace that one class binding (and its caps-lock variant). Bound per widget
    *class*, so it covers every current and future Entry/Text -- the general comment box, the per-item
    comment fields, and the room-name entries -- without touching each widget.
    """

    def _entry_select_all(event):
        event.widget.select_range(0, "end")
        event.widget.icursor("end")
        return "break"

    def _text_select_all(event):
        event.widget.tag_add("sel", "1.0", "end-1c")
        event.widget.mark_set("insert", "end-1c")
        return "break"

    for seq in ("<Control-a>", "<Control-A>"):
        root.bind_class("Entry", seq, _entry_select_all)
        root.bind_class("Text", seq, _text_select_all)


def build_scrollable(tk, parent, *, padx: int = 12, pady: tuple[int, int] = (6, 0)):
    """The panel's scrolling region, filling whatever space is left; returns ``(inner, sync)``.

    Pack the **footer first**, ``side="bottom"``, and call this after: Tk's packer hands out parcels
    in packing order, so a footer that is packed last gets whatever cavity the content above it did
    not eat -- which is nothing, once a caller passes a long enough ``message``. That is how the
    submit buttons went missing. With the footer's parcel already claimed, everything variable (the
    title, the message, the tool row, the item rows) goes in ``inner`` and scrolls instead of pushing.

    ``sync`` re-measures: call it after adding or removing rows programmatically, since a change made
    outside the geometry manager's own resize path produces no ``<Configure>``. The scrollbar is only
    mapped while the content actually overflows, so a short panel looks exactly as it did before.

    The wheel is bound on ``bind_all`` between ``<Enter>`` and ``<Leave>`` of the region rather than
    on the canvas: the pointer is almost always over a row's frame/label/entry, not the canvas, and a
    per-widget binding misses every one of them. Binding only while the pointer is inside the region
    is what leaves the wheel over the drawing canvas free to keep zooming.
    """
    wrap = tk.Frame(parent, bg=PANEL)
    wrap.pack(side="top", fill="both", expand=True, padx=padx, pady=pady)
    bar = tk.Scrollbar(
        wrap,
        orient="vertical",
        bg="#3a3a3a",
        troughcolor=PANEL,
        activebackground="#555555",
        relief="flat",
        bd=0,
        highlightthickness=0,
        width=10,
    )
    canvas = tk.Canvas(wrap, bg=PANEL, highlightthickness=0, yscrollcommand=bar.set)
    bar.config(command=canvas.yview)
    canvas.pack(side="left", fill="both", expand=True)
    inner = tk.Frame(canvas, bg=PANEL)
    window = canvas.create_window((0, 0), window=inner, anchor="nw")

    def sync(_event=None) -> None:
        canvas.configure(scrollregion=canvas.bbox("all"))
        # Match the inner frame to the viewport, or a `fill="x"` child would size to its own request
        # and the wrapped labels would never learn how wide they are allowed to be.
        canvas.itemconfigure(window, width=canvas.winfo_width())
        overflow = inner.winfo_reqheight() > canvas.winfo_height()
        if overflow and not bar.winfo_ismapped():
            # before the canvas, so it reserves the right edge -- the canvas's expand would else eat it
            bar.pack(side="right", fill="y", before=canvas)
        elif not overflow and bar.winfo_ismapped():
            bar.pack_forget()

    inner.bind("<Configure>", sync)
    canvas.bind("<Configure>", sync)

    def _wheel(event):
        if inner.winfo_reqheight() <= canvas.winfo_height():
            return  # nothing to scroll; leave the event alone
        down = getattr(event, "num", 0) == 5 or getattr(event, "delta", 0) < 0
        canvas.yview_scroll(1 if down else -1, "units")

    def _grab(_event):
        for seq in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            canvas.bind_all(seq, _wheel)

    def _release(_event):
        for seq in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            canvas.unbind_all(seq)

    wrap.bind("<Enter>", _grab)
    wrap.bind("<Leave>", _release)
    return inner, sync


def build_comment_box(tk, parent, height: int = 5, label: str = "Comment"):
    """A labelled multi-line ``Text`` widget; returns the ``Text``. ``label`` heads it (default
    ``"Comment"`` for the feedback box; pass e.g. ``"Scene description"`` for a persistent field)."""
    tk.Label(parent, text=label, bg=PANEL, fg=MUTED, anchor="w").pack(
        fill="x", padx=12, pady=(10, 2)
    )
    text = tk.Text(
        parent,
        height=height,
        bg=ENTRY_BG,
        fg=FG,
        insertbackground=FG,
        highlightthickness=1,
        highlightbackground=BORDER,
        relief="flat",
        wrap="word",
    )
    text.pack(fill="x", padx=12)
    return text


def add_tooltip(tk, widget, text: str) -> None:
    """Show ``text`` in a small popup while the cursor hovers ``widget`` -- for carrying a longer hint
    off a short button label. No-op for empty ``text``."""
    if not text:
        return
    state = {"win": None}

    def show(_event=None):
        if state["win"] is not None:
            return
        win = tk.Toplevel(widget)
        win.wm_overrideredirect(True)
        win.wm_geometry(
            f"+{widget.winfo_rootx() + 8}+{widget.winfo_rooty() + widget.winfo_height() + 2}"
        )
        tk.Label(
            win,
            text=text,
            bg="#333333",
            fg=FG,
            relief="solid",
            bd=1,
            justify="left",
            font=("TkDefaultFont", 9),
            padx=6,
            pady=3,
        ).pack()
        state["win"] = win

    def hide(_event=None):
        if state["win"] is not None:
            state["win"].destroy()
            state["win"] = None

    widget.bind("<Enter>", show)
    widget.bind("<Leave>", hide)


def build_button_row(tk, parent, buttons: Sequence[tuple[str, Callable[[], None], str]]) -> None:
    """A bottom row of equal-width action buttons: ``(text, command, bg)`` each."""
    row = tk.Frame(parent, bg=PANEL)
    row.pack(fill="x", padx=12, pady=12, side="bottom")
    for i, (text, command, bg) in enumerate(buttons):
        pad = (0, 6) if i == 0 and len(buttons) > 1 else (6, 0) if i else (0, 0)
        tk.Button(
            row,
            text=text,
            command=command,
            bg=bg,
            fg="#ffffff",
            activebackground=bg,
            activeforeground="#ffffff",
            relief="flat",
            font=("TkDefaultFont", 13, "bold"),
        ).pack(side="left", expand=True, fill="x", padx=pad, ipady=8)
