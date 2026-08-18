# SPDX-License-Identifier: Apache-2.0
"""The ``sketch_floorplan_by_human`` MCP tool.

Opens the native 2D floorplan-sketch window (:mod:`floorplan_window`) and blocks until the human
sends their sketch. Like ``review_scene_by_human`` the window is a **subprocess** (via
``window_runner``) because tkinter owns the main thread. The returned sketch is the input the
deterministic world generator (``roqsim scenes floorplan-to-world``) consumes -- this
tool captures human intent, it does not build a world.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from roqsim_scene_builder.window_runner import run_window_subprocess


def sketch_floorplan_by_human(
    message: str = "",
    initial: dict | None = None,
    timeout_s: float | None = None,
    title: str = "",
) -> dict:
    """Ask a human to author a 2D floorplan in a native top-view window; return the structured sketch.

    **Use this tool whenever the human wants to AUTHOR OR EDIT a floorplan interactively** -- not only
    to draw walls, but also to **write or revise room/scene descriptions or rename rooms on an
    existing floorplan** ("let me describe the rooms", "I want to add descriptions", "rename these
    rooms"). Editing is the same call: pass the current floorplan as ``initial`` so the window opens on
    it and the human types the descriptions/names in the window. Do NOT collect descriptions in chat
    and hand-edit the JSON when the human has asked to describe them -- open the window and let them.

    The window has five modes -- **draw** a wall (freehand drag, straightened into lines the moment
    the pencil lifts), **move** a point, place a **door** opening, **mark** a prop (drop a point and
    name/comment it; hold and drag a direction to also set the prop's heading), **delete** a
    wall/door/marker. The canvas is an unbounded plane (wheel zooms,
    right-drag pans, a scale bar shows the length legend) -- there is no overall room size. When the
    human clicks Send you get back a **finished, structured** sketch (no raw-stroke round to process):
    it goes straight to the deterministic generator (``roqsim scenes floorplan-to-world``, which sizes
    the floor from the walls); do not hand-build the world yourself. Props can be marked here (Mark mode) and/or
    added later from comment dots in the 3D review (``review_scene_by_human``).

    Walls are **lines**: each an independent segment ``{"id", "x0_m", "y0_m", "x1_m", "y1_m"}`` with
    its own two endpoints and a **stable id** (shared corners stay separate points, so "move line 3
    two metres left" always means the same wall). Walls that enclose a loop are reported as **rooms**.

    Two ways to start:

    * **human-draws** -- call with ``initial=None``; the human draws from a blank canvas.
    * **review an LLM-made floorplan** -- when the human asks for a layout in words ("a 4-room
      apartment, 120 m2"), YOU draft the walls and pass them as ``initial``; the window opens
      auto-zoomed to your floorplan for the human to comment on and edit, and returns the edited
      sketch. Re-seed with the returned sketch to iterate. Explain what you drafted or changed in
      ``message`` -- **not** in ``initial["comment"]``: the comment box is the human's reply channel
      and always opens empty (any ``comment`` in ``initial`` is ignored), so a note left there would
      just sit in the field the human is meant to type their feedback into.

    **Descriptions vs the comment.** The sketch also carries an object-placement ``description``
    (floorplan-level) and a per-room ``description`` on each room -- free text saying what the space
    is for, so an agent choosing which props to place has the human's intent instead of re-guessing
    it from geometry every time. Unlike ``comment``, these **are** seeded from ``initial`` and
    survive into the scene: draft a ``description`` (e.g. "small startup office: open-plan desks, a
    meeting room, a kitchenette") and per-room ones ("meeting room: seats 6, wall-mounted TV") for
    the human to refine, and they come back edited. The returned sketch is persisted as the scene's
    ``floorplan.json`` (the single authored source of truth, which ``scene.json`` only references), so
    the descriptions live there and re-seed on the next edit -- and because they are not baked into
    geometry, revising a description later is a plain edit to ``floorplan.json`` with no re-run.

    **Merge descriptions/names back by wall-set, do not overwrite the file wholesale.** The window
    re-detects rooms from the walls, so a room that is not a clean closed loop can vanish from the
    returned ``rooms`` and ids can be renumbered; coordinates are also rounded and a door's ``height_m``
    can drop. When the human only edited descriptions/names, match each returned room to the existing
    floorplan by its ``line_ids`` set and copy the ``name``/``description`` across, leaving the
    geometry (all rooms, door ``height_m``, exact coordinates) untouched.

    ``initial`` uses the exact schema this tool returns. Minimal example (one 12x10 m room with a
    door on the south wall)::

        {"lines": [{"id": 1, "x0_m": 0,  "y0_m": 0,  "x1_m": 12, "y1_m": 0},
                   {"id": 2, "x0_m": 12, "y0_m": 0,  "x1_m": 12, "y1_m": 10},
                   {"id": 3, "x0_m": 12, "y0_m": 10, "x1_m": 0,  "y1_m": 10},
                   {"id": 4, "x0_m": 0,  "y0_m": 10, "x1_m": 0,  "y1_m": 0}],
         "doors": [{"id": 1, "line_id": 1, "t": 0.5, "width_m": 0.9}],
         "rooms": [{"id": 1, "name": "living room", "line_ids": [1, 2, 3, 4]}]}

    A valid ``initial``: each line has a unique integer ``id`` and numeric ``x0_m/y0_m/x1_m/y1_m``
    (metres, y up); each door's ``line_id`` matches a line and ``t`` is 0..1 along that wall; a room's
    ``line_ids`` reference existing lines forming a closed loop. Malformed input surfaces as a
    ``RuntimeError`` naming the field the window choked on -- fix it and call again.

    Args:
        title: a short heading shown atop the panel in a larger font (e.g. "Apartment -- 3 rooms");
            optional. Use it for the *what*, and ``message`` for the *instruction*.
        message: instruction shown beside the canvas, and where you explain a drafted/edited
            ``initial`` ("Revised: added a hallway + storeroom -- adjust and send back"). Optional
            but recommended.
        initial: a sketch dict (schema above) to pre-populate the window -- the LLM-drafted floorplan
            to review, or the previous round's returned sketch. ``lines`` keep their ids, ``rooms``
            restore names **and descriptions**, ``markers`` are shown and editable, and the top-level
            ``description`` seeds the scene-description box. Its top-level ``comment`` is ignored (the
            human's comment box always opens empty).
        timeout_s: seconds to wait for a sketch before raising ``TimeoutError`` (default 600).

    Returns:
        ``{"comment": str, "description"?: str, "rooms": [...], "lines": [...], "doors": [...],
        "markers": [...]}`` -- ``comment`` is the human's free-text feedback typed in the window
        (their reply to you; may be empty); ``description`` is the floorplan-level object-placement
        intent (present only when non-empty). rooms first, then lines (no overall dimensions; the
        canvas is unbounded). Each **room** is ``{"id", "name", "line_ids", "description"?}`` (a
        closed loop of walls; ``name`` defaults to ``"room N"``; ``description`` present only when
        non-empty).
        Each **line** is ``{"id", "x0_m", "y0_m", "x1_m", "y1_m"}`` (independent wall segment). Each
        **door** is ``{"id", "line_id", "t", "width_m"}`` -- a standard-width OPENING attached to a
        wall at fraction ``t`` (the generator cuts it out). Each **marker** is
        ``{"id", "x_m", "y_m", "comment", "in_room"}`` -- a prop point whose ``comment`` names the
        model to place, and ``in_room`` is the id of the room containing it (computed, ``null`` if
        outside every room) -- plus ``yaw_deg`` **only when a heading was dragged** (degrees about +Z,
        0 = +x, CCW). Coordinates are metres (2 decimals), y measured from the bottom.

    Raises:
        TimeoutError: if no sketch arrives within the timeout.
        RuntimeError: if the window closed without sending or failed to start.
    """
    extra: list[str] = ["--message", message]
    if title:
        extra += ["--title", title]

    if initial is not None:
        with tempfile.NamedTemporaryFile(
            "w", suffix=".json", prefix="floorplan-seed-", delete=False
        ) as fh:
            json.dump(initial, fh)
            seed_path = fh.name
        try:
            return run_window_subprocess(
                "sketch-floorplan", [*extra, "--initial-json", seed_path], timeout_s=timeout_s
            )
        finally:
            Path(seed_path).unlink(missing_ok=True)

    return run_window_subprocess("sketch-floorplan", extra, timeout_s=timeout_s)


def register(mcp) -> None:
    """Register this module's tools on the FastMCP server."""
    mcp.tool()(sketch_floorplan_by_human)
