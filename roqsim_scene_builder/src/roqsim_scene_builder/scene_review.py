# SPDX-License-Identifier: Apache-2.0
"""The ``review_scene_by_human`` MCP tool.

Opens a native MuJoCo scene-review window and blocks until the human returns a verdict. The window
is a **subprocess** (``roqsim-scene-builder review-scene``) because tkinter owns the main thread, so it
cannot run inside the FastMCP worker thread; the subprocess writes the verdict JSON to a temp file
which this tool reads back. Keeping the GUI out-of-process also means a crash in rendering can never
take the MCP server down.
"""

from __future__ import annotations

from roqsim_scene_builder.targets import require_existing
from roqsim_scene_builder.window_runner import run_window_subprocess


def review_scene_by_human(
    target: str,
    message: str = "",
    settle_steps: int = 0,
    timeout_s: float | None = None,
    title: str = "",
    focus_object: str = "",
) -> dict:
    """Ask a human to review a 3D scene in a native window and return their verdict.

    Opens the same thing ``roqsim`` would open -- a world YAML, an MJCF ``.xml`` scene, or a
    model/robot reference (``<pkg>:<name>``) shown in an empty room -- in a window the human can
    navigate first-person -- **left-drag looks, WASD or the arrows walk** (``Q``/``E`` or Page
    Up/Down for down/up, Shift faster),
    dropping numbered comment *dots* on anything worth noting and (in Move Objects
    mode) dragging props to reposition them. Blocks until they submit -- Pass, Fail, or Enter in the
    comment box for a neutral note. Use it to get a human judgement on a ported scene, a robot
    placement, or a world layout that a single rendered image cannot convey.

    A dot is dropped by double-clicking a surface; holding that second click and **dragging a
    direction** gives the dot a heading (``yaw_deg``), useful when the dot marks a prop to place --
    take it as the prop's yaw. A plain double-click leaves the dot headingless.

    Turning on **Move Objects** lets the human grab a ``spawn_model`` prop and slide it across the floor
    (drag) or turn it (Shift-drag); on release the prop's new pose comes back under ``moves``. The
    window never mutates the world -- it reports intent, and the caller (the ``scene-update`` skill)
    writes the pose into the world YAML. Only props move; baked walls and floor cannot.

    Pass ``message`` with the specific thing to judge ("Is the shelving reachable and not clipping
    the wall?"); the human sees it beside the scene and answers in the comment and dots. Pressing
    Enter in the comment box (with text) submits a neutral ``"comment"`` verdict -- a note without a
    pass/fail call.

    Args:
        target: World YAML, MJCF ``.xml``, or a model reference -- anything ``roqsim`` accepts.
        title: A short heading shown atop the panel in a larger font; optional. Use it for the
            *what* under review, and ``message`` for the *question*.
        message: The question/instruction shown in the window. Optional but recommended.
        settle_steps: Advance physics this many steps before showing the scene (let it come to
            rest). Default 0 (show the loaded state as-is).
        focus_object: Optional name of a scene object to point the initial camera at -- the same
            name used in a world's ``spawn_model`` ``name:`` (and returned under ``moves``). The
            window opens zoomed to fill the view with that object, from a viewing angle chosen to
            have a clear line of sight to it (it looks over/around walls so the object is not hidden
            behind one). Empty (the default) keeps the automatic camera -- a model preview for a bare
            model ref, otherwise the world's ``sim.view`` or MuJoCo's default. An unknown name is not
            an error: it warns and falls back to the automatic camera.
        timeout_s: Seconds to wait for a verdict before raising ``TimeoutError`` (default 600).

    Returns:
        ``{"verdict": "pass" | "fail" | "comment", "comment": str, "annotations": [...],
        "moves": [...]}`` (``"comment"`` = an Enter-submitted note, neither pass nor fail). Each
        annotation is a dot the human dropped on a surface: ``{"id", "world": [x, y, z], "target":
        {"geom", "body"} | null, "comment"}`` -- ``world`` is the 3D hit point and ``target`` names
        the geom/body it landed on -- plus ``yaw_deg`` (heading about +Z, 0 = +x, CCW) **only when a
        heading was dragged**. Each move is a prop the human repositioned in Move-Objects mode:
        ``{"entity": str, "model": str, "pos": [x, y, z], "yaw_deg": float}`` -- ``entity`` matches the
        world YAML ``spawn_model`` ``name:``. ``moves`` is empty when nothing was moved.

    Raises:
        FileNotFoundError: If ``target`` is a path to a file that does not exist.
        TimeoutError: If no verdict arrives within the timeout.
        RuntimeError: If the review window closed without a verdict or failed to start.
    """
    require_existing(target, "Scene to review")
    extra = ["--message", message, "--settle-steps", str(int(settle_steps))]
    if title:
        extra += ["--title", title]
    if focus_object:
        extra += ["--focus-object", focus_object]
    return run_window_subprocess("review-scene", extra, timeout_s=timeout_s, target=target)


def register(mcp) -> None:
    """Register this module's tools on the FastMCP server."""
    mcp.tool()(review_scene_by_human)
