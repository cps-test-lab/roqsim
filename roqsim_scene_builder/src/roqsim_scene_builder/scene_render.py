# SPDX-License-Identifier: Apache-2.0
"""The ``render_scene`` MCP tool: look at a world, or at a moment from a recorded run.

Shells out to ``roqsim render`` rather than importing it, matching the precedent the review windows already
set (:mod:`roqsim_scene_builder.window_runner`). Four reasons, and they all point the same way: this MCP
server is long-lived and must not acquire or leak a GL context per call; every render gets a fresh
offscreen context; a MuJoCo compile failure cannot take the server down; and the CLI and this tool cannot
drift, because there is only one implementation of the rendering itself.

**Returns a path by default, not an image.** Every other tool on this server returns a plain ``dict``, and
that convention is worth keeping: an agent reads the returned path with its own file-reading tool, so the
image's tokens (~700 for 960x540, since image tokens run about ``w*h/750``) are paid only if it actually
looks. Ask for ``inline=True`` when the picture should appear in the conversation itself, and prefer a
smaller ``size`` when the question is merely "is anything moving" -- 640x360 is about 310 tokens.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from fastmcp.tools import ToolResult

from roqsim_scene_builder.targets import require_existing

#: Offscreen backend for the subprocess when the host has no display. ``egl`` is the GPU path and the
#: same default the windowed launcher uses; a caller on a GPU-less machine sets ``MUJOCO_GL=osmesa``
#: themselves, which is honoured because this only fills in a value that is absent.
_DEFAULT_GL = "egl"


def _rst() -> str:
    """The ``roqsim`` executable, preferring the one from this interpreter's own environment."""
    candidate = Path(sys.executable).with_name("roqsim")
    if candidate.exists():
        return str(candidate)
    found = shutil.which("roqsim")
    if found is None:
        raise RuntimeError(
            "the 'roqsim' command is not on PATH, so nothing can be rendered. Install the roqsim package "
            "into this environment (`make venv` at the top of this repository)."
        )
    return found


def render_scene(
    target: str = "",
    state: str = "",
    at: float | None = None,
    out: str = "",
    size: str = "960x540",
    view: list[str] | None = None,
    focus: str = "",
    camera: str = "",
    no_ceiling: bool = False,
    inline: bool = False,
) -> dict | ToolResult:
    """Render a world, model or recorded moment to a PNG and return where it is.

    Use this to *see* a scene: whether a ported world looks right, where a robot ended up, what a run
    looked like at some moment. It needs no display and no human -- for a judgement call from a person,
    use ``review_scene_by_human`` instead, which opens a navigable window.

    Args:
        target: what to render -- a world YAML, an MJCF ``.xml`` scene, a model reference
            (``<pkg>:<name>``), or a raw mesh (``.obj``/``.stl``). Optional when ``state`` is given,
            because a recording names the world it came from.
        state: a recording written by ``roqsim sim --record``. With this, the image is a moment from that
            run rather than the world's initial state.
        at: which moment, in *simulated* seconds. Snaps to the nearest recorded sample and reports which
            one it used, so a caller can see it landed a few milliseconds off rather than assume it did
            not. Omit to get the last sample -- "what did this run end up looking like".
        out: where to write the PNG. Defaults to a temp file, whose path comes back in the result.
        size: ``WxH``. Smaller is cheaper to look at: 640x360 is roughly 310 image tokens against 700
            for 960x540.
        view: camera overrides, as ``KEY=VALUE`` strings using the world's own ``sim.view`` vocabulary
            (``lookat``, ``distance``, ``azimuth``, ``elevation``, ``track``, ``follow_heading``). Each
            key given replaces just that one, so ``["elevation=-85"]`` looks down without disturbing the
            rest. A vector is written comma- or space-separated: ``["lookat=-3.2 -1.3 1.9"]``.
        focus: an entity or body to frame on, searching for a viewpoint with a clear line of sight --
            which is what you want indoors, where a wall is usually between the default camera and the
            thing you care about.
        camera: render through a fixed MJCF ``<camera>`` (what a robot's own camera sees). Owns its pose,
            so it cannot be combined with ``view``/``focus``.
        no_ceiling: drop a roofed world's ceiling, to look into it from above.
        inline: return the image itself so it appears in the conversation, instead of only its path.
            Costs the image's tokens; the default costs about forty.

    Returns:
        ``{"path", "width", "height", "camera", "nbody", "ngeom"}``, plus ``{"sim_time",
        "sample_index", "requested_at", "at_error"}` when rendering from a recording. With
        ``inline=True`` the same record comes back as the result's structured content, with the image
        itself alongside it as an image block.

    Raises:
        FileNotFoundError: if ``target`` or ``state`` names a file that does not exist.
        RuntimeError: if rendering fails -- the message is ``roqsim render``'s own.
    """
    if target:
        require_existing(target, "target")
    # A recording is always a path -- there is no reference form of one, so it is checked outright.
    if state and not Path(state).exists():
        raise FileNotFoundError(f"state: no such file: {state}")
    if not target and not state:
        raise ValueError("give a target to render, or a state recording to render from")

    destination = (
        Path(out) if out else Path(tempfile.mkdtemp(prefix="roqsim-render-")) / "render.png"
    )
    argv = [_rst(), "render", "--out", str(destination), "--size", size]
    if target:
        argv.insert(2, target)
    if state:
        argv += ["--state", state]
    if at is not None:
        argv += ["--at", str(at)]
    if view:
        argv += ["--view", *view]
    if focus:
        argv += ["--focus", focus]
    if camera:
        argv += ["--camera", camera]
    if no_ceiling:
        argv.append("--no-ceiling")

    env = dict(os.environ)
    env.setdefault("MUJOCO_GL", _DEFAULT_GL)
    proc = subprocess.run(argv, capture_output=True, text=True, env=env, check=False)  # noqa: S603
    if proc.returncode != 0:
        # `roqsim render`'s own message is the useful one -- it names the missing camera, the unknown view
        # key, or the GL backend to set. Passing it through beats paraphrasing it.
        detail = (proc.stderr or proc.stdout or "").strip().splitlines()
        raise RuntimeError(
            detail[-1] if detail else f"roqsim render failed (exit {proc.returncode})"
        )

    record = json.loads(proc.stdout.strip().splitlines()[-1])
    if not inline:
        return record
    from fastmcp.utilities.types import Image

    # An image has to be its own content block, so the inline case returns the result explicitly rather
    # than putting the Image in the dict: a dict holding one is serialized like any other value, so the
    # picture arrived as the repr of a Python object *and* the record lost its structured form (an Image
    # is not JSON-serializable, so the whole dict was dropped from structuredContent while the tool still
    # advertised an output schema -- the client then rejected the call it had just made succeed).
    return ToolResult(content=[Image(path=record["path"])], structured_content=record)


def register(mcp) -> None:
    """Register this module's tools on the FastMCP server."""
    # The output schema is stated rather than inferred: the inline branch returns a ToolResult, which
    # tells FastMCP nothing about the record's shape, and a client that validates would otherwise lose
    # the promise that a render always answers with an object.
    mcp.tool(output_schema={"type": "object", "additionalProperties": True})(render_scene)
