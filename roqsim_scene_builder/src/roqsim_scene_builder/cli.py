# SPDX-License-Identifier: Apache-2.0
"""CLI for the roqsim scene builder.

Three commands, reachable both as ``roqsim-scene-builder <cmd>`` and as ``roqsim builder <cmd>`` (this
group is what the ``roqsim.commands`` entry point registers):

* ``serve`` starts the MCP server -- what an MCP client is configured to launch over stdio.
* ``review-scene`` opens the native 3D scene-review window directly, with no MCP client in the loop.
* ``sketch-floorplan`` opens the 2D floorplan-sketch window the same way.

The two window commands are both the human debug entry and what the corresponding MCP tool spawns as
a subprocess.
"""

from __future__ import annotations

import logging
import sys
import tempfile
from pathlib import Path

import click


def _parse_size(size: str, example: str) -> tuple[int, int]:
    """``"960x720"`` -> ``(960, 720)``, as a click parameter error when it is not that.

    Not :func:`roqsim.render.parse_size`, which is the *video* frame parser: it rounds up to an even
    size because ``yuv420p`` demands it and raises ``RenderError``. A window has neither constraint,
    and a CLI wants ``BadParameter``.
    """
    try:
        w, h = (int(v) for v in size.lower().split("x", 1))
    except ValueError:
        raise click.BadParameter(f"--size must look like {example}, got {size!r}") from None
    return w, h


@click.group()
def main() -> None:
    """Open, or serve, the windows a human uses to build and judge an roqsim scene."""


@main.command(name="serve")
@click.option(
    "--transport",
    type=click.Choice(["stdio", "sse", "streamable-http"]),
    default="stdio",
    show_default=True,
    help="Transport to use. stdio is what an MCP client launches.",
)
@click.option(
    "--host",
    default="127.0.0.1",
    show_default=True,
    help="Host to bind when using an HTTP transport.",
)
@click.option(
    "--port",
    default=8812,
    show_default=True,
    type=int,
    help="Port to bind when using an HTTP transport.",
)
def serve(transport: str, host: str, port: int) -> None:
    """Start the scene-builder MCP server."""
    from roqsim_scene_builder.server import create_server

    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("mcp").setLevel(logging.CRITICAL)

    server = create_server()
    try:
        if transport in ("sse", "streamable-http"):
            server.run(transport=transport, host=host, port=port)
        else:
            server.run(transport=transport)
    except KeyboardInterrupt:
        pass


@main.command(name="review-scene")
@click.argument("target")
@click.option("--title", default="", help="Title shown atop the panel, in a larger font.")
@click.option("--message", "-m", default="", help="Question/instruction shown beside the scene.")
@click.option(
    "--settle-steps",
    default=0,
    type=int,
    help="Advance physics this many steps before showing the scene.",
)
@click.option(
    "--focus-object",
    "focus_object",
    default="",
    help="Name of a scene object to point the initial camera at (zoomed, unoccluded). "
    "Empty = automatic camera.",
)
@click.option(
    "--size", default="960x720", show_default=True, help="Render size WxH, e.g. 1280x800."
)
@click.option(
    "--json-out",
    "json_out",
    default=None,
    type=click.Path(),
    help="Write the verdict JSON to this file (used by the MCP tool).",
)
def review_scene(
    target: str,
    title: str,
    message: str,
    settle_steps: int,
    focus_object: str,
    size: str,
    json_out: str | None,
) -> None:
    """Open the scene-review window for TARGET (world YAML, MJCF, or model ref).

    Prints the verdict JSON and exits: 0 pass (or a neutral Enter-submitted comment), 1 fail, 2
    no-display/load-error, 3 closed without a verdict. TARGET accepts exactly what ``roqsim``
    accepts.
    """
    from roqsim_scene_builder.scene_window import run_window

    w, h = _parse_size(size, "960x720")

    # Always render to a file so we can echo the verdict for the human; the MCP tool passes its own
    # --json-out and reads that instead of stdout.
    out = json_out
    tmp = None
    if out is None:
        tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        tmp.close()
        out = tmp.name

    code = run_window(
        target,
        message=message,
        settle_steps=settle_steps,
        json_out=out,
        size=(w, h),
        title=title,
        focus_object=focus_object,
    )
    if code in (0, 1):
        click.echo(Path(out).read_text(encoding="utf-8"))
    if tmp is not None:
        Path(out).unlink(missing_ok=True)
    sys.exit(code)


@main.command(name="sketch-floorplan")
@click.option("--title", default="", help="Title shown atop the panel, in a larger font.")
@click.option("--message", "-m", default="", help="Instruction shown beside the canvas.")
@click.option(
    "--initial-json",
    "initial_json",
    default=None,
    type=click.Path(exists=True),
    help="A sketch JSON to pre-seed the window with (review an LLM-made floorplan).",
)
@click.option("--size", default="760x760", show_default=True, help="Canvas size WxH, e.g. 760x760.")
@click.option(
    "--json-out",
    "json_out",
    default=None,
    type=click.Path(),
    help="Write the sketch JSON to this file (used by the MCP tool).",
)
def sketch_floorplan(title, message, initial_json, size, json_out):
    """Open the 2D floorplan-sketch window and print the sketch JSON on send.

    Exits: 0 sent, 2 no-display, 3 closed without sending.
    """
    import json

    from roqsim_scene_builder.floorplan_window import run_window

    w, h = _parse_size(size, "760x760")

    initial = json.loads(Path(initial_json).read_text(encoding="utf-8")) if initial_json else None

    out = json_out
    tmp = None
    if out is None:
        tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        tmp.close()
        out = tmp.name

    code = run_window(message=message, initial=initial, json_out=out, size=(w, h), title=title)
    if code == 0:
        click.echo(Path(out).read_text(encoding="utf-8"))
    if tmp is not None:
        Path(out).unlink(missing_ok=True)
    sys.exit(code)


if __name__ == "__main__":
    main()
