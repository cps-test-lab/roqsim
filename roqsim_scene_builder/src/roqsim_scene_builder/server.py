# SPDX-License-Identifier: Apache-2.0
"""FastMCP server for the roqsim scene builder.

Thin: each feature module exposes ``register(mcp)`` and :func:`create_server` calls them from a
hardcoded list. A new review mode is a new module plus one line here.
"""

from __future__ import annotations

from fastmcp import FastMCP

from roqsim_scene_builder import floorplan_sketch, scene_render, scene_review

_INSTRUCTIONS = """\
Eyes on an roqsim scene: two native windows a human answers in, and one headless render you look at
yourself.

Use review_scene_by_human to open a MuJoCo scene (world, MJCF, or model reference) in a 3D window
where a human can look around and drop comment dots, and block until they return a pass/fail verdict.
With Move Objects mode on, the human can also grab a spawn_model prop and drag it across the floor
(Shift-drag to rotate); the new poses come back under "moves" for the caller to write into the world.

Use render_scene to LOOK at a world yourself -- no window, no human. It renders a world, model, mesh or
a moment from a recorded run (`roqsim sim --record`) to a PNG and returns the path, so reading the image is
your choice and its tokens are only spent when you look. Ask for inline=True to have the picture appear
in the conversation. This is the right tool for "does this world look right", "where did the robot end
up", "what did the run look like at t=12.5"; review_scene_by_human is for when a *person* must judge.

Use sketch_floorplan_by_human to open a 2D top-view window where a human draws the WALLS of a
floorplan (freehand strokes are straightened into lines immediately) and returns a finished,
structured sketch in metres: rooms (closed loops, nameable) + lines (independent wall segments with
stable ids) + door openings. Props are not placed there -- they come from comment dots in the 3D
review. The returned sketch feeds the deterministic world generator (`roqsim scenes floorplan-to-world`)
directly; the tool captures human intent, it does not build the world.
"""

# Feature modules contributing tools. Extend as new review modes are added.
_FEATURE_MODULES = (scene_review, floorplan_sketch, scene_render)


def create_server() -> FastMCP:
    """Create and configure the MCP server instance."""
    mcp = FastMCP(name="roqsim scene builder", instructions=_INSTRUCTIONS)
    for module in _FEATURE_MODULES:
        module.register(mcp)
    return mcp
