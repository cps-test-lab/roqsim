"""``roqsim scenes`` -- build the world a simulation runs in, from whatever the source happens to be.

The tools here are the front-ends and the bake behind them. A world arrives as a published format
(SDF, USD), as an occupancy grid, as a floorplan someone drew, or only as a picture of a map; each
front-end turns one of those into the same ``scene.json``/floorplan contract, and one shared stage
bakes that into the MJCF a world YAML points at. Two more read a finished world back out, as a plan
view or as the Nav2 occupancy grid a planner needs.

Every command wraps a module under :mod:`roqsim_scenes.cli`, keeps that module's own ``--help``, and is
listed by the first line of its docstring. For the reasoning behind any one of them -- the frame
conventions, the failure it refuses, the measurements behind a default -- read the module itself::

    python -m pydoc roqsim_scenes.cli.gridmap_to_world
"""

from __future__ import annotations

import click

from roqsim.commands import tool


@click.group("scenes")
def scenes() -> None:
    """Import, generate and bake the worlds a simulation runs in."""


# Stage 1 -- a source becomes scene.json (+ world-space meshes)
scenes.add_command(tool("roqsim_scenes.cli.sdf_to_scene"))
scenes.add_command(tool("roqsim_scenes.cli.usd_to_scene", blender=True))
scenes.add_command(tool("roqsim_scenes.cli.mjcf_to_world"))
# A Floorplan-DSL room has TWO sources, and only one of them can be collided: the fused mesh for
# looks, the json-ld beside it for the walls. Importing such a room through sdf-to-scene hulls every
# doorway shut -- read this module before reaching for that one.
scenes.add_command(tool("roqsim_scenes.cli.jsonld_to_scene"))

# Stage 1 -- generated rather than imported: no source file, no meshes
scenes.add_command(tool("roqsim_scenes.cli.gridmap_to_world"))
scenes.add_command(tool("roqsim_scenes.cli.gridmap_to_floorplan"))
scenes.add_command(tool("roqsim_scenes.cli.mapimage_to_floorplan"))
scenes.add_command(tool("roqsim_scenes.cli.floorplan_to_world"))
scenes.add_command(tool("roqsim_scenes.dxf_to_floorplan"))

# Stage 2 -- scene.json becomes the MJCF a world loads
scenes.add_command(tool("roqsim_scenes.cli.scene_to_mjcf"))

# Reading a finished world back out
scenes.add_command(tool("roqsim_scenes.cli.scene_to_map"))
scenes.add_command(tool("roqsim_scenes.cli.scene_to_floorplan"))
scenes.add_command(tool("roqsim_scenes.floorplan_to_png"))
scenes.add_command(tool("roqsim_scenes.cad_to_png"))

# Assets a scene depends on
scenes.add_command(tool("roqsim_scenes.cli.fuel_fetch"))
scenes.add_command(tool("roqsim_scenes.cli.world_inputs", "inputs"))
# The other half of the same question: what a world PROVIDES -- its addressable override
# paths, and (on request) the entities it compiles. A caller validating an override cannot
# resolve a world's extends chain without roqsim, so it asks the image that has it.
scenes.add_command(tool("roqsim_scenes.cli.world_describe", "describe"))
