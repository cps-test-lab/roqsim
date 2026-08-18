"""``roqsim assets`` -- bring an external 3D model in and make it scene-ready.

The pipeline runs in one direction and each step is independently useful::

    sketchfab-helper download  ->  reduce-mesh  ->  finalize-mujoco  ->  inspect-prop
    licence-checked source         triangle budget   origin + textures   the verdict

``inspect-prop`` is the one that decides: it is the deterministic ground truth on scale, origin and
materials, and a prop that has not passed it is not ready to place. For a *look* at a mesh, use
``roqsim render prop.obj --out prop.png`` -- rendering is one tool for everything now, and it reads the
mesh through the same preview scene (:mod:`roqsim.mesh_preview`) that ``render-thumbnails`` here uses.
``render-thumbnails`` refreshes the catalogue images after a model changes.

For the reasoning behind any of them -- what a check refuses and why -- read the module::

    python -m pydoc roqsim_assets.cli.inspect_prop
"""

from __future__ import annotations

import click

from roqsim.commands import tool


@click.group("assets")
def assets() -> None:
    """Import, reduce and inspect the props a scene places."""


assets.add_command(tool("roqsim_assets.sketchfab", "sketchfab-helper"))
assets.add_command(tool("roqsim_assets.cli.reduce_mesh"))
assets.add_command(tool("roqsim_assets.cli.finalize_mujoco"))
assets.add_command(tool("roqsim_assets.cli.inspect_prop"))
assets.add_command(tool("roqsim_assets.cli.render_thumbnails"))
