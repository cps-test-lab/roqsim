"""``roqsim walker`` -- bring rigged human actors and their motion into the substrate.

A walker is a skinned character moved by clips, so its import differs from a prop's: the mesh is
only half of it, and the skeleton has to survive the conversion. ``import-actor`` brings the
character, ``import-clip`` a motion capture it plays.

    python -m pydoc roqsim_walker.cli.import_actor
    python -m pydoc roqsim_walker.cli.import_clip
"""

from __future__ import annotations

import click

from roqsim.commands import tool


@click.group("walker")
def walker() -> None:
    """Import and prepare the human actors a scene animates, and their motion."""


walker.add_command(tool("roqsim_walker.cli.import_actor"))
walker.add_command(tool("roqsim_walker.cli.import_clip"))
walker.add_command(tool("roqsim_walker.cli.bench_hands"))
