"""``roqsim walker`` -- bring a rigged human actor into the substrate.

A walker is a skinned character with a locomotion clip, so its import differs from a prop's: the mesh
is only half of it, and the skeleton has to survive the conversion.

    python -m pydoc roqsim_walker.cli.import_actor
"""

from __future__ import annotations

import click

from roqsim.commands import tool


@click.group("walker")
def walker() -> None:
    """Import and prepare the human actors a scene animates."""


walker.add_command(tool("roqsim_walker.cli.import_actor"))
