"""``roqsim arm`` -- what a recorded arm did, and the vendor files an arm is held to.

Every command wraps a module under :mod:`roqsim_manipulation.cli` and keeps its own ``--help``; the
reasoning behind one is in its module (``python -m pydoc roqsim_manipulation.cli.margins``).
"""

from __future__ import annotations

import click

from roqsim.commands import tool


@click.group("arm")
def arm() -> None:
    """Measure what a recorded arm did; fetch the vendor files an arm is held to."""


arm.add_command(tool("roqsim_manipulation.cli.margins"))
arm.add_command(tool("roqsim_manipulation.cli.franka_upstream"))
