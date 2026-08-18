"""``roqsim sensors`` -- what the robot can actually see.

Coverage is the one sensor question a static world can answer offline: how much of a room, and which
objects, are observed by how many sensors. Everything else about a sensor is a runtime plugin.

    python -m pydoc roqsim_sensors.coverage.cli
"""

from __future__ import annotations

import click

from roqsim.commands import tool


@click.group("sensors")
def sensors() -> None:
    """Estimate and optimise sensor coverage of a fixed world."""


sensors.add_command(tool("roqsim_sensors.coverage.cli", "coverage"))
