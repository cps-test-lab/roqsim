"""Place entities and make them present or absent, as commands.

``roqsim sim`` adds this plugin itself, as ``sim.entities`` beside ``sim.run_control``, so a client of
the control socket can do what a scenario's ``set_entity_state``, ``spawn_entity`` and
``delete_entity`` do::

    sim/entities/set_state      command   place an entity (and set its velocity)
    sim/entities/set_presence   command   make an entity present (optionally placed) or absent

The operations are :mod:`roqsim.entity_control`'s, which the in-process route calls too, so a
refusal reads the same on both. An entity's pose is the core's ``sim/entities/<name>/pose``.
"""

from __future__ import annotations

from typing import Annotated

import numpy as np
from numpy.typing import NDArray

from .. import endpoint, entity_control
from ..context import SimContext
from ..endpoint import Shape, Unit
from ..plugin import Plugin


class EntityControlPlugin(Plugin):
    """Entity placement and presence, served as commands."""

    # It builds nothing and holds no simulation state: a consumer that wants the scene drops it.
    transport_only = True

    def configure(self, ctx: SimContext) -> None:
        self._ctx = ctx

    @endpoint.command("set_state")
    def set_state(
        self,
        entity: Annotated[str, "the world's `name:` for it"],
        position: Annotated[NDArray[np.float64], Shape(3), Unit("m"), "world frame"],
        orientation: Annotated[NDArray[np.float64], Shape(4), "quaternion (w, x, y, z)"],
        linear_velocity: Annotated[
            NDArray[np.float64] | None, Shape(3), Unit("m/s"), "zero unless stated"
        ] = None,
        angular_velocity: Annotated[
            NDArray[np.float64] | None, Shape(3), Unit("rad/s"), "zero unless stated"
        ] = None,
    ) -> str:
        """Place a free-jointed or mocap entity, at rest unless velocities are stated."""
        return entity_control.set_state(
            self._ctx, entity, position, orientation, linear_velocity, angular_velocity
        )

    @endpoint.command("set_presence")
    def set_presence(
        self,
        entity: Annotated[str, "the world's `name:` for it"],
        present: Annotated[bool, "true: perceivable and collidable; false: absent"],
        position: Annotated[
            NDArray[np.float64] | None, Shape(3), Unit("m"), "where it appears"
        ] = None,
        orientation: Annotated[
            NDArray[np.float64] | None, Shape(4), "quaternion (w, x, y, z)"
        ] = None,
    ) -> str:
        """Make an entity the world declared present (placed, where a pose is given) or absent."""
        return entity_control.set_presence(self._ctx, entity, present, position, orientation)

    def validate_config(self, config: dict) -> list[str]:
        return [f"entity_control takes no config, got {sorted(config)}"] if config else []
