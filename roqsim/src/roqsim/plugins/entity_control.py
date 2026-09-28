"""Place entities and make them present or absent, as commands.

``roqsim sim`` adds this plugin itself, as ``sim.entities`` beside ``sim.run_control``, so a client of
the control socket can do what a scenario's ``set_entity_state``, ``spawn_entity`` and
``delete_entity`` do::

    sim/entities/set_state      command   place an entity (and set its velocity)
    sim/entities/set_presence   command   make an entity present (optionally placed) or absent

The operations are :mod:`roqsim.entity_control`'s, which the in-process route calls too, so a
refusal reads the same on both. An entity's pose is the core's ``sim/entities/<name>/pose``. They are
served over the control socket only (``ros2=None``): ROS has its own simulation-control services.
"""

from __future__ import annotations

from .. import endpoint, entity_control
from ..context import SimContext
from ..plugin import Plugin
from ..types import AngularVelocity3, Point3, Quaternion, Velocity3


class EntityControlPlugin(Plugin):
    """Entity placement and presence, served as commands."""

    # It builds nothing and holds no simulation state: a consumer that wants the scene drops it.
    transport_only = True

    def configure(self, ctx: SimContext) -> None:
        self._ctx = ctx

    @endpoint.command(ros2=None)
    def set_state(
        self,
        entity: str,
        position: Point3,
        orientation: Quaternion,
        linear_velocity: Velocity3 | None = None,
        angular_velocity: AngularVelocity3 | None = None,
    ) -> str:
        """Place a free-jointed or mocap entity, at rest unless velocities are stated.

        Args:
            entity: the world's `name:` for it
            position: world frame
            orientation: quaternion (w, x, y, z), world frame
            linear_velocity: zero unless stated
            angular_velocity: zero unless stated
        """
        return entity_control.set_state(
            self._ctx, entity, position, orientation, linear_velocity, angular_velocity
        )

    @endpoint.command(ros2=None)
    def set_presence(
        self,
        entity: str,
        present: bool,
        position: Point3 | None = None,
        orientation: Quaternion | None = None,
    ) -> str:
        """Make an entity the world declared present (placed, where a pose is given) or absent.

        Args:
            entity: the world's `name:` for it
            present: true: perceivable and collidable; false: absent
            position: where it appears, world frame
            orientation: quaternion (w, x, y, z), world frame
        """
        return entity_control.set_presence(self._ctx, entity, present, position, orientation)

    def validate_config(self, config: dict) -> list[str]:
        return [f"entity_control takes no config, got {sorted(config)}"] if config else []
