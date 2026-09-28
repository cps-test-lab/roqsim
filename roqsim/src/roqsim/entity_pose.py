"""Every entity's ground-truth pose, as a core ``out`` endpoint.

Each registered entity whose body is in the compiled model gets one endpoint, addressed
``sim/entities/<name>/pose`` in the endpoint tree -- owner :data:`OWNER`, name
``entities/<name>/pose`` (:func:`endpoint_name`). No plugin declares it: the engine registers it
after the plugin that registered the entity has configured, so a world needs no entry for it.

What it reads (:class:`EntityPose`) is the entity body's true state in the world frame, from the
physics: ``data.xpos`` and ``data.xquat`` (MuJoCo's ``(w, x, y, z)`` order), and the linear and
angular velocity of the body origin, from ``cvel`` through ``mj_objectVelocity``. All four are
copies, so a reader on another thread holds values that do not change under it.

**Computed only when read.** Nothing runs per step: the endpoint carries no backend hint, so no
bridge publishes it unless asked, and its ``read`` is the whole cost, paid by whoever calls it.

**Presence is reflected**: an entity deleted at run time reads ``None`` until it is spawned again
(``Endpoint.read`` returning ``None`` is "nothing to publish"), and an entity registered after a
bridge bound still gets its endpoint (``on_demand``), for a consumer that looks it up by name.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated

import mujoco
import numpy as np
from numpy.typing import NDArray

from .context import Endpoint
from .endpoint import Shape, Unit, value_type

if TYPE_CHECKING:
    from .context import SimContext

#: The owner of every core endpoint that describes the simulation rather than one plugin.
OWNER = "sim"


def endpoint_name(entity: str) -> str:
    """The endpoint name of *entity*'s pose under :data:`OWNER`."""
    return f"entities/{entity}/pose"


@dataclass(frozen=True)
class EntityPose:
    """An entity body's true state in the world frame."""

    position: Annotated[NDArray[np.float64], Shape(3), Unit("m"), "body origin, world frame"]
    orientation: Annotated[NDArray[np.float64], Shape(4), "quaternion (w, x, y, z), world frame"]
    linear_velocity: Annotated[
        NDArray[np.float64], Shape(3), Unit("m/s"), "of the body origin, world frame"
    ]
    angular_velocity: Annotated[NDArray[np.float64], Shape(3), Unit("rad/s"), "world frame"]


_RESULT = value_type(EntityPose)


class _PoseReader:
    """``read`` of one entity's pose endpoint; physics thread, like every ``read``."""

    __slots__ = ("_bid", "_ctx", "_name", "_vel")

    def __init__(self, ctx: SimContext, name: str, bid: int) -> None:
        self._ctx, self._name, self._bid = ctx, name, bid
        self._vel = np.zeros(6)

    def __call__(self) -> EntityPose | None:
        ctx = self._ctx
        entity = ctx.entities.get(self._name)
        if entity is None or not entity.present:
            return None
        m, d, bid = ctx.model, ctx.data, self._bid
        mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, bid, self._vel, 0)
        return EntityPose(
            position=d.xpos[bid].copy(),
            orientation=d.xquat[bid].copy(),
            linear_velocity=self._vel[3:].copy(),
            angular_velocity=self._vel[:3].copy(),
        )


def register(ctx: SimContext) -> None:
    """Add a pose endpoint for each entity that has none yet. Idempotent; the engine calls it after
    every plugin's ``configure``.

    An entity with no body, or whose body is not in the compiled model, has no pose to read and
    gets no endpoint, so a lookup of it fails rather than returning a pose of something else.
    """
    done = ctx.entity_poses
    for entity in ctx.entities.all():
        if entity.name in done or not entity.body:
            continue
        done.add(entity.name)
        bid = mujoco.mj_name2id(ctx.model, mujoco.mjtObj.mjOBJ_BODY, entity.body)
        if bid < 0:
            continue
        ctx.interface.add(
            Endpoint(
                name=endpoint_name(entity.name),
                direction="out",
                owner=OWNER,
                read=_PoseReader(ctx, entity.name, bid),
                result=_RESULT,
            ),
            on_demand=True,
        )
