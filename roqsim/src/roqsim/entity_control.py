"""Place an entity, or make it present or absent -- once, for every route that asks.

:func:`set_state` and :func:`set_presence` are what a scenario's ``set_entity_state``,
``spawn_entity`` and ``delete_entity`` do, and what the ``entity_control`` plugin serves as the
commands ``sim/entities/set_state`` and ``sim/entities/set_presence`` over the control socket. Both
routes call these functions, so a refusal reads the same whichever one a scenario took.

Physics thread only. A name the world does not carry raises :class:`UnknownEntity` -- an
authoring error; a request the entity cannot take (a welded body placed, an entity already in the
state asked for) raises :class:`EntityRefused` -- a fact about this trial's world.
"""

from __future__ import annotations

import numpy as np

from .placement import PLACEABLE_MODES_HINT, base_joint_of, place_body


class UnknownEntity(LookupError):
    """The world carries no entity by that name."""


class EntityRefused(RuntimeError):
    """The entity exists and cannot take what was asked of it."""


def require_entity(ctx, name: str, *, spawning: bool = False):
    """The entity called *name*, or :class:`UnknownEntity` saying what the name must be."""
    entity = ctx.entities.get(name)
    if entity is not None:
        return entity
    if spawning:
        raise UnknownEntity(
            f"the simulator has no entity called {name!r}. A spawn ACTIVATES what the world "
            "already declares -- it does not create one -- so the name must be a `name:` in the "
            "world, and a world that declares no such entity cannot be made to have it."
        )
    raise UnknownEntity(
        f"the simulator has no entity called {name!r}. The name is the world's `name:` for that "
        "spawn, not a body name and not a TF frame."
    )


def unplaceable(name: str, joint_name: str | None) -> str:
    """Why a pose could not be applied, in terms of what the world would have to say instead."""
    return (
        f"entity {name!r} is welded scenery: it has neither a mocap body nor a free joint named "
        f"{joint_name!r}, so no pose can be written to it. {PLACEABLE_MODES_HINT}"
    )


def set_state(ctx, name: str, position, orientation, linear_velocity=None, angular_velocity=None):
    """Place *name* at ``position`` / ``orientation`` (w, x, y, z), moving at the velocities
    (zero unless stated: a body put somewhere is at rest). Returns what was done."""
    import mujoco

    entity = require_entity(ctx, name)
    pos = np.asarray(position, dtype=float)
    quat = np.asarray(orientation, dtype=float)
    vel = np.asarray(
        [
            *(linear_velocity if linear_velocity is not None else (0.0, 0.0, 0.0)),
            *(angular_velocity if angular_velocity is not None else (0.0, 0.0, 0.0)),
        ],
        dtype=float,
    )
    if not place_body(ctx, entity, pos, quat, vel):
        raise EntityRefused(unplaceable(name, base_joint_of(entity)))
    mujoco.mj_forward(ctx.model, ctx.data)
    moving = "" if not vel.any() else f", moving at {vel.tolist()}"
    return f"placed at {pos.tolist()}{moving}"


def set_presence(ctx, name: str, present: bool, position=None, orientation=None) -> str:
    """Make *name* present (placing it, in the same transaction, where a pose is given) or
    absent. Returns what was done.

    One transaction because that is the reason to spawn rather than teleport: a flip and a pose
    applied separately leave the entity perceivable for a step wherever the world compiled it.
    Refused, before anything is written, for an entity already in the state asked for.
    """
    import mujoco

    from .presence import set_present

    entity = require_entity(ctx, name, spawning=True)
    if bool(getattr(entity, "present", True)) == bool(present):
        raise EntityRefused(f"entity {name!r} is already {'present' if present else 'absent'}")
    pos = None if position is None else np.asarray(position, dtype=float)
    if pos is not None:
        quat = np.asarray(orientation if orientation is not None else (1.0, 0, 0, 0), dtype=float)
        # At rest, like a teleport: an entity that has just appeared has no history.
        if not place_body(ctx, entity, pos, quat):
            raise EntityRefused(
                unplaceable(name, base_joint_of(entity))
                + " Or spawn it without a pose, to activate it where the world put it."
            )
    if not set_present(ctx, entity, present):
        raise EntityRefused(f"entity {name!r} did not change presence")
    mujoco.mj_forward(ctx.model, ctx.data)
    where = "" if pos is None else f" at {pos.tolist()}"
    return f"{'present' if present else 'absent'}{where}"
