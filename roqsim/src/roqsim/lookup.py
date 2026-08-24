"""Resolve a NAME a caller was given to a body in the compiled model.

An entity's name is not its body's name, and that indirection is roqsim's own: ``spawn_model:
{model: graspable_carton, name: parcel}`` registers entity ``parcel`` on body
``<prefix><root body of the MJCF>``, so ``mj_name2id(model, mjOBJ_BODY, "parcel")`` returns -1
for a world that plainly contains a parcel. Every consumer that takes an object name from
outside -- a sensor's config, a scoring rule, an ``.osc`` action naming what to watch -- has to
walk the same two steps, and four of them had already written it out by hand with four slightly
different fallbacks.

It lives in the core because the convention is the core's: the day ``spawn_model`` changes how it
prefixes a body, one function moves with it instead of every caller drifting.

**It refuses what cannot answer the question**, rather than returning an id whose pose is a
constant. Three refusals, each measured:

*no body*
    ``Entity.body`` is optional -- an entity may be a pure marker. ``mj_name2id(None)`` is a
    crash several frames from the cause.
*absent*
    :mod:`roqsim.presence` leaves an absent entity in the model at its true pose, so ``xpos`` reads
    perfectly well while nothing can see or touch it. A caller waiting for an invisible obstacle
    to move is a scenario bug, not a long wait.
*welded to the world*
    ``body_weldid == 0`` and no mocap id means the body is rigidly fixed: its ``xpos`` is a
    compile-time constant and no amount of simulation will change it. Measured on a three-body
    model -- welded: ``weldid 0, mocapid -1, dofnum 0``; free: ``weldid 2, mocapid -1, dofnum 6``;
    mocap: ``mocapid 0, dofnum 0`` with ``weldid`` 0 up to MuJoCo 3.11 and the body's own id from
    3.12 on. The mocap row is why the test is not ``weldid == 0`` alone: a walker has zero DOFs and
    moves every step through ``mocap_pos``, and on 3.11 its ``weldid`` is indistinguishable from
    scenery's -- only ``mocapid`` separates the two there.

The refusals are opt-out (``movable=False``) for a caller that only wants a frame to read -- a
static TF publisher, an export -- rather than something it expects to move.
"""

from __future__ import annotations

import mujoco

from .context import SimContext


class LookupError_(RuntimeError):
    """Raised when a name cannot be resolved to a usable body. Named so it is not the builtin."""


def resolve_body_id(ctx: SimContext, name: str, *, movable: bool = True, what: str = "") -> int:
    """The body id *name* refers to: an entity's body first, then a raw body name.

    *name* is tried as an ENTITY first, because that is the name a world's ``name:`` keys and a
    campaign's overrides use, and it is the one a user has in hand. A raw body name still works,
    which is what lets a caller watch something the world never registered as an entity (a robot
    link, a piece of inherited scenery).

    *what* is the caller's own label, so the error says which config key was wrong rather than
    only which name. With *movable* the body must be able to move at all (see the module
    docstring); pass ``False`` when a fixed frame is a legitimate answer.
    """
    label = f"{what}: " if what else ""
    entity = ctx.entities.get(name)
    if entity is not None:
        if not entity.body:
            raise LookupError_(
                f"{label}entity {name!r} is registered but has no body, so it has no pose to "
                "read. Only an entity backed by geometry (a spawn_model prop, a spawn_robot "
                "base) can be resolved to one."
            )
        if not entity.present:
            raise LookupError_(
                f"{label}entity {name!r} is ABSENT (roqsim.presence): it is in the compiled model at "
                "its true pose, but nothing can see or touch it, so its pose is not a fact about "
                "the trial. Make it present first."
            )
        body = entity.body
        origin = f"entity {name!r} -> body {body!r}"
    else:
        body = name
        origin = f"body {name!r} (no entity of that name)"

    bid = mujoco.mj_name2id(ctx.model, mujoco.mjtObj.mjOBJ_BODY, body)
    if bid < 0:
        raise LookupError_(f"{label}{origin} is not in this world.{_hint(ctx, name)}")
    if movable and not _can_move(ctx.model, bid):
        raise LookupError_(
            f"{label}{origin} is welded to the world, so its pose is a compile-time constant and "
            "waiting for it to move never ends. Give it a free joint (`free: true` on a "
            "spawn_model), or name something that can actually move."
        )
    return bid


def _can_move(model, bid: int) -> bool:
    """Can this body's ``xpos`` ever change? See the module docstring for the measurements."""
    return int(model.body_weldid[bid]) != 0 or int(model.body_mocapid[bid]) >= 0


def _hint(ctx: SimContext, name: str) -> str:
    """Near misses over both namespaces, because the caller does not know which one they meant."""
    from .state import _closest, _names_of

    bodies = _names_of(ctx.model, mujoco.mjtObj.mjOBJ_BODY)
    entities = ctx.entities.names()
    close = _closest(name, entities) + _closest(name, bodies)
    if close:
        return f" Closest: {', '.join(dict.fromkeys(close))}."
    return f" Known entities: {', '.join(entities[:12]) or 'none'}."
