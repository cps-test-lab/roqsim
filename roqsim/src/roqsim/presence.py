"""Making an entity absent without removing it from the model.

roqsim never recompiles the model at runtime, and ``simulation_interfaces`` has no way to add a
body to a compiled one. So an obstacle that must *appear* mid-trial is compiled in up front and
made **absent** until it is wanted: nothing can see it, nothing can touch it, and the control
plane does not list it.

A world declares which entities start that way (``present: false`` on the entry that registers the
entity); ``SpawnEntity`` and ``DeleteEntity`` move them either way at run time.

Presence is not position
-------------------------

The obvious trick is to park the entity somewhere harmless -- below the floor, off the map --
and teleport it back. It has a real defect: a ``free`` body keeps accelerating under gravity for
however long it is parked, so it returns with metres per second of accumulated velocity, and a
long trial can drop it far enough to fall out of any sensible bound. It also stays in the
world's contact set, so two parked props can collide with each other.

Absence here is *perceptual and physical*, and the pose never moves. Three model fields decide what
can perceive or touch the entity, all runtime-mutable, so nothing recompiles:

``rgba`` alpha
    zeroed. **This is what actually removes the entity from a raycast**: ``mj_ray``/``mj_multiRay``
    skip a geom exactly when its *resolved* alpha is 0, independently of any ``geomgroup`` mask
    (verified; note "resolved" -- a material overrides the geom's own ``rgba``, so a geom carrying an
    opaque material is not hidden by zeroing the field). Contact is a separate axis and ignores
    appearance entirely, which is why the next two fields are also needed.
``contype`` / ``conaffinity``
    zeroed, so nothing collides with it. Raycasts ignore these, so this alone would leave an "absent"
    obstacle perfectly visible to a navigation stack.
``geom_group``
    moved to :data:`ABSENT_GEOM_GROUP`, which the renderer excludes and which every raycast
    excludes too -- :func:`roqsim.raycast.cast` masks it by default, so a caster has to opt *in* to
    seeing absent geometry. This is the axis a **viewer** answers to, and it is the one that
    survives an entity whose material makes the alpha trick inapplicable.

A **flex** the entity carries (a ``<flexcomp>`` in a prop or a tool) has its own copy of each:
``flex_rgba``, ``flex_contype``/``flex_conaffinity`` and ``flex_group``, written the same way. Each
takes effect on the next step without a recompile -- measured: zeroing the pair ends every contact
of a flex resting on a floor, and either the group or the alpha alone takes it out of a rendered
frame. No raycast needs telling: ``mj_ray`` does not intersect a flex (measured: a ray aimed
through a soft block reaches the floor behind it).

The originals are saved once, so returning is exact rather than a guess at what the world declared.

Presence composes
-----------------

A body is present only while every entity around it is. An entity's subtree can hold other entities
(an arm on a robot, a ``part_state`` state on a part), and each can go absent on its own, in any
order. So each geom, flex and body's gravity compensation is held rather than overwritten: the first
entity to go absent saves the original, each later one adds a hold, and the last to return puts the
original back. The result depends on which entities are absent, never on the order they changed in,
so a reset or a replay can apply them in any order.

An absent entity is also frozen
-------------------------------

Taking a body out of the contact set puts a ``free`` one in free fall -- the same accumulating
velocity the parking trick was rejected for, arrived at from the other direction. So absence also
compensates the subtree's gravity and zeroes its velocity (:func:`_freeze`), which together leave a
passive entity exactly where it was. The gravity half needs :func:`arm_gravity_compensation` to have
run before the world compiled; that is the engine's job, once per world.

Parking a visual-only body
--------------------------

The defect above is a property of a body with physics. A **visual-only mocap body** -- one whose
geoms collide with nothing, laid by a plugin as a marker or a deposit -- has neither: nothing
integrates its pose, so it stays exactly where it is put. A plugin that keeps a fixed pool of them,
because the model cannot grow at run time, keeps the unused ones at :data:`PARK_Z`, out of every
camera's view, and moves one into place when it is used. Being plain ``mocap_pos``, that is in the
recorded state, so a replay shows the same bodies in the same places.

Everything that frames the scene skips a geom below :data:`PARKED_BELOW_Z`
(:func:`roqsim.rendering.scene_corners`, :func:`roqsim.rendering.bounding_sphere`), so a parked pool
pulls no camera a hundred metres down to frame it. Compile such a pool at its place of use, not at
:data:`PARK_Z`: the model's extent is taken at compile and sets every camera's clipping planes.

Who calls this
--------------

The physics thread only, like every other write to ``model``/``data`` -- a control-plane
service posts through :meth:`~roqsim.context.SimContext.post` rather than calling here directly.
"""

from __future__ import annotations

import logging

import mujoco

from .flex import entity_flex_ids

_log = logging.getLogger(__name__)

#: Geom group reserved for absent entities. Groups 0-1 are visual, 2 is both sensor FOV
#: visualisation and the Menagerie convention for a robot's visual meshes, and 3 is
#: collision-only, so 4 is the first free one. MuJoCo's default ``MjvOption.geomgroup`` is
#: ``[1,1,1,0,0,0]``, so no renderer draws it without being told to.
ABSENT_GEOM_GROUP = 4

#: Height at which a visual-only mocap body waits while unused (see "Parking a visual-only body").
PARK_Z = -100.0

#: A geom whose world position lies below this height is parked and frames nothing. Well above
#: :data:`PARK_Z`, and well below the floor of any world a robot stands in.
PARKED_BELOW_Z = -50.0

#: Key under which an entity's held geoms are listed while it is absent.
_SAVED = "_presence_saved"

#: Key under which an entity's held bodies are listed while it is absent.
_SAVED_GRAVCOMP = "_presence_saved_gravcomp"

#: Key under which an entity's held flexes are listed while it is absent.
_SAVED_FLEX = "_presence_saved_flex"


def _take(holds: dict, ids, read) -> list:
    """Add a hold on each of ``ids``, saving the original on the first; those newly held."""
    first = []
    for i in ids:
        held = holds.get(i)
        if held is not None:
            held[1] += 1
        else:
            holds[i] = [read(i), 1]
            first.append(i)
    return first


def _release(holds: dict, ids) -> list:
    """Drop a hold on each of ``ids``; ``(id, original)`` of those no entity holds any more."""
    freed = []
    for i in ids:
        held = holds[i]
        held[1] -= 1
        if not held[1]:
            freed.append((i, held[0]))
            del holds[i]
    return freed


def subtree_body_ids(model, root: int) -> list[int]:
    """Body ``root`` and every body below it, ascending.

    The one subtree walk in the tree: an entity's absence, its contact scope, a camera's class
    mask, an arm's collapsed links and a robot's own geoms all mean "this body and its
    descendants", and each answering it with its own loop is how two of them come to disagree on
    a walker's limbs or a prop welded to a shelf.
    """
    bodies = {int(root)}
    # bodyid is topologically ordered (a parent precedes its children), so one forward pass
    # collects the whole subtree.
    for body in range(int(root) + 1, model.nbody):
        if int(model.body_parentid[body]) in bodies:
            bodies.add(body)
    return sorted(bodies)


def subtree_geom_ids(model, root: int) -> list[int]:
    """Every geom of body ``root`` and its descendants, ascending."""
    bodies = set(subtree_body_ids(model, root))
    return [g for g in range(model.ngeom) if int(model.geom_bodyid[g]) in bodies]


def entity_body_ids(model, body_name: str) -> list[int]:
    """*body_name* and every body below it; empty when no body has that name.

    Descendants included because an entity is rarely one body -- a prop with a free joint, a
    walker's limbs -- and leaving a child behind would make "absent" mean "mostly absent".
    """
    root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if root < 0:
        return []
    return subtree_body_ids(model, root)


def entity_geom_ids(model, body_name: str) -> list[int]:
    """Every geom of *body_name* and its descendants; empty when no body has that name."""
    root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if root < 0:
        return []
    return subtree_geom_ids(model, root)


def arm_gravity_compensation(spec) -> None:
    """Make ``body_gravcomp`` writable at run time, by marking the world body before compile.

    MuJoCo gates the whole gravity-compensation path on ``model.ngravcomp``, a count the COMPILER
    derives from the bodies that carry a nonzero ``gravcomp``. It is read-only afterwards, so in a
    model compiled without one, writing ``body_gravcomp`` at run time is silently inert -- and
    :func:`_freeze` depends on that write to stop an absent body falling.

    The world body is the one that can carry the mark for free: it has no degrees of freedom and
    infinite mass, so compensating its weight is arithmetic on a body that cannot move. Every other
    body keeps the compiled value it declared, so nothing is compensated until presence asks for it.

    Called once per world, immediately before ``spec.compile()``.
    """
    spec.worldbody.gravcomp = 1.0


def set_present(ctx, entity, present: bool) -> bool:
    """Make *entity* perceivable and collidable, or not. Physics thread only.

    Returns whether anything changed, so a caller can report "already absent" rather than
    claiming to have done something.
    """
    if entity is None or bool(getattr(entity, "present", True)) == bool(present):
        return False
    if not entity.body:
        # An entity with no body is a thing the registry knows about that has no geometry to
        # hide -- flip the flag and let the control plane's listing follow it.
        entity.present = bool(present)
        return True

    geoms = entity_geom_ids(ctx.model, entity.body)
    flexes = entity_flex_ids(ctx.model, entity.body)
    if present:
        _restore(ctx, entity)
        _restore_flexes(ctx, entity)
    else:
        _hide(ctx, entity, geoms)
        _hide_flexes(ctx, entity, flexes)
        _freeze(ctx, entity)
    entity.present = bool(present)
    # A transition writes model fields, while a recording's samples store `mjData` state
    # (roqsim.capture's STATE_SPEC is MuJoCo's keyframe notion), and the pose deliberately does
    # not move. A recording carries it in its provenance (StateRecorder notes the absent set per
    # sample); a run without one has this line, stamped with sim time so it lands on the run's
    # clock like every other event.
    _log.info(
        "presence: %s %s at t=%.3f (%d geoms, %d flexes)",
        entity.name,
        "present" if present else "absent",
        float(ctx.data.time),
        len(geoms),
        len(flexes),
    )
    return True


def _hide(ctx, entity, geoms) -> None:
    model = ctx.model

    def read(gid):
        return (
            int(model.geom_group[gid]),
            int(model.geom_contype[gid]),
            int(model.geom_conaffinity[gid]),
            float(model.geom_rgba[gid][3]),
        )

    for gid in _take(ctx.presence_holds.setdefault("geom", {}), geoms, read):
        model.geom_group[gid] = ABSENT_GEOM_GROUP
        model.geom_contype[gid] = 0
        model.geom_conaffinity[gid] = 0
        model.geom_rgba[gid][3] = 0.0
    entity.meta[_SAVED] = list(geoms)


def _hide_flexes(ctx, entity, flexes) -> None:
    """The flex counterpart of :func:`_hide`: the same three axes, on the flex's own fields."""
    model = ctx.model

    def read(fid):
        return (
            int(model.flex_group[fid]),
            int(model.flex_contype[fid]),
            int(model.flex_conaffinity[fid]),
            float(model.flex_rgba[fid][3]),
        )

    for fid in _take(ctx.presence_holds.setdefault("flex", {}), flexes, read):
        model.flex_group[fid] = ABSENT_GEOM_GROUP
        model.flex_contype[fid] = 0
        model.flex_conaffinity[fid] = 0
        model.flex_rgba[fid][3] = 0.0
    entity.meta[_SAVED_FLEX] = list(flexes)


def _restore_flexes(ctx, entity) -> None:
    """Put back what :func:`_hide_flexes` saved, for each flex no other entity holds."""
    model = ctx.model
    held = entity.meta.pop(_SAVED_FLEX, None) or []
    for fid, original in _release(ctx.presence_holds.setdefault("flex", {}), held):
        group, contype, conaffinity, alpha = original
        model.flex_group[fid] = group
        model.flex_contype[fid] = contype
        model.flex_conaffinity[fid] = conaffinity
        model.flex_rgba[fid][3] = alpha


def _freeze(ctx, entity) -> None:
    """Stop an absent entity moving, so "not there" does not mean "falling out of the world".

    Zeroing ``contype``/``conaffinity`` takes an absent body out of the contact set, and a body with
    no contacts and a free joint is in free fall: nothing holds it up. Over a trial that is metres
    per second and kilometres of drift, and the pose it comes back at is whatever the caller states
    on spawn -- so the fall is invisible while it happens and silently corrected afterwards, which
    is what makes it worth writing down. What it costs meanwhile is real: ``GetEntityState`` reports
    a pose far below the world, a recording stores the diverging qpos, and a long enough run reaches
    the magnitudes MuJoCo resets a diverged simulation for.

    Two writes, both restored on return:

    ``body_gravcomp``
        1.0, so gravity is exactly cancelled for the whole subtree rather than approximately
        resisted. Requires :func:`arm_gravity_compensation` to have marked the world body before
        compile -- without it MuJoCo skips the gravcomp path and this write does nothing. The root
        is compensated with the rest, which is what tells the engine's
        :class:`~roqsim.actuators.GravityReaction` to leave a frozen robot's arm weightless rather
        than hand its weight to a base the floor no longer touches.
    the subtree's velocities
        zeroed, because compensation removes the force but not the motion the entity already had.
        A prop deleted mid-flight would otherwise coast in a straight line forever.

    Together those two make an absent PASSIVE entity stationary: no force acts on it, and it starts
    from rest. An absent entity whose actuators are still being commanded is a separate matter --
    its controller is driving it, and presence does not silence controllers.
    """
    model, data = ctx.model, ctx.data
    bodies = entity_body_ids(model, entity.body)
    holds = ctx.presence_holds.setdefault("gravcomp", {})
    for bid in _take(holds, bodies, lambda b: float(model.body_gravcomp[b])):
        model.body_gravcomp[bid] = 1.0
    for bid in bodies:
        adr, num = int(model.body_dofadr[bid]), int(model.body_dofnum[bid])
        if num:
            data.qvel[adr : adr + num] = 0.0
    entity.meta[_SAVED_GRAVCOMP] = list(bodies)


def _thaw(ctx, entity) -> None:
    """Give the subtree back the gravity it was compiled with.

    Velocity is deliberately not restored: an entity returns from absence either at rest or at the
    pose and velocity its spawn stated, never carrying momentum from before it left.
    """
    held = entity.meta.pop(_SAVED_GRAVCOMP, None) or []
    for bid, gravcomp in _release(ctx.presence_holds.setdefault("gravcomp", {}), held):
        ctx.model.body_gravcomp[bid] = gravcomp


def _restore(ctx, entity) -> None:
    _thaw(ctx, entity)
    model = ctx.model
    held = entity.meta.pop(_SAVED, None) or []
    for gid, original in _release(ctx.presence_holds.setdefault("geom", {}), held):
        group, contype, conaffinity, alpha = original
        model.geom_group[gid] = group
        model.geom_contype[gid] = contype
        model.geom_conaffinity[gid] = conaffinity
        model.geom_rgba[gid][3] = alpha


def visible_geomgroup_mask(include_absent: bool = False):
    """A ``mj_multiRay`` ``geomgroup`` mask that excludes absent entities.

    Raycasting sensors pass this instead of ``None``. ``None`` means "every group", and an absent
    entity is still a return under it.

    Every raycaster in the tree gets it *by default*, because they all go through
    :func:`roqsim.raycast.cast` and that is the default there -- so "every group, absent entities
    included" takes an explicit ``geomgroup=None``. The default is that way round because the
    raycasters are many (the 2D ``lidar``, the 3D lidars, ``roqsim.rendering``'s line-of-sight probe,
    ``roqsim_assets``' ``moving_box``) and each one that forgot the mask would report an absent
    entity as present, with nothing downstream able to tell.

    The alpha zeroing in :func:`set_present` remains the first field written rather than a backstop,
    and the two are complementary rather than redundant: alpha covers a caller that legitimately
    asks for every group, while the group mask covers the case alpha cannot reach -- a geom whose
    *material* supplies an opaque colour, where zeroing the geom's own ``rgba`` hides nothing.
    """
    import numpy as np

    mask = np.ones(mujoco.mjNGROUP, dtype=np.uint8)
    if not include_absent:
        mask[ABSENT_GEOM_GROUP] = 0
    return mask
