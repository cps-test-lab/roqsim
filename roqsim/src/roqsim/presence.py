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

The originals are saved on the entity, so returning is exact rather than a guess at what the
world declared.

An absent entity is also frozen
-------------------------------

Taking a body out of the contact set puts a ``free`` one in free fall -- the same accumulating
velocity the parking trick was rejected for, arrived at from the other direction. So absence also
compensates the subtree's gravity and zeroes its velocity (:func:`_freeze`), which together leave a
passive entity exactly where it was. The gravity half needs :func:`arm_gravity_compensation` to have
run before the world compiled; that is the engine's job, once per world.

Who calls this
--------------

The physics thread only, like every other write to ``model``/``data`` -- a control-plane
service posts through :meth:`~roqsim.context.SimContext.post` rather than calling here directly.
"""

from __future__ import annotations

import logging

import mujoco

_log = logging.getLogger(__name__)

#: Geom group reserved for absent entities. Groups 0-1 are visual, 2 is both sensor FOV
#: visualisation and the Menagerie convention for a robot's visual meshes, and 3 is
#: collision-only, so 4 is the first free one. MuJoCo's default ``MjvOption.geomgroup`` is
#: ``[1,1,1,0,0,0]``, so no renderer draws it without being told to.
ABSENT_GEOM_GROUP = 4

#: Key under which an entity's saved geom appearance lives while it is absent.
_SAVED = "_presence_saved"

#: Key under which an entity's saved per-body gravity compensation lives while it is absent.
_SAVED_GRAVCOMP = "_presence_saved_gravcomp"


def entity_body_ids(model, body_name: str) -> list[int]:
    """*body_name* and every body below it.

    Descendants included because an entity is rarely one body -- a prop with a free joint, a
    walker's limbs -- and leaving a child behind would make "absent" mean "mostly absent".
    """
    root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if root < 0:
        return []
    bodies = {root}
    # bodyid is topologically ordered (a parent precedes its children), so one forward pass
    # collects the whole subtree.
    for body in range(root + 1, model.nbody):
        if int(model.body_parentid[body]) in bodies:
            bodies.add(body)
    return sorted(bodies)


def entity_geom_ids(model, body_name: str) -> list[int]:
    """Every geom of *body_name* and its descendants."""
    bodies = set(entity_body_ids(model, body_name))
    if not bodies:
        return []
    return [g for g in range(model.ngeom) if int(model.geom_bodyid[g]) in bodies]


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
    if present:
        _restore(ctx.model, entity, geoms)
    else:
        _hide(ctx.model, entity, geoms)
        _freeze(ctx.model, ctx.data, entity)
    entity.present = bool(present)
    # A transition leaves NO other trace. It writes model fields, while a recording stores
    # `mjData` state (roqsim.capture's STATE_SPEC is MuJoCo's keyframe notion), and the pose
    # deliberately does not move -- so afterwards nothing in a run's recorded data can say
    # whether an obstacle ever appeared. That made "did it spawn?" unanswerable on a campaign
    # whose service call had returned OK. Until presence rides in the capture, this line is
    # the record: stamped with sim time, so it lands on the run's clock like every other event.
    _log.info(
        "presence: %s %s at t=%.3f (%d geoms)",
        entity.name,
        "present" if present else "absent",
        float(ctx.data.time),
        len(geoms),
    )
    return True


def _hide(model, entity, geoms) -> None:
    saved = {}
    for gid in geoms:
        saved[gid] = (
            int(model.geom_group[gid]),
            int(model.geom_contype[gid]),
            int(model.geom_conaffinity[gid]),
            float(model.geom_rgba[gid][3]),
        )
        model.geom_group[gid] = ABSENT_GEOM_GROUP
        model.geom_contype[gid] = 0
        model.geom_conaffinity[gid] = 0
        model.geom_rgba[gid][3] = 0.0
    entity.meta[_SAVED] = saved


def _freeze(model, data, entity) -> None:
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
        compile -- without it MuJoCo skips the gravcomp path and this write does nothing.
    the subtree's velocities
        zeroed, because compensation removes the force but not the motion the entity already had.
        A prop deleted mid-flight would otherwise coast in a straight line forever.

    Together those two make an absent PASSIVE entity stationary: no force acts on it, and it starts
    from rest. An absent entity whose actuators are still being commanded is a separate matter --
    its controller is driving it, and presence does not silence controllers.
    """
    bodies = entity_body_ids(model, entity.body)
    saved = {}
    for bid in bodies:
        saved[bid] = float(model.body_gravcomp[bid])
        model.body_gravcomp[bid] = 1.0
        adr, num = int(model.body_dofadr[bid]), int(model.body_dofnum[bid])
        if num:
            data.qvel[adr : adr + num] = 0.0
    entity.meta[_SAVED_GRAVCOMP] = saved


def _thaw(model, entity) -> None:
    """Give the subtree back the gravity it was compiled with.

    Velocity is deliberately not restored: an entity returns from absence either at rest or at the
    pose and velocity its spawn stated, never carrying momentum from before it left.
    """
    for bid, gravcomp in (entity.meta.pop(_SAVED_GRAVCOMP, None) or {}).items():
        model.body_gravcomp[bid] = gravcomp


def _restore(model, entity, geoms) -> None:
    _thaw(model, entity)
    saved = entity.meta.pop(_SAVED, None) or {}
    for gid in geoms:
        original = saved.get(gid)
        if original is None:
            # Never hidden by us: leave it exactly as the world declared it rather than
            # inventing a group and a contact mask it never had.
            continue
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
