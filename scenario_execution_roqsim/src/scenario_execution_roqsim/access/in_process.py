# Copyright (C) 2026 Frederik Pasch
# SPDX-License-Identifier: Apache-2.0

"""The stepped backend: the simulator is this process, so the world is an object graph.

Reads are direct -- ``data.xpos`` between two ``mj_step``s is consistent by construction. Writes are
not: only the physics thread may touch ``model``/``data``, so an apply goes through
:meth:`~roqsim.context.SimContext.post` and is observed one step later. That is roqsim's single-writer
rule (architecture.rst §7), and it holds here even though the stepped runner ticks the tree on the
same thread that steps -- because the rule is the plugin's contract, not this caller's convenience,
and the ROS bridge's own service handler takes the identical path.
"""

from __future__ import annotations

import numpy as np

from . import (
    AccessError,
    NavCall,
    NavOutcome,
    OverrideCall,
    OverrideOutcome,
    Pose,
    SpawnCall,
    SpawnOutcome,
    TeleportCall,
    TeleportOutcome,
    WorldAccess,
)

_MISSING = object()


def _place_body(ctx, entity, joint_name, pos, quat, vel=None) -> bool:
    """Put *entity* at a pose. Physics thread only; ``False`` if the world compiled it welded.

    Two kinds of body can take a pose, and they differ in who owns it afterwards.

    A **mocap** body (``motion: driven``) is placed through ``mocap_pos``/``mocap_quat``. It has
    no degrees of freedom, so the solver never owns its pose: it stays where it is put, nothing
    that bumps into it moves it off the placement a campaign chose, and a placement that happens
    to intersect other geometry is not answered by launching it. A stated velocity is dropped
    rather than refused -- there is no DOF to carry one, and the placement itself was applied in
    full.

    A **free** body (``motion: physics``) is placed by writing its base joint. From the next step
    its pose is the solver's, which is what a trial wants only when the obstacle is meant to
    move, fall or be pushed.

    Welded scenery has neither and cannot be placed at all.
    """
    import mujoco

    bid = mujoco.mj_name2id(ctx.model, mujoco.mjtObj.mjOBJ_BODY, getattr(entity, "body", "") or "")
    mocapid = int(ctx.model.body_mocapid[bid]) if bid >= 0 else -1
    if mocapid >= 0:
        ctx.data.mocap_pos[mocapid] = pos
        ctx.data.mocap_quat[mocapid] = quat
        return True

    jid = mujoco.mj_name2id(ctx.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name) if joint_name else -1
    if jid < 0 or ctx.model.jnt_type[jid] != mujoco.mjtJoint.mjJNT_FREE:
        return False
    q = ctx.model.jnt_qposadr[jid]
    ctx.data.qpos[q : q + 3] = pos
    ctx.data.qpos[q + 3 : q + 7] = quat
    dof = ctx.model.jnt_dofadr[jid]
    ctx.data.qvel[dof : dof + 6] = 0.0 if vel is None else vel
    return True


def _unplaceable(name, joint_name) -> str:
    """Why a pose could not be applied, in terms of what the WORLD would have to say instead."""
    return (
        f"entity {name!r} is welded scenery: it has neither a mocap body nor a free joint named "
        f"{joint_name!r}, so no pose can be written to it. Give it 'motion: driven' in the world "
        "(placeable and immovable) or 'motion: physics' (placeable and owned by the solver from "
        "the next step)."
    )


class _PostedRoute(NavCall):
    """A route sent to a navigator, polled on its sequence number.

    ``wait=False`` succeeds as soon as the route is *applied* -- fire-and-forget traffic, where the
    scenario wants the mover moving and has something else to be doing. Even then it waits for the
    sequence to land rather than returning immediately, because the route is marshalled onto the
    physics thread and "accepted" must mean the simulator has it.
    """

    def __init__(self, handle, seq: int, *, wait: bool):
        self._handle = handle
        self._seq = seq
        self._wait = wait

    def poll(self):
        applied, finished, _goals, _dist = self._handle.status()
        if applied > self._seq:
            return NavOutcome(False, "a newer route preempted this one")
        if applied < self._seq:
            return None  # not applied yet: the post has not been drained
        if not self._wait:
            return NavOutcome(True, "route accepted")
        return NavOutcome(True, "arrived") if finished else None

    def cancel(self) -> None:
        self._handle.cancel()


class InProcessAccess(WorldAccess):
    transport = "in-process"

    def __init__(self, sim):
        self._sim = sim
        #: body id per (compiled model, name). The model is part of the key because a scenario that
        #: resets with different `world_overrides` gets a NEW model, in which ids are not stable.
        self._bids: dict[tuple[int, str], int] = {}

    # -- the world ------------------------------------------------------------------------------
    def _ctx(self):
        ctx = getattr(self._sim, "context", _MISSING)
        if ctx is _MISSING:
            raise AccessError(
                f"the simulation adapter {type(self._sim).__name__!r} has no `context`. roqsim's "
                "`MujocoSim` publishes it as the in-process seam; an adapter of your own must too "
                "(return the running world's SimContext, or None before it is built)."
            )
        return ctx

    def ready(self) -> bool:
        return self._ctx() is not None

    def entity_pose(self, name: str) -> Pose | None:
        ctx = self._ctx()
        if ctx is None:
            return None
        bid = self._body_id(ctx, name)
        return Pose(pos=np.array(ctx.data.xpos[bid]), quat=np.array(ctx.data.xquat[bid]))

    def _body_id(self, ctx, name: str) -> int:
        key = (id(ctx.model), name)
        if key not in self._bids:
            # Imported HERE rather than at module scope: importing `roqsim.lookup` pulls in MuJoCo, and
            # the behaviour tree is built before any world is compiled. Same reason the actions
            # compare the plugin's verdict strings by value instead of importing its constants.
            from roqsim.lookup import LookupError_, resolve_body_id

            try:
                self._bids[key] = resolve_body_id(ctx, name, what="entity")
            except LookupError_ as err:
                raise AccessError(str(err)) from None
        return self._bids[key]

    # -- the fault ------------------------------------------------------------------------------
    def apply_override(self, instance: str, active: bool, kind: str = "model") -> OverrideCall:
        ctx = self._ctx()
        if ctx is None:
            raise AccessError("the world is not built yet; call ready() first")
        prefix = self.OVERRIDE_KINDS[kind]
        key = f"{prefix}:{instance}"
        handle = ctx.blackboard.get(key)
        if handle is None:
            published_by = (
                "a `model_override` plugin instance in the world, whose `name:` must match"
                if kind == "model"
                else "a sensor carrying a `fault:` block, addressed by its COMPONENT ADDRESS "
                "(`robot.lidar`, not `lidar`) -- a sensor with no `fault:` publishes nothing"
            )
            offered = sorted(
                k.split(":", 1)[1]
                for k in getattr(ctx.blackboard, "_data", {})
                if k.startswith(prefix + ":")
            )
            raise AccessError(
                f"nothing on the blackboard under {key!r}. It is published by {published_by}. "
                f"This world offers: {', '.join(offered) if offered else '(none)'}. "
                "Check that the campaign's config is the world carrying the fault; "
                "`roqsim scenes describe <world>` lists what a world offers."
            )
        if bool(handle.is_active()) == bool(active):
            # Nothing to do, and nothing to WAIT for. A call that waited for a transition here would
            # hang forever: `set_active` returns early when the state already matches, so `changes`
            # never increments (measured in the plugin, model_override.set_active).
            return _Settled(
                OverrideOutcome(
                    ok=True,
                    verified=str(getattr(handle.read_state(), "verified", "") or ""),
                    detail=f"already {'active' if active else 'nominal'}",
                )
            )
        before = int(handle.read_state().changes)
        ctx.post(lambda _ctx: handle.set_active(bool(active)))
        return _PostedCall(handle, before)

    # -- navigation --------------------------------------------------------------------------------
    def navigate(self, name: str, goal_poses, *, wait: bool, action_name: str = "") -> NavCall:
        ctx = self._ctx()
        if ctx is None:
            raise AccessError("the world is not built yet; call ready() first")
        handle = ctx.blackboard.get(f"nav:{name}:handle")
        if handle is None:
            offered = sorted(
                k.split(":", 1)[1].removesuffix(":handle")
                for k in getattr(ctx.blackboard, "_data", {})
                if k.startswith("nav:") and k.endswith(":handle")
            )
            raise AccessError(
                f"entity {name!r} has no navigator, so nothing can drive it. A `navigator` component "
                f"must be nested under the entry that provides it (spawn_robot, spawn_model with "
                f"`mocap: true`, or walker). This world can navigate: "
                f"{', '.join(offered) if offered else '(nothing)'}."
            )
        poses = [(float(p[0]), float(p[1])) for p in goal_poses]
        if poses:
            seq = handle.send_goals(poses)
        else:
            seq = handle.start()
        return _PostedRoute(handle, seq, wait=wait)

    # -- teleport ---------------------------------------------------------------------------------
    def set_entity_state(
        self, name: str, pos: np.ndarray, quat: np.ndarray, lin=None, ang=None
    ) -> TeleportCall:
        # Imported HERE, not at module scope -- see the note on `_body_id`: this pulls in MuJoCo,
        # and the behaviour tree is built before any world is compiled.
        import mujoco

        ctx = self._ctx()
        if ctx is None:
            raise AccessError("the world is not built yet; call ready() first")
        entity = ctx.entities.get(name)
        if entity is None:
            raise AccessError(
                f"the simulator has no entity called {name!r}. The name is the world's `name:` for "
                "that spawn, not a body name and not a TF frame."
            )
        joint_name = (entity.meta or {}).get("base_joint")
        outcome_box: dict = {}

        def _write(
            _ctx,
            joint_name=joint_name,
            pos=np.asarray(pos, dtype=float),
            quat=np.asarray(quat, dtype=float),
            vel=np.asarray(
                [
                    *(lin if lin is not None else (0.0, 0.0, 0.0)),
                    *(ang if ang is not None else (0.0, 0.0, 0.0)),
                ],
                dtype=float,
            ),
        ):
            # The velocity the caller asked for, defaulting to zero: a body PUT somewhere is at
            # rest unless the caller says otherwise, which is what distinguishes a placement from
            # a launch.
            if not _place_body(_ctx, entity, joint_name, pos, quat, vel):
                outcome_box["outcome"] = TeleportOutcome(
                    ok=False, detail=_unplaceable(name, joint_name)
                )
                return
            mujoco.mj_forward(_ctx.model, _ctx.data)
            moving = "" if not vel.any() else f", moving at {vel.tolist()}"
            outcome_box["outcome"] = TeleportOutcome(
                ok=True, detail=f"placed at {pos.tolist()}{moving}"
            )

        ctx.post(_write)
        return _PostedTeleport(outcome_box)

    def set_entity_presence(self, name: str, present: bool, pos=None, quat=None) -> SpawnCall:
        """Flip presence and place the entity in ONE posted callback.

        One callback, not two, because that is the whole reason to spawn rather than teleport: a
        flip and a pose applied in separate transactions leave the entity perceivable for a step at
        wherever the world compiled it, and a free body accelerating under gravity in between.
        """
        import mujoco

        from roqsim.presence import set_present

        ctx = self._ctx()
        if ctx is None:
            raise AccessError("the world is not built yet; call ready() first")
        entity = ctx.entities.get(name)
        if entity is None:
            raise AccessError(
                f"the simulator has no entity called {name!r}. A spawn ACTIVATES what the world "
                "already declares -- it does not create one -- so the name must be a `name:` in "
                "the world, and a world that declares no such entity cannot be made to have it."
            )
        joint_name = (entity.meta or {}).get("base_joint")
        outcome_box: dict = {}

        def _apply(
            _ctx,
            joint_name=joint_name,
            pos=None if pos is None else np.asarray(pos, dtype=float),
            quat=None if quat is None else np.asarray(quat, dtype=float),
        ):
            # Refused BEFORE the pose is written, and refused at all: `SpawnEntity` over ROS answers
            # RESULT_OPERATION_FAILED for an entity that is already in the state asked for, and two
            # transports must not answer one question differently -- a scenario is written once and
            # does not learn which shape it is running in. Checked first because refusing after the
            # write would leave the entity moved by a call that reported failure.
            if bool(getattr(entity, "present", True)) == bool(present):
                outcome_box["outcome"] = SpawnOutcome(
                    ok=False,
                    detail=f"entity {name!r} is already {'present' if present else 'absent'}",
                )
                return
            if pos is not None:
                # The velocity is left at zero for the same reason a teleport zeroes it: an entity
                # that has just appeared has no history, and a velocity carried over from before it
                # was hidden is one this trial never applied.
                if not _place_body(_ctx, entity, joint_name, pos, quat):
                    outcome_box["outcome"] = SpawnOutcome(
                        ok=False,
                        detail=_unplaceable(name, joint_name)
                        + " Or spawn it without a pose, to activate it where the world put it.",
                    )
                    return
            # The return value is the confirmation that something changed, so it is what the
            # outcome is built from. Ignoring it would report success for a no-op.
            if not set_present(_ctx, entity, present):
                outcome_box["outcome"] = SpawnOutcome(
                    ok=False, detail=f"entity {name!r} did not change presence"
                )
                return
            mujoco.mj_forward(_ctx.model, _ctx.data)
            where = "" if pos is None else f" at {pos.tolist()}"
            outcome_box["outcome"] = SpawnOutcome(
                ok=True, detail=f"{'present' if present else 'absent'}{where}"
            )

        ctx.post(_apply)
        return _PostedSpawn(outcome_box)


class _PostedSpawn(SpawnCall):
    """Waits for the queued presence flip, then reports what ``_apply`` recorded.

    Same box-as-confirmation shape as :class:`_PostedTeleport`, and safe unlocked for the same
    reason: the stepped runner ticks the tree and steps physics on one thread, alternating.
    """

    def __init__(self, outcome_box: dict):
        self._box = outcome_box

    def poll(self) -> SpawnOutcome | None:
        return self._box.get("outcome")


class _PostedTeleport(TeleportCall):
    """Waits for the queued pose write, then reports what ``_write`` recorded.

    No ``changes`` counter to key on (that is `model_override`'s own bookkeeping) -- the outcome box
    IS the confirmation, filled by the same callback that performs the write. Safe unlocked: the
    stepped runner ticks the tree and steps physics on the same thread, alternating, so the callback
    has either not run yet (box empty) or has fully run (box filled) by the time ``poll`` reads it.
    """

    def __init__(self, outcome_box: dict):
        self._box = outcome_box

    def poll(self) -> TeleportOutcome | None:
        return self._box.get("outcome")


class _Settled(OverrideCall):
    """An outcome that was known immediately (nothing to apply)."""

    def __init__(self, outcome: OverrideOutcome):
        self._outcome = outcome

    def poll(self) -> OverrideOutcome | None:
        return self._outcome


class _PostedCall(OverrideCall):
    """Waits for the queued write, then reports the plugin's own verdict.

    Keyed on ``changes``, not on ``active``: the report's ``active`` is the state, and in the restore
    direction it already reads the value being asked for before the queue has drained -- so a caller
    watching ``active`` would report a restore that has not happened. ``changes`` only moves when the
    plugin actually wrote.

    One tick of latency by construction: the command is drained in the next ``pre_step``, and the
    verdict is computed in that same step's ``post_step``, so the tick that sees ``changes`` move also
    sees a final ``verified``.
    """

    def __init__(self, handle, changes_before: int):
        self._handle = handle
        self._before = changes_before

    def poll(self) -> OverrideOutcome | None:
        report = self._handle.read_state()
        if int(report.changes) <= self._before:
            return None
        return OverrideOutcome(
            ok=True,
            verified=str(report.verified or ""),
            detail=f"at t={float(report.since):.2f} s",
        )
