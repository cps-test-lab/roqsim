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
    OverrideCall,
    OverrideOutcome,
    Pose,
    TeleportCall,
    TeleportOutcome,
    WorldAccess,
)

_MISSING = object()


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
    def apply_override(self, instance: str, active: bool) -> OverrideCall:
        ctx = self._ctx()
        if ctx is None:
            raise AccessError("the world is not built yet; call ready() first")
        key = f"model_override:{instance}"
        handle = ctx.blackboard.get(key)
        if handle is None:
            raise AccessError(
                f"nothing on the blackboard under {key!r}. It is published by a `model_override` "
                "plugin instance in the world -- check that the campaign's config is the world "
                "carrying the fault, and that the instance's `name:` matches what the scenario "
                "asks for. `roqsim scenes describe <world>` lists what a world offers an override."
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

    # -- teleport ---------------------------------------------------------------------------------
    def set_entity_pose(self, name: str, pos: np.ndarray, quat: np.ndarray) -> TeleportCall:
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
        ):
            jid = (
                mujoco.mj_name2id(_ctx.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
                if joint_name
                else -1
            )
            if jid < 0 or _ctx.model.jnt_type[jid] != mujoco.mjtJoint.mjJNT_FREE:
                outcome_box["outcome"] = TeleportOutcome(
                    ok=False,
                    detail=f"entity {name!r} has no free joint named {joint_name!r} -- it cannot be "
                    "teleported (a static prop, or a model without a base_joint in its meta).",
                )
                return
            q = _ctx.model.jnt_qposadr[jid]
            _ctx.data.qpos[q : q + 3] = pos
            _ctx.data.qpos[q + 3 : q + 7] = quat
            dof = _ctx.model.jnt_dofadr[jid]
            _ctx.data.qvel[dof : dof + 6] = 0.0
            mujoco.mj_forward(_ctx.model, _ctx.data)
            outcome_box["outcome"] = TeleportOutcome(ok=True, detail=f"placed at {pos.tolist()}")

        ctx.post(_write)
        return _PostedTeleport(outcome_box)


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
