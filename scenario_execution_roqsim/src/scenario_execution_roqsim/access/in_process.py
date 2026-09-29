# Copyright (C) 2026 Frederik Pasch
# SPDX-License-Identifier: Apache-2.0

"""The stepped backend: the simulator is this process, so the world is an object graph.

Reads are direct -- ``data.xpos`` between two ``mj_step``s is consistent by construction. Writes are
not: only the physics thread may touch ``model``/``data``, so a write goes through
:meth:`~roqsim.context.SimContext.post` (or an endpoint's own marshalled ``write``) and is observed
one step later. That is roqsim's single-writer rule (architecture.rst §7), and it holds here even
though the stepped runner ticks the tree on the same thread that steps -- because the rule is the
plugin's contract, not this caller's convenience, and the control socket's bridge takes the
identical path.
"""

from __future__ import annotations

import numpy as np

from . import (
    AccessError,
    CommandCall,
    CommandOutcome,
    NavCall,
    NavOutcome,
    Pose,
    ReportCall,
    ReportReading,
    SpawnCall,
    SpawnOutcome,
    TeleportCall,
    TeleportOutcome,
    WorldAccess,
    find_endpoint,
    no_navigator,
    no_report,
    published_field,
    report_value,
)

_MISSING = object()


def _rows(ctx) -> list[dict]:
    """The world's endpoints as :func:`~scenario_execution_roqsim.access.find_endpoint` reads them."""
    from roqsim.context import endpoint_kind
    from roqsim.ipc import path_of

    return [
        {
            "path": path_of(ep),
            "owner": ep.owner,
            "name": ep.name,
            "kind": endpoint_kind(ep),
            "ep": ep,
        }
        for ep in ctx.interface.all()
    ]


class _InProcessReport(ReportCall):
    """A report read straight from its endpoint, resolved again on every poll.

    Resolved every time rather than held: a reset that rebuilds the world replaces the plugins and
    their endpoints, and a held endpoint would go on reading the torn-down one. ``read()`` runs on
    the physics thread, which in the stepped shape is the thread that ticks the tree.
    """

    def __init__(self, access: InProcessAccess, entity: str, report: str, field: str):
        self._access = access
        self._entity, self._report, self._field = entity, report, field

    def poll(self) -> ReportReading | None:
        ctx = self._access._ctx()
        if ctx is None:
            return None
        rows = _rows(ctx)
        try:
            row = find_endpoint(rows, self._entity, self._report, kind="out")
        except AccessError:
            raise no_report(
                rows,
                self._entity,
                self._report,
                is_entity=ctx.entities.get(self._entity) is not None,
            ) from None
        ep = row["ep"]
        payload = ep.read()
        if payload is None:  # the producer has nothing to report yet
            return None
        return report_value(
            f"{self._entity}.{self._report}",
            payload,
            self._field,
            published_field(ep.backend),
            "in-process",
        )


class _InProcessCommand(CommandCall):
    """A command written through its endpoint, and its confirmation read after the step.

    The write goes through the endpoint the way a bridge's does: a marshalled one (declared with
    :mod:`roqsim.endpoint`) queues itself, any other is submitted to the physics thread. The
    stepped runner ticks the tree between steps, so by the time the command's future is done the
    step that drained it has run its ``post_step``, and a confirmation recorded there is current.
    """

    def __init__(self, ctx, row: dict, confirm, value):
        from roqsim.context import CommandFuture

        ep = row["ep"]
        self._confirm = confirm
        self._stream = row["kind"] == "stream"
        self._outcome: CommandOutcome | None = None
        self._future = None
        try:
            if ep.marshalled:
                outcome = ep.write(value)
            else:
                outcome = ctx.submit(lambda _c, w=ep.write, v=value: w(v))
        except Exception as exc:  # noqa: BLE001 - a stream's parameters are refused right here
            self._outcome = CommandOutcome(ok=False, detail=str(exc))
            return
        if self._stream:
            self._outcome = CommandOutcome(ok=True, detail="queued")
        elif isinstance(outcome, CommandFuture):
            self._future = outcome
        else:
            self._outcome = CommandOutcome(ok=True, result=outcome)

    def poll(self) -> CommandOutcome | None:
        from roqsim.context import CommandFuture

        if self._outcome is not None:
            return self._outcome
        if not self._future.done():
            return None
        try:
            result = self._future.result(0)
            if isinstance(result, CommandFuture):
                self._future = result
                return None if not result.done() else self.poll()
        except Exception as exc:  # noqa: BLE001 - the producer's own refusal is the outcome
            self._outcome = CommandOutcome(ok=False, detail=str(exc))
            return self._outcome
        if self._confirm is None:
            self._outcome = CommandOutcome(ok=True, result=result)
            return self._outcome
        confirmation = self._confirm.read()
        self._outcome = CommandOutcome(
            ok=True,
            result=result,
            confirmation=confirmation,
            verified=_verdict(confirmation),
        )
        return self._outcome


def _verdict(confirmation) -> str:
    """A confirmation's ``verified`` field, where it has one."""
    if isinstance(confirmation, dict):
        return str(confirmation.get("verified") or "")
    return str(getattr(confirmation, "verified", "") or "")


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
        #: body id per name, valid for :attr:`_bids_model` only: a reset with other `world_overrides`
        #: compiles a new model with other ids. The model is held, not its `id()`, which a freed
        #: model may pass on to the next one.
        self._bids: dict[str, int] = {}
        self._bids_model = None

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
        if ctx.model is not self._bids_model:
            self._bids, self._bids_model = {}, ctx.model
        if name not in self._bids:
            # Imported HERE rather than at module scope: importing `roqsim.lookup` pulls in MuJoCo, and
            # the behaviour tree is built before any world is compiled. Same reason the actions
            # compare the plugin's verdict strings by value instead of importing its constants.
            from roqsim.lookup import LookupError_, resolve_body_id

            try:
                self._bids[name] = resolve_body_id(ctx, name, what="entity")
            except LookupError_ as err:
                raise AccessError(str(err)) from None
        return self._bids[name]

    # -- reports ----------------------------------------------------------------------------------
    def entity_report(self, entity: str, report: str, field: str = "") -> ReportCall:
        return _InProcessReport(self, entity, report, field)

    # -- commands ---------------------------------------------------------------------------------
    def call_endpoint(self, entity: str, endpoint: str, value=None) -> CommandCall:
        ctx = self._ctx()
        if ctx is None:
            raise AccessError("the world is not built yet; call ready() first")
        rows = _rows(ctx)
        row = find_endpoint(rows, entity, endpoint, kind="in")
        confirm = None
        if row["ep"].confirm:
            sibling = row["path"].rpartition("/")[0] + "/" + row["ep"].confirm
            confirm = next((r["ep"] for r in rows if r["path"] == sibling), None)
        return _InProcessCommand(ctx, row, confirm, value)

    # -- navigation --------------------------------------------------------------------------------
    def navigate(self, name: str, goal_poses, *, wait: bool) -> NavCall:
        handle = self._nav_handle(name)
        poses = [(float(p[0]), float(p[1])) for p in goal_poses]
        try:
            seq = handle.send_goals(poses)
        except ValueError as err:
            raise AccessError(str(err)) from None
        return _PostedRoute(handle, seq, wait=wait)

    def start_route(self, name: str, *, wait: bool) -> NavCall:
        handle = self._nav_handle(name)
        try:
            seq = handle.start()
        except ValueError as err:
            raise AccessError(str(err)) from None
        return _PostedRoute(handle, seq, wait=wait)

    def _nav_handle(self, name: str):
        ctx = self._ctx()
        if ctx is None:
            raise AccessError("the world is not built yet; call ready() first")
        handle = ctx.blackboard.get(f"nav:{name}:handle")
        if handle is None:
            offered = [
                k.split(":", 1)[1].removesuffix(":handle")
                for k in getattr(ctx.blackboard, "_data", {})
                if k.startswith("nav:") and k.endswith(":handle")
            ]
            raise no_navigator(name, offered)
        return handle

    # -- placement and presence ------------------------------------------------------------------
    def set_entity_state(
        self, name: str, pos: np.ndarray, quat: np.ndarray, lin=None, ang=None
    ) -> TeleportCall:
        from roqsim import entity_control

        ctx = self._ctx()
        if ctx is None:
            raise AccessError("the world is not built yet; call ready() first")
        try:
            entity_control.require_entity(ctx, name)
        except entity_control.UnknownEntity as err:
            raise AccessError(str(err)) from None
        box: dict = {}

        def _write(_ctx):
            try:
                box["outcome"] = TeleportOutcome(
                    ok=True, detail=entity_control.set_state(_ctx, name, pos, quat, lin, ang)
                )
            except entity_control.EntityRefused as err:
                box["outcome"] = TeleportOutcome(ok=False, detail=str(err))

        ctx.post(_write)
        return _PostedTeleport(box)

    def set_entity_presence(self, name: str, present: bool, pos=None, quat=None) -> SpawnCall:
        """Flip presence and place the entity in ONE posted callback
        (:func:`roqsim.entity_control.set_presence`)."""
        from roqsim import entity_control

        ctx = self._ctx()
        if ctx is None:
            raise AccessError("the world is not built yet; call ready() first")
        try:
            entity_control.require_entity(ctx, name, spawning=True)
        except entity_control.UnknownEntity as err:
            raise AccessError(str(err)) from None
        box: dict = {}

        def _apply(_ctx):
            try:
                box["outcome"] = SpawnOutcome(
                    ok=True,
                    detail=entity_control.set_presence(_ctx, name, present, pos, quat),
                )
            except entity_control.EntityRefused as err:
                box["outcome"] = SpawnOutcome(ok=False, detail=str(err))

        ctx.post(_apply)
        return _PostedSpawn(box)


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
