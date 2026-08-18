"""Backend-agnostic bridge base: wire a robot's declared interface to a transport.

A concrete bridge (ROS 2, zenoh, zmq, ...) subclasses :class:`BridgeBase`, sets ``BACKEND`` to its
key, and implements the small set of backend hooks below. Everything else -- discovering endpoints,
rate-gating, the per-tick publish loop, and marshalling inbound data onto the physics thread -- lives
here and is shared across backends.

The bridge reads :class:`roqsim.context.Endpoint`s registered by the robot's plugins; it never
imports the robot package or hardcodes topic/stream names. Backend particulars (message type, topic,
QoS, frames) come from each endpoint's ``backend[BACKEND]`` hint block, so adding an interface is a
one-line endpoint registration on the producer with zero bridge edits.

Threading (see docs/architecture.rst > Concurrency): ``_setup``/``configure``/``post_step``/
``shutdown`` run on the physics thread. Inbound transport callbacks run on the backend's own thread
and MUST NOT touch ``data`` -- they call the ``on_payload`` handed to :meth:`_make_input`, which
marshals the write onto the physics thread via ``ctx.post``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .plugin import Plugin

if TYPE_CHECKING:
    from .context import Endpoint, SimContext


class _RateGate:
    """Emit at most once per ``1/rate`` of sim-time. ``rate <= 0`` => every step (ungated)."""

    def __init__(self, rate_hz: float) -> None:
        self.rate_hz = float(rate_hz)
        self._last: float | None = None

    def due(self, t: float) -> bool:
        if self.rate_hz <= 0.0:
            return True
        if self._last is None or (t - self._last) >= (1.0 / self.rate_hz) - 1e-9:
            self._last = t
            return True
        return False

    def reset(self) -> None:
        self._last = None


@dataclass
class _Output:
    endpoint: Endpoint
    handle: Any  # opaque, created by the subclass (_make_output)
    gate: _RateGate


class BridgeBase(Plugin):
    """Base for transport bridges. Subclass, set ``BACKEND``, implement the backend hooks."""

    #: Backend key selecting which ``endpoint.backend[...]`` hint block applies (e.g. "ros2").
    BACKEND: str = ""

    # A bridge publishes what the other plugins built; it adds nothing to the scene itself. Declared
    # here rather than per bridge so any transport -- including out-of-tree ones -- is renderable
    # without its middleware installed.
    transport_only = True

    def __init__(self, config=None, *, name=None):
        super().__init__(config, name=name)
        self._ctx: SimContext | None = None
        self._outputs: list[_Output] = []
        self._ready = False
        # Optional owner filter (``owner``: a name or list of names; omit to serve all endpoints).
        # The common case is ONE bridge serving everything -- per-robot scoping comes from each
        # endpoint's ``namespace``, not from running one filtered bridge per robot. The filter stays
        # for the rare split (e.g. two transports, or excluding a robot from ROS entirely).
        owner = self.config.get("owner")
        if owner is None:
            self._owners: set[str] | None = None
        else:
            self._owners = {owner} if isinstance(owner, str) else set(owner)

    # -- lifecycle --------------------------------------------------------------------------------
    def configure(self, ctx: SimContext) -> None:
        self._ctx = ctx
        self._setup(ctx)
        self._bind(ctx)
        self._ready = True

    def _bind(self, ctx: SimContext) -> None:
        # Producers register their endpoints in configure(); the bridge is loaded last (world YAML
        # convention), so ctx.interface is fully populated by now. Closing the registry makes that
        # convention enforced rather than assumed: a producer listed after the bridge now raises
        # instead of quietly never being published.
        ctx.interface.mark_bound(self.name)
        rate_overrides = self.config.get("rates", {})
        for ep in ctx.interface.all():
            hints = ep.backend.get(self.BACKEND)
            if hints is None:
                continue
            if self._owners is not None and ep.owner not in self._owners:
                continue
            rate = float(rate_overrides.get(ep.name, hints.get("rate_hz", ep.rate_hz)))
            if ep.direction == "out":
                if ep.read is None:
                    ctx.logger.warning("bridge: out endpoint %r has no read(); skipped", ep.name)
                    continue
                handle = self._make_output(ep, hints)
                self._outputs.append(_Output(ep, handle, _RateGate(rate)))
            elif ep.direction == "in":
                if ep.write is None:
                    ctx.logger.warning("bridge: in endpoint %r has no write(); skipped", ep.name)
                    continue
                self._make_input(ep, hints, self._inbound(ep))
            else:
                ctx.logger.warning(
                    "bridge: endpoint %r has bad direction %r", ep.name, ep.direction
                )

    def _inbound(self, ep: Endpoint):
        """Return a thread-safe callback that marshals a neutral payload onto the physics thread."""

        def on_payload(payload) -> None:
            ctx = self._ctx
            if ctx is not None:
                ctx.post(lambda c, w=ep.write, p=payload: w(p))

        return on_payload

    def on_reset(self, ctx: SimContext) -> None:
        for out in self._outputs:
            out.gate.reset()

    def post_step(self, ctx: SimContext) -> None:
        if not self._ready:
            return
        t = ctx.sim_time
        stamp = self._now(t)
        for out in self._outputs:
            if out.gate.due(t):
                payload = out.endpoint.read()
                if payload is not None:
                    self._publish(out.handle, payload, stamp)
        self._tick(ctx, t, stamp)

    def shutdown(self, ctx: SimContext) -> None:
        self._teardown(ctx)

    # -- backend hooks (subclass implements) ------------------------------------------------------
    def _setup(self, ctx: SimContext) -> None:
        """Initialise the transport (open a node/session, start any spin thread)."""

    def _make_output(self, ep: Endpoint, hints: dict) -> Any:
        """Create a publisher for an ``out`` endpoint; return an opaque handle for :meth:`_publish`."""
        raise NotImplementedError

    def _make_input(self, ep: Endpoint, hints: dict, on_payload) -> None:
        """Subscribe for an ``in`` endpoint; call ``on_payload(neutral_payload)`` on each message."""
        raise NotImplementedError

    def _publish(self, handle: Any, payload: Any, stamp: Any) -> None:
        """Serialise ``payload`` through ``handle`` and send it. Reuse buffers for the hot path."""
        raise NotImplementedError

    def _now(self, t: float) -> Any:
        """Backend timestamp for sim-time ``t`` (computed once per tick). Default: the float itself."""
        return t

    def _tick(self, ctx: SimContext, t: float, stamp: Any) -> None:
        """Optional per-tick extras owned by the backend (e.g. a clock/time source)."""

    def _teardown(self, ctx: SimContext) -> None:
        """Release transport resources."""
