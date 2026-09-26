"""Backend-agnostic bridge base: wire a robot's declared interface to a transport.

A concrete bridge (ROS 2, zenoh, zmq, ...) subclasses :class:`BridgeBase`, sets ``BACKEND`` to its
key, and implements the small set of backend hooks below. Everything else -- discovering endpoints,
rate-gating (on the world's physics grid, see :meth:`BridgeBase._rate_gate`), skipping endpoints that
opted out of publishing to nobody (``Endpoint.lazy``), the per-tick publish loop, marshalling
inbound data onto the physics thread, and the map of what the bridge publishes
(:meth:`BridgeBase.endpoint_map`) -- lives here and is shared across backends.

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
from .rates import SNAP_NOTABLE, SNAP_QUIET, GridRate, snap_rate

if TYPE_CHECKING:
    from collections.abc import Callable

    from .context import Endpoint, SimContext

#: Where a bridge advertises what it publishes (:meth:`BridgeBase.endpoint_map`), relative to the
#: bridge's own scope -- for a ROS bridge, its node namespace. A reader in the same scope finds it
#: by this name without knowing anything else about the deployment.
ENDPOINT_MAP = "roqsim/endpoints"


class _RateGate:
    """Emit once per ``every`` physics steps. ``rate <= 0`` => every step (ungated).

    ``due`` is asked exactly once per physics step, by the publish loop below and by a backend's own
    per-step work -- so a gate that knows how many steps apart its firings are needs no clock at all:
    it counts them. :meth:`BridgeBase._rate_gate` is what supplies that count, by putting the
    requested rate on one this world can hold (``physics_rate / every``). Counting rather than
    comparing sim-time is what makes the spacing exactly ``every`` for the whole run: accumulated
    float time cannot drift off a step count, and no epsilon has to guard the comparison.

    A gate built straight from a rate carries no count (``every`` is ``None``) and falls back to the
    time comparison, which fires at the first step at or past the period -- i.e. at the SLOWER
    neighbour of any rate that is not a whole number of steps. That path exists for a gate built
    before a model is compiled, where there is no grid to snap against yet.
    """

    def __init__(self, rate_hz: float, *, every: int | None = None) -> None:
        self.rate_hz = float(rate_hz)
        self.every = every
        self._last: float | None = None
        self._since = 0  # steps since the last firing, for a gate that carries its step count

    def due(self, t: float) -> bool:
        if self.rate_hz <= 0.0:
            return True
        if self.every is not None:
            fire = self._since == 0
            self._since = (self._since + 1) % self.every
            return fire
        if self._last is None or (t - self._last) >= (1.0 / self.rate_hz) - 1e-9:
            self._last = t
            return True
        return False

    def reset(self) -> None:
        self._last = None
        self._since = 0


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

    def __init__(self, config=None, *, name=None, entity=None, label=None):
        super().__init__(config, name=name, entity=entity, label=label)
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
            if ep.direction == "out":
                if ep.read is None:
                    ctx.logger.warning("bridge: out endpoint %r has no read(); skipped", ep.name)
                    continue
                requested = float(rate_overrides.get(ep.name, hints.get("rate_hz", ep.rate_hz)))
                handle = self._make_output(ep, hints)
                gate = self._rate_gate(ctx, requested, f"endpoint {ep.name!r}")
                self._outputs.append(_Output(ep, handle, gate))
                self._record_rate(ctx, ep.name, ep.owner, ep.namespace, requested, gate)
            elif ep.direction == "in":
                if ep.write is None:
                    ctx.logger.warning("bridge: in endpoint %r has no write(); skipped", ep.name)
                    continue
                self._make_input(ep, hints, self._inbound(ep))
            else:
                ctx.logger.warning(
                    "bridge: endpoint %r has bad direction %r", ep.name, ep.direction
                )

    def endpoint_map(self, describe: Callable[[_Output], dict]) -> dict:
        """What this bridge publishes, keyed as the world names it: ``(owner, endpoint name)``.

        A scenario addresses a plugin's report by the entity that owns it and the endpoint's name;
        the transport carries it under whatever the backend made of that -- for ROS a topic after the
        endpoint's namespace, a ``topics:`` rename, a stripped namespace and a ground-truth prefix.
        Only the bridge knows the result exactly, because it is what resolved it, so the bridge says
        it rather than a reader re-deriving it. ``describe`` is the backend's half: where and how one
        bound output travels (a ROS bridge: its topic, message type and published field).

        ``owners`` is the owner filter (``None`` = every owner), so a reader can tell "that entity
        publishes no such report" from "this bridge does not serve that entity".
        """
        return {
            "owners": None if self._owners is None else sorted(self._owners),
            "endpoints": [
                {"owner": out.endpoint.owner, "name": out.endpoint.name, **describe(out)}
                for out in self._outputs
            ],
        }

    def _rate_gate(self, ctx: SimContext, rate_hz: float, subject: str) -> _RateGate:
        """A gate at the nearest rate this world can hold, announced in proportion to the move.

        A publication is tested once per physics step, so the rates a world can hold are exactly
        ``physics_rate / k`` for integer ``k >= 1``. A request between two of them is served at one of
        them for the whole run -- a constant that is not the requested one, rather than jitter that
        averages out -- so the rate is put on the grid here, where the timestep is compiled and the
        request is known. Left to the gate alone it would land on the same grid anyway and go on
        calling itself the requested rate, which is the number the world document, the campaign's
        factor level and the result all quote.

        Same grid, same bands and same vocabulary as a capture rate (:func:`roqsim.rates.snap_rate`),
        because it is the same constraint. It snaps to the NEAREST achievable rate, which may be the
        faster neighbour -- so a rate meant as a ceiling has to be stated as one this world can hold;
        an ungated gate could only ever have reached the slower one.

        **Not the other fix.** A gate that advanced its deadline BY the period instead of to the
        firing time would hold the requested rate on average and alternate its spacing to do it (30 Hz
        on a 2 ms step as 17, 17, 16, 17 steps). That trades a wrong constant for a rate that is right
        on average and wrong at every single step, and the spacing is the part a consumer reads: a
        speed, a rate, anything per second is differentiated from it, and a filter's dt is set from it,
        so an alternating one is read as the robot's behaviour rather than as the gate's. A ROS bridge
        already warns where a coarse ``/clock`` imposes exactly that pattern on an output's stamps. One
        exact rate a run can state is worth more here than a mean nobody observes.

        ``rate_hz <= 0`` is left alone: it means every step / event-driven, which is on the grid by
        construction.
        """
        model = getattr(ctx, "model", None)
        if rate_hz <= 0.0:
            return _RateGate(rate_hz)
        if model is None:
            # An embedding driver's bare context: bound before compile, so there is no timestep and
            # no grid. Said out loud rather than silently left off the grid, because this is the one
            # path where the realised rate is neither snapped nor recorded.
            ctx.logger.debug(
                "bridge: %s keeps its requested %g Hz unsnapped and unrecorded -- no model is "
                "compiled yet, so this world has no step grid to put it on",
                subject,
                rate_hz,
            )
            return _RateGate(rate_hz)
        snapped = snap_rate(rate_hz, float(model.opt.timestep))
        self._report_rate_snap(ctx, snapped, subject)
        return _RateGate(float(snapped.hz), every=snapped.every)

    @staticmethod
    def _report_rate_snap(ctx: SimContext, rate: GridRate, subject: str) -> None:
        """Say how far the snap moved the rate: nothing, a note, or a warning naming the neighbours.

        The bands are :mod:`roqsim.rates`' (:data:`~roqsim.rates.SNAP_QUIET`,
        :data:`~roqsim.rates.SNAP_NOTABLE`), so one move is announced equally loudly whether it hits a
        recording or a topic. The common case stays silent -- a sensor at 10 Hz on a 2 ms step is
        exact -- because a line per endpoint would bury the one endpoint that is not.

        Every rate in the message is one that can be given back to the producer or to ``rates:``: a
        neighbour is stated as a decimal and as its ``k``, and typing either back reaches this same
        gate. The warning names the other remedy too -- a step rate that is a whole multiple of the
        request serves it exactly -- because moving the world is the way to KEEP a rate that came from
        a paper rather than from us.
        """
        if rate.deviation <= 0:
            return
        detail = f"{float(rate.hz):.4g} Hz (every {rate.every} steps, exactly {rate.rational()})"
        if rate.deviation < SNAP_QUIET:
            ctx.logger.debug("bridge: %s %g Hz -> %s", subject, float(rate.requested), detail)
        elif rate.deviation < SNAP_NOTABLE:
            ctx.logger.info(
                "bridge: %s asked for %g Hz, published at %s",
                subject,
                float(rate.requested),
                detail,
            )
        else:
            nearby = ", ".join(f"{float(n.hz):g} Hz (k={n.every})" for n in rate.neighbours())
            ctx.logger.warning(
                "bridge: %s asked for %g Hz and publishes at %s -- a publish lands on a physics "
                "step and this world steps at %g Hz, so the request is %.2f steps apart and no "
                "whole number of steps gives it. The spacing is exact, so there is no drift. "
                "Nearby: %s. To keep %g Hz itself, step this world at a multiple of it -- the "
                "nearest is %g Hz (sim.timestep %.17g).",
                subject,
                float(rate.requested),
                detail,
                float(rate.physics),
                float(rate.physics / rate.requested),
                nearby,
                float(rate.requested),
                float(rate.exact_step_rate()),
                1.0 / float(rate.exact_step_rate()),
            )

    def _record_rate(
        self,
        ctx: SimContext,
        name: str,
        owner: str | None,
        namespace: str,
        requested_hz: float,
        gate: _RateGate,
    ) -> None:
        """Write one gated publication's requested and realised rate into the run's record.

        Beside the requested rate rather than in place of it: the world document, a campaign's factor
        level and whatever a result quotes all carry what was asked for, and the only other way to
        learn what the run published at is to measure the arrival times of a stream nobody kept. The
        log line above says it once; this is the half that reaches a reader who never opens the log.

        Every gate a bridge fires on belongs here, not only the ones an endpoint owns: a backend that
        publishes a stream of its own -- a merged joint state, a clock -- has a rate too, and that one
        appears in no world document at all, so the record is the only place it can be read.

        A requested ``0`` is "every step", so its realised rate is the world's step rate. Nothing is
        recorded before a model exists, there being no grid to state the rate on.
        """
        model = getattr(ctx, "model", None)
        if model is None:
            return
        dt = float(model.opt.timestep)
        ctx.endpoint_rates.append(
            {
                "name": name,
                "owner": owner,
                "namespace": namespace,
                "backend": self.BACKEND,
                "requested_hz": float(requested_hz),
                "realised_hz": float(gate.rate_hz) if gate.rate_hz > 0.0 else 1.0 / dt,
                "every_steps": gate.every or 1,
            }
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
            if not out.gate.due(t):
                continue
            if self._skip_unsubscribed(out.endpoint):
                continue
            payload = out.endpoint.read()
            if payload is not None:
                self._publish(out.handle, payload, stamp)
        self._tick(ctx, t, stamp)

    @staticmethod
    def _skip_unsubscribed(ep: Endpoint) -> bool:
        """Whether to skip an opted-in (``lazy``) endpoint because nothing is listening.

        Opt-in rather than applied to every output, because a publish can carry more than its own
        message: :meth:`_publish` may derive TF from an odometry payload, so a lazy ``odom`` would
        silently stop broadcasting ``odom -> base_link`` whenever nothing subscribed to the topic.
        Cheap payloads (scan, joint_states) gain nothing from the check either.

        ``has_subscribers is None`` = no introspection available (no transport wired it, or a backend
        that cannot tell) => publish, the same "assume yes" convention producers use for the
        render-side check.
        """
        return ep.lazy and ep.has_subscribers is not None and not ep.has_subscribers()

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
