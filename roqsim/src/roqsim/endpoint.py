"""Endpoints declared on a plugin's methods.

A plugin marks the methods that are its I/O ports, and the base :class:`~roqsim.plugin.Plugin`
turns them into :class:`~roqsim.context.Endpoint`\\ s once its ``configure`` has run::

    from roqsim import endpoint

    class ForceTorquePlugin(Plugin):
        @endpoint.out("wrench", rate_hz=lambda self: self.rate_hz,
                      ros2=lambda self: {"type": "geometry_msgs.msg.WrenchStamped", ...})
        def read_pair(self):
            ...

        @endpoint.command("tare", ros2={"service": "std_srvs.srv.Trigger"})
        def tare(self, _payload=None):
            ...

Three kinds, by what the caller needs:

``out``
    The method returns the neutral payload. It runs on the physics thread, whenever a bridge reads.
``command``
    A request with an outcome. The endpoint's ``write`` is callable from any thread: it queues the
    method for the physics thread and returns a :class:`~roqsim.context.CommandFuture` holding
    what the method returned or raised.
``stream``
    An inbound stream. ``write`` stores the payload in the endpoint's latest-value slot, and the
    physics thread hands the newest one to the method once per step (while the run is paused, in
    the driver's idle loop). Values superseded within a step are never applied.

Keyword arguments other than the documented ones are backend hints, keyed by backend name
(``ros2=...``). Each is a dict, or a callable taking the plugin and returning the dict (or ``None`` to
leave that backend out) -- hints often need values known only after ``configure``. ``rate_hz`` takes
a number or such a callable too, and ``when`` a callable deciding whether this instance has the
endpoint at all. The owner, namespace and default name come from the plugin
(:attr:`~roqsim.plugin.Plugin.endpoint_owner`, :meth:`~roqsim.plugin.Plugin.endpoint_namespace`,
the method's name).

:func:`declared` lists a class's endpoints without building a world.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .context import Endpoint

if TYPE_CHECKING:
    from .context import SimContext
    from .plugin import Plugin

#: The kinds, and the :class:`~roqsim.context.Endpoint` direction each one is.
DIRECTIONS = {"out": "out", "command": "in", "stream": "in"}

_ATTR = "__roqsim_endpoints__"


@dataclass(frozen=True)
class EndpointSpec:
    """One decorated endpoint, as declared on the class."""

    kind: str
    name: str
    attr: str
    rate_hz: float | Callable[[Any], float] = 0.0
    lazy: bool = False
    when: Callable[[Any], bool] | None = None
    backend: dict[str, dict | Callable[[Any], dict | None]] = field(default_factory=dict)

    @property
    def direction(self) -> str:
        return DIRECTIONS[self.kind]

    def describe(self, cls: type) -> dict:
        """What can be said of it without an instance: static values, never a hint's contents."""
        doc = (getattr(cls, self.attr).__doc__ or "").strip().splitlines()
        row = {
            "name": self.name,
            "kind": self.kind,
            "direction": self.direction,
            "backends": sorted(self.backend),
            "conditional": self.when is not None,
            "doc": doc[0] if doc else None,
        }
        if self.kind == "out" and not callable(self.rate_hz):
            row["rate_hz"] = float(self.rate_hz)
        return row


def _decorator(kind: str, name: str | None, **fields) -> Callable:
    def mark(fn):
        specs = list(getattr(fn, _ATTR, ()))
        specs.append((kind, name, fields))
        setattr(fn, _ATTR, tuple(specs))
        return fn

    return mark


def out(
    name: str | None = None,
    *,
    rate_hz: float | Callable[[Any], float] = 0.0,
    lazy: bool = False,
    when: Callable[[Any], bool] | None = None,
    **backend,
) -> Callable:
    """Declare the decorated method as an ``out`` endpoint; see the module docstring."""
    return _decorator("out", name, rate_hz=rate_hz, lazy=lazy, when=when, backend=backend)


def command(name: str | None = None, *, when: Callable[[Any], bool] | None = None, **backend):
    """Declare the decorated method as a command (``in``, with an outcome).

    The method takes the payload as its one argument.
    """
    return _decorator("command", name, when=when, backend=backend)


def stream(name: str | None = None, *, when: Callable[[Any], bool] | None = None, **backend):
    """Declare the decorated method as an inbound stream (``in``, latest value wins per step).

    The method takes the payload as its one argument.
    """
    return _decorator("stream", name, when=when, backend=backend)


def declared(cls: type) -> list[EndpointSpec]:
    """The endpoints *cls* declares, base classes first, in definition order.

    A subclass that redefines a decorated method replaces its declarations: with its own, or with
    none when the redefinition is undecorated.
    """
    by_attr: dict[str, list[EndpointSpec]] = {}
    for klass in reversed(cls.__mro__):
        for attr, value in vars(klass).items():
            marks = getattr(value, _ATTR, None)
            by_attr.pop(attr, None)
            if marks is None:
                continue
            by_attr[attr] = [
                EndpointSpec(kind=kind, name=name or attr, attr=attr, **fields)
                for kind, name, fields in marks
            ]
    return [spec for specs in by_attr.values() for spec in specs]


def _value(value, plugin):
    return value(plugin) if callable(value) else value


def build(plugin: Plugin, ctx: SimContext) -> list[Endpoint]:
    """The :class:`~roqsim.context.Endpoint`\\ s *plugin* declares, bound to it and to *ctx*."""
    endpoints = []
    specs = declared(type(plugin))
    if not specs:
        return endpoints
    owner = plugin.endpoint_owner
    namespace = plugin.endpoint_namespace(ctx)
    for spec in specs:
        if spec.when is not None and not spec.when(plugin):
            continue
        method = getattr(plugin, spec.attr)
        backend = {}
        for key, hints in spec.backend.items():
            resolved = _value(hints, plugin)
            if resolved is not None:
                backend[key] = dict(resolved)
        ep = Endpoint(
            name=spec.name,
            direction=spec.direction,
            owner=owner,
            namespace=namespace,
            backend=backend,
        )
        if spec.kind == "out":
            ep.read = method
            ep.rate_hz = float(_value(spec.rate_hz, plugin))
            ep.lazy = spec.lazy
        elif spec.kind == "command":
            ep.write = _submitter(ctx, method)
            ep.marshalled = True
        else:
            ep.write = ctx.stream_slot(f"{owner}/{spec.name}", method).put
            ep.marshalled = True
        endpoints.append(ep)
    return endpoints


def _submitter(ctx: SimContext, method: Callable[[Any], Any]) -> Callable:
    def write(payload=None):
        return ctx.submit(lambda _ctx: method(payload))

    return write
