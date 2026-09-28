"""Endpoints declared on a plugin's methods, typed by the method's signature.

A plugin marks the methods that are its I/O ports, and the base :class:`~roqsim.plugin.Plugin`
turns them into :class:`~roqsim.context.Endpoint`\\ s once its ``configure`` has run::

    from typing import Annotated

    from roqsim import endpoint
    from roqsim.endpoint import Unit

    class BasePlugin(Plugin):
        @endpoint.command(ros2={"service": "std_srvs.srv.Trigger"})
        def tare(self) -> None:
            ...

        @endpoint.stream("cmd_vel", ros2=lambda self: {"type": "geometry_msgs.msg.Twist", ...})
        def command_twist(
            self,
            vx: Annotated[float, Unit("m/s"), "forward speed"],
            w: Annotated[float, Unit("rad/s"), "yaw rate"] = 0.0,
        ) -> None:
            ...

        @endpoint.out("odom", rate_hz=lambda self: self.rate_hz)
        def read_odom(self) -> Odometry:
            ...

Three kinds, by what the caller needs:

``out``
    The method takes no parameters and returns the neutral payload. It runs on the physics thread,
    whenever a bridge reads. Its return annotation is the payload's type (:attr:`Endpoint.result`).
``command``
    A request with an outcome. The endpoint's ``write`` is callable from any thread: it checks the
    named parameters against the method's signature, queues the method for the physics thread and
    returns a :class:`~roqsim.context.CommandFuture` holding what the method returned or raised.
``stream``
    An inbound stream. ``write`` checks the parameters the same way and stores them in the
    endpoint's latest-value slot, and the physics thread calls the method with the newest ones once
    per step (while the run is paused, in the driver's idle loop). Values superseded within a step
    are never applied.

**The signature is the schema.** A command's or a stream's parameters are the method's own:
annotated types, defaults (a parameter without one is required), and per parameter a unit and a
line of documentation through :class:`typing.Annotated` -- a :class:`Unit` and a plain string. A
bridge passes them as a mapping of names to values; ``None`` is the empty mapping. What ``write``
accepts is checked before anything is queued: a missing required parameter, an unknown one (with
the nearest known name), and a value of the wrong type are refused with one :class:`ParameterError`
naming all of them -- raised into the returned future for a command, raised to the caller for a
stream. An ``int`` passes for a ``float``, a ``bool`` for neither, and a sequence for a declared
``numpy`` array of the declared dtype and :class:`Shape`. The resulting schema is data on the
endpoint (:attr:`Endpoint.params`, :attr:`Endpoint.result`), so any bridge can read it.

**A family** is one declaration for a set decided by config -- one endpoint per joint, per wheel,
per instance. ``each`` is a callable taking the plugin and returning the items; each item makes one
endpoint named ``<name>/<item>`` (or the name with ``{item}`` substituted), and the item is passed
to the method as its first argument, which is not a parameter of the endpoint::

    @endpoint.out("joints/{item}/effort", each=lambda self: self.joint_names)
    def effort(self, joint: str) -> Annotated[float, Unit("N*m")]:
        ...

In a family every callable option -- a backend hint, ``rate_hz``, ``when``, ``owner``,
``namespace`` -- takes ``(plugin, item)``.

Keyword arguments other than the documented ones are backend hints, keyed by backend name
(``ros2=...``). Each is a dict, or a callable taking the plugin and returning the dict (or ``None`` to
leave that backend out) -- hints often need values known only after ``configure``. ``rate_hz`` takes
a number or such a callable too, and ``when`` a callable deciding whether this instance has the
endpoint at all. The owner, namespace and default name come from the plugin
(:attr:`~roqsim.plugin.Plugin.endpoint_owner`, :meth:`~roqsim.plugin.Plugin.endpoint_namespace`,
the method's name). An endpoint that belongs to another entity than the plugin's other ones names it
with ``owner=`` (and ``namespace=`` where its scope differs too), a value or a callable.

:func:`declared` lists a class's endpoints, with their schema, without building a world.
"""

from __future__ import annotations

import dataclasses
import functools
import inspect
import types
import typing
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from .context import CommandFuture, Endpoint
from .document import nearest

if TYPE_CHECKING:
    from .context import SimContext
    from .plugin import Plugin

#: The kinds, and the :class:`~roqsim.context.Endpoint` direction each one is.
DIRECTIONS = {"out": "out", "command": "in", "stream": "in"}

_ATTR = "__roqsim_endpoints__"


# -- the markers a signature is annotated with ---------------------------------------------------
class Unit:
    """The unit of a parameter or a result value: ``Annotated[float, Unit("m/s")]``.

    A plain string, spelled as :class:`roqsim.schema.Field`'s ``unit`` is, so a plugin's config and
    its endpoints state units in one vocabulary.
    """

    __slots__ = ("symbol",)

    def __init__(self, symbol: str) -> None:
        self.symbol = str(symbol)

    def __repr__(self) -> str:
        return f"Unit({self.symbol!r})"

    def __eq__(self, other) -> bool:
        return isinstance(other, Unit) and other.symbol == self.symbol

    def __hash__(self) -> int:
        return hash(("Unit", self.symbol))


class Shape:
    """The shape of a numpy array value: ``Annotated[NDArray[np.float64], Shape(3)]``.

    ``None`` leaves a dimension free (``Shape(None, 3)`` is any number of 3-vectors).
    """

    __slots__ = ("dims",)

    def __init__(self, *dims: int | None) -> None:
        self.dims = tuple(None if d is None else int(d) for d in dims)

    def __repr__(self) -> str:
        return f"Shape{self.dims!r}"

    def __eq__(self, other) -> bool:
        return isinstance(other, Shape) and other.dims == self.dims

    def __hash__(self) -> int:
        return hash(("Shape", self.dims))


class ParameterError(ValueError):
    """What a bridge passed does not fit the endpoint's parameters; the message names each misfit."""


# -- the schema, as data ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ValueType:
    """The type of one parameter, one result, or one field of either.

    ``kind`` is what a bridge dispatches on: ``float``, ``int``, ``bool``, ``str``, ``array`` (numpy,
    with ``dtype``/``shape`` where declared), ``list`` (``items`` has the one element type),
    ``tuple`` (``items`` has one type per position), ``dict``, ``struct`` (a dataclass; ``fields``),
    ``none``, ``any`` (unannotated, or not describable further; ``name`` still says what it is).
    """

    kind: str
    name: str
    unit: str = ""
    doc: str = ""
    optional: bool = False
    dtype: str | None = None
    shape: tuple[int | None, ...] | None = None
    items: tuple[ValueType, ...] = ()
    fields: tuple[Param, ...] = ()
    cls: type | None = field(default=None, compare=False, repr=False)

    def describe(self) -> dict:
        """JSON-friendly: the same shape for every endpoint, whichever plugin declared it."""
        row: dict[str, Any] = {"type": self.name}
        if self.unit:
            row["unit"] = self.unit
        if self.doc:
            row["doc"] = self.doc
        if self.optional:
            row["optional"] = True
        if self.dtype is not None:
            row["dtype"] = self.dtype
        if self.shape is not None:
            row["shape"] = list(self.shape)
        if self.items:
            row["items"] = [item.describe() for item in self.items]
        if self.fields:
            row["fields"] = [f.describe() for f in self.fields]
        return row


@dataclass(frozen=True)
class Param:
    """One named parameter of a command or a stream (or one field of a struct)."""

    name: str
    type: ValueType
    required: bool = True
    default: Any = None

    def describe(self) -> dict:
        row = {"name": self.name, **self.type.describe(), "required": self.required}
        if not self.required:
            row["default"] = _plain(self.default)
        return row


def _plain(value):
    """A default as JSON can carry it."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return [_plain(v) for v in value]
    return value


_SCALARS = {float: "float", int: "int", bool: "bool", str: "str"}


def value_type(hint: Any, *, where: str = "") -> ValueType:
    """The :class:`ValueType` an annotation declares (``Annotated`` markers included)."""
    unit, doc, shape = "", "", None
    if typing.get_origin(hint) is typing.Annotated:
        hint, *meta = typing.get_args(hint)
        for m in meta:
            if isinstance(m, Unit):
                unit = m.symbol
            elif isinstance(m, Shape):
                shape = m.dims
            elif isinstance(m, str):
                doc = m
    base = _base_type(hint, where)
    return dataclasses.replace(
        base, unit=unit or base.unit, doc=doc or base.doc, shape=shape or base.shape
    )


def _base_type(hint: Any, where: str) -> ValueType:
    if hint is inspect.Parameter.empty or hint is Any:
        return ValueType("any", "any")
    if hint is None or hint is type(None):
        return ValueType("none", "none")
    origin = typing.get_origin(hint)
    args = typing.get_args(hint)
    if hasattr(origin, "__value__"):  # a generic ``type`` alias, numpy's NDArray among them
        return _base_type(origin.__value__[args], where)
    if hasattr(hint, "__value__"):
        return _base_type(hint.__value__, where)
    if origin is typing.Union or origin is types.UnionType:
        rest = [a for a in args if a is not type(None)]
        if len(rest) == 1 and len(rest) < len(args):
            inner = value_type(rest[0], where=where)
            return dataclasses.replace(inner, optional=True)
        return ValueType("any", " | ".join(_name(a) for a in args))
    if hint in _SCALARS:
        return ValueType(_SCALARS[hint], _SCALARS[hint], cls=hint)
    if hint is np.ndarray or origin is np.ndarray:
        dtype = None
        if len(args) == 2:
            (dt,) = typing.get_args(args[1]) or (None,)
            if isinstance(dt, type) and issubclass(dt, np.generic):
                dtype = np.dtype(dt).name
        return ValueType("array", "array", dtype=dtype)
    if origin in (list, Sequence) or hint in (list, Sequence):
        item = value_type(args[0], where=where) if args else ValueType("any", "any")
        return ValueType("list", f"list[{item.name}]", items=(item,))
    if origin is tuple or hint is tuple:
        if len(args) == 2 and args[1] is Ellipsis:
            item = value_type(args[0], where=where)
            return ValueType("list", f"tuple[{item.name}, ...]", items=(item,))
        items = tuple(value_type(a, where=where) for a in args)
        return ValueType("tuple", f"tuple[{', '.join(i.name for i in items)}]", items=items)
    if origin in (dict, Mapping) or hint in (dict, Mapping):
        return ValueType("dict", "dict")
    if isinstance(hint, type) and dataclasses.is_dataclass(hint):
        return ValueType("struct", hint.__name__, fields=_dataclass_fields(hint), cls=hint)
    if isinstance(hint, type):
        return ValueType("any", hint.__name__, cls=hint)
    return ValueType("any", _name(hint))


def _name(hint: Any) -> str:
    return getattr(hint, "__name__", None) or str(hint).replace("typing.", "")


@functools.cache
def _dataclass_fields(cls: type) -> tuple[Param, ...]:
    hints = _hints(cls, cls.__qualname__)
    out = []
    for f in dataclasses.fields(cls):
        has_default = (
            f.default is not dataclasses.MISSING or f.default_factory is not dataclasses.MISSING
        )
        default = f.default if f.default is not dataclasses.MISSING else None
        out.append(
            Param(
                f.name,
                value_type(hints.get(f.name, inspect.Parameter.empty), where=cls.__qualname__),
                required=not has_default,
                default=default,
            )
        )
    return tuple(out)


def _hints(obj: Any, where: str) -> dict:
    try:
        return typing.get_type_hints(obj, include_extras=True)
    except NameError as exc:
        # Loudly: the signature is the endpoint's schema, so an annotation that cannot be resolved
        # is a schema nobody can read -- most often a type imported only under TYPE_CHECKING.
        raise TypeError(
            f"{where}: an annotation names {exc.name!r}, which is not importable when the endpoint "
            f"is described; import it at module level (not under TYPE_CHECKING)"
        ) from exc


@functools.cache
def signature(fn: Callable, kind: str, family: bool) -> tuple[tuple[Param, ...], ValueType | None]:
    """``(params, result)`` of a decorated method: the endpoint's schema, read off its signature.

    The first parameter is ``self``, and in a family the second is the item. An ``out`` takes no
    other parameter; a ``stream`` has no result. ``*args``, ``**kwargs`` and positional-only
    parameters are refused: a bridge names every parameter it passes.
    """
    where = fn.__qualname__
    hints = _hints(fn, where)
    sig = inspect.signature(fn)
    params = list(sig.parameters.values())[2 if family else 1 :]
    out = []
    for p in params:
        if p.kind not in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY):
            raise TypeError(
                f"{where}: endpoint parameter {p.name!r} is {p.kind.description}; an endpoint's "
                f"parameters are named, so declare each one explicitly"
            )
        out.append(
            Param(
                p.name,
                value_type(hints.get(p.name, inspect.Parameter.empty), where=where),
                required=p.default is inspect.Parameter.empty,
                default=None if p.default is inspect.Parameter.empty else p.default,
            )
        )
    if kind == "out" and out:
        raise TypeError(
            f"{where}: an out endpoint takes no parameters, but declares "
            f"{', '.join(p.name for p in out)}"
        )
    result = None
    if kind != "stream" and sig.return_annotation is not inspect.Signature.empty:
        result = value_type(hints.get("return"), where=where)
    return tuple(out), result


# -- checking what a bridge passed -----------------------------------------------------------------
class _Mismatch(Exception):
    pass


def _check(vt: ValueType, value: Any) -> Any:
    """*value* as *vt* declares it (an ``int`` made a ``float``, a list an array), or raise."""
    if value is None:
        if vt.optional or vt.kind in ("none", "any"):
            return None
        raise _Mismatch(f"must be {vt.name}, got None")
    kind = vt.kind
    if kind == "float":
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number)):
            raise _Mismatch(f"must be float, got {type(value).__name__} ({value!r})")
        return float(value)
    if kind == "int":
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise _Mismatch(f"must be int, got {type(value).__name__} ({value!r})")
        return int(value)
    if kind == "bool":
        if not isinstance(value, (bool, np.bool_)):
            raise _Mismatch(f"must be bool, got {type(value).__name__} ({value!r})")
        return bool(value)
    if kind == "str":
        if not isinstance(value, str):
            raise _Mismatch(f"must be str, got {type(value).__name__} ({value!r})")
        return value
    if kind == "array":
        if isinstance(value, (str, bytes, Mapping)):
            raise _Mismatch(f"must be an array, got {type(value).__name__}")
        try:
            array = np.asarray(value, dtype=vt.dtype)
        except (TypeError, ValueError) as exc:
            raise _Mismatch(f"must be an array of {vt.dtype or 'numbers'}: {exc}") from None
        if vt.shape is not None and (
            array.ndim != len(vt.shape)
            or any(d is not None and d != n for d, n in zip(vt.shape, array.shape, strict=True))
        ):
            want = "(" + ", ".join("*" if d is None else str(d) for d in vt.shape) + ")"
            raise _Mismatch(f"must have shape {want}, got {array.shape}")
        return array
    if kind in ("list", "tuple"):
        if isinstance(value, (str, bytes, Mapping)) or not isinstance(
            value, (Sequence, np.ndarray)
        ):
            raise _Mismatch(f"must be {vt.name}, got {type(value).__name__}")
        if kind == "tuple" and len(value) != len(vt.items):
            raise _Mismatch(f"must have {len(vt.items)} entries, got {len(value)}")
        items = vt.items if kind == "tuple" else vt.items * len(value)
        checked = []
        for i, (item, v) in enumerate(zip(items, value, strict=True)):
            try:
                checked.append(_check(item, v))
            except _Mismatch as exc:
                raise _Mismatch(f"entry {i} {exc}") from None
        return tuple(checked) if kind == "tuple" else checked
    if kind == "dict":
        if not isinstance(value, Mapping):
            raise _Mismatch(f"must be a mapping, got {type(value).__name__}")
        return value
    if kind == "struct":
        if isinstance(value, vt.cls):
            return value
        if not isinstance(value, Mapping):
            raise _Mismatch(
                f"must be {vt.name} or a mapping of its fields, got {type(value).__name__}"
            )
        try:
            return vt.cls(**_bind(vt.fields, value, vt.name))
        except ParameterError as exc:
            raise _Mismatch(str(exc)) from None
    if vt.cls is not None and not isinstance(value, vt.cls):
        raise _Mismatch(f"must be {vt.name}, got {type(value).__name__}")
    return value


def _bind(params: tuple[Param, ...], payload: Any, where: str) -> dict:
    if payload is None:
        payload = {}
    if not isinstance(payload, Mapping):
        raise ParameterError(
            f"{where} takes named parameters (a mapping), got {type(payload).__name__} "
            f"({payload!r}). It takes {_listing(params)}."
        )
    known = {p.name: p for p in params}
    errors = []
    for key in payload:
        if key not in known:
            near = nearest(key, known)
            hint = f" (did you mean {near!r}?)" if near else ""
            errors.append(f"unknown parameter {key!r}{hint}")
    bound = {}
    for p in params:
        if p.name not in payload:
            if p.required:
                errors.append(f"missing parameter {p.name!r} ({_summary(p)})")
            continue
        try:
            bound[p.name] = _check(p.type, payload[p.name])
        except _Mismatch as exc:
            errors.append(f"parameter {p.name!r} {exc}")
    if errors:
        raise ParameterError(f"{where}: {'; '.join(errors)}. It takes {_listing(params)}.")
    return bound


def _summary(p: Param) -> str:
    return f"{p.type.name}, {p.type.unit}" if p.type.unit else p.type.name


def _listing(params: tuple[Param, ...]) -> str:
    if not params:
        return "no parameters"
    return ", ".join(
        f"{p.name}: {_summary(p)}" + ("" if p.required else f" = {p.default!r}") for p in params
    )


def bind(params: tuple[Param, ...], payload: Any, where: str = "endpoint") -> dict:
    """Check *payload* against *params*; the keyword arguments to call the method with.

    Raises :class:`ParameterError` naming every missing, unknown or mistyped parameter at once.
    """
    return _bind(params, payload, where)


# -- declaring -------------------------------------------------------------------------------------
@dataclass(frozen=True)
class EndpointSpec:
    """One decorated endpoint, as declared on the class."""

    kind: str
    name: str
    attr: str
    rate_hz: float | Callable[..., float] = 0.0
    lazy: bool = False
    when: Callable[..., bool] | None = None
    each: Callable[[Any], Any] | None = None
    owner: str | Callable[..., str] | None = None
    namespace: str | Callable[..., str] | None = None
    backend: dict[str, dict | Callable[..., dict | None]] = field(default_factory=dict)

    @property
    def direction(self) -> str:
        return DIRECTIONS[self.kind]

    def signature(self, cls: type) -> tuple[tuple[Param, ...], ValueType | None]:
        """``(params, result)`` read off the method's signature on *cls*."""
        return signature(getattr(cls, self.attr), self.kind, self.each is not None)

    def endpoint_name(self, item: Any = None) -> str:
        if self.each is None:
            return self.name
        if "{item}" in self.name:
            return self.name.replace("{item}", str(item))
        return f"{self.name}/{item}"

    def describe(self, cls: type) -> dict:
        """What can be said of it without an instance: its schema and static values, never a
        hint's contents."""
        doc = (getattr(cls, self.attr).__doc__ or "").strip().splitlines()
        params, result = self.signature(cls)
        row = {
            "name": self.endpoint_name("{item}") if self.each is not None else self.name,
            "kind": self.kind,
            "direction": self.direction,
            "backends": sorted(self.backend),
            "conditional": self.when is not None,
            "doc": doc[0] if doc else None,
        }
        if self.each is not None:
            row["family"] = True
        if self.kind != "out":
            row["params"] = [p.describe() for p in params]
        if self.kind != "stream":
            row["result"] = result.describe() if result is not None else None
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


_Owner = str | Callable[..., str] | None


def out(
    name: str | None = None,
    *,
    rate_hz: float | Callable[..., float] = 0.0,
    lazy: bool = False,
    when: Callable[..., bool] | None = None,
    each: Callable[[Any], Any] | None = None,
    owner: _Owner = None,
    namespace: _Owner = None,
    **backend,
) -> Callable:
    """Declare the decorated method as an ``out`` endpoint; see the module docstring."""
    return _decorator(
        "out",
        name,
        rate_hz=rate_hz,
        lazy=lazy,
        when=when,
        each=each,
        owner=owner,
        namespace=namespace,
        backend=backend,
    )


def command(
    name: str | None = None,
    *,
    when: Callable[..., bool] | None = None,
    each: Callable[[Any], Any] | None = None,
    owner: _Owner = None,
    namespace: _Owner = None,
    **backend,
) -> Callable:
    """Declare the decorated method as a command (``in``, with an outcome).

    The method's parameters are the command's; what it returns is the future's result.
    """
    return _decorator(
        "command", name, when=when, each=each, owner=owner, namespace=namespace, backend=backend
    )


def stream(
    name: str | None = None,
    *,
    when: Callable[..., bool] | None = None,
    each: Callable[[Any], Any] | None = None,
    owner: _Owner = None,
    namespace: _Owner = None,
    **backend,
) -> Callable:
    """Declare the decorated method as an inbound stream (``in``, latest value wins per step).

    The method's parameters are the stream's; it is called with the newest ones once per step.
    """
    return _decorator(
        "stream", name, when=when, each=each, owner=owner, namespace=namespace, backend=backend
    )


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


# -- building ----------------------------------------------------------------------------------------
def _value(value, *args):
    return value(*args) if callable(value) else value


def build(plugin: Plugin, ctx: SimContext) -> list[Endpoint]:
    """The :class:`~roqsim.context.Endpoint`\\ s *plugin* declares, bound to it and to *ctx*."""
    endpoints = []
    specs = declared(type(plugin))
    if not specs:
        return endpoints
    default_owner = plugin.endpoint_owner
    default_namespace = plugin.endpoint_namespace(ctx)
    for spec in specs:
        params, result = spec.signature(type(plugin))
        method = getattr(plugin, spec.attr)
        if spec.each is None:
            instances = [((plugin,), method)]
        else:
            instances = [
                ((plugin, item), functools.partial(method, item)) for item in spec.each(plugin)
            ]
        for args, call in instances:
            if spec.when is not None and not spec.when(*args):
                continue
            name = spec.endpoint_name(args[1] if len(args) > 1 else None)
            owner = default_owner if spec.owner is None else _value(spec.owner, *args)
            if spec.namespace is not None:
                namespace = _value(spec.namespace, *args)
            elif spec.owner is not None:
                namespace = plugin.endpoint_namespace(ctx, owner)
            else:
                namespace = default_namespace
            backend = {}
            for key, hints in spec.backend.items():
                resolved = _value(hints, *args)
                if resolved is not None:
                    backend[key] = dict(resolved)
            ep = Endpoint(
                name=name,
                direction=spec.direction,
                owner=owner,
                namespace=namespace,
                backend=backend,
                result=result,
            )
            where = f"{owner}/{name}"
            if spec.kind == "out":
                ep.read = call
                ep.rate_hz = float(_value(spec.rate_hz, *args))
                ep.lazy = spec.lazy
            elif spec.kind == "command":
                ep.params = params
                ep.write = _submitter(ctx, call, params, where)
                ep.marshalled = True
            else:
                ep.params = params
                slot = ctx.stream_slot(where, lambda kwargs, _call=call: _call(**kwargs))
                ep.write = _streamer(slot, params, where)
                ep.marshalled = True
            endpoints.append(ep)
    return endpoints


def _submitter(ctx: SimContext, method: Callable, params: tuple[Param, ...], where: str):
    def write(payload=None) -> CommandFuture:
        try:
            kwargs = _bind(params, payload, where)
        except ParameterError as exc:
            # Logged here as well as handed back: a transport that fires and forgets (a topic
            # feeding a command) never reads the future, and the refusal must not vanish.
            ctx.logger.warning("%s", exc)
            refused = CommandFuture()
            refused._resolve(error=exc)
            return refused
        return ctx.submit(lambda _ctx: method(**kwargs))

    return write


def _streamer(slot, params: tuple[Param, ...], where: str):
    put = slot.put

    def write(payload=None) -> None:
        put(_bind(params, payload, where))

    return write
