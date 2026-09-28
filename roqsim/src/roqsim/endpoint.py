"""Endpoints declared on a plugin's methods, typed by the method's signature.

A plugin marks the methods that are its I/O ports; the engine registers them as
:class:`~roqsim.context.Endpoint`\\ s after the plugin's ``configure``::

    from roqsim import endpoint
    from roqsim.types import AngularSpeed, JointState, Odometry, Speed, Twist

    class DiffDrive(Plugin):
        @endpoint.stream(Twist)
        def cmd_vel(self, vx: Speed, vy: Speed = 0.0, wz: AngularSpeed = 0.0) -> None:
            \"\"\"Body-frame velocity command, applied once per step.

            Args:
                vx: forward speed
                vy: sideways speed; a differential drive drops it
                wz: yaw rate
            \"\"\"

        @endpoint.out(rate="odom_rate_hz", ros2={"emit_tf": True})
        def odom(self) -> Odometry:
            \"\"\"Wheel odometry.\"\"\"

        @endpoint.out(rate="odom_rate_hz", when="publish_joint_states")
        def joint_states(self) -> JointState:
            \"\"\"The wheels' positions and velocities.\"\"\"

        @endpoint.command
        def tare(self) -> None:
            \"\"\"Zero the sensor at its current load.\"\"\"

**The method is the endpoint**: its name is the endpoint's name, the first line of its docstring
the endpoint's documentation, and its signature the endpoint's schema. Three kinds, by what the
caller needs:

``out``
    The method takes no parameters and returns the payload, on the physics thread, whenever a
    bridge reads. Its return annotation is the payload's type (:attr:`Endpoint.result`).
``command``
    A request with an outcome. ``write`` is callable from any thread: it checks the named parameters
    against the signature, queues the method for the physics thread and returns a
    :class:`~roqsim.context.CommandFuture` holding what the method returned or raised.
``stream``
    An inbound stream. ``write`` checks the parameters the same way and keeps them in a latest-value
    slot; the physics thread calls the method with the newest ones once per step (while paused, in
    the driver's idle loop). Values superseded within a step are never applied.

**Payloads are dataclasses**: the neutral types of :mod:`roqsim.types`, or any dataclass of the
plugin's own. A ``command`` or ``stream`` names the type it takes as the decorator's first argument
-- ``@endpoint.stream(Twist)`` -- and its parameters are that type's fields it uses, by name; or it
takes one parameter annotated with the type, which receives the whole value. A bridge maps the type
to its transport (the ROS bridge: a message type, a topic named after the endpoint, converters both
ways), so a plugin writes no transport type.

**Parameters.** Annotated types (the unit aliases of :mod:`roqsim.types`, or ``Annotated[float,
Unit("m/s")]`` directly), defaults (a parameter without one is required), and a line of
documentation per parameter in the docstring's ``Args:`` section; a dataclass documents its fields
in an ``Attributes:`` section. A bridge passes parameters as a mapping of names to values (``None``
for none). ``write`` refuses a missing required parameter, an unknown one (naming the nearest known)
and a value of the wrong type with one :class:`ParameterError` naming all of them -- into the future
for a command, raised to the caller for a stream -- before anything is queued. An ``int`` passes for
a ``float``, a ``bool`` for neither, and a sequence for a ``numpy`` array of the declared dtype and
:class:`Shape`. The schema is data on the endpoint (:attr:`Endpoint.params`, :attr:`Endpoint.result`,
:attr:`Endpoint.payload_type`), for any bridge to read.

**Options**:

``name``
    The endpoint's name, where it is not the method's.
``rate``
    Publish rate of an ``out``, Hz: a number, or the name of the plugin attribute (else config key)
    holding it -- ``rate="odom_rate_hz"``.
``when``
    Whether this instance has the endpoint at all: the name of a boolean attribute or config key --
    ``when="publish_joint_states"``.
``lazy``
    Whether an ``out``'s read is skipped while nobody subscribes: ``True``, or the name of the
    plugin attribute (else config key) that says so per instance -- ``lazy="lazy"``.
``each``
    A family: one endpoint per item of the named attribute (or of what a callable returns), named
    ``<name>/<item>`` or ``name`` with ``{item}`` substituted; the method gets the item as its first
    argument, which is not an endpoint parameter::

        @endpoint.out(name="joints/{item}/effort", each="joint_names")
        def effort(self, joint: str) -> Torque: ...

``confirm``
    For a ``command``: the name of an ``out`` endpoint of the same plugin whose value, read in the
    ``post_step`` of the step that applied the command, confirms it.
``owner``, ``namespace``
    The entity an endpoint belongs to and its transport scope, where they are not the plugin's
    (:attr:`~roqsim.plugin.Plugin.endpoint_owner`, :meth:`~roqsim.plugin.Plugin.endpoint_namespace`).
``ros2=`` (any other keyword)
    A backend's hints, for what the type's default mapping does not cover: a frame id, a stamped
    variant, a QoS, a TF to emit. A dict, a callable of the plugin returning one, or ``None`` to keep
    the endpoint off that backend.

``rate``, ``when``, ``lazy``, ``each``, ``owner``, ``namespace`` and the hints also take a callable
of the plugin (``(plugin, item)`` in a family) for a value that has to be computed.

**A topic derived from a sibling's**: a hint's ``topic`` (or a service's ``name``) may name another
endpoint of the same plugin in braces. It is rendered when the plugin registers, from where that
endpoint is carried after the world's renames, so a world that renames the one moves the other with
it; ``..`` steps out of the sibling's last segment, as in a path::

    @endpoint.out(ros2={"type": "sensor_msgs.msg.CompressedImage", "topic": "{image}/compressed"})
    def image_compressed(self) -> Image: ...

    @endpoint.out(ros2={"topic": "{depth}/../camera_info"})
    def depth_camera_info(self) -> CameraInfo: ...

**The world renames and tunes**: a plugin's ``topics:`` config renames an endpoint on every
transport (:attr:`Endpoint.topic`) and its ``qos:`` config sets the endpoint's quality of service
(:attr:`Endpoint.qos`) -- a preset name of :data:`QOS_PRESETS` or a mapping of ``reliability``,
``durability``, ``history`` and ``depth``.

:func:`declared` lists a class's endpoints, with their schema, without building a world.
"""

from __future__ import annotations

import dataclasses
import functools
import inspect
import re
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

#: Entry-point group of the transports that describe how they carry an endpoint: each entry loads a
#: callable taking an :class:`~roqsim.context.Endpoint` and returning a JSON-ready dict.
TRANSPORTS_GROUP = "roqsim.transports"

_ATTR = "__roqsim_endpoints__"


# -- the markers a signature is annotated with ---------------------------------------------------
class Unit:
    """The unit of a parameter or a value: ``Annotated[float, Unit("m/s")]``.

    A plain string, spelled as :class:`roqsim.schema.Field`'s ``unit`` is, so a plugin's config and
    its endpoints state units in one vocabulary. :mod:`roqsim.types` has aliases for the common ones.
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


# -- quality of service ----------------------------------------------------------------------------
#: The named QoS profiles an endpoint's hint or a world's ``qos:`` may give. ``default`` is what an
#: endpoint gets when nothing names one.
QOS_PRESETS: dict[str, dict[str, Any]] = {
    "default": {
        "reliability": "reliable",
        "durability": "volatile",
        "history": "keep_last",
        "depth": 10,
    },
    "sensor_data": {
        "reliability": "best_effort",
        "durability": "volatile",
        "history": "keep_last",
        "depth": 5,
    },
    "services_default": {
        "reliability": "reliable",
        "durability": "volatile",
        "history": "keep_last",
        "depth": 10,
    },
    "latched": {
        "reliability": "reliable",
        "durability": "transient_local",
        "history": "keep_last",
        "depth": 1,
    },
}

_QOS_CHOICES = {
    "reliability": ("reliable", "best_effort"),
    "durability": ("volatile", "transient_local"),
    "history": ("keep_last", "keep_all"),
}


def qos_profile(spec: str | Mapping[str, Any]) -> dict[str, Any]:
    """The full profile *spec* names: a preset of :data:`QOS_PRESETS`, or a mapping of some of
    ``reliability``, ``durability``, ``history`` and ``depth`` over ``default``.

    Raises :class:`ValueError` naming what is wrong.
    """
    if isinstance(spec, str):
        if spec not in QOS_PRESETS:
            near = nearest(spec, QOS_PRESETS)
            hint = f" (did you mean {near!r}?)" if near else ""
            raise ValueError(
                f"unknown QoS preset {spec!r}{hint}; the presets are {', '.join(QOS_PRESETS)}"
            )
        return dict(QOS_PRESETS[spec])
    if not isinstance(spec, Mapping):
        raise ValueError(
            f"a QoS is a preset name ({', '.join(QOS_PRESETS)}) or a mapping of reliability, "
            f"durability, history and depth, got {type(spec).__name__}"
        )
    profile = dict(QOS_PRESETS["default"])
    errors = []
    for key, value in spec.items():
        if key in _QOS_CHOICES:
            if value not in _QOS_CHOICES[key]:
                errors.append(f"{key} must be one of {', '.join(_QOS_CHOICES[key])}, got {value!r}")
        elif key == "depth":
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                errors.append(f"depth must be an integer >= 1, got {value!r}")
        else:
            near = nearest(str(key), [*_QOS_CHOICES, "depth"])
            errors.append(
                f"unknown QoS key {key!r}" + (f" (did you mean {near!r}?)" if near else "")
            )
        profile[key] = value
    if errors:
        raise ValueError("; ".join(errors))
    return profile


def validate_qos_config(config: Mapping[str, Any]) -> list[str]:
    """Errors of a plugin's ``qos:`` config: a mapping of endpoint name to a QoS."""
    qos = config.get("qos")
    if qos is None:
        return []
    if not isinstance(qos, Mapping):
        return ["'qos' must be a mapping of endpoint name -> QoS preset or profile"]
    errors = []
    for name, spec in qos.items():
        try:
            qos_profile(spec)
        except ValueError as exc:
            errors.append(f"qos[{name!r}]: {exc}")
    return errors


# -- the schema, as data ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ValueType:
    """The type of one parameter, one result, or one field of either.

    ``kind`` is what a bridge dispatches on: ``float``, ``int``, ``bool``, ``str``, ``array`` (numpy,
    with ``dtype``/``shape`` where declared), ``list`` (``items`` has the one element type),
    ``tuple`` (``items`` has one type per position), ``dict``, ``struct`` (a dataclass; ``fields``,
    and ``cls`` the class), ``none``, ``any`` (unannotated, or not describable further; ``name``
    still says what it is).
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


def value_type(hint: Any, *, where: str = "", doc: str = "") -> ValueType:
    """The :class:`ValueType` an annotation declares, with its :class:`Unit` and :class:`Shape`.

    Documentation comes from a docstring (``doc``), never from the annotation: a plain string in
    ``Annotated`` is refused, so a parameter is documented in one place.
    """
    unit, shape = "", None
    if typing.get_origin(hint) is typing.Annotated:
        hint, *meta = typing.get_args(hint)
        for m in meta:
            if isinstance(m, Unit):
                unit = m.symbol
            elif isinstance(m, Shape):
                shape = m.dims
            elif isinstance(m, str):
                raise TypeError(
                    f"{where}: {m!r} documents a value inside its annotation; document a parameter "
                    f"in the docstring's Args: section and a dataclass field in Attributes:"
                )
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
    if origin is CommandFuture or hint is CommandFuture:
        # A command that answers later: its result is what the future resolves to.
        return value_type(args[0], where=where) if args else ValueType("any", "any")
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
        return ValueType(
            "struct", hint.__name__, doc=doc_summary(hint), fields=_dataclass_fields(hint), cls=hint
        )
    if isinstance(hint, type):
        return ValueType("any", hint.__name__, cls=hint)
    return ValueType("any", _name(hint))


def _name(hint: Any) -> str:
    return getattr(hint, "__name__", None) or str(hint).replace("typing.", "")


# -- docstrings --------------------------------------------------------------------------------------
_SECTION = re.compile(r"^(\s*)(Args|Attributes):\s*$")
_ENTRY = re.compile(r"^(\s*)(\w+)(?:\s*\([^)]*\))?\s*:\s*(.*)$")


def doc_summary(obj: Any) -> str:
    """The first line of *obj*'s own docstring, or ``""``."""
    doc = inspect.cleandoc(obj.__doc__ or "") if getattr(obj, "__doc__", None) else ""
    return doc.splitlines()[0].strip() if doc else ""


def doc_prose(obj: Any) -> str:
    """*obj*'s docstring without its ``Args:`` and ``Attributes:`` sections, which the schema
    carries entry by entry."""
    lines = inspect.cleandoc(obj.__doc__ or "").splitlines()
    kept, base = [], None
    for line in lines:
        if base is not None:
            if not line.strip() or len(line) - len(line.lstrip()) > base:
                continue
            base = None
        head = _SECTION.match(line)
        if head is not None:
            base = len(head.group(1))
            continue
        kept.append(line)
    return "\n".join(kept).strip()


@functools.cache
def doc_entries(obj: Any, section: str) -> dict[str, str]:
    """The ``name: text`` entries of *obj*'s docstring section (``Args`` or ``Attributes``).

    Google style: the section header on its own line, one entry per name indented below it, and a
    continuation indented further.
    """
    doc = inspect.cleandoc(obj.__doc__ or "")
    entries: dict[str, str] = {}
    lines = doc.splitlines()
    i = 0
    while i < len(lines):
        head = _SECTION.match(lines[i])
        i += 1
        if head is None or head.group(2) != section:
            continue
        base = len(head.group(1))
        entry_indent = None
        current = None
        while i < len(lines):
            line = lines[i]
            if not line.strip():
                i += 1
                continue
            indent = len(line) - len(line.lstrip())
            if indent <= base:
                break
            match = _ENTRY.match(line)
            if match and (entry_indent is None or indent == entry_indent):
                entry_indent = indent
                current = match.group(2)
                entries[current] = match.group(3).strip()
            elif current is not None:
                entries[current] = f"{entries[current]} {line.strip()}".strip()
            i += 1
    return entries


def _refuse_undeclared(documented: dict[str, str], names: Sequence[str], where: str, section: str):
    extra = [n for n in documented if n not in names]
    if extra:
        raise TypeError(
            f"{where}: the docstring's {section}: section documents {', '.join(map(repr, extra))}, "
            f"which {'is' if len(extra) == 1 else 'are'} not among {', '.join(names) or 'nothing'}"
        )


@functools.cache
def _dataclass_fields(cls: type) -> tuple[Param, ...]:
    where = cls.__qualname__
    hints = _hints(cls, where)
    docs = doc_entries(cls, "Attributes")
    names = [f.name for f in dataclasses.fields(cls)]
    _refuse_undeclared(docs, names, where, "Attributes")
    out = []
    for f in dataclasses.fields(cls):
        has_default = (
            f.default is not dataclasses.MISSING or f.default_factory is not dataclasses.MISSING
        )
        if f.default is not dataclasses.MISSING:
            default = f.default
        elif f.default_factory is not dataclasses.MISSING:
            default = f.default_factory()
        else:
            default = None
        out.append(
            Param(
                f.name,
                value_type(
                    hints.get(f.name, inspect.Parameter.empty),
                    where=f"{where}.{f.name}",
                    doc=docs.get(f.name, ""),
                ),
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


# -- a signature, read -------------------------------------------------------------------------------
@dataclass(frozen=True)
class Signature:
    """What a decorated method declares: its parameters, its result, and the payload it carries."""

    params: tuple[Param, ...]
    result: ValueType | None
    #: The type a transport carries: an ``out``'s result, or the type a ``command``/``stream`` takes
    #: (its decorator's, or its one dataclass parameter's). ``None`` for an inbound endpoint that
    #: takes plain parameters of no declared type.
    payload: ValueType | None


@functools.cache
def signature(fn: Callable, kind: str, family: bool, msg: type | None = None) -> Signature:
    """The endpoint's schema, read off a decorated method.

    The first parameter is ``self``, and in a family the second is the item. An ``out`` takes no
    other parameter; a ``stream`` has no result. ``*args``, ``**kwargs`` and positional-only
    parameters are refused: a bridge names every parameter it passes. With *msg*, each parameter is
    one of its fields, of the same kind and unit.
    """
    where = fn.__qualname__
    hints = _hints(fn, where)
    docs = doc_entries(fn, "Args")
    sig = inspect.signature(fn)
    params = list(sig.parameters.values())[2 if family else 1 :]
    _refuse_undeclared(docs, [p.name for p in params], where, "Args")
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
                value_type(
                    hints.get(p.name, inspect.Parameter.empty),
                    where=f"{where}({p.name})",
                    doc=docs.get(p.name, ""),
                ),
                required=p.default is inspect.Parameter.empty,
                default=None if p.default is inspect.Parameter.empty else p.default,
            )
        )
    params_t = tuple(out)
    if kind == "out" and params_t:
        raise TypeError(
            f"{where}: an out endpoint takes no parameters, but declares "
            f"{', '.join(p.name for p in params_t)}"
        )
    result = None
    if kind != "stream" and sig.return_annotation is not inspect.Signature.empty:
        result = value_type(hints.get("return"), where=f"{where} -> return")
    if kind == "out":
        return Signature(params_t, result, result)
    return Signature(params_t, result, _inbound_payload(params_t, msg, where))


def _inbound_payload(params: tuple[Param, ...], msg: type | None, where: str) -> ValueType | None:
    if msg is None:
        if len(params) == 1 and params[0].type.kind == "struct":
            return params[0].type
        return None
    if not (isinstance(msg, type) and dataclasses.is_dataclass(msg)):
        raise TypeError(f"{where}: an endpoint's payload type is a dataclass, got {msg!r}")
    carried = value_type(msg, where=where)
    if len(params) == 1 and params[0].type.cls is msg:
        return carried
    fields = {f.name: f for f in carried.fields}
    errors = []
    for p in params:
        f = fields.get(p.name)
        if f is None:
            near = nearest(p.name, fields)
            errors.append(
                f"{p.name!r} is not a field of {msg.__name__}"
                + (f" (did you mean {near!r}?)" if near else "")
            )
        elif f.type.kind != p.type.kind and not (f.type.kind == "float" and p.type.kind == "int"):
            errors.append(
                f"{p.name!r} is {p.type.name} but {msg.__name__}.{p.name} is {f.type.name}"
            )
        elif f.type.unit and p.type.unit and f.type.unit != p.type.unit:
            errors.append(
                f"{p.name!r} is in {p.type.unit} but {msg.__name__}.{p.name} in {f.type.unit}"
            )
    if errors:
        raise TypeError(
            f"{where}: the parameters of an endpoint taking {msg.__name__} are its fields, by name "
            f"({', '.join(fields)}): {'; '.join(errors)}"
        )
    return carried


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
_Option = Any  # a value, an attribute/config key name, or a callable of the plugin


@dataclass(frozen=True)
class EndpointSpec:
    """One decorated endpoint, as declared on the class."""

    kind: str
    name: str
    attr: str
    msg: type | None = None
    rate: _Option = 0.0
    lazy: _Option = False
    when: _Option = None
    each: _Option = None
    owner: str | Callable[..., str] | None = None
    namespace: str | Callable[..., str] | None = None
    backend: dict[str, Any] = field(default_factory=dict)
    confirm: str = ""

    @property
    def direction(self) -> str:
        return DIRECTIONS[self.kind]

    def options(self, cls: type) -> dict[str, Any]:
        """Where the ``rate``, ``lazy``, ``when`` and ``each`` options of the plugin class *cls*
        come from, for a reader of a built endpoint (:attr:`~roqsim.context.Endpoint.options`):
        ``rate`` and ``lazy`` as :meth:`describe` gives them, the others the key named, else
        ``"computed"``."""
        out: dict[str, Any] = {}
        if self.kind == "out":
            if not isinstance(self.rate, (int, float)):
                out["rate"] = _describe_option(cls, self.rate, float)
            if self.lazy is not False:
                out["lazy"] = _describe_option(cls, self.lazy, bool)
        if self.when is not None:
            out["when"] = _option_name(self.when)
        if self.each is not None:
            out["family"] = _option_name(self.each)
        return out

    def signature(self, cls: type) -> Signature:
        """The schema, read off the method on *cls*."""
        return signature(getattr(cls, self.attr), self.kind, self.each is not None, self.msg)

    def endpoint_name(self, item: Any = None) -> str:
        if self.each is None:
            return self.name
        if "{item}" in self.name:
            return self.name.replace("{item}", str(item))
        return f"{self.name}/{item}"

    def prototype(self, cls: type) -> Endpoint:
        """An unbound :class:`Endpoint` of what the class alone says: its schema and static hints.

        A hint computed at configure time is not evaluated; the backend's entry says so.
        """
        sig = self.signature(cls)
        backend = {}
        for key, hints in self.backend.items():
            if hints is None or isinstance(hints, Mapping):
                backend[key] = None if hints is None else dict(hints)
            else:
                backend[key] = {}
        return Endpoint(
            name=self.endpoint_name("{item}") if self.each is not None else self.name,
            direction=self.direction,
            owner="",
            backend=backend,
            params=None if self.kind == "out" else sig.params,
            result=sig.result,
            payload_type=sig.payload,
            transport=True,
            rate_hz=float(self.rate) if isinstance(self.rate, (int, float)) else 0.0,
            lazy=self.lazy if isinstance(self.lazy, bool) else False,
        )

    def describe(self, cls: type) -> dict:
        """What can be said of it without an instance: its schema, its options, and how each
        installed transport carries it."""
        sig = self.signature(cls)
        row: dict[str, Any] = {
            "name": self.endpoint_name("{item}") if self.each is not None else self.name,
            "kind": self.kind,
            "direction": self.direction,
            "backends": sorted(self.backend),
            "conditional": self.when is not None,
            "doc": doc_summary(getattr(cls, self.attr)) or None,
        }
        if self.each is not None:
            row["family"] = _option_name(self.each)
        if self.when is not None:
            row["when"] = _option_name(self.when)
        if self.confirm:
            row["confirm"] = self.confirm
        if self.kind != "out":
            row["params"] = [p.describe() for p in sig.params]
        if self.kind != "stream":
            row["result"] = sig.result.describe() if sig.result is not None else None
        if sig.payload is not None:
            row["payload"] = sig.payload.name
        if self.kind == "out":
            row["rate_hz"] = _describe_option(cls, self.rate, float)
            if self.lazy is not False:
                row["lazy"] = _describe_option(cls, self.lazy, bool)
        computed = sorted(k for k, v in self.backend.items() if v is not None and callable(v))
        if computed:
            row["hints_at_configure"] = computed
        transports = describe_transports(self.prototype(cls))
        if transports:
            row["transports"] = transports
        return row


def _option_name(option: _Option) -> str:
    return option if isinstance(option, str) else "computed"


def _describe_option(cls: type, value: _Option, literal: type):
    """A value option as describe shows it: the value, the key it is read from (with the config
    default), or ``"computed"``."""
    if isinstance(value, (bool, int, float)):
        return literal(value)
    if isinstance(value, str):
        field_spec = (getattr(cls, "CONFIG_SCHEMA", None) or {}).get(value)
        row: dict[str, Any] = {"from": value}
        if field_spec is not None and field_spec.default is not None:
            row["default"] = field_spec.default
        return row
    return "computed"


def describe_transports(ep: Endpoint) -> dict:
    """How each installed transport (:data:`TRANSPORTS_GROUP`) carries *ep*."""
    from .registry import _entry_points

    out = {}
    for entry in _entry_points(TRANSPORTS_GROUP):
        try:
            out[entry.name] = entry.load()(ep)
        except Exception as exc:  # noqa: BLE001 - a transport that cannot say is reported, not fatal
            out[entry.name] = {"error": f"{type(exc).__name__}: {exc}"}
    return out


def _decorator(kind: str, fn: Callable | None, **fields) -> Callable:
    def mark(method):
        specs = list(getattr(method, _ATTR, ()))
        specs.append((kind, fields))
        setattr(method, _ATTR, tuple(specs))
        return method

    return mark(fn) if fn is not None else mark


def _split(first: Any, kind: str) -> tuple[Callable | None, type | None]:
    """``@endpoint.out`` bare (the method), or ``@endpoint.stream(Twist)`` (the payload type)."""
    if first is None:
        return None, None
    if inspect.isfunction(first):
        return first, None
    if kind == "out":
        raise TypeError(
            f"endpoint.out takes no positional argument but the method it decorates, got {first!r}; "
            f"the payload type is the method's return annotation, and a name is name=..."
        )
    return None, first


_Owner = str | Callable[..., str] | None


def out(
    fn: Callable | None = None,
    /,
    *,
    name: str | None = None,
    rate: _Option = 0.0,
    lazy: _Option = False,
    when: _Option = None,
    each: _Option = None,
    owner: _Owner = None,
    namespace: _Owner = None,
    **backend,
) -> Callable:
    """Declare the decorated method as an ``out`` endpoint; see the module docstring."""
    method, _ = _split(fn, "out")
    return _decorator(
        "out",
        method,
        name=name,
        rate=rate,
        lazy=lazy,
        when=when,
        each=each,
        owner=owner,
        namespace=namespace,
        backend=backend,
    )


def command(
    msg: type | Callable | None = None,
    /,
    *,
    name: str | None = None,
    when: _Option = None,
    each: _Option = None,
    owner: _Owner = None,
    namespace: _Owner = None,
    confirm: str = "",
    **backend,
) -> Callable:
    """Declare the decorated method as a command (``in``, with an outcome).

    *msg* is the payload type, when the parameters are its fields. What the method returns is the
    future's result. ``confirm`` names an ``out`` endpoint of the same plugin whose value, read in
    the ``post_step`` of the step that applied the command, confirms it.
    """
    method, payload = _split(msg, "command")
    return _decorator(
        "command",
        method,
        name=name,
        msg=payload,
        when=when,
        each=each,
        owner=owner,
        namespace=namespace,
        confirm=confirm,
        backend=backend,
    )


def stream(
    msg: type | Callable | None = None,
    /,
    *,
    name: str | None = None,
    when: _Option = None,
    each: _Option = None,
    owner: _Owner = None,
    namespace: _Owner = None,
    **backend,
) -> Callable:
    """Declare the decorated method as an inbound stream (``in``, latest value wins per step).

    *msg* is the payload type, when the parameters are its fields. The method is called with the
    newest parameters once per step.
    """
    method, payload = _split(msg, "stream")
    return _decorator(
        "stream",
        method,
        name=name,
        msg=payload,
        when=when,
        each=each,
        owner=owner,
        namespace=namespace,
        backend=backend,
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
            specs = []
            for kind, fields in marks:
                fields = dict(fields)
                specs.append(
                    EndpointSpec(kind=kind, name=fields.pop("name") or attr, attr=attr, **fields)
                )
            by_attr[attr] = specs
    return [spec for specs in by_attr.values() for spec in specs]


# -- building ----------------------------------------------------------------------------------------
def option(plugin: Plugin, value: _Option, *args):
    """An option's value for *plugin*: a callable's result, or the named attribute (else config key),
    or the value itself."""
    if callable(value):
        return value(*args)
    if isinstance(value, str):
        if hasattr(plugin, value):
            return getattr(plugin, value)
        if value in plugin.config:
            return plugin.config[value]
        raise TypeError(
            f"{type(plugin).__name__}: an endpoint option names {value!r}, which is neither an "
            f"attribute of the plugin nor a key of its config"
        )
    return value


def build(plugin: Plugin, ctx: SimContext) -> list[Endpoint]:
    """The :class:`~roqsim.context.Endpoint`\\ s *plugin* declares, bound to it and to *ctx*."""
    endpoints = []
    specs = declared(type(plugin))
    if not specs:
        return endpoints
    default_owner = plugin.endpoint_owner
    default_namespace = plugin.endpoint_namespace(ctx)
    for spec in specs:
        sig = spec.signature(type(plugin))
        if spec.attr in vars(plugin):
            raise TypeError(
                f"{type(plugin).__name__}.{spec.attr}: an instance attribute of that name hides the "
                f"endpoint method {spec.name!r}; rename the attribute"
            )
        options = spec.options(type(plugin))
        method = getattr(plugin, spec.attr)
        if spec.each is None:
            instances = [((plugin,), method)]
        else:
            items = option(plugin, spec.each, plugin)
            instances = [((plugin, item), functools.partial(method, item)) for item in items]
        for args, call in instances:
            if spec.when is not None and not option(plugin, spec.when, *args):
                continue
            name = spec.endpoint_name(args[1] if len(args) > 1 else None)
            owner = default_owner if spec.owner is None else _value(spec.owner, *args)
            if spec.namespace is not None:
                namespace = _value(spec.namespace, *args)
            elif spec.owner is not None:
                namespace = plugin.endpoint_namespace(ctx, owner)
            else:
                namespace = default_namespace
            backend: dict[str, dict | None] = {}
            for key, hints in spec.backend.items():
                resolved = _value(hints, *args)
                backend[key] = None if resolved is None else dict(resolved)
            ep = Endpoint(
                name=name,
                direction=spec.direction,
                owner=owner,
                namespace=namespace,
                backend=backend,
                result=sig.result,
                payload_type=sig.payload,
                topic=plugin.topic_override(name),
                transport=True,
                kind=spec.kind,
                producer=plugin.address,
                confirm=spec.confirm,
                doc=doc_prose(method),
                options=options,
            )
            where = f"{owner}/{name}"
            if spec.kind == "out":
                ep.read = call
                ep.rate_hz = float(option(plugin, spec.rate, *args))
                ep.lazy = bool(option(plugin, spec.lazy, *args))
            elif spec.kind == "command":
                ep.params = sig.params
                ep.write = _submitter(ctx, call, sig.params, where)
                ep.marshalled = True
            else:
                ep.params = sig.params
                slot = ctx.stream_slot(where, lambda kwargs, _call=call: _call(**kwargs))
                ep.write = _streamer(slot, sig.params, where)
                ep.slot = slot
                ep.marshalled = True
            endpoints.append(ep)
    _derive_topics(endpoints, type(plugin).__name__)
    return endpoints


def _value(value, *args):
    return value(*args) if callable(value) else value


# -- where a transport carries it ------------------------------------------------------------------
_OFF = object()


def hints_for(ep: Endpoint, backend: str) -> dict | None:
    """A copy of *ep*'s hint block for *backend*: ``{}`` for a decorated endpoint that gave none, and
    ``None`` when the endpoint is not on that backend (a hint block of ``None``, or a hand-built
    endpoint without one)."""
    raw = ep.backend.get(backend, _OFF)
    if raw is None:
        return None
    if raw is _OFF:
        return {} if ep.transport else None
    return dict(raw)


def is_service(ep: Endpoint, hints: Mapping) -> bool:
    """Whether a transport binds *ep* as a request with a reply rather than a topic: its hints name a
    ``service`` or an ``action``, or it is a command without parameters and without a message
    ``type``."""
    if "service" in hints or "action" in hints:
        return True
    return ep.direction == "in" and ep.params == () and "type" not in hints


def topic_of(ep: Endpoint, backend: str) -> str | None:
    """The topic (a service's name, for a service) *backend* carries *ep* on, before its namespace.

    The world's ``topics:`` rename (:attr:`Endpoint.topic`), else the hint's ``topic`` (``name``
    for a service), else the endpoint's name. ``None`` when the endpoint is not on that backend, or
    is a ``static`` one, which has no topic of its own.
    """
    hints = hints_for(ep, backend)
    if hints is None or hints.get("static"):
        return None
    key = "name" if is_service(ep, hints) else "topic"
    return ep.topic or hints.get(key) or ep.name


_SIBLING = re.compile(r"\{([^{}]+)\}")


def _derive_topics(endpoints: Sequence[Endpoint], where: str) -> None:
    """Render every hint ``topic`` (or ``name``) that names a sibling endpoint in braces.

    ``{image}`` is the sibling's :func:`topic_of` on the same backend, after the world's renames;
    ``..`` then steps out of its last segment, as in a path -- ``{depth}/../camera_info`` is the
    ``camera_info`` beside the depth image. The sibling must be one of *endpoints* (the plugin's own)
    and in the same namespace, unless its topic is absolute.
    """
    by_name: dict[str, list[Endpoint]] = {}
    for ep in endpoints:
        by_name.setdefault(ep.name, []).append(ep)

    def render(ep: Endpoint, stack: tuple) -> None:
        for backend, hints in ep.backend.items():
            if not isinstance(hints, dict):
                continue
            for key in ("topic", "name"):
                value = hints.get(key)
                if isinstance(value, str) and "{" in value:
                    hints[key] = derive(ep, backend, key, value, stack + (ep,))

    def derive(ep: Endpoint, backend: str, key: str, value: str, stack: tuple) -> str:
        label = f"{where}: {ep.name}'s {backend} {key} {value!r}"

        def sibling_topic(match) -> str:
            name = match.group(1)
            found = by_name.get(name, [])
            if len(found) != 1:
                has = ", ".join(sorted(by_name)) or "none"
                how = "registers more than once" if found else "does not register"
                raise ValueError(f"{label} names {name!r}, which this plugin {how}; it has {has}")
            sibling = found[0]
            if sibling in stack:
                raise ValueError(f"{label} names {name!r}, whose topic is derived from it in turn")
            render(sibling, stack)
            topic = topic_of(sibling, backend)
            if topic is None:
                raise ValueError(f"{label} names {name!r}, which has no {backend} topic")
            if sibling.namespace != ep.namespace and not topic.startswith("/"):
                raise ValueError(
                    f"{label} names {name!r}, whose relative topic is in another namespace "
                    f"({sibling.namespace!r}, not {ep.namespace!r})"
                )
            return topic

        text = _SIBLING.sub(sibling_topic, value)
        absolute = text.startswith("/")
        parts: list[str] = []
        for part in text.split("/"):
            if part == "..":
                if not parts:
                    raise ValueError(f"{label} steps out of {text!r}'s first segment")
                parts.pop()
            elif part:
                parts.append(part)
        return ("/" if absolute else "") + "/".join(parts)

    for ep in endpoints:
        render(ep, ())


def apply_world_qos(plugin: Plugin, endpoints: Sequence[Endpoint]) -> None:
    """Set each endpoint's :attr:`~roqsim.context.Endpoint.qos` from *plugin*'s ``qos:`` config.

    *endpoints* are the ones the plugin registered. A key naming none of them is refused, naming the
    ones it has: a QoS that silently applied to nothing would leave the world believing it tuned a
    stream it did not.
    """
    wanted = plugin.config.get("qos") or {}
    if not wanted:
        return
    by_name: dict[str, list[Endpoint]] = {}
    for ep in endpoints:
        by_name.setdefault(ep.name, []).append(ep)
    unknown = [key for key in wanted if key not in by_name]
    if unknown:
        hints = [
            f"{k!r}" + (f" (did you mean {n!r}?)" if (n := nearest(k, by_name)) else "")
            for k in unknown
        ]
        raise ValueError(
            f"{plugin.label}: qos names {', '.join(hints)}, which it does not register; its "
            f"endpoints are {', '.join(sorted(by_name)) or 'none'}"
        )
    for key, spec in wanted.items():
        profile = qos_profile(spec)
        for ep in by_name[key]:
            ep.qos = dict(profile)


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

    def write(payload=None, source: str | None = None) -> None:
        # *source*: the transport writing, so two driving one stream are told apart.
        put(_bind(params, payload, where), source)

    return write
