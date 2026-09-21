# Copyright (C) 2026 Frederik Pasch
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""A plugin's config, declared once: checked at load, and readable by a machine.

Every plugin validates its own config (:meth:`roqsim.plugin.Plugin.validate_config`). A schema is
how a plugin also *says* what that config is. Without one the catalog (``roqsim plugins describe``)
has only the ``Config::`` block parsed out of a docstring: a name, an example and a comment, with no
type, no range, and no way to tell a required key from one with a default -- so a caller writing a
world, whether a person, a campaign generator or an agent, reads prose, guesses, and finds out by
running.

A plugin that declares :data:`Plugin.CONFIG_SCHEMA` gets both from one place: :func:`validate` turns
the declaration into the same error strings the hand-written checks produce, and the introspection
API publishes the fields with their types, defaults, units and bounds -- as ``schema``, and as the
``parameters`` and docs block that a plugin without one parses from its docstring. The plugin reads
its config through it too (:class:`Settings`, ``self.settings``), so a default is written once.

Opt-in. A plugin without a schema is unchecked by it, and a plugin with one still owns
``validate_config`` for whatever else it knows (that two lists must be the same length, that a file
must exist). The schema is not a validation framework covering every rule; it is the part that is
the same everywhere, written once rather than in every plugin.

**Declaring a schema is what enforces it**: ``instantiate_plugins`` checks it, so there is no call
for a plugin author to remember and no way to publish a contract through the catalog that nothing
verifies -- which would leave the schema exactly as trustworthy as the docstring it replaces.

Declaring it::

    CONFIG_SCHEMA = {
        "mass": Field(float, required=True, minimum=0.0, unit="kg", doc="added to the body's own"),
        "body": Field(str, default="", doc="body to load (default: the entity's root body)"),
        "mode": Field(str, default="soft", choices=("soft", "rigid")),
        "pos": Field(list, length=3, unit="m", doc="offset in the body frame"),
        "gain": Field((float, dict), default=0.0, minimum=0.0, doc="one value, or one per joint"),
    }

What it checks: a required key is present, a value has the declared type (with ``int`` accepted for
``float``, since YAML writes ``1`` for a one-metre offset), a number is within ``minimum``/
``maximum``, a string is one of ``choices``, a sequence has ``length``, and -- with ``strict_keys``,
which every plugin's schema is checked with unless it says why not -- that no key is unknown, which
is the typo check nothing else can do.

A key that takes one of several shapes declares a tuple of types, as ``isinstance`` does: ``gain``
above is a number or a mapping. The value must be one of them, and each rule applies to the shapes it
has a meaning for -- a bound to a number, a length to a sequence -- so a mapping's entries are the
plugin's to check in ``validate_config``.

**An unknown key is refused by default.** A plugin that declares a schema says what its config is,
and a key outside it is a typo that would otherwise leave a setting at its default and look
configured. What a component carries without the world's author writing it -- a manifest's
``prefix``, the transport keys, a sensor's fault block, ``present`` -- is known here
(:data:`INJECTED_KEYS`), so a complete schema need not list it. A plugin whose schema cannot be
complete (one that passes keys through to something else) sets ``STRICT_KEYS = False`` and says why
in ``OPEN_KEYS``; the guard test over every shipped plugin refuses the first without the second.
"""

from __future__ import annotations

import copy
from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from typing import Any

#: Config keys a component may carry without its own schema mentioning them, because something other
#: than the plugin owns them: the spawn plugins' ``prefix``, the transport scope, the topic hardwire
#: map, a sensor's runtime fault block, and ``present``, which the base class reads and checks for
#: every plugin (:meth:`roqsim.plugin.Plugin.validate_presence` refuses it, with the reason, on one
#: that registers no entity). A plugin that declares one of these in its own schema (with a type or
#: a default) overrides the entry here.
INJECTED_KEYS = frozenset({"prefix", "namespace", "topics", "fault", "robot", "arm", "present"})

#: How a type is named in the published schema -- the vocabulary a caller matches on, not Python's.
_TYPE_NAMES = {bool: "bool", int: "int", float: "float", str: "str", list: "list", dict: "dict"}


@dataclass(frozen=True)
class Field:
    """One config key: what it holds, what it defaults to, and what it may not be.

    ``default`` is the value the plugin uses when the key is absent, and it is published so a reader
    does not have to find it in the code. ``required=True`` means there is no sensible default --
    the two are mutually exclusive, and a schema that sets both is refused when it is read rather
    than producing an error message no world can act on.
    """

    #: One type, or a tuple of them for a key that takes several shapes (``(float, dict)``).
    type: type | tuple[type, ...]
    default: Any = None
    required: bool = False
    minimum: float | None = None
    maximum: float | None = None
    choices: Sequence[Any] | None = None
    length: int | None = None
    unit: str = ""
    doc: str = ""
    #: Keys whose value this plugin reads once at configure -- documented as such so a caller knows
    #: writing it later takes effect nowhere (what `model_override` documents about geom_size).
    static: bool = dataclass_field(default=False)

    def describe(self, name: str) -> dict:
        """The published form: JSON-friendly, and the same shape for every plugin."""
        # A union publishes a list of names, as JSON Schema writes one, so a caller matching on a
        # single name never mistakes "float or dict" for a float.
        names = [_type_name(t) for t in _types(self.type)]
        described = {
            "name": name,
            "type": names[0] if len(names) == 1 else names,
            "required": self.required,
        }
        if not self.required:
            described["default"] = self.default
        for key in ("minimum", "maximum", "length"):
            value = getattr(self, key)
            if value is not None:
                described[key] = value
        if self.choices is not None:
            described["choices"] = list(self.choices)
        if self.unit:
            described["unit"] = self.unit
        if self.doc:
            described["doc"] = self.doc
        if self.static:
            described["static"] = True
        return described


class Settings:
    """A plugin's config, read through its schema: ``plugin.settings.rate_hz``.

    What :attr:`roqsim.plugin.Plugin.settings` returns for a plugin that declares a schema. Three
    things a ``config.get("rate_hz", 5.0)`` does not do:

    * **The default is the schema's.** A key the world left out reads as its declared default, so
      the default is written once -- where ``describe`` publishes it -- and cannot differ between
      ``__init__``, ``validate_config`` and the catalog. A mutable default is copied per read.
    * **An undeclared name is an AttributeError**, naming what is declared. ``config.get`` of a
      misspelt key returns ``None`` and the plugin runs on it.
    * **It is read-only.** Assigning to it raises; the config a world stated is the record of the
      run, and a plugin that wants a derived value keeps it on itself.

    An ``int`` given for a ``float`` key reads as a ``float``, the one coercion the schema already
    accepts from YAML. Nothing else is converted: a value of the wrong type reads as given, and
    the schema check reports it -- a view that fell back to the default would run the plugin on a
    value nobody stated.

    Only declared keys are here. The keys another owner injects (:data:`INJECTED_KEYS`) stay where
    their owner reads them, on ``self.config``.
    """

    __slots__ = ("_config", "_owner", "_schema")

    def __init__(self, schema: dict[str, Field], config: dict, owner: str = "this plugin"):
        object.__setattr__(self, "_schema", schema)
        object.__setattr__(self, "_config", config)
        object.__setattr__(self, "_owner", owner)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            # Private and dunder lookups (copy, pickle) are not settings; answering them from the
            # schema would recurse on a view whose slots are not filled yet.
            raise AttributeError(name)
        spec = self._schema.get(name)
        if spec is None:
            raise AttributeError(
                f"{self._owner} declares no setting {name!r}. Declared: {', '.join(self._schema)}"
            )
        if name not in self._config:
            return copy.deepcopy(spec.default)
        value = self._config[name]
        if float in _types(spec.type) and isinstance(value, int) and not isinstance(value, bool):
            return float(value)
        return value

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError(f"{self._owner}'s settings are read-only; {name!r} was not assigned")

    def __delattr__(self, name: str) -> None:
        raise AttributeError(f"{self._owner}'s settings are read-only; {name!r} was not deleted")

    def __dir__(self) -> list[str]:
        return list(self._schema)

    def __repr__(self) -> str:
        values = ", ".join(f"{name}={getattr(self, name)!r}" for name in self._schema)
        return f"Settings({values})"


def describe(schema: dict[str, Field]) -> list[dict]:
    """A whole schema as a list of published fields, in declaration order."""
    return [spec.describe(name) for name, spec in schema.items()]


def validate(schema: dict[str, Field], config: dict, *, strict_keys: bool = False) -> list[str]:
    """Config errors for *config* against *schema*, in the same voice as a hand-written check.

    Every error names the key, because a message that does not is a message a caller has to bisect a
    world file to act on. Errors accumulate rather than raising at the first: a world with three
    mistakes should take one run to find them, which is the same reason ``instantiate_plugins``
    aggregates across plugins.
    """
    errors: list[str] = []
    for name, spec in schema.items():
        if spec.required and spec.default is not None:
            errors.append(
                f"schema error: '{name}' is required AND has a default, which cannot both be true"
            )
        if name not in config:
            if spec.required:
                doc = f" -- {spec.doc}" if spec.doc else ""
                errors.append(f"'{name}' is required{doc}")
            continue
        errors += _check_value(name, spec, config[name])

    if strict_keys:
        known = set(schema) | INJECTED_KEYS
        for key in config:
            if key not in known:
                near = _nearest(key, schema)
                suggestion = f" -- did you mean '{near}'?" if near else ""
                errors.append(
                    f"'{key}' is not a setting of this component{suggestion}. Known: "
                    f"{', '.join(sorted(schema))}"
                )
    return errors


def _check_value(name: str, spec: Field, value: Any) -> list[str]:
    errors: list[str] = []
    wanted = _types(spec.type)
    if not any(_has_type(value, t) for t in wanted):
        expected = " or ".join(_type_name(t) for t in wanted)
        errors.append(f"'{name}' must be {expected}, got {type(value).__name__} ({value!r})")
        return errors  # a wrong type makes every other check meaningless

    # Each rule applies to the shapes it means something for: on a union, a mapping has no length
    # and no bound, and its entries are the plugin's own to check.
    number = _is_number(value)
    if spec.length is not None and isinstance(value, Sequence) and len(value) != spec.length:
        errors.append(f"'{name}' must have exactly {spec.length} entries, got {len(value)}")
    if spec.choices is not None and value not in spec.choices:
        errors.append(f"'{name}' must be one of {', '.join(map(str, spec.choices))}, got {value!r}")
    if number and spec.minimum is not None and value < spec.minimum:
        errors.append(f"'{name}' must be >= {spec.minimum}{_unit(spec)}, got {value}")
    if number and spec.maximum is not None and value > spec.maximum:
        errors.append(f"'{name}' must be <= {spec.maximum}{_unit(spec)}, got {value}")
    return errors


def _types(declared: type | tuple[type, ...]) -> tuple[type, ...]:
    return declared if isinstance(declared, tuple) else (declared,)


def _type_name(wanted: type) -> str:
    return _TYPE_NAMES.get(wanted, getattr(wanted, "__name__", str(wanted)))


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _unit(spec: Field) -> str:
    return f" {spec.unit}" if spec.unit else ""


def _has_type(value: Any, wanted: type) -> bool:
    """Type check with the two coercions YAML forces on us, and no others.

    A YAML ``1`` for a metre is an ``int`` and must satisfy a ``float`` field -- refusing it would
    make every world write ``1.0`` to please a checker. A ``bool`` must NOT satisfy ``int`` or
    ``float`` even though Python says it does: ``rate_hz: true`` is a mistake, and 1 Hz is not what
    it meant.
    """
    if isinstance(value, bool):
        return wanted is bool
    if wanted is float:
        return isinstance(value, (int, float))
    return isinstance(value, wanted)


def _nearest(key: str, schema: dict[str, Field]) -> str | None:
    """The closest declared key to a typo, or None when nothing is close.

    A typo suggestion is worth having only when it is nearly certain: 'radius' for 'radius_m' helps,
    while 'body' for 'mass' sends someone to the wrong line. The cutoff is deliberately tight.
    """
    from difflib import get_close_matches

    matches = get_close_matches(key, list(schema), n=1, cutoff=0.8)
    return matches[0] if matches else None
