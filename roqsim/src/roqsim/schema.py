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

Every plugin validates its own config -- that part is deliberate and stays (:meth:`roqsim.plugin.
Plugin.validate_config`). What was missing is a way to *say* what the config IS. The catalog
(``roqsim plugins describe``) reconstructs it by parsing the ``Config::`` block out of a docstring,
which yields a name, an example and a comment: no type, no range, no way to tell a required key from
one with a default. A caller writing a world -- a person, a campaign generator, an agent -- has to
read prose and guess, and finds out by running.

A plugin that declares :data:`Plugin.CONFIG_SCHEMA` gets both from one place: :func:`validate` turns
the declaration into the same error strings the hand-written checks produce, and the introspection
API publishes the fields with their types, defaults, units and bounds.

Opt-in, and additive. A plugin without a schema behaves exactly as before, and a plugin WITH one
still owns ``validate_config`` -- it calls :func:`validate` for the mechanical part and keeps
whatever else it knows (that two lists must be the same length, that a file must exist). The schema
is not a validation framework growing to cover every rule; it is the part that is the same
everywhere, written once instead of forty times.

Declaring it::

    CONFIG_SCHEMA = {
        "mass": Field(float, required=True, minimum=0.0, unit="kg", doc="added to the body's own"),
        "body": Field(str, default="", doc="body to load (default: the entity's root body)"),
        "mode": Field(str, default="soft", choices=("soft", "rigid")),
        "pos": Field(list, length=3, unit="m", doc="offset in the body frame"),
    }

What it checks: a required key is present, a value has the declared type (with ``int`` accepted for
``float``, since YAML writes ``1`` for a one-metre offset), a number is within ``minimum``/
``maximum``, a string is one of ``choices``, a sequence has ``length``, and -- for a plugin that asks
for it with ``strict_keys`` -- that no key is unknown, which is the typo check nothing else can do.

**Unknown keys are opt-in for one reason.** A component's config does not only come from the world:
a model's manifest injects ``prefix``, a spawn fills in the entity, and a fault block arrives from
elsewhere. Rejecting what a schema does not mention would break those the moment a plugin adopted a
schema, so the shared keys are known here (:data:`INJECTED_KEYS`) and a plugin opts in when it is
sure its own list is complete.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from typing import Any

#: Config keys a component may carry without its own schema mentioning them, because something other
#: than the world's author put them there: the spawn plugins' ``prefix``, the transport scope, the
#: topic hardwire map, and a sensor's runtime fault block. A plugin that declares one of these in its
#: own schema (with a type or a default) overrides the entry here.
INJECTED_KEYS = frozenset({"prefix", "namespace", "topics", "fault", "robot", "arm"})

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

    type: type
    default: Any = None
    required: bool = False
    minimum: float | None = None
    maximum: float | None = None
    choices: Sequence[Any] | None = None
    length: int | None = None
    unit: str = ""
    doc: str = ""
    #: Keys whose value this plugin reads once at configure -- documented as such so a caller knows
    #: writing it later takes effect nowhere (the lesson `model_override` records about geom_size).
    static: bool = dataclass_field(default=False)

    def describe(self, name: str) -> dict:
        """The published form: JSON-friendly, and the same shape for every plugin."""
        described = {
            "name": name,
            "type": _TYPE_NAMES.get(self.type, getattr(self.type, "__name__", str(self.type))),
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
    if not _has_type(value, spec.type):
        expected = _TYPE_NAMES.get(spec.type, str(spec.type))
        errors.append(f"'{name}' must be {expected}, got {type(value).__name__} ({value!r})")
        return errors  # a wrong type makes every other check meaningless

    if spec.length is not None and len(value) != spec.length:
        errors.append(f"'{name}' must have exactly {spec.length} entries, got {len(value)}")
    if spec.choices is not None and value not in spec.choices:
        errors.append(f"'{name}' must be one of {', '.join(map(str, spec.choices))}, got {value!r}")
    if spec.minimum is not None and value < spec.minimum:
        errors.append(f"'{name}' must be >= {spec.minimum}{_unit(spec)}, got {value}")
    if spec.maximum is not None and value > spec.maximum:
        errors.append(f"'{name}' must be <= {spec.maximum}{_unit(spec)}, got {value}")
    return errors


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
