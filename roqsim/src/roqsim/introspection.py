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

"""Programmatic, JSON-friendly introspection of the roqsim.plugins registry.

Two public entry points: a one-liner list, full detail on request.

* :func:`list_plugins` -- every registered ``roqsim.plugins`` entry, one line per plugin.
* :func:`get_plugin_details` -- one plugin's full detail, including its ``Config::``
  block parsed into structured fields.

And one check that the detail is complete: :func:`undeclared_config_reads`, the keys a plugin reads
that its published entry does not list.

The doc-extraction helpers here (:func:`_own_or_module_doc`, :func:`_summary_and_config`,
:func:`_flags`, :func:`_dist_name`) are the same ones the Sphinx ``.. roqsim-plugins::``
directive (``docs/_ext/plugin_docs.py``) uses to build its documentation page -- moved
here so there is exactly one place that extracts a plugin's docs, not two independently
reimplementing the same parsing.

Also runnable as a module, so it can be executed inside a runtime container image and
have its JSON output parsed by a caller on the host::

    python -m roqsim.introspection list
    python -m roqsim.introspection describe <name>
"""

from __future__ import annotations

import ast
import inspect
import json
import re
import sys
import textwrap

from roqsim.registry import ENTRY_POINT_GROUP, _entry_points

# Some plugins qualify the header ("Config (in addition to camera_common.CameraPlugin's)::",
# "Config (same keys as livox_mid360; only the defaults differ)::") rather than writing a bare
# "Config::", and a long qualifier wraps over two or three source lines. The "::" therefore does
# not have to sit on the same line as the word: the header is a line opening with "Config" plus
# however many lines it takes to reach "::".
_CONFIG_HEADER_START_RE = re.compile(r"\s*Config\b")
_CONFIG_HEADER_END_RE = re.compile(r".*::\s*$")
#: How far a wrapped header may run before it is read as prose rather than a header. Three is the
#: longest in the tree; the bound is what stops an ordinary sentence opening with the word from
#: swallowing the docstring behind it.
_MAX_HEADER_LINES = 4
# A Config:: field line: "  name: example_value  # trailing doc comment". The
# example is whatever text sits between the colon and an optional trailing
# comment -- kept as raw text (not parsed as YAML) since this is documentation,
# not a live config value.
_CONFIG_FIELD_RE = re.compile(r"^\s+([A-Za-z_]\w*):\s*(.+?)\s*(?:#\s*(.*))?$")
# A bare comment line, no leading "name:" -- a trailing doc comment too long for
# one line wraps onto a second line shaped exactly like this (see ceiling.py's
# "enabled" field for a real example), so it must extend the previous field's
# doc rather than end the block.
_COMMENT_ONLY_RE = re.compile(r"^\s*#\s?(.*)$")
# A key that opens a nested mapping rather than carrying a value: "  sample:" with the keys under
# it indented further. The world YAML nests, so the block does too, and ending the parse at a key
# like this would report a plugin's first few keys and silently drop the rest.
_CONFIG_NEST_RE = re.compile(r"^\s+([A-Za-z_]\w*):\s*(?:#\s*(.*))?$")
# The line naming the plugin itself, which a block opens with one level above its keys. Written
# either as a plain key ("sensor_coverage_probe:"), as the list entry a world YAML's
# "components:" actually takes ("- spawn_sensor:"), or -- in a base class documenting keys its
# subclasses inherit -- as a placeholder standing in for whichever name they register under
# ("<plugin short name>:"). All three are skipped rather than parsed, so the keys underneath are
# read at the level they are written.
_CONFIG_WRAPPER_RE = re.compile(r"^\s+-?\s*(?:[A-Za-z_]\w*|<[^>]+>):\s*(?:#\s*(.*))?$")
# A YAML sequence item ("- {id: 0, x0_m: 0.0}"), which a block writes under a key that takes a list
# to show one example entry. It is the value of that key, not a key of its own.
_SEQUENCE_ITEM_RE = re.compile(r"^\s+-(?:\s|$)")


def _config_header_span(lines: list[str]) -> tuple[int, int] | None:
    """``(first, last)`` line indices of the ``Config::`` header, or ``None``.

    Bounded by :data:`_MAX_HEADER_LINES` and never crossing a blank line or a field line, so a
    sentence that merely opens with the word cannot be read as a header.
    """
    for i, ln in enumerate(lines):
        if not _CONFIG_HEADER_START_RE.match(ln):
            continue
        for j in range(i, min(i + _MAX_HEADER_LINES, len(lines))):
            if not lines[j].strip() or (j > i and _CONFIG_FIELD_RE.match(lines[j])):
                break
            if _CONFIG_HEADER_END_RE.match(lines[j]):
                return i, j
    return None


def _own_or_module_doc(cls) -> str:
    """The plugin's *own* docstring, else its module's -- never the inherited base ``Plugin`` one.

    ``inspect.getdoc`` walks the MRO and would return ``Plugin``'s boilerplate for a plugin that has
    no docstring of its own; that base text is useless in a per-plugin catalog. Most plugins put
    their description (+ ``Config::``) at module level, so that is the fallback.
    """
    own_raw = cls.__dict__.get("__doc__")
    own = inspect.cleandoc(own_raw) if own_raw and own_raw.strip() else ""
    module = inspect.getmodule(cls)
    mod = inspect.cleandoc(module.__doc__) if module and module.__doc__ else ""
    # A class docstring that documents no config while its module does is a pointer to the module
    # ("See the module docstring."), and preferring it publishes the pointer and hides the block.
    # The catalog exists to say what a plugin's config keys are, so the docstring that has them wins.
    if (
        own
        and mod
        and _config_header_span(own.splitlines()) is None
        and _config_header_span(mod.splitlines()) is not None
    ):
        return mod
    return own or mod


def _summary_and_config(doc: str) -> list[str]:
    """A short description (first paragraph) plus the ``Config::`` block, dropping the middle prose."""
    lines = doc.splitlines()
    summary: list[str] = []
    for ln in lines:
        if not ln.strip():
            break
        summary.append(ln)
    config: list[str] = []
    span = _config_header_span(lines)
    if span is not None:
        config = lines[span[0] :]
    out = list(summary)
    if config:
        out += ["", *config]
    return out


def _flags(cls) -> list[str]:
    out = []
    if getattr(cls, "parallel_safe", False):
        out.append("parallel_safe")
    if getattr(cls, "provides_world", False):
        out.append("provides_world")
    return out


def _dist_name(ep) -> str:
    dist = getattr(ep, "dist", None)
    return dist.name if dist is not None else "unknown"


def _block_resumes(rest: list[str], base: int) -> bool:
    """Whether the block goes on after a blank line: the next text is still indented at its keys.

    A blank line groups a long block into sections; the prose after a block sits at the docstring's
    own margin, shallower than any key, so a blank line followed by that ends it.
    """
    for ln in rest:
        if ln.strip():
            return len(ln) - len(ln.lstrip()) >= base
    return False


def _parse_config_block(doc: str) -> list[dict]:
    """Parse a docstring's ``Config::`` block into structured fields.

    Each field line is ``  name: example_value  # trailing doc comment`` (see any
    plugin's ``Config::`` block, e.g. ``roqsim/plugins/contact_monitor.py``, for the
    convention). A doc comment too long for one line wraps onto a bare ``#``
    continuation line (e.g. ``ceiling.py``'s ``enabled`` field), which extends the
    previous field's doc rather than ending the block.

    A key that opens a **nested mapping** (``sample:``, with its keys indented under it) is
    reported itself and then its children, each under the dotted path a world YAML writes it
    at (``sample.resolution``). The world YAML nests, so a reader that stopped at the first
    nested key described the plugin's first few options and silently omitted the rest.
    A sequence item under such a key (``lines:`` over ``- {id: 0, ...}``) is an example of
    its value and is skipped the same way.

    The block ends at a blank line followed by text shallower than its keys, or at the first
    line that is neither a field, a nested key, nor a comment continuation, encountered *after*
    at least one field -- which keeps the trailing prose paragraphs common after a ``Config::``
    block from being read as more fields, while a blank line that only groups the block's keys
    into sections does not end it.
    """
    lines = doc.splitlines()
    span = _config_header_span(lines)
    if span is None:
        return []
    body = lines[span[1] + 1 :]

    # The indent the plugin's own keys sit at. A block opens with the plugin key itself
    # ("sensor_coverage_probe:"), one level shallower than the keys under it; anchoring on the
    # first key that carries a value tells the two apart without knowing the plugin's name.
    base = None
    for ln in body:
        if not ln.strip() or _COMMENT_ONLY_RE.match(ln):
            continue
        if _CONFIG_FIELD_RE.match(ln):
            base = len(ln) - len(ln.lstrip())
            break
        if not _CONFIG_WRAPPER_RE.match(ln):
            break
    if base is None:
        return []

    fields: list[dict] = []
    in_block = False
    #: (indent, key) of each mapping currently open, so a nested key is reported under the
    #: dotted path a world YAML would actually write it at.
    open_maps: list[tuple[int, str]] = []
    for i, ln in enumerate(body):
        if not ln.strip():
            if in_block and not _block_resumes(body[i + 1 :], base):
                break
            continue
        comment_only = _COMMENT_ONLY_RE.match(ln)
        if comment_only and in_block and fields:
            extra = comment_only.group(1).strip()
            if extra:
                last = fields[-1]
                last["doc"] = f"{last['doc']} {extra}" if last["doc"] else extra
            continue
        indent = len(ln) - len(ln.lstrip())
        if indent < base:
            # The wrapper naming the plugin, above its keys. Anything else out here has left
            # the block.
            if not in_block and _CONFIG_WRAPPER_RE.match(ln):
                continue
            break
        if open_maps and indent >= open_maps[-1][0] and _SEQUENCE_ITEM_RE.match(ln):
            # An example item of the list the open key takes ("lines:" over "- {id: 0, ...}",
            # indented or not, as YAML allows both): part of that key's value, so the keys after
            # it are still the plugin's.
            continue
        while open_maps and indent <= open_maps[-1][0]:
            open_maps.pop()
        prefix = "".join(f"{key}." for _, key in open_maps)
        # A nested key first: "floor:   # appearance" also fits the field pattern, with its
        # comment read as the value, and would leave the keys under it outside any mapping.
        nested = _CONFIG_NEST_RE.match(ln)
        if nested:
            name, comment = nested.groups()
            fields.append(
                {
                    "name": prefix + name,
                    "example": None,
                    "doc": comment.strip() if comment else None,
                }
            )
            open_maps.append((indent, name))
            in_block = True
            continue
        match = _CONFIG_FIELD_RE.match(ln)
        if match:
            name, example, comment = match.groups()
            fields.append(
                {
                    "name": prefix + name,
                    "example": example.strip(),
                    "doc": comment.strip() if comment else None,
                }
            )
            in_block = True
            continue
        if in_block:
            break
    return fields


def list_plugins() -> dict:
    """Every registered ``roqsim.plugins`` entry, one line per plugin.

    Returns ``{"items": [{name, kind: "plugin", doc, flags, package}, ...]}``,
    sorted by name. A plugin whose class fails to import is still listed (with
    ``doc: None`` and an ``error`` note) rather than sinking the whole catalog --
    the same "one broken entry must not sink the rest" rule the Sphinx directive
    and :func:`roqsim.registry.resolve_plugin` already follow.
    """
    items = []
    for ep in _entry_points(ENTRY_POINT_GROUP):
        try:
            cls = ep.load()
        except Exception as exc:  # noqa: BLE001 - one broken plugin must not sink the rest
            items.append(
                {
                    "name": ep.name,
                    "kind": "plugin",
                    "doc": None,
                    "flags": [],
                    "package": _dist_name(ep),
                    "error": f"could not import {ep.value!r}: {exc}",
                }
            )
            continue
        doc = _own_or_module_doc(cls)
        summary = doc.splitlines()[0].strip() if doc.strip() else None
        items.append(
            {
                "name": ep.name,
                "kind": "plugin",
                "doc": summary,
                "flags": _flags(cls),
                "package": _dist_name(ep),
            }
        )
    items.sort(key=lambda item: item["name"])
    return {"items": items}


#: The key :class:`~roqsim.plugin.Plugin` reads on every plugin that registers an entity.
_PRESENT_FIELD = {
    "name": "present",
    "example": "true",
    "doc": "whether the entity this entry registers is perceivable from the first step",
}


def _parameters(cls) -> list[dict]:
    """The ``Config::`` fields of *cls* and of every plugin base it inherits keys from.

    A base documents the keys it reads once, for all its subclasses (``camera_common.CameraPlugin``
    under every camera), and a subclass's own block then lists only what it adds. A subclass reads
    the inherited keys all the same, so a catalog that stopped at the subclass's own block would
    publish a camera without its ``width`` or ``rate_hz``. The subclass's own entry for a key wins
    over a base's, since it is the more specific statement of the default.

    ``present`` is read by :class:`~roqsim.plugin.Plugin` itself, for exactly the plugins that
    register an entity (:meth:`~roqsim.plugin.Plugin.validate_presence`), so it is published for
    those from that declaration rather than from each one's docstring.
    """
    from roqsim.plugin import Plugin

    fields: list[dict] = []
    seen_names: set[str] = set()
    seen_docs: set[str] = set()
    for klass in inspect.getmro(cls):
        if klass is Plugin or not (klass is cls or issubclass(klass, Plugin)):
            continue
        doc = _own_or_module_doc(klass)
        if not doc or doc in seen_docs:
            continue
        seen_docs.add(doc)
        for field in _parse_config_block(doc):
            if field["name"] not in seen_names:
                seen_names.add(field["name"])
                fields.append(field)
    if getattr(cls, "provides_entity", False) and "present" not in seen_names:
        fields.append(dict(_PRESENT_FIELD))
    return fields


def _declared_schema(cls) -> list[dict] | None:
    """The plugin's own ``CONFIG_SCHEMA``, published -- or ``None`` when it declares none.

    Published BESIDE the docstring-parsed ``config`` rather than instead of it: the parsed block is
    all most plugins have, and a caller that can read only one of the two should get the one that is
    always there. Where both exist the declared one is authoritative -- it is what validation runs
    on, so it cannot drift from behaviour the way a comment can.
    """
    schema = getattr(cls, "CONFIG_SCHEMA", None)
    if not schema:
        return None
    from roqsim.schema import describe

    return describe(schema)


def get_plugin_details(name: str) -> dict:
    """One plugin's full detail, or an error if *name* isn't a registered ``roqsim.plugins`` entry.

    Returns ``{name, kind: "plugin", doc, parameters, flags, package, class}`` where
    ``parameters`` is the ``Config::`` block parsed by :func:`_parse_config_block`, the plugin's
    own and each plugin base's it inherits (:func:`_parameters`) -- empty if none of them has
    one, whether because it takes no config or because nobody wrote one -- or ``{"error": "..."}``.

    A plugin that declares :data:`roqsim.plugin.Plugin.CONFIG_SCHEMA` also gets ``schema``: the same
    keys with their TYPES, defaults, units and bounds, which is what a caller generating a world
    needs and what prose cannot give it. It is authoritative where it exists, because validation
    runs on it -- unlike a docstring, it cannot drift from behaviour.
    """
    matches = [ep for ep in _entry_points(ENTRY_POINT_GROUP) if ep.name == name]
    if not matches:
        return {"error": f"no roqsim.plugins entry named {name!r}"}
    ep = matches[0]
    try:
        cls = ep.load()
    except Exception as exc:  # noqa: BLE001 - report, never raise
        return {"error": f"could not import {ep.value!r}: {exc}"}

    doc = _own_or_module_doc(cls)
    summary_lines = []
    for ln in doc.splitlines():
        if not ln.strip():
            break
        summary_lines.append(ln)
    details = {
        "name": ep.name,
        "kind": "plugin",
        "doc": " ".join(line.strip() for line in summary_lines) or None,
        "parameters": _parameters(cls),
        "flags": _flags(cls),
        "package": _dist_name(ep),
        "class": ep.value,
    }
    schema = _declared_schema(cls)
    if schema is not None:
        details["schema"] = schema
        details["strict_keys"] = bool(getattr(cls, "STRICT_KEYS", False))
    return details


def _is_config(node, *, in_init: bool) -> bool:
    """``self.config``, or -- inside ``__init__``, where it is the same mapping -- ``config``."""
    if isinstance(node, ast.Attribute):
        return (
            node.attr == "config" and isinstance(node.value, ast.Name) and node.value.id == "self"
        )
    return in_init and isinstance(node, ast.Name) and node.id == "config"


def _literal(node) -> str | None:
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None


def config_reads(cls) -> set[str]:
    """The config keys *cls* and its plugin bases read by literal name, from their source.

    A read is ``self.config.get("k")``, ``self.config["k"]`` or ``"k" in self.config``, and the same
    on ``config`` inside ``__init__``, which is handed the component's mapping before it is stored.
    ``config`` anywhere else -- ``validate_config``'s argument -- is not read: a key named there may
    be one the plugin refuses. A key reached
    through an alias or a computed name is not found -- this is a lower bound on what a plugin
    reads, which is what checking its published keys against needs.
    """
    from roqsim.plugin import Plugin

    keys: set[str] = set()
    for klass in inspect.getmro(cls):
        if klass is Plugin or not (klass is cls or issubclass(klass, Plugin)):
            continue
        try:
            tree = ast.parse(textwrap.dedent(inspect.getsource(klass)))
        except (OSError, TypeError):
            continue
        for func in ast.walk(tree):
            if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            in_init = func.name == "__init__"
            for node in ast.walk(func):
                key = None
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr in ("get", "pop", "setdefault")
                    and _is_config(node.func.value, in_init=in_init)
                    and node.args
                ):
                    key = _literal(node.args[0])
                elif isinstance(node, ast.Subscript) and _is_config(node.value, in_init=in_init):
                    key = _literal(node.slice)
                elif (
                    isinstance(node, ast.Compare)
                    and len(node.ops) == 1
                    and isinstance(node.ops[0], (ast.In, ast.NotIn))
                    and _is_config(node.comparators[0], in_init=in_init)
                ):
                    key = _literal(node.left)
                if key is not None:
                    keys.add(key)
    return keys


def undeclared_config_reads(cls) -> list[str]:
    """The keys *cls* reads (:func:`config_reads`) that its published catalog entry does not list.

    What a caller learns from :func:`get_plugin_details` is all it has to go on: a key the plugin
    reads but does not publish looks, from outside, like a key the plugin ignores. A key roqsim
    lets any component carry (:data:`roqsim.schema.INJECTED_KEYS`) is known without being listed,
    and a key with a leading underscore is the plugin's own bookkeeping rather than a setting.
    """
    from roqsim.schema import INJECTED_KEYS

    published = {field["name"].split(".", 1)[0] for field in _parameters(cls)}
    published |= {field["name"] for field in _declared_schema(cls) or []}
    return sorted(
        key
        for key in config_reads(cls)
        if key not in published and key not in INJECTED_KEYS and not key.startswith("_")
    )


# ── Module CLI (python -m roqsim.introspection <subcommand>) ─────────────────────


def main(argv=None):
    import argparse  # pylint: disable=import-outside-toplevel

    parser = argparse.ArgumentParser(
        prog="python -m roqsim.introspection",
        description="JSON introspection of the roqsim.plugins registry.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="List every registered roqsim.plugins entry, as JSON.")

    p_describe = sub.add_parser("describe", help="One plugin's full detail, as JSON.")
    p_describe.add_argument("name", help="Exact plugin entry-point name, e.g. 'contact_monitor'")

    args = parser.parse_args(argv)
    if args.command == "list":
        print(json.dumps(list_plugins(), indent=2))
    else:  # describe
        result = get_plugin_details(args.name)
        print(json.dumps(result, indent=2))
        sys.exit(1 if "error" in result else 0)


if __name__ == "__main__":
    main()
