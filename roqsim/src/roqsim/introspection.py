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

import inspect
import json
import re
import sys

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
# it indented further. The world YAML nests, so the block does too, and a key like this used to end
# the parse -- reporting a plugin's first few keys and silently dropping the rest.
_CONFIG_NEST_RE = re.compile(r"^\s+([A-Za-z_]\w*):\s*(?:#\s*(.*))?$")
# The line naming the plugin itself, which a block opens with one level above its keys. Written
# either as a plain key ("sensor_coverage_probe:"), as the list entry a world YAML's
# "components:" actually takes ("- spawn_sensor:"), or -- in a base class documenting keys its
# subclasses inherit -- as a placeholder standing in for whichever name they register under
# ("<plugin short name>:"). All three are skipped rather than parsed, so the keys underneath are
# read at the level they are written.
_CONFIG_WRAPPER_RE = re.compile(r"^\s+-?\s*(?:[A-Za-z_]\w*|<[^>]+>):\s*(?:#\s*(.*))?$")


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

    The block ends at the first blank line, or the first line that is neither a field, a
    nested key, nor a comment continuation, encountered *after* at least one field -- which
    keeps the trailing prose paragraphs common after a ``Config::`` block from being read as
    more fields.
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
    for ln in body:
        if not ln.strip():
            if in_block:
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
        while open_maps and indent <= open_maps[-1][0]:
            open_maps.pop()
        prefix = "".join(f"{key}." for _, key in open_maps)
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
    ``parameters`` is :func:`_parse_config_block`'s output (empty if the plugin has
    no ``Config::`` block, whether because it takes no config or because nobody
    wrote one) -- or ``{"error": "..."}``.

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
        "parameters": _parse_config_block(doc),
        "flags": _flags(cls),
        "package": _dist_name(ep),
        "class": ep.value,
    }
    schema = _declared_schema(cls)
    if schema is not None:
        details["schema"] = schema
        details["strict_keys"] = bool(getattr(cls, "STRICT_KEYS", False))
    return details


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
