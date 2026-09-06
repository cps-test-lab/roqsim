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
# "Config (same keys as livox_mid360; only the defaults differ)::") rather than a bare "Config::" --
# matched too, as long as the "::" that starts the block is on the same source line as "Config".
_CONFIG_HEADER_RE = re.compile(r"\s*Config\b.*::\s*$")
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


def _own_or_module_doc(cls) -> str:
    """The plugin's *own* docstring, else its module's -- never the inherited base ``Plugin`` one.

    ``inspect.getdoc`` walks the MRO and would return ``Plugin``'s boilerplate for a plugin that has
    no docstring of its own; that base text is useless in a per-plugin catalog. Most plugins put
    their description (+ ``Config::``) at module level, so that is the fallback.
    """
    own = cls.__dict__.get("__doc__")
    if own and own.strip():
        return inspect.cleandoc(own)
    module = inspect.getmodule(cls)
    return inspect.cleandoc(module.__doc__) if module and module.__doc__ else ""


def _summary_and_config(doc: str) -> list[str]:
    """A short description (first paragraph) plus the ``Config::`` block, dropping the middle prose."""
    lines = doc.splitlines()
    summary: list[str] = []
    for ln in lines:
        if not ln.strip():
            break
        summary.append(ln)
    config: list[str] = []
    for i, ln in enumerate(lines):
        if _CONFIG_HEADER_RE.match(ln):
            config = lines[i:]
            break
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
    previous field's doc rather than ending the block. Otherwise best-effort: a
    line that doesn't match either shape (a nested/multi-line value, a bare
    mapping header) is skipped rather than crashing the whole parse -- the block
    ends at the first blank line, or the first line that is neither a field nor a
    comment continuation, encountered *after* at least one field has been parsed
    -- which is what keeps trailing prose paragraphs (common after a ``Config::``
    block) from being mistaken for more fields.
    """
    lines = doc.splitlines()
    start = None
    for i, ln in enumerate(lines):
        if _CONFIG_HEADER_RE.match(ln):
            start = i + 1
            break
    if start is None:
        return []

    fields: list[dict] = []
    in_block = False
    for ln in lines[start:]:
        if not ln.strip():
            if in_block:
                break
            continue
        match = _CONFIG_FIELD_RE.match(ln)
        if match:
            name, example, comment = match.groups()
            fields.append(
                {
                    "name": name,
                    "example": example.strip(),
                    "doc": comment.strip() if comment else None,
                }
            )
            in_block = True
            continue
        comment_only = _COMMENT_ONLY_RE.match(ln)
        if comment_only and in_block and fields:
            extra = comment_only.group(1).strip()
            if extra:
                last = fields[-1]
                last["doc"] = f"{last['doc']} {extra}" if last["doc"] else extra
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
