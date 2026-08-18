# Copyright (C) 2026 Frederik Pasch
#
# SPDX-License-Identifier: Apache-2.0

"""``.. roqsim-plugins::`` -- generate the plugin catalog from the ``roqsim.plugins`` registry.

Enumerates the same entry-point group the runtime loader uses
(:data:`roqsim.registry.ENTRY_POINT_GROUP`), groups plugins by the package that ships them, and
renders each plugin's short name (the key used in a world YAML), its lifecycle flags, a one-line
summary, and its ``Config::`` block. A plugin whose class fails to import is still listed, with a note
-- so the ``-W`` docs build never aborts on an optional/heavy dependency.
"""

from __future__ import annotations

from _render import docstring_lines, parse_rst, rubric
from docutils.parsers.rst import Directive

from roqsim.introspection import _dist_name, _flags as _plugin_flags
from roqsim.introspection import _own_or_module_doc, _summary_and_config

# Core package first, then alphabetical -- mirrors the Makefile's PKGS sort and the hand-written page.
_CORE = "roqsim"


def _package_label(dist_name: str) -> str:
    return f"Core (``{dist_name}``)" if dist_name == _CORE else f"``{dist_name}``"


def _flags(cls) -> list[str]:
    """RST-quoted flag names for the docs page -- ``roqsim.introspection``'s own :func:`_flags`
    returns plain names (JSON has no markup to escape); this wraps each in double-backticks."""
    return [f"``{flag}``" for flag in _plugin_flags(cls)]


class SimSuitePluginsDirective(Directive):
    has_content = False

    def run(self):
        from roqsim.registry import ENTRY_POINT_GROUP, _entry_points

        groups: dict[str, list] = {}
        for ep in _entry_points(ENTRY_POINT_GROUP):
            groups.setdefault(_dist_name(ep), []).append(ep)

        order = sorted(groups, key=lambda d: (d != _CORE, d))
        lines: list[str] = []
        for dist_name in order:
            lines += rubric(_package_label(dist_name))
            for ep in sorted(groups[dist_name], key=lambda e: e.name):
                lines += rubric(f"``{ep.name}``")
                try:
                    cls = ep.load()
                except Exception as exc:  # noqa: BLE001 - optional/heavy deps must not break docs
                    lines += [
                        f".. note:: Could not introspect ``{ep.name}`` "
                        f"({type(exc).__name__}: {exc}).",
                        "",
                    ]
                    continue
                flags = _flags(cls)
                if flags:
                    lines += [f"*Flags:* {', '.join(flags)}", ""]
                doc = _own_or_module_doc(cls)
                lines += docstring_lines("\n".join(_summary_and_config(doc)))
                lines += [""]
        return parse_rst(self, lines)


def setup(app):
    app.add_directive("roqsim-plugins", SimSuitePluginsDirective)
    return {"version": "0.1", "parallel_read_safe": True}
