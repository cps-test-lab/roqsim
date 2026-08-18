# Copyright (C) 2026 Frederik Pasch
#
# SPDX-License-Identifier: Apache-2.0

"""``.. roqsim-worlds::`` -- generate the world catalog.

Covers both kinds of ``sim.world``: the built-in *world definitions* built in code
(:func:`roqsim.world.available_worlds`, e.g. ``empty_room``) and the *provider scenes* baked as
MJCF and registered in the ``roqsim.worlds`` entry-point group (e.g. ``roqsim_scenes:depot``).
Each gets a rendered thumbnail (generated once) beside a synthesised description.
"""

from __future__ import annotations

from importlib import import_module, metadata
from pathlib import Path

from _render import catalog_table, parse_rst, rubric
from docutils.parsers.rst import Directive


def _summary(dist_name: str) -> str:
    try:
        return metadata.metadata(dist_name).get("Summary", "") or ""
    except metadata.PackageNotFoundError:
        return ""


def _scene_worlds(worlds_dir: Path) -> list[tuple[str, Path]]:
    """(name, mjcf path) per world under ``WORLDS_DIR``: ``<name>/<name>.xml`` or flat ``<name>.xml``."""
    found: dict[str, Path] = {}
    for sub in worlds_dir.iterdir():
        nested = sub / f"{sub.name}.xml"
        if sub.is_dir() and nested.is_file():
            found[sub.name] = nested
    for flat in worlds_dir.glob("*.xml"):
        found.setdefault(flat.stem, flat)
    return sorted(found.items())


class SimSuiteWorldsDirective(Directive):
    has_content = False

    def run(self):
        from roqsim.world import _world_entry_points, available_worlds

        env = self.state.document.settings.env
        lines: list[str] = []

        builtins = available_worlds()
        if builtins:
            # Code-built worlds have no on-disk MJCF, so no co-located thumbnail (None -> text only).
            rows = [
                (
                    None,
                    [
                        f"**{name}** — built-in world definition",
                        "",
                        "Ground plane + ceiling light"
                        + (", enclosed by four perimeter walls" if name == "empty_room" else "")
                        + ". Selected with ``sim.world`` in the world YAML.",
                    ],
                )
                for name in builtins
            ]
            lines += rubric("Built-in world definitions")
            lines += catalog_table(env, rows)

        for ep in sorted(_world_entry_points(), key=lambda e: e.name):
            dist = getattr(ep, "dist", None)
            dist_name = dist.name if dist is not None else ep.name
            try:
                module = import_module(
                    ep.value.split(":")[0] if isinstance(ep.value, str) else ep.value
                )
                worlds_dir = Path(module.WORLDS_DIR)
                names = _scene_worlds(worlds_dir)
            except Exception as exc:  # noqa: BLE001 - a broken provider must not break docs
                lines += rubric(f"``{dist_name}``")
                lines += [
                    f".. note:: Could not load world provider ``{ep.name}`` "
                    f"({type(exc).__name__}: {exc}).",
                    "",
                ]
                continue
            if not names:
                continue
            summary = _summary(dist_name)
            rows = []
            for name, mjcf in names:
                body = [f"**{name}** — ``{dist_name}``", ""]
                if summary:
                    body += [summary, ""]
                body += [
                    f"Baked MJCF scene. Reference as ``{ep.name}:{name}`` in ``sim.world``.",
                    "",
                ]
                while body and body[-1] == "":
                    body.pop()
                rows.append((mjcf.parent / f"{mjcf.stem}.thumb.png", body))
            lines += rubric(f"``{dist_name}``")
            lines += catalog_table(env, rows)

        return parse_rst(self, lines)


def setup(app):
    app.add_directive("roqsim-worlds", SimSuiteWorldsDirective)
    return {"version": "0.1", "parallel_read_safe": True}
