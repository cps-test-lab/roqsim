# Copyright (C) 2026 Frederik Pasch
#
# SPDX-License-Identifier: Apache-2.0

"""``.. roqsim-models::`` -- generate the model catalog from the ``roqsim.models`` registry.

Enumerates every registered model provider, scans each provider's ``MODELS_DIR`` for ``<model>.xml``
files (plus walker ``people/<blueprint>/`` folders, which have no MJCF), and renders one catalog row
per model: its rendered thumbnail (generated once by ``make thumbnails``) beside a description
*synthesised* from the model name, the owning package's summary, and the model's
``<model>.manifest.yaml`` (bundled plugins + borrowed asset providers). No per-model prose is
authored anywhere; everything is derived from what already ships.
"""

from __future__ import annotations

from importlib import import_module, metadata
from pathlib import Path

import yaml
from _render import catalog_table, parse_rst, rubric
from docutils.parsers.rst import Directive


def _summary(dist_name: str) -> str:
    try:
        return metadata.metadata(dist_name).get("Summary", "") or ""
    except metadata.PackageNotFoundError:
        return ""


def _manifest_bits(model_xml: Path) -> tuple[list[str], list[str]]:
    """(bundled plugin names, borrowed asset providers) from ``<model>.manifest.yaml``, if any."""
    manifest = model_xml.parent / f"{model_xml.stem}.manifest.yaml"
    if not manifest.exists():
        return [], []
    data = yaml.safe_load(manifest.read_text()) or {}
    plugins = []
    for entry in data.get("plugins", []) or []:
        if isinstance(entry, str):
            plugins.append(entry)
        elif isinstance(entry, dict) and entry:
            plugins.append(next(iter(entry)))
    assets = data.get("assets")
    assets = [assets] if isinstance(assets, str) else list(assets or [])
    return plugins, assets


def _license_file(model_dir: Path, stem: str) -> str | None:
    """A LICENSE sidecar relevant to this model (a ``<stem>*LICENSE*`` or a lone ``*LICENSE*``)."""
    licenses = [p.name for p in model_dir.iterdir() if p.is_file() and "LICENSE" in p.name.upper()]
    if not licenses:
        return None
    stem_l = stem.lower()
    for name in licenses:
        if stem_l in name.lower():
            return name
    return licenses[0] if len(licenses) == 1 else None


def _body(
    name: str, dist_name: str, summary: str, plugins, assets, license_name, extra=None
) -> list[str]:
    body = [f"**{name}** — ``{dist_name}``", ""]
    if summary:
        body += [summary, ""]
    if extra:
        body += [extra, ""]
    if plugins:
        body += ["Bundled plugins: " + ", ".join(f"``{p}``" for p in plugins) + ".", ""]
    if assets:
        body += ["Borrows assets from: " + ", ".join(f"``{a}``" for a in assets) + ".", ""]
    if license_name:
        body += [f"License: ``{license_name}``.", ""]
    while body and body[-1] == "":
        body.pop()
    return body


def _credits(folder: Path) -> str | None:
    """First (attribution) line of a ``CREDITS.txt`` beside a prop, if present."""
    credits = folder / "CREDITS.txt"
    if not credits.exists():
        return None
    for raw in credits.read_text().splitlines():
        line = raw.strip()
        if line:
            return line
    return None


def _thumb(model_file: Path) -> Path:
    """Co-located thumbnail beside a model MJCF: ``<dir>/<stem>.thumb.png`` (see render_thumbnails)."""
    return model_file.parent / f"{model_file.stem}.thumb.png"


def _provider_rows(models_dir: Path, dist_name: str, summary: str) -> list[tuple[Path, list[str]]]:
    rows: list[tuple[Path, list[str]]] = []
    for xml in sorted(models_dir.glob("*.xml")):
        stem = xml.stem
        plugins, assets = _manifest_bits(xml)
        license_name = _license_file(models_dir, stem)
        rows.append((_thumb(xml), _body(stem, dist_name, summary, plugins, assets, license_name)))
    # Nested <name>/<name>.xml models: a prop (self-contained MJCF + mesh + CREDITS per folder), or a
    # robot laid out one folder per model (roqsim_manipulation_assets) -- which keeps its vendor LICENSE
    # sidecar in the folder, so look for one there as well as flat beside the MJCF.
    for sub in sorted(p for p in models_dir.iterdir() if p.is_dir()):
        nested = sub / f"{sub.name}.xml"
        if not nested.is_file():
            continue
        plugins, assets = _manifest_bits(nested)
        rows.append(
            (
                _thumb(nested),
                _body(
                    sub.name,
                    dist_name,
                    summary,
                    plugins,
                    assets,
                    _license_file(sub, sub.name),
                    extra=_credits(sub),
                ),
            )
        )
    people = models_dir / "people"
    if people.is_dir():
        for blueprint in sorted(p for p in people.iterdir() if p.is_dir()):
            if not any(blueprint.glob("*.walker.json")):
                continue
            extra = "Procedural skinned pedestrian blueprint (no MJCF; body + skin generated at build time)."
            rows.append(
                (
                    blueprint / f"{blueprint.name}.thumb.png",
                    _body(blueprint.name, dist_name, summary, [], [], None, extra),
                )
            )
    return rows


class SimSuiteModelsDirective(Directive):
    has_content = False

    def run(self):
        from roqsim.models import ENTRY_POINT_GROUP, _entry_points, _provider_dirs

        lines: list[str] = []
        for ep in sorted(_entry_points(ENTRY_POINT_GROUP), key=lambda e: e.name):
            dist = getattr(ep, "dist", None)
            dist_name = dist.name if dist is not None else ep.name
            try:
                module = import_module(
                    ep.value.split(":")[0] if isinstance(ep.value, str) else ep.value
                )
                models_dir, _mesh, _tex = _provider_dirs(module)
            except Exception as exc:  # noqa: BLE001 - a broken provider must not break docs
                lines += rubric(f"``{dist_name}``")
                lines += [
                    f".. note:: Could not load model provider ``{ep.name}`` "
                    f"({type(exc).__name__}: {exc}).",
                    "",
                ]
                continue
            rows = _provider_rows(models_dir, dist_name, _summary(dist_name))
            if not rows:
                continue
            lines += rubric(f"``{dist_name}``")
            lines += catalog_table(self.state.document.settings.env, rows)
        return parse_rst(self, lines)


def setup(app):
    app.add_directive("roqsim-models", SimSuiteModelsDirective)
    return {"version": "0.1", "parallel_read_safe": True}
