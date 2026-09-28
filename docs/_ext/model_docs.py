# Copyright (C) 2026 Frederik Pasch
#
# SPDX-License-Identifier: Apache-2.0

"""``.. roqsim-models::`` -- generate the model catalog from the ``roqsim.models`` registry.

Enumerates every registered model provider, scans each provider's ``MODELS_DIR`` for ``<model>.xml``
files (plus walker ``people/<blueprint>/`` folders, which have no MJCF), and renders one catalog row
per model: its rendered thumbnail (generated once by ``make thumbnails``) beside a description
*derived* from what already ships -- the first sentence of the model file's own header comment, the
model's ``<model>.manifest.yaml`` (bundled plugins + borrowed asset providers), and the licence read
out of its ``LICENSE`` sidecar. No per-model prose is authored for the docs: a model that reads
poorly in the catalog is fixed by giving its MJCF header a first sentence saying what it is.
"""

from __future__ import annotations

import os
import re
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


def _license_files(model_xml: Path) -> list[Path]:
    """The licence sidecars covering a model: what its manifest declares, else what sits beside it.

    The fallback (a ``<stem>*LICENSE*``, or a lone ``*LICENSE*``) reads a folder-per-model provider
    correctly and cannot read a flat one, whose single directory holds several vendors' terms -- so
    an ambiguous model declares its own (``license:`` in the manifest) and that answer wins.
    """
    from roqsim.manifest import manifest_license

    if declared := manifest_license(model_xml):
        return declared
    model_dir = model_xml.parent
    licenses = [p for p in model_dir.iterdir() if p.is_file() and "LICENSE" in p.name.upper()]
    stem = model_xml.stem.lower()
    matching = [p for p in licenses if stem in p.name.lower()]
    if matching:
        return matching[:1]
    return licenses if len(licenses) == 1 else []


# Vendor licence texts are verbatim upstream files, so the licence a model carries is read out of the
# text rather than declared a second time in metadata that could drift from it. Ordered: the first
# marker that matches wins, so a text naming a more specific licence is not swallowed by a generic
# redistribution clause.
_LICENSE_MARKERS: tuple[tuple[str, str], ...] = (
    ("MIT", r"\bMIT License\b"),
    ("MPL-2.0", r"Mozilla Public License,?\s+Version 2\.0"),
    ("Apache-2.0", r"Apache License,?\s+Version 2\.0|Apache License 2\.0"),
    (
        "BSD-3-Clause",
        r"BSD 3-Clause|Software License Agreement \(BSD\)|"
        r"Redistribution and use in source and binary forms",
    ),
    ("CC-BY-4.0", r"Creative Commons Attribution 4\.0|CC[ -]BY[ -]4\.0"),
)


def _license_holder(text: str) -> str | None:
    """The copyright holder from a licence text: ``Copyright (c) 2023, Someone`` -> ``Someone``.

    A notice is only read where ``Copyright`` is followed by what makes it one -- a year, a ``(c)``,
    or the ``Copyright:`` of a port header. That is what separates a real notice from the prose of a
    stock licence body ("...retain the above copyright notice..."), and the unfilled placeholder an
    Apache appendix carries is rejected outright, so a stock text with no holder filled in yields
    none and the catalog names the licence alone.
    """
    for raw in text.splitlines():
        notice = re.search(r"copyright\b\s*(?::|\(c\)|©|(?=\d{4}))\s*(.+)", raw.strip(), flags=re.I)
        if not notice:
            continue
        holder = re.sub(r"^\d{4}(\s*[-–]\s*\d{4})?[,\s]*", "", notice.group(1))
        holder = re.sub(r"\s*(<[^>]*>|\([^)]*@[^)]*\))", "", holder)  # contact address
        holder = re.sub(r"[.,;]?\s*All rights reserved\.?$", "", holder, flags=re.I)
        holder = holder.strip(" ,;")
        if len(holder) < 3 or "[" in holder or "owner" in holder.lower():
            continue
        return holder
    return None


def _license_clause(env, license_path: Path) -> str:
    """``<name> (© <holder>) -- <download link>`` read out of one licence sidecar.

    What a reader needs is the licence itself -- its name and whose copyright it carries -- so both
    are read out of the sidecar's text, and the file is offered as a download for the full terms. A
    text whose licence is not one of the recognised ones (a vendor's own permission note about a
    mesh) is named by file alone, because naming it anything else would be a guess about terms.
    """
    text = license_path.read_text(errors="replace")
    name = next((n for n, marker in _LICENSE_MARKERS if re.search(marker, text, re.I)), None)
    rel = os.path.relpath(license_path, env.srcdir)
    link = f":download:`{license_path.name} <{rel}>`"
    if name is None:
        return f"see {link}"
    holder = _license_holder(text)
    return f"{name}{f' (© {holder})' if holder else ''} -- {link}"


def _license_line(env, license_paths: list[Path]) -> str | None:
    """``License: ...`` for a model's licence sidecars, or none when it ships no licence."""
    if not license_paths:
        return None
    return "License: " + "; ".join(_license_clause(env, path) for path in license_paths) + "."


def _headline(model_xml: Path) -> str | None:
    """First sentence of the MJCF's opening comment -- what this model *is*, in the model's own words.

    Only the comment that opens the ``<mujoco>`` element counts: that is where every model file says
    what the thing is, while a comment further down explains one element and describes nothing. That
    sentence is the catalog's per-model description, so a row says something about the model rather
    than repeating the owning package's summary once per row; a file whose opening comment is a
    generator banner (or that has none) falls back to the package summary.
    """
    text = model_xml.read_text(errors="replace")
    comment = re.search(r"<mujoco\b[^>]*>\s*<!--(.*?)-->", text, re.S)
    if not comment:
        return None
    paragraph: list[str] = []
    for raw in comment.group(1).splitlines():
        line = raw.strip()
        if not line:
            if paragraph:
                break
            continue
        paragraph.append(line)
    sentence = re.split(r"(?<=[.])\s", " ".join(paragraph), maxsplit=1)[0].strip()
    if (
        not re.match(r"[A-Z]", sentence)
        or not sentence.endswith(".")
        or sentence.startswith("GENERATED")
    ):
        return None
    return sentence if 15 <= len(sentence) <= 300 else None


def _body(
    name: str, dist_name: str, description: str, plugins, assets, license_line, extra=None
) -> list[str]:
    body = [f"**{name}** — ``{dist_name}``", ""]
    if description:
        body += [description, ""]
    if extra:
        body += [extra, ""]
    if plugins:
        body += ["Bundled plugins: " + ", ".join(f"``{p}``" for p in plugins) + ".", ""]
    if assets:
        body += ["Borrows assets from: " + ", ".join(f"``{a}``" for a in assets) + ".", ""]
    if license_line:
        body += [license_line, ""]
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


def _provider_rows(
    env, models_dir: Path, dist_name: str, summary: str
) -> list[tuple[Path, list[str]]]:
    rows: list[tuple[Path, list[str]]] = []
    for xml in sorted(models_dir.glob("*.xml")):
        plugins, assets = _manifest_bits(xml)
        rows.append(
            (
                _thumb(xml),
                _body(
                    xml.stem,
                    dist_name,
                    _headline(xml) or summary,
                    plugins,
                    assets,
                    _license_line(env, _license_files(xml)),
                ),
            )
        )
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
                    _headline(nested) or summary,
                    plugins,
                    assets,
                    _license_line(env, _license_files(nested)),
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
                    _body(blueprint.name, dist_name, "", [], [], None, extra),
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
            env = self.state.document.settings.env
            summary = _summary(dist_name)
            rows = _provider_rows(env, models_dir, dist_name, summary)
            if not rows:
                continue
            lines += rubric(f"``{dist_name}``")
            # The package summary describes the package, so it is said once for the group -- each row
            # carries its own model's headline instead.
            if summary:
                lines += [summary, ""]
            lines += catalog_table(env, rows)
        return parse_rst(self, lines)


def setup(app):
    app.add_directive("roqsim-models", SimSuiteModelsDirective)
    return {"version": "0.1", "parallel_read_safe": True}
