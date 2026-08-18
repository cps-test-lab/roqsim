# Copyright (C) 2026 Frederik Pasch
#
# SPDX-License-Identifier: Apache-2.0

"""``.. roqsim-textures::`` -- generate the shared-texture catalog from ``roqsim_assets``.

``roqsim_assets`` holds reusable PBR surface **textures** (``assets/<Name>/`` -- a colour map +
``manifest.yaml`` with ``reflectance``/``physical_size`` + CC0 ``CREDITS.txt``). The swatch shown is
the texture's own ``*_Color.png``, referenced in place (no separate render). The package's *props*
are placeable models and are documented in the model catalog (``.. roqsim-models::``), not here.
"""

from __future__ import annotations

from pathlib import Path

from _render import catalog_table, parse_rst, rubric
from docutils.parsers.rst import Directive

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None


def _credits_lines(folder: Path) -> list[str]:
    """CREDITS.txt as RST paragraphs (attribution / license / source), or empty."""
    credits = folder / "CREDITS.txt"
    if not credits.exists():
        return []
    out: list[str] = []
    for raw in credits.read_text().splitlines():
        line = raw.strip()
        if line:
            out += [line, ""]
    return out


def _texture_rows(assets_dir: Path) -> list[tuple[Path | None, list[str]]]:
    rows: list[tuple[Path | None, list[str]]] = []
    for folder in sorted(p for p in assets_dir.iterdir() if p.is_dir()):
        name = folder.name
        body = [f"**{name}**", ""]
        manifest = folder / "manifest.yaml"
        if yaml is not None and manifest.exists():
            data = yaml.safe_load(manifest.read_text()) or {}
            facts = []
            if "physical_size" in data:
                facts.append(f"tiles at {data['physical_size']} m")
            if "reflectance" in data:
                facts.append(f"reflectance {data['reflectance']}")
            if facts:
                body += ["Surface: " + ", ".join(facts) + ".", ""]
        body += _credits_lines(folder)
        while body and body[-1] == "":
            body.pop()
        # The swatch is the texture's own colour map -- no separate thumbnail is generated.
        colors = sorted(folder.glob("*_Color.png")) or sorted(folder.glob("*.png"))
        rows.append((colors[0] if colors else None, body))
    return rows


class SimSuiteTexturesDirective(Directive):
    has_content = False

    def run(self):
        try:
            import roqsim_assets
        except Exception as exc:  # noqa: BLE001
            return parse_rst(
                self,
                [
                    f".. note:: ``roqsim_assets`` is not installed "
                    f"({type(exc).__name__}: {exc}).",
                    "",
                ],
            )
        base = Path(roqsim_assets.__file__).parent
        env = self.state.document.settings.env
        lines: list[str] = []
        textures = base / "assets"
        if textures.is_dir():
            rows = _texture_rows(textures)
            if rows:
                lines += rubric("Textures")
                lines += catalog_table(env, rows)
        return parse_rst(self, lines)


def setup(app):
    app.add_directive("roqsim-textures", SimSuiteTexturesDirective)
    return {"version": "0.1", "parallel_read_safe": True}
