"""Every prop and texture folder ships a CREDITS.txt that names its licence.

Attribution for third-party content lives in the CREDITS.txt beside the files it credits, not in a
central list. A folder of our own work says so in its CREDITS.txt with the OWN_WORK line, so every
folder carries one: a folder without it cannot be told apart from third-party content that lost its
attribution.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_DIR = PACKAGE_ROOT / "src" / "roqsim_assets"

OWN_WORK = "Licence: Apache-2.0, with the rest of this repository."
# Licences a third-party asset here may carry (THIRD_PARTY.md): CC0, CC-BY, CC-BY-SA -- in the
# wording of ambientCG / Poly Haven (CC0 1.0) and of Sketchfab's licence names.
REDISTRIBUTABLE = re.compile(
    r"\bCC0 1\.0\b|\bCC Attribution(-ShareAlike)?\b|\bCC-BY(-SA)?(-4\.0)?\b(?!-)"
)
EXCLUDED = re.compile(r"NonCommercial|NoDeriv|\bCC-BY(-SA)?-N[CD]\b")


def _asset_folders() -> list[Path]:
    folders = []
    for kind in ("assets", "models"):
        folders += [
            d
            for d in sorted((PACKAGE_DIR / kind).iterdir())
            if d.is_dir() and not d.name.startswith((".", "__"))
        ]
    return folders


def _package_data_globs() -> list[str]:
    tomllib = pytest.importorskip("tomllib")  # stdlib from 3.11
    with open(PACKAGE_ROOT / "pyproject.toml", "rb") as f:
        pyproject = tomllib.load(f)
    return pyproject["tool"]["setuptools"]["package-data"]["roqsim_assets"]


def test_asset_folders_found():
    kinds = {d.parent.name for d in _asset_folders()}
    assert kinds == {"assets", "models"}


@pytest.mark.parametrize("folder", _asset_folders(), ids=lambda d: f"{d.parent.name}/{d.name}")
def test_folder_has_credits_naming_its_licence(folder):
    credits = folder / "CREDITS.txt"
    assert credits.is_file(), (
        f"{folder.relative_to(PACKAGE_DIR)} has no CREDITS.txt: third-party content needs its "
        f"licence and attribution beside it, and our own work says {OWN_WORK!r}"
    )
    text = credits.read_text(encoding="utf-8")
    assert not EXCLUDED.search(text), (
        f"{credits.relative_to(PACKAGE_DIR)}: NC/ND content is not admitted"
    )
    assert OWN_WORK in text or REDISTRIBUTABLE.search(text), (
        f"{credits.relative_to(PACKAGE_DIR)} names no licence: expected {OWN_WORK!r} for our own "
        f"work, or CC0 1.0 / CC Attribution / CC-BY for third-party content"
    )


def test_every_credits_file_ships():
    globs = _package_data_globs()
    shipped = {p for pattern in globs for p in PACKAGE_DIR.glob(pattern)}
    credits = sorted(PACKAGE_DIR.rglob("CREDITS.txt"))
    assert credits
    missing = [str(p.relative_to(PACKAGE_DIR)) for p in credits if p not in shipped]
    assert not missing, f"CREDITS.txt not covered by package-data in pyproject.toml: {missing}"
