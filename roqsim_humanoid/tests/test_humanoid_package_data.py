# Copyright (C) 2026 Frederik Pasch
# SPDX-License-Identifier: Apache-2.0

"""Every file shipped under ``models/`` is one a ``[tool.setuptools.package-data]`` glob matches.

A file no glob matches installs cleanly from an editable checkout and is absent from the wheel, so
it fails at RUN time inside a container, where the checkout is not there to paper over it -- a mesh
the MJCF loads, a licence, or the ``<model>.thumb.png`` that ``roqsim catalog`` reports.
"""

from __future__ import annotations

import fnmatch
from pathlib import Path

import pytest


def test_package_data_globs_cover_every_shipped_model_file():
    tomllib = pytest.importorskip("tomllib")  # stdlib from 3.11; package supports 3.10
    root = Path(__file__).resolve().parents[1]
    data = tomllib.loads((root / "pyproject.toml").read_text())
    globs = data["tool"]["setuptools"]["package-data"]["roqsim_humanoid"]
    pkg = root / "src" / "roqsim_humanoid"
    missing = [
        str(p.relative_to(pkg))
        for p in (pkg / "models").rglob("*")
        if p.is_file()
        # __init__.py is a module (and .pyc build litter).
        and p.suffix not in (".py", ".pyc")
        and not any(fnmatch.fnmatch(str(p.relative_to(pkg)), g) for g in globs)
    ]
    assert not missing, f"not covered by package-data (would be absent from the wheel): {missing}"
