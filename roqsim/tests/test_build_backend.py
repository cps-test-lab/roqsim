# Copyright (C) 2026 Frederik Pasch
# SPDX-License-Identifier: Apache-2.0
"""Every pip-installed package builds with the one build backend, and its sdist carries it.

The backend (``_build/roqsim_build_backend.py``) is ``setuptools.build_meta`` with the setuptools a
package declares, even where an installer without build isolation offers an older one. A backend
must sit inside the project it builds, so each package carries a copy; these tests hold the copies
identical, named by every ``pyproject.toml`` and listed in every ``MANIFEST.in``.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
REFERENCE = REPO / "roqsim" / "_build" / "roqsim_build_backend.py"
PACKAGES = sorted(
    p.parent
    for pattern in ("roqsim*/pyproject.toml", "scenario_execution_*/pyproject.toml")
    for p in REPO.glob(pattern)
)


def test_the_repository_has_packages_to_check():
    assert len(PACKAGES) > 5 and REPO / "roqsim" in PACKAGES


@pytest.mark.parametrize("package", PACKAGES, ids=lambda p: p.name)
def test_a_package_builds_with_the_one_backend_and_ships_it_in_its_sdist(package):
    pyproject = (package / "pyproject.toml").read_text()
    assert 'build-backend = "roqsim_build_backend"' in pyproject
    assert 'backend-path = ["_build"]' in pyproject
    assert '"setuptools>=64"' in pyproject
    copy = package / "_build" / "roqsim_build_backend.py"
    assert copy.read_bytes() == REFERENCE.read_bytes(), f"{copy} differs from {REFERENCE}"
    assert "include _build/roqsim_build_backend.py" in (package / "MANIFEST.in").read_text()


def test_with_a_current_setuptools_it_is_setuptools_build_meta():
    import setuptools
    import setuptools.build_meta

    if int(setuptools.__version__.split(".")[0]) < 64:
        pytest.skip(f"setuptools {setuptools.__version__} here: the backend would fetch 64+")
    spec = importlib.util.spec_from_file_location("roqsim_build_backend_under_test", REFERENCE)
    backend = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(backend)
    for hook in (
        "build_wheel",
        "build_sdist",
        "build_editable",
        "prepare_metadata_for_build_wheel",
    ):
        assert getattr(backend, hook) is getattr(setuptools.build_meta, hook)
