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
import os
from pathlib import Path
from types import SimpleNamespace

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
    backend = _backend()
    for hook in (
        "build_wheel",
        "build_sdist",
        "build_editable",
        "prepare_metadata_for_build_wheel",
    ):
        assert getattr(backend, hook) is getattr(setuptools.build_meta, hook)


def _backend():
    spec = importlib.util.spec_from_file_location("roqsim_build_backend_under_test", REFERENCE)
    backend = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(backend)
    return backend


def _pip_writes(backend, monkeypatch, complete: bool, returncode: int = 0):
    """Stand in for pip: what it would leave in the directory ``--target`` names."""

    def run(argv, **_kwargs):
        where = Path(argv[argv.index("--target") + 1])
        (where / "partial").write_text("x")
        if complete:
            (where / "setuptools").mkdir()
            (where / "setuptools" / "__init__.py").write_text("")
        return SimpleNamespace(returncode=returncode)

    monkeypatch.setattr(backend.subprocess, "run", run)


def test_what_it_imports_is_in_the_user_s_own_cache_not_a_shared_temporary_directory(
    tmp_path, monkeypatch
):
    backend = _backend()
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    assert Path(backend._cache()).is_relative_to(tmp_path)
    monkeypatch.delenv("XDG_CACHE_HOME")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    assert Path(backend._cache()).is_relative_to(tmp_path / "home")


def test_a_failed_install_leaves_nothing_a_later_build_would_import(tmp_path, monkeypatch):
    backend = _backend()
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    _pip_writes(backend, monkeypatch, complete=True, returncode=1)
    with pytest.raises(RuntimeError, match="setuptools>=64"):
        backend._declared_setuptools()
    assert os.listdir(Path(backend._cache()).parent) == []


def test_an_install_appears_whole_and_is_used_again(tmp_path, monkeypatch):
    backend = _backend()
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    _pip_writes(backend, monkeypatch, complete=True)
    target = Path(backend._declared_setuptools())
    assert (target / "setuptools" / "__init__.py").is_file()
    assert os.listdir(target.parent) == [target.name]
    monkeypatch.setattr(backend.subprocess, "run", None)  # a second build installs nothing
    assert Path(backend._declared_setuptools()) == target
