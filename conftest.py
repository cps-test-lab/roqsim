# SPDX-License-Identifier: Apache-2.0
"""Hooks for running every package's tests in one process, as ``make test`` does. A run whose rootdir
is one package (``make test-<pkg>``) does not load this file.

A built world is a reference cycle: the ``SimContext`` holds the plugins' endpoints and blackboard
entries, and those hold the plugins, which hold the context. So a finished test's ``MjModel``,
``MjData`` and renderer buffers are freed only when the cycle collector reaches them. Their memory
is allocated in C, invisible to the collector's thresholds, which count Python objects, and from
Python 3.14 on the collector reaches old cycles so late that a test process grows by gigabytes of
dead worlds. So each test is followed by a full collection.

A full collection walks every tracked object, and most of them are the modules imported by the time
the first test ends, which live as long as the process. They are frozen out of the collector once,
after the first test, so a collection costs what the tests left behind; what is alive then is never
collected (a module imported later, a world a fixture of the first file still holds).
"""

from __future__ import annotations

import functools
import gc
import importlib.util
import os
from pathlib import Path

import pytest

_frozen = False
_ROOT = Path(__file__).resolve().parent


@functools.cache
def _requirements_check():
    """``roqsim/tests/installed_requirements.py``, the rule the tests over every installed package
    already follow, loaded by path: this file is no package's."""
    spec = importlib.util.spec_from_file_location(
        "_roqsim_installed_requirements", _ROOT / "roqsim" / "tests" / "installed_requirements.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@functools.cache
def _missing(package_dir: Path) -> str:
    """Why *package_dir*'s distribution is installed without what it declares, or ``""``."""
    pyproject = package_dir / "pyproject.toml"
    if not pyproject.is_file():
        return ""
    name = _requirements_check().project_name(pyproject)
    if name is None:
        return ""
    missing = _requirements_check().missing_requirements(name)
    if not missing:
        return ""
    return (
        f"{name} is installed without its declared requirement(s) {', '.join(missing)} "
        "(installed with --no-deps?)"
    )


#: Set (to anything) where every declared requirement must be installed -- roqsim's CI: there a
#: package installed without one fails its collection instead of being skipped.
REQUIRE_ALL = "ROQSIM_TESTS_REQUIRE_ALL"


class _NotInstalledWhole(pytest.Module):
    """A test module of a package installed without what it declares: skipped, naming it, or a
    failure where :data:`REQUIRE_ALL` is set."""

    reason = ""

    def collect(self):
        if os.environ.get(REQUIRE_ALL):
            pytest.fail(f"{self.reason}, and {REQUIRE_ALL} is set", pytrace=False)
        pytest.skip(self.reason, allow_module_level=True)


def pytest_pycollect_makemodule(module_path, parent):
    """A package installed without a requirement it declares (``pip install --no-deps``, a partial
    image) fails its tests for a reason that is the installation's, not the code's; its test modules
    are skipped with the requirement named, read in ``-rs``, as the tests over every installed
    package skip it (``roqsim/tests/installed_requirements.py``). A requirement behind an extra does
    not count, and an import that fails for any other reason still fails."""
    path = Path(module_path).resolve()
    try:
        package = _ROOT / path.relative_to(_ROOT).parts[0]
    except (ValueError, IndexError):
        return None
    reason = _missing(package)
    if not reason:
        return None
    module = _NotInstalledWhole.from_parent(parent, path=path)
    module.reason = reason
    return module


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_teardown(item, nextitem):
    """Once a test's fixtures are torn down, free the worlds it built."""
    global _frozen
    yield
    gc.collect()
    if not _frozen:
        gc.freeze()
        _frozen = True
