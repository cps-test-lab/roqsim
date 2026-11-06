# Copyright (C) 2026 Frederik Pasch
# SPDX-License-Identifier: Apache-2.0
"""This package's build backend: ``setuptools.build_meta``, with the setuptools the package declares.

The package's metadata is a ``[project]`` table, and installing it in place (``pip install -e``)
needs PEP 660's ``build_editable`` hook: both need setuptools 64 or later, which ``[build-system]
requires`` states. An installer with build isolation provides it. One without
(``pip install --no-build-isolation``) builds with whatever setuptools the environment has, and an
older one builds an ``UNKNOWN-0.0.0`` wheel or refuses an editable install. So where the importable
setuptools is older than :data:`REQUIRED`, this installs the declared one into a cache directory of
its own -- what build isolation would have installed -- and builds with that; anywhere else it is
``setuptools.build_meta`` and nothing more.

Every pip-installed package of this repository carries this file unchanged under ``_build/`` (a
build backend must sit inside the project it builds, so a published sdist builds on its own);
``roqsim/tests/test_build_backend.py`` holds the copies identical.
"""

from __future__ import annotations

import importlib
import os
import shutil
import subprocess
import sys
import tempfile

#: The setuptools ``[build-system] requires`` states: the first to read ``[project]`` and to offer
#: ``build_editable``.
REQUIRED = 64
SPEC = f"setuptools>={REQUIRED}"
_OWN = ("setuptools", "pkg_resources", "_distutils_hack", "distutils")


def _major(version: str) -> int:
    try:
        return int(version.split(".")[0])
    except ValueError:
        return 0


def _cache() -> str:
    """The declared setuptools' directory, in this user's cache: it is imported, so nobody else may
    be able to put it there first."""
    root = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    return os.path.join(root, "roqsim-build", f"setuptools-{REQUIRED}")


def _declared_setuptools() -> str:
    """A directory holding the declared setuptools, installed there on first use.

    It is installed beside the directory and moved into place, so a build running at the same time
    never imports half of it.
    """
    target = _cache()
    marker = os.path.join(target, "setuptools", "__init__.py")
    if os.path.isfile(marker):
        return target
    parent = os.path.dirname(target)
    os.makedirs(parent, exist_ok=True)
    scratch = tempfile.mkdtemp(prefix=".setuptools-", dir=parent)
    try:
        done = subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--quiet",
                "--disable-pip-version-check",
                "--no-warn-script-location",
                "--target",
                scratch,
                SPEC,
            ],
            stdout=sys.stderr,
            check=False,
        )
        if done.returncode != 0 or not os.path.isfile(
            os.path.join(scratch, "setuptools", "__init__.py")
        ):
            raise RuntimeError(
                f"this package builds with {SPEC}; the importable setuptools is older, and "
                f"installing {SPEC} into {target} failed (pip exited {done.returncode}). Install "
                f"{SPEC} into the environment, or build with build isolation."
            )
        try:
            os.rename(scratch, target)
        except OSError:
            if not os.path.isfile(marker):  # not another build having put it there first
                raise
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    return target


def _build_meta():
    import setuptools

    if _major(setuptools.__version__) >= REQUIRED:
        return importlib.import_module("setuptools.build_meta")
    target = _declared_setuptools()
    # The older setuptools is already imported (and its distutils shim installed); drop both, so
    # every name below resolves to the declared one.
    for name in list(sys.modules):
        if name.split(".")[0] in _OWN:
            del sys.modules[name]
    sys.meta_path[:] = [f for f in sys.meta_path if type(f).__module__.split(".")[0] not in _OWN]
    sys.path.insert(0, target)
    meta = importlib.import_module("setuptools.build_meta")
    import setuptools as declared

    if _major(declared.__version__) < REQUIRED:
        raise RuntimeError(f"{target} holds setuptools {declared.__version__}, not {SPEC}")
    return meta


_meta = _build_meta()

get_requires_for_build_wheel = _meta.get_requires_for_build_wheel
get_requires_for_build_sdist = _meta.get_requires_for_build_sdist
get_requires_for_build_editable = _meta.get_requires_for_build_editable
prepare_metadata_for_build_wheel = _meta.prepare_metadata_for_build_wheel
prepare_metadata_for_build_editable = _meta.prepare_metadata_for_build_editable
build_wheel = _meta.build_wheel
build_sdist = _meta.build_sdist
build_editable = _meta.build_editable
