# Copyright (C) 2026 Frederik Pasch
# SPDX-License-Identifier: Apache-2.0
"""Whether an installed distribution has what it declares: for the tests that walk every installed one.

A test that loads every ``roqsim.plugins`` entry or every ``roqsim`` command group of this repository
meets each of its packages, and a package installed without its declared requirements (``pip install
--no-deps``, a partial image) fails its import for a reason that is the installation's, not the
code's. Those tests fail naming the requirement it lacks, so the fix is the one the message asks
for; none of them skips it, so a run never ends green having checked less. Only a requirement the
distribution itself (or one it requires, in turn) **declares** counts: an import that fails for any
other reason fails as it does. Another repository's packages, installed beside this one's, are that
repository's to check (:func:`ours`).
"""

from __future__ import annotations

import functools
import re
from importlib import metadata
from pathlib import Path

from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name


def _installed(name: str) -> metadata.Distribution | None:
    try:
        return metadata.distribution(name)
    except metadata.PackageNotFoundError:
        return None


@functools.cache
def missing_requirements(dist_name: str) -> tuple[str, ...]:
    """The requirements *dist_name* declares, directly or through those it requires, not installed.

    Requirements behind an extra are not counted: an extra is optional by definition.
    """
    missing: list[str] = []
    seen: set[str] = set()
    todo = [dist_name]
    while todo:
        name = canonicalize_name(todo.pop())
        if name in seen:
            continue
        seen.add(name)
        dist = _installed(name)
        if dist is None:
            continue
        for spec in dist.requires or ():
            try:
                req = Requirement(spec)
            except InvalidRequirement:
                continue
            if req.marker is not None and not req.marker.evaluate({"extra": ""}):
                continue
            if _installed(req.name) is None:
                missing.append(req.name)
            else:
                todo.append(req.name)
    return tuple(sorted(set(missing)))


def not_installed_with_its_requirements(ep) -> str | None:
    """Why entry point *ep*'s package cannot be used here, when it was installed without what it
    declares; ``None`` when nothing it declares is missing."""
    dist = getattr(ep, "dist", None)
    if dist is None:
        return None
    missing = missing_requirements(dist.name)
    if not missing:
        return None
    return (
        f"{dist.name} is installed without its declared requirement(s) {', '.join(missing)} "
        f"(installed with --no-deps?)"
    )


def project_name(pyproject: Path) -> str | None:
    """The ``[project]`` table's ``name`` of *pyproject*, canonical; ``None`` without one. Read by
    pattern, not ``tomllib``, which Python 3.10 lacks."""
    table = re.search(r"^\[project\]\s*$(.*?)(?=^\[|\Z)", pyproject.read_text(), re.M | re.S)
    found = table and re.search(r"^name\s*=\s*[\"']([^\"']+)[\"']", table.group(1), re.M)
    return canonicalize_name(found.group(1)) if found else None


def repository_distributions(root: Path) -> frozenset[str]:
    """The distributions a repository ships, canonical: each top-level package's ``pyproject.toml``
    and each ROS package's ``package.xml`` under ``ros2_ws/src``."""
    names = set()
    for pyproject in root.glob("*/pyproject.toml"):
        name = project_name(pyproject)
        if name is None:
            raise ValueError(f"{pyproject}: no [project] name could be read")
        names.add(name)
    for xml in root.glob("ros2_ws/src/*/package.xml"):
        found = re.search(r"<name>\s*([^<\s]+)\s*</name>", xml.read_text())
        if found:
            names.add(canonicalize_name(found.group(1)))
    return frozenset(n for n in names if n)


#: This repository's root: the directory above ``roqsim/tests``.
ROOT = Path(__file__).resolve().parents[2]


@functools.cache
def _ours() -> frozenset[str]:
    return repository_distributions(ROOT)


def ours(ep) -> bool:
    """Whether entry point *ep* belongs to one of this repository's distributions; one with no
    distribution (registered by hand) counts as this repository's."""
    dist = getattr(ep, "dist", None)
    return dist is None or canonicalize_name(dist.name) in _ours()
