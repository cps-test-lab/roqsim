# Copyright (C) 2026 Frederik Pasch
# SPDX-License-Identifier: Apache-2.0
"""Whether an installed distribution has what it declares: for the tests that walk every installed one.

A test that loads every ``roqsim.plugins`` entry or every ``roqsim`` command group meets each
installed package, and a package installed without its declared requirements (``pip install
--no-deps``, a partial image) fails its import for a reason that is the installation's, not the
code's. Those tests skip such a package and name the requirement it lacks, so the skip is read in
``-rs`` and nothing is hidden. Only a requirement the distribution itself (or one it requires, in
turn) **declares** counts: an import that fails for any other reason is a defect and still fails.
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
    names = {project_name(p) for p in root.glob("*/pyproject.toml")}
    for xml in root.glob("ros2_ws/src/*/package.xml"):
        found = re.search(r"<name>\s*([^<\s]+)\s*</name>", xml.read_text())
        if found:
            names.add(canonicalize_name(found.group(1)))
    return frozenset(n for n in names if n)
