# Copyright (C) 2026 Frederik Pasch
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""Which commit this roqsim was built from.

The version (``0.1.0``) names a release line, not a build: every image built from ``main`` between
two releases carries the same one, so two images that behave differently cannot be told apart by it.
The build identity is the git commit the installed sources came from, plus whether the tree had
uncommitted changes to tracked files at the time (``-dirty``).

Where it comes from, in order:

1. **Baked at build.** A regular (non-editable) install writes :data:`BAKED_FILE` into the
   installed package (``roqsim/setup.py``). The commit is taken from :data:`ENV_VAR` when set --
   which is how a container build passes it, since ``.git`` is not part of the build context -- and
   otherwise from ``git`` in the source tree. When neither can say, the build **fails**: an
   install that cannot name its commit is refused rather than stamped with a placeholder.
2. **Read from the checkout.** An editable install runs the working tree itself, so the commit is
   asked of ``git`` each time and follows the tree as it moves.

Anything else -- a package directory with neither -- reports the identity as absent, with the reason,
rather than a guess.

Standard library only at import time: ``setup.py`` loads this file by path at build time, before
the package or any of its dependencies exist.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

#: The file a regular install carries beside this module. Never present in a source tree: the build
#: writes it into the build directory, so a checkout is always read through ``git``.
BAKED_FILE = "_build_identity.json"

#: The build's explicit commit, as ``<40 hex>`` or ``<40 hex>-dirty``. A container build passes it
#: as a build argument of the same name.
ENV_VAR = "ROQSIM_GIT_SHA"

_BUILD_RE = re.compile(r"[0-9a-f]{40}(-dirty)?")
_PACKAGE_DIR = Path(__file__).resolve().parent


class BuildIdentityError(RuntimeError):
    """The commit a build needs cannot be determined, or what was supplied is not one."""


@dataclass(frozen=True)
class BuildIdentity:
    """One build: the commit, whether the tree was dirty, and how this process learned it."""

    commit: str
    dirty: bool
    #: ``"baked"`` (written by the build) or ``"checkout"`` (asked of git in the source tree).
    source: str

    @property
    def build(self) -> str:
        """``<commit>`` or ``<commit>-dirty`` -- the one-word form ``roqsim --version`` prints."""
        return f"{self.commit}-dirty" if self.dirty else self.commit


def parse_build(text: str, *, origin: str) -> tuple[str, bool]:
    """``(commit, dirty)`` from ``<40 hex>[-dirty]``; refuses anything else, naming *origin*.

    A full commit and not an abbreviation: an abbreviated one is ambiguous as a repository grows,
    and this value is recorded next to results for as long as they are kept.
    """
    value = (text or "").strip()
    if not _BUILD_RE.fullmatch(value):
        raise BuildIdentityError(
            f"{origin} is {value!r}, which is not a build identity: expected the full 40-character "
            "lowercase commit, optionally followed by '-dirty'"
        )
    return value[:40], value.endswith("-dirty")


def _git(source_dir: Path, *args: str) -> str | None:
    """``git -C source_dir <args>``'s stdout, or ``None`` where git cannot answer there."""
    try:
        result = subprocess.run(
            ["git", "-C", str(source_dir), *args],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout


def from_checkout(source_dir: Path, *, tracked: str = "") -> tuple[str, bool] | None:
    """``(commit, dirty)`` of the git work tree holding *source_dir*, or ``None`` if there is none.

    Dirty means a tracked file differs from the commit. Untracked files do not count: a build
    directory or a cache inside the tree says nothing about the sources that were installed.

    With *tracked*, that file in *source_dir* must also be tracked by the work tree. Being *inside*
    a work tree is not enough to be its sources: a virtualenv kept in a repository sits inside that
    repository's tree, and a package installed there would otherwise report its commit.
    """
    if tracked and _git(source_dir, "ls-files", "--error-unmatch", tracked) is None:
        return None
    head = _git(source_dir, "rev-parse", "--verify", "HEAD")
    if head is None:
        return None
    status = _git(source_dir, "status", "--porcelain", "--untracked-files=no")
    if status is None:
        return None
    commit, _ = parse_build(head, origin=f"git rev-parse HEAD in {source_dir}")
    return commit, bool(status.strip())


def for_build(source_dir: Path) -> dict:
    """The record a build bakes into the package, or raise :class:`BuildIdentityError`.

    :data:`ENV_VAR` wins when set, because the caller stated it; a malformed value is refused rather
    than passed over for git. Without it, git in *source_dir*. Neither is a failed build.
    """
    explicit = os.environ.get(ENV_VAR, "").strip()
    if explicit:
        commit, dirty = parse_build(explicit, origin=ENV_VAR)
    else:
        found = from_checkout(source_dir)
        if found is None:
            raise BuildIdentityError(
                f"cannot determine which commit {source_dir} is: it is not inside a git work tree "
                f"that git can read, and {ENV_VAR} is not set. Set {ENV_VAR} to the full commit of "
                "these sources (a container build passes it as --build-arg "
                f"{ENV_VAR}=$(git rev-parse HEAD)); a build that cannot name its commit is refused "
                "rather than recorded as unknown."
            )
        commit, dirty = found
    return {"commit": commit, "dirty": dirty}


def build_identity() -> BuildIdentity | None:
    """This installation's build identity, or ``None`` when it has none (see :func:`describe`)."""
    baked = _PACKAGE_DIR / BAKED_FILE
    if baked.is_file():
        record = json.loads(baked.read_text(encoding="utf-8"))
        text = record["commit"] + ("-dirty" if record.get("dirty") else "")
        commit, dirty = parse_build(text, origin=str(baked))
        return BuildIdentity(commit=commit, dirty=dirty, source="baked")
    found = from_checkout(_PACKAGE_DIR, tracked="__init__.py")
    if found is not None:
        return BuildIdentity(commit=found[0], dirty=found[1], source="checkout")
    return None


def _absent_reason() -> str:
    return (
        f"{_PACKAGE_DIR} carries no {BAKED_FILE} and is not a git checkout of roqsim -- it was not "
        "installed by roqsim's own build"
    )


def _version() -> str:
    # Imported here, not at the top: setup.py loads this file by path, where no package exists.
    import roqsim

    return roqsim.__version__


def describe() -> dict:
    """The version and build identity as JSON-ready fields.

    ``{"version", "commit", "dirty", "source"}`` when the identity is known, and
    ``{"version", "commit": None, "reason"}`` when it is not -- absent is stated, never filled in.
    """
    identity = build_identity()
    if identity is None:
        return {"version": _version(), "commit": None, "reason": _absent_reason()}
    return {
        "version": _version(),
        "commit": identity.commit,
        "dirty": identity.dirty,
        "source": identity.source,
    }


def version_line() -> str:
    """What ``roqsim --version`` prints: one line, stable enough to be read by a program.

    ``roqsim, version <v>, build <commit>[-dirty]``, or ``roqsim, version <v>, build not recorded
    (<reason>)``.
    """
    identity = build_identity()
    build = identity.build if identity is not None else f"not recorded ({_absent_reason()})"
    return f"roqsim, version {_version()}, build {build}"
