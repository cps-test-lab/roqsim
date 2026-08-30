"""Resolve a pinned upstream source tree for the per-model converters.

Every converter in this directory rebuilds a vendored model from an upstream repository at a pinned
commit. Each used to resolve that source its own way, and both ways were broken: ``build_oli.py``
defaulted to an absolute path inside a long-dead agent session scratchpad (so the documented rebuild
never worked on any machine), and ``build_g2_mjcf.py`` used bare relative filenames (so it only
worked from one CWD). This module is the single answer: sources land in ``external/sources/<name>/``,
pinned by commit, and a converter names what it needs rather than where it happens to sit.

A checkout is reused if it is already at the requested commit, so re-running a converter costs
nothing. Anything unexpected -- no network, a moved commit, a missing subdirectory -- raises with the
command to run by hand, rather than silently falling back to a stale or partial tree: a converter
that quietly builds from the wrong revision produces a model whose port log lies about its provenance.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

# external/sources/, a sibling of external/convert/. Git-ignored: upstream trees are fetched, never
# committed -- only the derived model + meshes are (see THIRD_PARTY.md for the per-source licence).
SOURCES_DIR = Path(__file__).resolve().parents[1] / "sources"


def _git(*args: str, cwd: Path | None = None) -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed:\n{proc.stderr.strip()}")
    return proc.stdout.strip()


def resolve_source(
    name: str,
    url: str,
    commit: str,
    *,
    subdir: str | None = None,
    sparse: str | None = None,
) -> Path:
    """Return a local path to *url* at *commit*, fetching it into ``external/sources/<name>``.

    ``sparse`` limits the checkout to one cone-mode path (the G1 description is a few MB inside a
    repository of hundreds), and ``subdir`` is appended to the returned path. Both are verified to
    exist before returning, so a converter never has to check.

    Raises RuntimeError -- never returns a tree at the wrong revision, and never returns a partial
    one. The message carries the exact clone command so a machine without network access can be
    prepared by hand.
    """
    dest = SOURCES_DIR / name
    manual = f"  git clone {url} {dest}\n  git -C {dest} checkout {commit}" + (
        f"\n  # only '{sparse}' is needed" if sparse else ""
    )

    if dest.exists():
        try:
            head = _git("rev-parse", "HEAD", cwd=dest)
        except RuntimeError as exc:  # present but not a usable checkout
            raise RuntimeError(
                f"{dest} exists but is not a git checkout ({exc}).\n"
                f"Remove it and re-run, or prepare it by hand:\n{manual}"
            ) from exc
        if not head.startswith(commit) and not commit.startswith(head):
            # A tree at the wrong revision is the dangerous case: it builds, and the port log then
            # records a commit the model was not built from. Refuse rather than silently re-point.
            raise RuntimeError(
                f"{dest} is at {head}, but {name} is pinned to {commit}.\n"
                f"Update it deliberately:\n  git -C {dest} fetch origin {commit}\n"
                f"  git -C {dest} checkout {commit}\n"
                f"If the pin itself should change, update it in the converter AND in "
                f"roqsim_humanoid/THIRD_PARTY.md -- the two must not drift."
            )
    else:
        SOURCES_DIR.mkdir(parents=True, exist_ok=True)
        try:
            if sparse:
                _git("clone", "--filter=blob:none", "--no-checkout", url, str(dest))
                _git("sparse-checkout", "init", "--cone", cwd=dest)
                _git("sparse-checkout", "set", sparse, cwd=dest)
            else:
                _git("clone", "--filter=blob:none", "--no-checkout", url, str(dest))
            _git("checkout", commit, cwd=dest)
        except RuntimeError as exc:
            raise RuntimeError(
                f"Could not fetch {name} from {url} at {commit} ({exc}).\n"
                f"With no network access, prepare the source by hand:\n{manual}"
            ) from exc

    # A sparse cone is set when the checkout is CREATED, so the second model converted out of the
    # same repository would otherwise find the tree at the right commit but without its directory --
    # reported as "the pinned commit may predate that path", which is both wrong and hard to act on.
    # Widen the cone instead: sparse-checkout add is idempotent, and one shared checkout serving
    # every model from a monorepo is the point of pinning by name.
    if sparse and not (dest / sparse).exists():
        try:
            _git("sparse-checkout", "add", sparse, cwd=dest)
        except RuntimeError as exc:
            raise RuntimeError(
                f"{name} is checked out without {sparse!r} and the cone could not be widened "
                f"({exc}).\nPrepare it by hand:\n  git -C {dest} sparse-checkout add {sparse}"
            ) from exc

    out = dest / subdir if subdir else dest
    if not out.exists():
        raise RuntimeError(
            f"{name} @ {commit} has no {subdir!r} (looked in {out}).\n"
            f"The pinned commit may predate that path, or the sparse pattern excluded it."
        )
    return out
