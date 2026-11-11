# Copyright (C) 2026 Frederik Pasch
#
# SPDX-License-Identifier: Apache-2.0
"""The package directories ``make test`` runs, of those it is given: each one installed with
everything it declares. Prints them on stdout, one line; names every package left out, and why, on
stderr.

    python tools/test_packages.py roqsim roqsim_humanoid ...

A package not installed, or installed without a requirement it declares (``pip install --no-deps``,
an image without network that could not fetch it), cannot pass its tests for a reason that is the
installation's. Run anyway, its modules fail at collection (the root ``conftest.py``) and the whole
suite stops there; left out, the rest runs. Leaving one out is never silent: each is named with what
it lacks, and the run's last lines say how many were left out, so a run that tested less reads as
such. ``make venv`` installs every package with what it declares.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _requirements():
    spec = importlib.util.spec_from_file_location(
        "_roqsim_installed_requirements", ROOT / "roqsim" / "tests" / "installed_requirements.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def why_not(package: str, req=None) -> str:
    """Why *package*'s tests cannot run here, or ``""`` when they can: the rule the tests over the
    whole tree apply (``installed_requirements.not_testable``)."""
    if not (ROOT / package / "pyproject.toml").is_file():
        raise SystemExit(f"make test: {package} is not a package directory of this repository")
    return (req or _requirements()).not_testable(package)


def main(argv: list[str]) -> int:
    req = _requirements()
    keep, left = [], []
    for package in argv:
        reason = why_not(package, req)
        (left if reason else keep).append((package, reason))
    for package, reason in left:
        print(f"make test: NOT TESTED {package}: {reason} (make venv installs it)", file=sys.stderr)
    if left:
        print(f"make test: {len(left)} of {len(argv)} packages not tested", file=sys.stderr)
    print(" ".join(p for p, _ in keep))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
