"""Fail the image build if any roqsim package's declared runtime dependency is not installed.

The images install our packages with ``--no-deps`` on purpose: third-party wheels go into an earlier,
stable layer so editing our code rebuilds only a cheap tail. The cost is that the third-party list is
maintained by hand, one copy per Dockerfile, and nothing noticed when it fell behind a pyproject.
It did: ``click`` was missing from both images, which is the entrypoint of the lean one, so
``docker run <image> --help`` -- its own CMD -- died with ModuleNotFoundError. ``requests``, ``scipy``
and ``onnxruntime`` were missing too.

So the list stays hand-written and this makes the drift loud at build time instead. Only our own
distributions are inspected, which keeps the check away from the system packages a ROS base image
brings and their pre-existing pip complaints.
"""

from __future__ import annotations

import importlib.metadata as md
import re
import sys

OURS = ("roqsim", "scenario_execution_roqsim")
# A requirement string down to its distribution name: "opencv-python-headless>=4.6" -> the name.
NAME_RE = re.compile(r"^[A-Za-z0-9._-]+")


def main() -> int:
    missing: set[str] = set()
    checked = 0
    for dist in md.distributions():
        name = (dist.metadata["Name"] or "").replace("-", "_")
        if not name.startswith(OURS):
            continue
        checked += 1
        for req in dist.requires or []:
            # Extras are opt-in by definition; only the unconditional requirements are the image's
            # problem. A marker mentioning `extra` is what pyproject's optional-dependencies become.
            requirement, _, marker = req.partition(";")
            if "extra" in marker:
                continue
            match = NAME_RE.match(requirement.strip())
            if not match:
                continue
            dep = match.group(0)
            try:
                md.version(dep)
            except md.PackageNotFoundError:
                missing.add(f"{name} requires {dep}, which is not installed")

    if not checked:
        print("ERROR: no roqsim distribution is installed -- this check ran too early", file=sys.stderr)
        return 1
    if missing:
        print(
            "ERROR: the image's third-party list has fallen behind a pyproject:\n  "
            + "\n  ".join(sorted(missing))
            + "\n\nAdd them to the stable pip layer in the Dockerfile.",
            file=sys.stderr,
        )
        return 1
    print(f"all runtime dependencies of {checked} roqsim distributions are satisfied")
    return 0


if __name__ == "__main__":
    sys.exit(main())
