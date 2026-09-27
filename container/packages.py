"""The roqsim packages an image installs: every package in this repository, minus named exclusions.

A package is found the way the Makefile's ``PKGS`` finds it -- a top-level ``roqsim*`` or
``scenario_execution_*`` directory with a ``pyproject.toml`` -- so a new package goes into both
images with no edit here. Leaving one out takes an entry in ``EXCLUDE`` with its reason.

Usage (from the repository root, or the directory the sources were copied to)::

    python3 container/packages.py lean|ros           # the package directories to install (./<dir>)
    python3 container/packages.py torch              # the package directories that require torch
    python3 container/packages.py lean|ros --check   # exit 1 unless exactly those are installed

Standard library only (3.11+, for tomllib): it runs in the image before anything of ours is installed.
"""

from __future__ import annotations

import importlib.metadata as md
import re
import sys
from pathlib import Path

import tomllib

ROOT = Path(__file__).resolve().parent.parent
PATTERNS = ("roqsim*/pyproject.toml", "scenario_execution_*/pyproject.toml")

TORCH = "requires torch; the lean image adds it only with INCLUDE_TORCH_PKGS=1"
GUI = "its tools are tkinter windows a person answers in; the images are headless"

EXCLUDE: dict[str, dict[str, str]] = {
    "lean": {
        "roqsim_humanoid": TORCH,
        "roqsim_quadruped": TORCH,
        "roqsim_scene_builder": GUI,
    },
    "ros": {
        "roqsim_scene_builder": GUI,
    },
}

_NAME = re.compile(r"^[A-Za-z0-9._-]+")


def discover(root: Path = ROOT) -> dict[str, dict]:
    """Every package directory in *root*, mapped to its ``[project]`` table."""
    found = {}
    for pattern in PATTERNS:
        for pyproject in root.glob(pattern):
            found[pyproject.parent.name] = tomllib.loads(pyproject.read_text())["project"]
    return dict(sorted(found.items()))


def requires(project: dict, dist: str) -> bool:
    """Whether *project* lists *dist* among its unconditional dependencies."""
    return any(_NAME.match(req.strip()).group(0) == dist for req in project.get("dependencies", []))


def _normalize(dist: str) -> str:
    return re.sub(r"[-_.]+", "-", dist).lower()


def for_image(image: str, root: Path = ROOT) -> list[str]:
    packages = discover(root)
    unknown = set(EXCLUDE[image]) - set(packages)
    if unknown:
        raise SystemExit(
            f"EXCLUDE[{image!r}] names no package in this repository: {sorted(unknown)}"
        )
    return [name for name in packages if name not in EXCLUDE[image]]


def check(image: str) -> int:
    """Compare the roqsim distributions installed here with what *image* should carry."""
    projects = discover()
    expected = {_normalize(projects[name]["name"]) for name in for_image(image)}
    ours = {_normalize(project["name"]) for project in projects.values()}
    installed = {_normalize(d.metadata["Name"] or "") for d in md.distributions()} & ours
    if installed != expected:
        print(
            f"ERROR: the {image} image's roqsim packages are not the repository's minus its "
            f"exclusions.\n  missing: {sorted(expected - installed)}\n"
            f"  unexpected: {sorted(installed - expected)}",
            file=sys.stderr,
        )
        return 1
    print(f"the {image} image carries all {len(expected)} of its roqsim packages")
    return 0


def main(argv: list[str]) -> int:
    if not argv or argv[0] not in (*EXCLUDE, "torch"):
        print(__doc__, file=sys.stderr)
        return 2
    if argv[0] == "torch":
        print("\n".join(f"./{name}" for name, p in discover().items() if requires(p, "torch")))
        return 0
    if argv[1:] == ["--check"]:
        return check(argv[0])
    print("\n".join(f"./{name}" for name in for_image(argv[0])))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
