"""Fetch, or check, the Franka upstream files the ``test_franka*`` tests read.

The simulated Franka Research 3 (the ``fr3`` model) is held to Franka's own
releases by tests that read Franka's files, never a copy typed from them. Those files are vendored
under ``roqsim_manipulation/tests/upstream/franka/``, one directory per repository at the pinned
commit, each with its ``LICENSE`` (all Apache-2.0), and listed with their sha256 in ``SOURCES.toml``
beside them. ``make test`` then needs no network.

    roqsim arm franka-upstream --check      # every listed file present, with its hash (offline)
    roqsim arm franka-upstream              # re-fetch every file at the commits SOURCES.toml pins

Moving a pin is: edit the commit in ``SOURCES.toml``, run this without ``--check``, and fix what
the tests then find, in one pull request.

**The generated URDF.** ``franka_description`` ships xacro, not URDF. ``fr3.urdf`` beside the other
files is what ``robots/fr3/fr3.urdf.xacro`` produces with ``hand:=false`` (the arm without the Franka
Hand), generated here with the ``xacro`` package. xacro resolves
``$(find franka_description)`` through ``ament_index_python``, which exists only in a ROS install,
so the tool answers that one lookup itself with the checkout's path. That needs ``pip install xacro``
and ``git``, at refresh time only.
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
import tempfile
import tomllib
import types
import urllib.request
from pathlib import Path

#: Where the vendored files live, relative to the roqsim checkout.
UPSTREAM = Path("roqsim_manipulation") / "tests" / "upstream" / "franka"


def _default_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / UPSTREAM / "SOURCES.toml").is_file():
            return parent / UPSTREAM
    raise SystemExit(
        "franka-upstream: no roqsim checkout above this module holds "
        f"{UPSTREAM}/SOURCES.toml; pass --dir"
    )


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_sources(root: Path) -> dict:
    return tomllib.loads((root / "SOURCES.toml").read_text(encoding="utf-8"))


def listed_files(sources: dict) -> list[tuple[str, str]]:
    """(directory, path) of every file SOURCES.toml lists, the generated ones included."""
    out = []
    for repo in sources["repo"]:
        out += [(repo["dir"], f) for f in repo["files"]]
    for gen in sources.get("generated", []):
        out.append((gen["dir"], gen["file"]))
    return out


def check(root: Path) -> list[str]:
    """What is missing or differs from its listed hash; empty when the vendored set is intact."""
    sources = load_sources(root)
    hashes = sources.get("sha256", {})
    problems = []
    for d, f in listed_files(sources):
        key = f"{d}/{f}"
        path = root / d / f
        if not path.is_file():
            problems.append(f"missing: {key}")
        elif key not in hashes:
            problems.append(f"no hash listed: {key}")
        elif sha256(path) != hashes[key]:
            problems.append(f"hash differs: {key}")
    return problems


def _raw_url(repo: dict, path: str) -> str:
    owner_repo = repo["url"].removeprefix("https://github.com/").removesuffix(".git")
    return f"https://raw.githubusercontent.com/{owner_repo}/{repo['commit']}/{path}"


def _generate_urdf(repo: dict, gen: dict, out: Path) -> None:
    import xacro  # refresh-time dependency only

    with tempfile.TemporaryDirectory() as tmp:
        checkout = Path(tmp) / "franka_description"
        subprocess.run(["git", "init", "-q", str(checkout)], check=True)
        subprocess.run(
            [
                "git",
                "-C",
                str(checkout),
                "fetch",
                "-q",
                "--depth",
                "1",
                repo["url"],
                repo["commit"],
            ],
            check=True,
        )
        subprocess.run(["git", "-C", str(checkout), "checkout", "-q", "FETCH_HEAD"], check=True)
        # xacro's $(find pkg) asks ament_index_python, a ROS-only module: answer the one lookup.
        packages = types.ModuleType("ament_index_python.packages")
        shares = {"franka_description": str(checkout)}
        packages.get_package_share_directory = lambda name: shares[name]
        index = types.ModuleType("ament_index_python")
        index.packages = packages
        sys.modules.setdefault("ament_index_python", index)
        sys.modules["ament_index_python.packages"] = packages
        doc = xacro.process_file(str(checkout / gen["xacro"]), mappings=dict(gen["mappings"]))
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(doc.toprettyxml(indent="  "), encoding="utf-8")


def fetch(root: Path) -> None:
    sources = load_sources(root)
    hashes: dict[str, str] = {}
    by_dir = {repo["dir"]: repo for repo in sources["repo"]}
    for repo in sources["repo"]:
        for f in repo["files"]:
            dest = root / repo["dir"] / f
            dest.parent.mkdir(parents=True, exist_ok=True)
            with urllib.request.urlopen(_raw_url(repo, f), timeout=60) as r:  # noqa: S310
                dest.write_bytes(r.read())
            hashes[f"{repo['dir']}/{f}"] = sha256(dest)
            print(f"fetched {repo['dir']}/{f}")
    for gen in sources.get("generated", []):
        dest = root / gen["dir"] / gen["file"]
        _generate_urdf(by_dir[gen["dir"]], gen, dest)
        hashes[f"{gen['dir']}/{gen['file']}"] = sha256(dest)
        print(f"generated {gen['dir']}/{gen['file']}")
    _write_hashes(root, hashes)


def _write_hashes(root: Path, hashes: dict[str, str]) -> None:
    path = root / "SOURCES.toml"
    text = path.read_text(encoding="utf-8")
    head = text.split("\n[sha256]\n")[0].rstrip("\n")
    lines = [f'"{k}" = "{v}"' for k, v in sorted(hashes.items())]
    path.write_text(head + "\n\n[sha256]\n" + "\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="roqsim arm franka-upstream", description=__doc__.split("\n")[0]
    )
    parser.add_argument("--check", action="store_true", help="verify the vendored files offline")
    parser.add_argument("--dir", type=Path, default=None, help="the vendored directory")
    args = parser.parse_args(argv)
    root = args.dir or _default_root()
    if not args.check:
        fetch(root)
    problems = check(root)
    for p in problems:
        print(p, file=sys.stderr)
    if not problems:
        print(f"franka-upstream: {len(listed_files(load_sources(root)))} files intact")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
