#!/usr/bin/env python3
"""Fetch, or check, the Franka upstream files the simulated FR3 is built from and checked against.

The simulated Franka Research 3 (``franka_fci``, the ``fr3`` model) is held to Franka's own
releases by tests that read Franka's files, never a copy typed from them. Those files, and the
ones the model's converter and the ``franka`` modules cite, are vendored
under ``roqsim_manipulation/tests/upstream/franka/``, one directory per repository at the pinned
commit, each with its ``LICENSE`` (all Apache-2.0), and listed with their sha256 in ``SOURCES.toml``
beside them. ``make test`` then needs no network.

    make franka-upstream-check   # every listed file present, with its hash (offline)
    make franka-upstream         # re-fetch every file at the commits SOURCES.toml pins

Like ``external_resources.py`` beside it, it belongs to the roqsim checkout, not to an installed
package: what it reads and writes exists only in a checkout.

Moving a pin is: edit the commit in ``SOURCES.toml``, run this without ``--check``, and fix what
the tests then find, in one pull request.

**The generated URDF.** ``franka_description`` ships xacro, not URDF. ``fr3.urdf`` beside the other
files is what ``robots/fr3/fr3.urdf.xacro`` produces with ``hand:=false`` (the arm without the Franka
Hand), generated here with the ``xacro`` package. xacro resolves
``$(find franka_description)`` through ``ament_index_python``, which exists only in a ROS install,
so the tool answers that one lookup itself with the checkout's path. xacro heads the file with a
comment naming the path it read, a temporary directory; that comment is dropped, so a refresh at
the same pin writes the same bytes. A generated file with ``serve`` set is also written there,
relative to the roqsim checkout: the URDF the simulated robot serves to its client library.

**The visual digest.** The COLLADA visuals are too large to vendor, so ``fr3_visual.json`` holds
what the tests compare the converted meshes against, read from each DAE primitive with pycollada:
its material and colours, its triangle count, its bounds, its open edges and its area-weighted
normal.

Refreshing needs ``git``, and ``pip install xacro pycollada``.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import re
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
    for gen in sources.get("generated", []) + sources.get("digest", []):
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


@contextlib.contextmanager
def _checkout(repo: dict):
    """A shallow checkout of ``repo`` at its pinned commit, in a temporary directory."""
    with tempfile.TemporaryDirectory() as tmp:
        checkout = Path(tmp) / repo["dir"]
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
        yield checkout


#: xacro's banner names the temporary path it read; dropping it keeps a refresh byte-identical.
_XACRO_BANNER = re.compile(
    r"<!-- =+ -->\n<!-- \|    This document was autogenerated by xacro.*?<!-- =+ -->\n", re.S
)


def _generate_urdf(checkout: Path, gen: dict, out: Path) -> None:
    import xacro  # refresh-time dependency only

    # xacro's $(find pkg) asks ament_index_python, a ROS-only module: answer the one lookup.
    packages = types.ModuleType("ament_index_python.packages")
    shares = {"franka_description": str(checkout)}
    packages.get_package_share_directory = lambda name: shares[name]
    index = types.ModuleType("ament_index_python")
    index.packages = packages
    sys.modules.setdefault("ament_index_python", index)
    sys.modules["ament_index_python.packages"] = packages
    doc = xacro.process_file(str(checkout / gen["xacro"]), mappings=dict(gen["mappings"]))
    text = _XACRO_BANNER.sub("", doc.toprettyxml(indent="  "), count=1)
    if "autogenerated by xacro" in text:
        raise SystemExit(f"franka-upstream: xacro's banner in {gen['file']} was not recognised")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")


def visual_digest(dae: Path) -> list[dict]:
    """Each primitive of a COLLADA file, one material each, in file order: what the tests hold the
    converted visual meshes to."""
    import collada  # refresh-time dependency only
    import numpy as np

    out = []
    for geom in collada.Collada(str(dae)).scene.objects("geometry"):
        for prim in geom.primitives():
            tris = prim.triangleset() if hasattr(prim, "triangleset") else prim
            corners = np.asarray(tris.vertex, float)[np.asarray(tris.vertex_index)].reshape(-1, 3)
            _, faces = np.unique(corners, axis=0, return_inverse=True)
            faces = faces.reshape(-1, 3)
            edges = np.sort(
                np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), 1
            )
            _, uses = np.unique(edges, axis=0, return_counts=True)
            tri = corners.reshape(-1, 3, 3)
            normal = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]).sum(0) / 2
            effect = prim.material.effect
            out.append(
                {
                    "material": prim.material.id.removesuffix("-material"),
                    "diffuse": [round(float(c), 7) for c in effect.diffuse],
                    "emission": [round(float(c), 7) for c in effect.emission],
                    "reflectivity": None
                    if effect.reflectivity is None
                    else round(float(effect.reflectivity), 7),
                    "triangles": len(faces),
                    "bounds": [
                        [round(float(c), 7) for c in corners.min(0)],
                        [round(float(c), 7) for c in corners.max(0)],
                    ],
                    "open_edges": int((uses == 1).sum()),
                    "area_normal": [round(float(c), 9) for c in normal],
                }
            )
    return out


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
    derived = sources.get("generated", []) + sources.get("digest", [])
    for d in sorted({g["dir"] for g in derived}):
        with _checkout(by_dir[d]) as checkout:
            for gen in sources.get("generated", []):
                if gen["dir"] != d:
                    continue
                dest = root / d / gen["file"]
                _generate_urdf(checkout, gen, dest)
                if "serve" in gen:
                    roqsim = root.parents[len(UPSTREAM.parts) - 1]
                    if roqsim / UPSTREAM != root:
                        raise SystemExit(
                            f"franka-upstream: {root} is not {UPSTREAM} of a roqsim checkout, so "
                            f"there is no {gen['serve']} to write"
                        )
                    served = roqsim / gen["serve"]
                    served.write_bytes(dest.read_bytes())
                    print(f"served {gen['serve']}")
                hashes[f"{d}/{gen['file']}"] = sha256(dest)
                print(f"generated {d}/{gen['file']}")
            for dig in sources.get("digest", []):
                if dig["dir"] != d:
                    continue
                dest = root / d / dig["file"]
                digest = {Path(m).stem: visual_digest(checkout / m) for m in dig["meshes"]}
                dest.write_text(json.dumps(digest, indent=1) + "\n", encoding="utf-8")
                hashes[f"{d}/{dig['file']}"] = sha256(dest)
                print(f"digested {d}/{dig['file']}")
    _write_hashes(root, hashes)


def _write_hashes(root: Path, hashes: dict[str, str]) -> None:
    path = root / "SOURCES.toml"
    text = path.read_text(encoding="utf-8")
    head = text.split("\n[sha256]\n")[0].rstrip("\n")
    lines = [f'"{k}" = "{v}"' for k, v in sorted(hashes.items())]
    path.write_text(head + "\n\n[sha256]\n" + "\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="franka_upstream.py", description=__doc__.split("\n")[0])
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
