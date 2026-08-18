"""Fetch + convert external assets declared in external/external_assets.yaml, and keep .gitignore
in sync. Driven by the ``external-*`` Makefile targets; see that manifest for the schema and rationale
(sources with unclear redistribution terms are regenerated locally, never committed).

Subcommands::

    list                          # show resources, their sources and targets
    fetch     [--resource NAME]   # download/verify sources (manual sources must be placed by hand)
    convert   [--resource NAME]   # fetch, then run each resource's conversion into its targets
    sync-gitignore                # rewrite the managed .gitignore block from the manifest
    add ...                       # append a resource to the manifest and re-sync .gitignore

Paths in the manifest are relative to the roqsim directory (this file's grandparent).
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request
import venv
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent  # the roqsim directory
MANIFEST = ROOT / "external" / "external_assets.yaml"
TOOLVENV = ROOT / "external" / ".venv-tools"
GITIGNORE = ROOT / ".gitignore"

_MARK_BEGIN = "# BEGIN external-resources (managed by external/external_resources.py -- do not edit)"
_MARK_END = "# END external-resources"


def _load() -> list[dict]:
    if not MANIFEST.exists():
        sys.exit(f"manifest not found: {MANIFEST}")
    data = yaml.safe_load(MANIFEST.read_text()) or {}
    return data.get("resources") or []


def _resolve(rel: str) -> Path:
    return (ROOT / rel).resolve()


def _rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _select(resources: list[dict], name: str | None) -> list[dict]:
    if name is None:
        return resources
    chosen = [r for r in resources if r["name"] == name]
    if not chosen:
        sys.exit(f"no such resource {name!r}; known: {', '.join(r['name'] for r in resources)}")
    return chosen


# -- fetch ---------------------------------------------------------------------------------------
def _format_license(text: str) -> str:
    body = "\n".join("      " + ln for ln in text.strip().splitlines())
    return "  license:\n" + body


def _fetch(res: dict, force: bool = False) -> bool:
    """Fetch/verify a resource's sources.

    Returns True if all are present, False if an *optional* resource was skipped (a missing manual
    source or a network error). For a non-optional resource those are hard errors (fail loudly)."""
    optional = bool(res.get("optional"))
    license_shown = False
    for src in res["sources"]:
        dst = _resolve(src["path"])
        if dst.exists() and not force and (not src.get("sha256") or _sha256(dst) == src["sha256"]):
            print(f"  have    {src['path']}")
            continue
        if src.get("manual"):
            msg = (f"manual source {src['path']} missing; obtain it from {src['url']} and place it "
                   f"at that path")
            if optional:
                print(f"  skip (optional): {msg}")
                return False
            sys.exit(f"  MISSING {msg}, then re-run.")
        if res.get("license") and not license_shown:
            print(_format_license(res["license"]))
            license_shown = True
        dst.parent.mkdir(parents=True, exist_ok=True)
        print(f"  fetch   {src['url']} -> {src['path']}")
        try:
            urllib.request.urlretrieve(src["url"], dst)  # noqa: S310 (declared http(s) source)
        except (urllib.error.URLError, OSError) as exc:
            if optional:
                print(f"  skip (optional): could not fetch {src['path']} ({exc})")
                return False
            sys.exit(f"  fetch failed for {src['path']}: {exc}")
        if src.get("sha256") and _sha256(dst) != src["sha256"]:
            sys.exit(f"  sha256 mismatch for {src['path']} (got {_sha256(dst)})")
    return True


# -- convert -------------------------------------------------------------------------------------
def _toolvenv_python() -> Path:
    py = TOOLVENV / "bin" / "python"
    if not py.exists():
        print(f"creating tool venv {_rel(TOOLVENV)}")
        venv.EnvBuilder(with_pip=True).create(TOOLVENV)
        subprocess.run([str(py), "-m", "pip", "install", "-q", "--upgrade", "pip"], check=True)
    return py


def _convert(res: dict, force: bool = False) -> None:
    if not _fetch(res, force):
        return  # optional resource skipped (missing source / offline)
    conv = res.get("convert")
    if not conv:
        print("  (download-only; no conversion)")
        return
    py = _toolvenv_python()
    deps = conv.get("pip_deps") or []
    if deps:
        print(f"  pip install {' '.join(deps)}")
        subprocess.run([str(py), "-m", "pip", "install", "-q", *deps], check=True)
    cmd = [
        str(py), str(_resolve(conv["script"])),
        "--sources", *[str(_resolve(s["path"])) for s in res["sources"]],
        "--targets", *[str(_resolve(t)) for t in res["targets"]],
    ]
    if conv.get("needs_blender"):
        cmd += ["--blender", os.environ.get("BLENDER", "blender")]
    print(f"  convert -> {', '.join(res['targets'])}")
    subprocess.run(cmd, check=True)
    missing = [t for t in res["targets"] if not _resolve(t).exists()]
    if missing:
        sys.exit(f"  conversion did not produce: {', '.join(missing)}")


# -- gitignore -----------------------------------------------------------------------------------
def _sync_gitignore() -> None:
    raw = [f"{_rel(TOOLVENV)}/"]
    for res in _load():
        raw += [s["path"] for s in res["sources"]]
        raw += list(res["targets"])
    entries = list(dict.fromkeys(raw))  # de-dup, preserve order (a download-only target == its source)
    block = "\n".join([_MARK_BEGIN, *entries, _MARK_END]) + "\n"
    text = GITIGNORE.read_text() if GITIGNORE.exists() else ""
    if _MARK_BEGIN in text:
        text = re.sub(
            re.escape(_MARK_BEGIN) + r".*?" + re.escape(_MARK_END) + r"\n?",
            block, text, flags=re.DOTALL,
        )
    else:
        text = (text.rstrip() + "\n\n" if text.strip() else "") + block
    GITIGNORE.write_text(text)
    print(f"synced {len(entries)} path(s) into {_rel(GITIGNORE)}")


# -- add -----------------------------------------------------------------------------------------
def _add(args: argparse.Namespace) -> None:
    resources = _load()
    if any(r["name"] == args.name for r in resources):
        sys.exit(f"resource {args.name!r} already exists in the manifest")
    sources = []
    for spec in args.source:  # "URL::PATH" or "URL::PATH::manual"
        parts = spec.split("::")
        if len(parts) < 2:
            sys.exit(f"--source must be URL::PATH[::manual], got {spec!r}")
        entry = {"url": parts[0], "path": parts[1]}
        if len(parts) > 2 and parts[2] == "manual":
            entry["manual"] = True
        sources.append(entry)
    res: dict = {"name": args.name, "description": args.description}
    if args.license:
        res["license"] = args.license
    res["sources"] = sources
    if args.script:
        conv: dict = {"script": args.script}
        if args.pip_dep:
            conv["pip_deps"] = args.pip_dep
        if args.needs_blender:
            conv["needs_blender"] = True
        res["convert"] = conv
    res["targets"] = args.target
    block = yaml.safe_dump([res], sort_keys=False, default_flow_style=False, width=100)
    with MANIFEST.open("a") as fh:
        fh.write(block if MANIFEST.read_text().endswith("\n") else "\n" + block)
    print(f"appended resource {args.name!r} to {_rel(MANIFEST)}")
    _sync_gitignore()


# -- cli -----------------------------------------------------------------------------------------
def _cmd_list(_: argparse.Namespace) -> None:
    for res in _load():
        print(f"{res['name']}: {res.get('description', '')}")
        for s in res["sources"]:
            tag = " (manual)" if s.get("manual") else ""
            print(f"    source {s['path']}{tag}")
        for t in res["targets"]:
            print(f"    target {t}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list").set_defaults(func=_cmd_list)

    p = sub.add_parser("fetch")
    p.add_argument("--resource")
    p.add_argument("--force", action="store_true", help="re-fetch even if the target already exists")
    p.set_defaults(func=lambda a: [_fetch(r, a.force) for r in _select(_load(), a.resource)])

    p = sub.add_parser("convert")
    p.add_argument("--resource")
    p.add_argument("--force", action="store_true", help="re-fetch + re-convert even if present")
    p.set_defaults(func=lambda a: [_convert(r, a.force) for r in _select(_load(), a.resource)])

    sub.add_parser("sync-gitignore").set_defaults(func=lambda a: _sync_gitignore())

    p = sub.add_parser("add")
    p.add_argument("--name", required=True)
    p.add_argument("--description", default="")
    p.add_argument("--license", default="")
    p.add_argument("--source", action="append", default=[], help="URL::PATH[::manual] (repeatable)")
    p.add_argument("--target", action="append", default=[], help="generated path (repeatable)")
    p.add_argument("--script", help="conversion script (omit for download-only)")
    p.add_argument("--pip-dep", action="append", default=[], help="pip dep for conversion (repeatable)")
    p.add_argument("--needs-blender", action="store_true")
    p.set_defaults(func=_add)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
