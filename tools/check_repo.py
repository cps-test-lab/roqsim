#!/usr/bin/env python3
"""Publication hygiene checks -- what has to stay true for this repo to be publishable.

Run via ``make check``. Every check states a POSITIVE invariant about tracked files, so this script
can live in a public repository without itself becoming a list of things to hide. (The complementary
scrub -- confirming no internal identifier survived -- belongs to whoever prepares a release, not
here, for exactly that reason.)

Checks
------
1. No Git LFS. Neither a ``filter=lfs`` rule in .gitattributes nor a committed pointer file. LFS
   bandwidth is metered even on public repositories and billed to the repo owner, so a pointer that
   sneaks back in becomes a clone failure for everyone once the monthly quota is spent.
2. Every asset folder carries attribution. A CC-BY asset without its CREDITS.txt is a licence
   violation the moment the repo is cloned, and the folders are added by hand, so this cannot be
   left to review.
3. Every vendored robot model carries a licence file next to it.
4. No absolute filesystem paths in tracked text -- they leak the author's machine layout and are
   never reproducible.
5. Every mesh and texture a bundled model references exists on disk. A model composed from a sibling
   prop's assets (the sanctioned intra-package pattern -- humanoid_gantry instances trolley_wheel's
   caster by a ``../`` path) breaks silently the moment the donor moves to another distribution: no
   world spawns it, so neither the test suite nor `make smoke` notices, and the failure surfaces only
   for whoever first spawns it.

   Checked by resolving ``file=`` rather than by compiling: several robot models declare contact pairs
   against the *world's* ``floor`` geom and cannot compile standalone by design, so a compile check
   reports them broken when they are not.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Folders whose contents are third-party and therefore need a CREDITS.txt. Globs are relative to ROOT.
ASSET_FOLDER_GLOBS = (
    "roqsim_assets/src/roqsim_assets/assets/*",
    "roqsim_assets/src/roqsim_assets/models/*",
    "roqsim_walker/src/roqsim_walker/models/people/*",
    "roqsim_scenes/src/roqsim_scenes/scenes/*",
)

# Model directories that vendor a robot description: each needs a licence file beside it. Matched by
# looking for a *.manifest.yaml and requiring a sibling whose name suggests a licence.
MODEL_MANIFEST_GLOBS = (
    "roqsim_mobile/src/roqsim_mobile/models/*/",
    "roqsim_manipulation_assets/src/roqsim_manipulation_assets/models/*/",
    "roqsim_mobile_manipulation/src/roqsim_mobile_manipulation/models/*/",
    "roqsim_sensors/src/roqsim_sensors/models/*/",
)

LICENCE_RE = re.compile(r"licen[cs]e|copyright|CC0|CC-BY|BSD|Apache|MIT|MPL", re.I)
# A path like /home/<user>/... or /Users/<user>/... in tracked text. Deliberately not matching
# /opt/ros or /usr/share, which are legitimate system paths in docs and launch files.
ABS_PATH_RE = re.compile(r"(?:/home/|/Users/)[a-z_][a-z0-9_-]*/", re.I)

TEXT_SUFFIXES = {
    ".py", ".md", ".rst", ".txt", ".yaml", ".yml", ".toml", ".cfg", ".xml", ".json",
    ".osc", ".sh", ".mk", ".ini", ".gitignore", ".gitattributes", "",
}


def tracked_files() -> list[str]:
    out = subprocess.run(
        ["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout
    return [p for p in out.split("\0") if p]


def check_no_lfs(files: list[str]) -> list[str]:
    problems = []
    attrs = ROOT / ".gitattributes"
    if attrs.exists():
        for i, line in enumerate(attrs.read_text().splitlines(), 1):
            if "filter=lfs" in line and not line.lstrip().startswith("#"):
                problems.append(f".gitattributes:{i}: filter=lfs rule -- LFS must stay disabled")
    # A committed LFS pointer is a small text file with this exact first line.
    for rel in files:
        p = ROOT / rel
        try:
            if p.stat().st_size > 300 or not p.is_file():
                continue
            with p.open("rb") as fh:
                if fh.readline().startswith(b"version https://git-lfs.github.com/spec/"):
                    problems.append(f"{rel}: committed Git LFS pointer")
        except OSError:
            continue
    return problems


def check_asset_attribution() -> list[str]:
    problems = []
    for glob in ASSET_FOLDER_GLOBS:
        for d in sorted(ROOT.glob(glob)):
            if not d.is_dir() or d.name in {"meshes", "textures", "__pycache__"}:
                continue
            # A folder with no payload (only subdirs we skip) is not an asset folder.
            if not any(f.is_file() for f in d.iterdir()):
                continue
            credits = d / "CREDITS.txt"
            if not credits.exists():
                problems.append(f"{d.relative_to(ROOT)}: no CREDITS.txt")
            elif not LICENCE_RE.search(credits.read_text(errors="replace")):
                problems.append(f"{credits.relative_to(ROOT)}: no licence named in it")
    return problems


def check_model_licences() -> list[str]:
    problems = []
    for glob in MODEL_MANIFEST_GLOBS:
        for d in sorted(ROOT.glob(glob)):
            if not d.is_dir() or not list(d.glob("*.manifest.yaml")):
                continue
            if not any(LICENCE_RE.search(f.name) for f in d.iterdir() if f.is_file()):
                problems.append(f"{d.relative_to(ROOT)}: vendored model with no licence file")
    return problems


FILE_REF_RE = re.compile(r'\bfile\s*=\s*"([^"]+)"')


def _is_ignored(path: Path) -> bool:
    """True if git deliberately ignores ``path`` -- i.e. its absence is declared, not accidental."""
    try:
        rel = path.resolve()
    except OSError:
        return False
    done = subprocess.run(
        ["git", "check-ignore", "-q", str(rel)], cwd=ROOT, capture_output=True
    )
    return done.returncode == 0


def _provider_roots() -> dict[str, tuple[Path, ...]]:
    """``roqsim.models`` provider name -> its models/meshes/texture dirs, for cross-package borrows."""
    try:
        from roqsim.models import providers
    except Exception:
        return {}
    out = {}
    for name, models_dir, meshdir, texturedir in providers():
        out[name] = (Path(models_dir), Path(meshdir), Path(texturedir))
    return out


def check_model_asset_refs() -> list[str]:
    """Every ``file=`` in a tracked MJCF must resolve, including a ``../`` borrow from a sibling."""
    problems = []
    for rel in tracked_files():
        if not rel.endswith(".xml"):
            continue
        p = ROOT / rel
        try:
            text = p.read_text(errors="strict")
        except (OSError, UnicodeDecodeError):
            continue
        if "<mujoco" not in text:
            continue
        # MuJoCo resolves file= against the <compiler> meshdir/texturedir, so honour whatever the
        # model declares -- props use "." (their own folder), baked scenes use "assets/". The
        # spawn_model loader can add further search dirs, so only flag what resolves nowhere.
        # The model's own folder, plus the package's models/ dir above it: a model may compose from a
        # SIBLING prop in the same package, by `../donor/...` (humanoid_gantry -> trolley_wheel) or by
        # `donor/meshes/...` (gen3 -> robotiq_2f85). Both are the sanctioned intra-package pattern.
        roots = [p.parent, p.parent / "meshes", p.parent / "textures", p.parent.parent]
        for attr in ("meshdir", "texturedir"):
            m = re.search(rf'{attr}\s*=\s*"([^"]+)"', text)
            if m:
                roots.append(p.parent / m.group(1))
        # A model may also borrow ACROSS packages by naming providers in its manifest's `assets:`
        # key (frankie takes the Panda's hand meshes that way). That is the documented cross-package
        # mechanism, so its dirs count as search roots too.
        manifest = p.parent / f"{p.stem}.manifest.yaml"
        if manifest.exists():
            m = re.search(r"^assets:\s*\[([^\]]*)\]", manifest.read_text(), re.M)
            if m:
                for name in (s.strip() for s in m.group(1).split(",") if s.strip()):
                    roots.extend(_provider_roots().get(name, ()))
        for ref in FILE_REF_RE.findall(text):
            if ref.startswith(("http://", "https://")):
                continue
            if any((root / ref).exists() for root in roots):
                continue
            # Some assets are DERIVED from vendor CAD whose redistribution terms are unclear, so they
            # are generated by `make external-resources` and git-ignored (the Livox, Zivid and Seyond
            # sensor meshes). A clean checkout legitimately lacks them, so an ignored target is a
            # declared absence, not a dangling reference. Anything else is a real break.
            if any(_is_ignored(root / ref) for root in roots):
                continue
            problems.append(f"{rel}: references missing asset {ref!r}")
    return problems


def check_no_absolute_paths(files: list[str]) -> list[str]:
    problems = []
    for rel in files:
        p = ROOT / rel
        if p.suffix.lower() not in TEXT_SUFFIXES:
            continue
        try:
            text = p.read_text(errors="strict")
        except (OSError, UnicodeDecodeError):
            continue
        for i, line in enumerate(text.splitlines(), 1):
            if ABS_PATH_RE.search(line):
                problems.append(f"{rel}:{i}: absolute path in tracked text")
    return problems


def main() -> int:
    files = tracked_files()
    groups = [
        ("Git LFS is disabled and no pointer is committed", check_no_lfs(files)),
        ("every third-party asset folder carries attribution", check_asset_attribution()),
        ("every vendored robot model carries a licence file", check_model_licences()),
        ("no absolute filesystem paths in tracked text", check_no_absolute_paths(files)),
        ("every model asset reference resolves", check_model_asset_refs()),
    ]
    failed = False
    for name, problems in groups:
        if problems:
            failed = True
            print(f"FAIL  {name}")
            for p in problems:
                print(f"        {p}")
        else:
            print(f"ok    {name}")
    if failed:
        print("\n`make check` failed -- see above.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
