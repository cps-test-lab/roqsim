"""Sketchfab: search, preview, licence-check and download models for the prop pipeline.

The single home for all Sketchfab interaction in the asset pipeline, exposed as the
``sketchfab_helper`` console command (installed via ``[project.scripts]``) with four subcommands::

    sketchfab_helper search "office chair" [--count 12]   # redistributable + downloadable only
    sketchfab_helper thumbs <uid|url> ...                 # <=12 preview images -> temp dir (path printed)
    sketchfab_helper info   <uid|url> [--out DIR]         # licence verdict + face count (no token)
    sketchfab_helper download <uid|url> [--out DIR] [--token T] [--force]
    sketchfab_helper import <uid|url> --blender <exe> [--preview] [--stage DIR] [--models-dir DIR]

``search`` / ``thumbs`` / ``info`` need no token -- they hit only public endpoints. ``download`` and
``import`` need a Sketchfab API token (``--token``, ``$SKETCHFAB_API_TOKEN``, or a
``SKETCHFAB_API_TOKEN=…`` line in the repo's git-ignored ``.env``; get one at
https://sketchfab.com/settings/password).

``import`` is the one-shot pipeline: it stages the download + reduction + MuJoCo preview in a scratch
dir **outside any asset library**, so you judge the real geometry first; only on your approval does it
copy the finished prop into ``--models-dir`` (default: the ``roqsim_assets`` library; point it at
another package's ``models/`` to import there). ``--preview`` stops after the view, importing nothing.
Reduction needs Blender and the preview needs MuJoCo + a display, so both are reached lazily
(``roqsim assets reduce-mesh`` in a subprocess, ``roqsim render`` in process) -- search / thumbs / info /
download work on a box that has neither.

**Licence matters for redistribution.** Only **CC0 / CC-BY / CC-BY-SA** may be committed to this
(Apache-2.0) repo -- CC-BY* require crediting the author, captured in the ``CREDITS.txt`` written into
the asset folder. Everything else (NC / ND / undeclared) is flagged; ``download`` refuses it unless
``--force`` (local-only use). ``download`` also refuses a model the author never enabled for API
download (``downloadable: false``) -- there is no glTF to fetch and ``--force`` does not bypass it.

``search`` applies both gates up front (server-side licence filter + ``downloadable=true``) so no
un-importable candidate is ever surfaced; the slug it prints is the filter value, hence authoritative
(the search payload's licence object carries only a label). Also reachable as ``roqsim assets
sketchfab-helper``; ``tools/sketchfab_helper.py`` is a run-from-the-folder wrapper onto this module.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
import zipfile

_API = "https://api.sketchfab.com/v3/models"
_SEARCH = "https://api.sketchfab.com/v3/search"

# Licence slugs safe to redistribute in this permissive repo. _OPEN_LICENSES is the search display
# order (CC0 first: no attribution burden); _REDISTRIBUTABLE is the membership test the download gate
# uses. One definition, used by both -- no drift.
_OPEN_LICENSES = ["cc0", "by", "by-sa"]
_REDISTRIBUTABLE = set(_OPEN_LICENSES)

_MAX_THUMBS = (
    12  # a preview grid a human can actually eyeball at once; also caps accidental fan-out
)


def _load_dotenv() -> None:
    """Populate os.environ from the nearest ``.env`` (walking up to the repo root), without clobbering.

    Lets ``$SKETCHFAB_API_TOKEN`` live in the git-ignored repo ``.env`` instead of the shell, so the
    token never has to be pasted on the command line. A value already in the environment wins, so an
    explicit ``export`` or ``--token`` still overrides the file. Pure stdlib KEY=VALUE parser (no
    ``export`` prefixes, no interpolation). Missing file is a silent no-op.
    """
    d = os.path.dirname(os.path.abspath(__file__))
    while True:
        env = os.path.join(d, ".env")
        if os.path.isfile(env):
            with open(env, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, _, val = line.partition("=")
                    key = key.strip()
                    val = val.strip().strip('"').strip("'")
                    if key and key not in os.environ:
                        os.environ[key] = val
            return
        parent = os.path.dirname(d)
        if parent == d:  # reached filesystem root without finding a .env
            return
        d = parent


# Load the repo .env before argparse evaluates its os.environ.get(...) token default.
_load_dotenv()


def _get(url: str, token: str | None = None) -> bytes:
    """GET raw bytes. Search / metadata / thumbnails are public; only the download endpoint needs a token."""
    req = urllib.request.Request(url)
    if token:
        req.add_header("Authorization", f"Token {token}")
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return resp.read()
    except urllib.error.HTTPError as exc:
        sys.exit(f"HTTP {exc.code} for {url}: {exc.read().decode(errors='replace')[:200]}")


def _uid(model: str) -> str:
    """Extract the 32-hex model uid from a Sketchfab URL, or accept a bare uid."""
    m = re.search(r"[0-9a-f]{32}", model.lower())
    if not m:
        sys.exit(f"could not find a 32-hex model uid in {model!r}")
    return m.group(0)


def _metadata(uid: str) -> dict:
    return json.loads(_get(f"{_API}/{uid}"))


def _license_verdict(slug: str | None) -> tuple[bool, str]:
    if not slug:
        return False, "no licence declared -- do NOT redistribute without checking with the author"
    if slug == "cc0":
        return True, "CC0: public domain, no conditions -- safe to redistribute"
    if slug in _REDISTRIBUTABLE:
        return True, f"{slug}: redistributable WITH attribution (see CREDITS.txt)"
    return False, f"{slug}: non-commercial or no-derivatives -- NOT recommended for the repo"


def _write_credit(out_dir: str, meta: dict) -> None:
    """Write the model's licence + attribution into the asset folder (the per-folder CREDITS.txt)."""
    lic = meta.get("license") or {}
    user = meta.get("user") or {}
    with open(os.path.join(out_dir, "CREDITS.txt"), "w") as fh:
        fh.write(
            f'"{meta.get("name")}" by {user.get("username")} ({user.get("profileUrl")})\n'
            f"licensed under {lic.get('label')} ({lic.get('slug')}) -- {lic.get('requirements')}\n"
            f"source: {meta.get('viewerUrl')}\n"
        )


def _download_gltf(uid: str, token: str, out_dir: str) -> str:
    info = json.loads(_get(f"{_API}/{uid}/download", token=token))
    if "gltf" not in info:
        sys.exit(f"no glTF download offered for this model (got: {sorted(info)})")
    data = _get(info["gltf"]["url"])  # signed URL -> a zip archive
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        zf.extractall(out_dir)
    # Sketchfab bundles its own license.txt; our CREDITS.txt supersedes it -- drop the duplicate.
    for dp, _, fs in os.walk(out_dir):
        for f in fs:
            if f.lower() == "license.txt":
                os.remove(os.path.join(dp, f))
    gltfs = [
        os.path.join(dp, f)
        for dp, _, fs in os.walk(out_dir)
        for f in fs
        if f.lower().endswith((".gltf", ".glb"))
    ]
    if not gltfs:
        sys.exit("download extracted but no .gltf/.glb found inside")
    return sorted(gltfs)[0]


def _search_one(query: str, license_slug: str, count: int) -> list[dict]:
    params = urllib.parse.urlencode(
        {
            "type": "models",
            "q": query,
            "downloadable": "true",
            "license": license_slug,
            "count": count,
            "sort_by": "-likeCount",  # surface the well-liked results first, not the newest upload
        }
    )
    return json.loads(_get(f"{_SEARCH}?{params}"))["results"]


def _search(query: str, count: int) -> list[dict]:
    """Redistributable + downloadable models for a query, best-liked first, deduped across licences."""
    seen: set[str] = set()
    merged: list[dict] = []
    # Sketchfab search takes ONE licence slug per request, so fetch each redistributable slug and merge.
    # Over-fetch per licence (count each) then trim, so a query dominated by one licence still fills up.
    for slug in _OPEN_LICENSES:
        for r in _search_one(query, slug, count):
            uid = r.get("uid")
            if uid and uid not in seen:
                seen.add(uid)
                r["_license_slug"] = (
                    slug  # authoritative: it is the filter value, not the label-only payload
                )
                merged.append(r)
    return merged[:count]


def _pick_thumb(images: list[dict], target_w: int = 1024) -> str | None:
    """URL of the preview closest to ``target_w`` wide without going huge -- big enough to judge, not 1080p."""
    usable = [im for im in images if im.get("url")]
    if not usable:
        return None
    small = [im for im in usable if (im.get("width") or 0) <= target_w]
    if small:
        return max(small, key=lambda im: im.get("width") or 0)["url"]
    return min(usable, key=lambda im: im.get("width") or 0)["url"]


def _print_meta(meta: dict) -> bool:
    """Print name / author / licence / face count / downloadability; return whether the licence is OK."""
    lic = meta.get("license") or {}
    ok, verdict = _license_verdict(lic.get("slug"))
    print(f"name:        {meta.get('name')}")
    print(f"author:      {(meta.get('user') or {}).get('username')}")
    print(f"license:     {lic.get('label')} ({lic.get('slug')})")
    print(f"requires:    {lic.get('requirements')}")
    print(f"faces:       {meta.get('faceCount')}   downloadable: {meta.get('isDownloadable')}")
    print(f"LICENCE:     {'OK  ' if ok else 'CHECK'} -- {verdict}")
    return ok


def cmd_search(args: argparse.Namespace) -> None:
    results = _search(args.query, args.count)
    if not results:
        print(f"no redistributable, downloadable models found for {args.query!r}")
        return
    print(
        f"{len(results)} redistributable + downloadable model(s) for {args.query!r} "
        f"(licence in {'/'.join(_OPEN_LICENSES)}):\n"
    )
    for r in results:
        faces = r.get("faceCount")
        faces_s = f"{faces:,}f" if isinstance(faces, int) else "?f"
        print(
            f"  {r.get('uid')}  [{r['_license_slug']:<5}] {faces_s:>10}  "
            f"{(r.get('name') or '').strip()[:50]}"
        )
        print(f"      by {(r.get('user') or {}).get('username')}  {r.get('viewerUrl')}")
    print(
        f"\nnext: sketchfab_helper thumbs {' '.join(r.get('uid') for r in results[:_MAX_THUMBS])}"
    )


def cmd_thumbs(args: argparse.Namespace) -> None:
    if len(args.models) > _MAX_THUMBS:
        sys.exit(f"at most {_MAX_THUMBS} models at a time (got {len(args.models)})")
    uids = [_uid(m) for m in args.models]
    out_dir = tempfile.mkdtemp(prefix="sketchfab_thumbs_")
    for uid in uids:
        meta = _metadata(uid)
        url = _pick_thumb((meta.get("thumbnails") or {}).get("images") or [])
        if not url:
            print(f"  {uid}  no thumbnail available", file=sys.stderr)
            continue
        ext = os.path.splitext(urllib.parse.urlparse(url).path)[1] or ".jpeg"
        dest = os.path.join(out_dir, f"{uid}{ext}")
        with open(dest, "wb") as fh:
            fh.write(_get(url))
        print(f"  {uid}  {(meta.get('name') or '').strip()[:50]:<52}  -> {os.path.basename(dest)}")
    # Last line is the temp dir alone, so a caller can capture it (e.g. to feed a media-review grid).
    print(out_dir)


def cmd_info(args: argparse.Namespace) -> None:
    meta = _metadata(_uid(args.model))
    _print_meta(meta)
    # Only touch the disk when asked -- a bare preflight check should leave nothing behind.
    if args.out:
        os.makedirs(args.out, exist_ok=True)
        _write_credit(args.out, meta)
        print(f"wrote attribution -> {os.path.join(args.out, 'CREDITS.txt')}")


def cmd_download(args: argparse.Namespace) -> None:
    uid = _uid(args.model)
    meta = _metadata(uid)
    ok = _print_meta(meta)
    # Confirm the licence permits committing to the repo BEFORE writing anything or downloading.
    if not ok and not args.force:
        sys.exit(
            "licence does not permit committing to the open-source repo (need CC0 / CC-BY / "
            "CC-BY-SA); re-run with --force for local-only use"
        )
    if not meta.get("isDownloadable"):
        sys.exit("model is not downloadable on Sketchfab")
    if not args.token:
        sys.exit("download needs a Sketchfab API token (--token or $SKETCHFAB_API_TOKEN)")

    os.makedirs(args.out, exist_ok=True)
    _write_credit(args.out, meta)
    print(f"wrote attribution -> {os.path.join(args.out, 'CREDITS.txt')}")
    path = _download_gltf(uid, args.token, args.out)
    print(f"downloaded glTF -> {path}")
    print(f"next: roqsim assets reduce-mesh --blender <path> {path} out.obj --target-faces 20000")


# --- import: the one-shot download -> reduce -> view -> finalize -> import-on-approval pipeline ------
_DEFAULT_MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") or "model"


def _yes(prompt: str) -> bool:
    return input(prompt).strip().lower() in ("", "y", "yes")


def _cleanup_intermediate(out_dir: str) -> None:
    """Delete the downloaded glTF source (kept only to build the OBJ) once the reduction is accepted."""
    for dp, _, fs in os.walk(out_dir):
        for f in fs:
            if f.lower().endswith((".gltf", ".glb", ".bin")):
                os.remove(os.path.join(dp, f))


def _run(*cmd: str) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)


def cmd_import(args: argparse.Namespace) -> None:
    # Blender + MuJoCo tooling is needed only here, so import it lazily: search / thumbs / info /
    # download stay usable on a box that has neither.
    from roqsim.render import main as render_main
    from roqsim_assets.cli import finalize_mujoco, reduce_mesh

    # Fail fast if Blender isn't available (before any network) -- reduction needs it.
    blender = reduce_mesh.blender_exe(args.blender)
    if blender is None:
        sys.exit(
            f"Blender not available as {args.blender!r} -- put it on PATH as 'blender' or pass "
            f"--blender /path/to/blender"
        )

    uid = _uid(args.model)
    meta = _metadata(uid)
    ok = _print_meta(meta)
    name = args.name or _slug(meta.get("name") or uid)
    if not ok and not args.force:
        sys.exit(
            "licence does not permit committing to the open-source repo (need CC0 / CC-BY / "
            "CC-BY-SA); re-run with --force for local-only use"
        )
    if not meta.get("isDownloadable"):
        sys.exit("model is not downloadable on Sketchfab")
    if not args.token:
        sys.exit("download needs a Sketchfab API token (--token or $SKETCHFAB_API_TOKEN)")

    # Stage everything OUTSIDE any asset library so a rejected candidate never lands in a package.
    explicit_stage = bool(args.stage)
    stage = args.stage or tempfile.mkdtemp(prefix=f"prop_{name}_")
    os.makedirs(stage, exist_ok=True)
    print(f"staging in {stage}")
    _write_credit(stage, meta)
    gltf = _download_gltf(uid, args.token, stage)
    print(f"downloaded glTF -> {gltf}")
    finalize_mujoco.pngify(stage, gltf)  # textures -> PNG up front, so the preview shows them

    # reduce + view, looping until the reduction quality is accepted
    obj = os.path.join(stage, f"{name}.obj")
    # Deliberately NOT inside the staging dir: everything in there is copied verbatim into the asset
    # library on approval, so a preview PNG left behind would be committed as part of the prop (and
    # trip inspect-prop's leftover-intermediates check).
    preview = os.path.join(tempfile.gettempdir(), f"prop_{name}_preview.png")
    target = args.target_faces
    while True:
        # A subprocess, not reduce_mesh.main(): outside Blender that call re-invokes Blender on itself
        # and exits with its return code, which would take this loop down with it on the first pass.
        _run(
            sys.executable,
            "-m",
            reduce_mesh.__name__,
            "--blender",
            blender,
            gltf,
            obj,
            "--target-faces",
            str(target),
            "--scale",
            str(args.scale),
        )
        render_main([obj, "--out", preview, "--show"])
        if _yes(f"Reduction to {target} faces OK? [Y/n]: "):
            break
        raw = input(f"New target-faces [{target}]: ").strip()
        if raw:
            try:
                target = int(raw)
            except ValueError:
                print("not a number; keeping", target)

    # finalize in the staging dir -- now a complete, loadable prop (<name>.xml + PNG textures)
    _cleanup_intermediate(stage)  # drop the glTF/bin now that the OBJ is accepted
    finalize_mujoco.finalize(stage, name)  # scale is already baked into the geometry by reduce

    # import decision -- the only step that touches an asset library
    if args.preview:
        print(f"\npreview staged (NOT imported) -> {stage}")
        print(f"re-view it any time: roqsim render {obj} --out {preview} --show")
        print("drop --preview on a run to be asked whether to import after viewing.")
        return
    dest = os.path.join(args.models_dir, name)
    where = "" if args.models_dir == _DEFAULT_MODELS_DIR else f" (models-dir: {args.models_dir})"
    if not _yes(f"Import into {dest}{where}? [Y/n]: "):
        print(f"not imported; staged prop left at {stage}")
        return
    if os.path.exists(dest):
        sys.exit(
            f"{dest} already exists -- remove it or pass --name to import under a different name"
        )
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    shutil.copytree(stage, dest)
    if not explicit_stage:
        shutil.rmtree(stage, ignore_errors=True)  # discard the temp; an explicit --stage is kept
    print(f"\ndone -> {dest} ({', '.join(sorted(os.listdir(dest)))})")


def main(argv: list | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("search", help="search redistributable, downloadable models")
    sp.add_argument("query", help="free-text search query")
    sp.add_argument("--count", type=int, default=12, help="max results to return (default 12)")
    sp.set_defaults(func=cmd_search)

    tp = sub.add_parser(
        "thumbs", help=f"download up to {_MAX_THUMBS} models' thumbnails to a temp dir"
    )
    tp.add_argument("models", nargs="+", help=f"1..{_MAX_THUMBS} model URLs or 32-hex uids")
    tp.set_defaults(func=cmd_thumbs)

    ip = sub.add_parser("info", help="print metadata + licence verdict (no token, no download)")
    ip.add_argument("model", help="Sketchfab model URL or 32-hex uid")
    ip.add_argument("--out", help="also write CREDITS.txt here (omit to only print)")
    ip.set_defaults(func=cmd_info)

    dp = sub.add_parser("download", help="download the glTF (licence-gated; needs a token)")
    dp.add_argument("model", help="Sketchfab model URL or 32-hex uid")
    dp.add_argument("--out", default="sketchfab_download", help="output directory")
    dp.add_argument("--token", default=os.environ.get("SKETCHFAB_API_TOKEN"))
    dp.add_argument("--force", action="store_true", help="download even if the licence is flagged")
    dp.set_defaults(func=cmd_download)

    mp = sub.add_parser(
        "import",
        help="one-shot: download -> reduce -> view -> import on approval",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    mp.add_argument("model", help="Sketchfab model URL or 32-hex uid")
    mp.add_argument(
        "--blender",
        default="blender",
        help="Blender binary: on PATH as 'blender' (default) or an explicit path",
    )
    mp.add_argument("--name", help="folder name under models/ (default: slug of the model name)")
    mp.add_argument(
        "--target-faces", type=int, default=20000, help="triangle budget (default 20000)"
    )
    mp.add_argument(
        "--scale", type=float, default=1.0, help="uniform scale to metres (default 1.0)"
    )
    mp.add_argument("--token", default=os.environ.get("SKETCHFAB_API_TOKEN"))
    mp.add_argument("--force", action="store_true", help="proceed even if the licence is flagged")
    mp.add_argument(
        "--preview",
        action="store_true",
        help="evaluate quality only: stage + reduce + view, then STOP without importing",
    )
    mp.add_argument(
        "--stage",
        help="staging dir for the evaluation (default: a temp dir), outside any "
        "asset library; an explicit dir is kept, a temp one is discarded",
    )
    mp.add_argument(
        "--models-dir",
        default=_DEFAULT_MODELS_DIR,
        help="destination models/ dir to import into on approval (default: the "
        "roqsim_assets library; point elsewhere to import into another package)",
    )
    mp.set_defaults(func=cmd_import)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
