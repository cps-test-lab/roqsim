"""Resolve, download and pin Gazebo/Ignition Fuel models referenced by an SDF world.

SDF worlds for ROS-era papers typically contain no geometry at all: they are a bill of materials of
``<include><uri>`` entries pointing at a model registry. Reconstructing such a world therefore depends
on a third-party server, and a campaign that re-fetches at run time is not reproducible. This module
fetches each model once into a local cache and records exactly what it got (version + sha256 +
licence) in an ``assets.lock.json``, so the port can be rebuilt byte-identically later — or audited
when the registry changes under us.

Use as a library (``resolve``/``fetch``) or a CLI::

    roqsim scenes fuel-fetch --uri "https://fuel.ignitionrobotics.org/1.0/OpenRobotics/models/Tunnel Tile 6"
    roqsim scenes fuel-fetch --world worlds/warehouse.sdf --lock simulation/scenes/s1/assets.lock.json

URI forms handled:

- ``https://fuel.<host>/1.0/<Owner>/models/<Model Name>`` (with or without percent-encoding)
- ``https://fuel.<host>/1.0/<Owner>/models/<Model Name>/<version>``
- ``model://<name>`` -- resolved against ``--model-path`` / ``GZ_SIM_RESOURCE_PATH`` /
  ``IGN_GAZEBO_RESOURCE_PATH`` / ``GAZEBO_MODEL_PATH``, never from the network.

``fuel.ignitionrobotics.org`` (legacy) and ``fuel.gazebosim.org`` serve the same content; we normalise
to the latter for fetching but record the URI as written in the world, because that string is the
paper's actual provenance claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import urllib.parse
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import requests

_FUEL_HOSTS = ("fuel.gazebosim.org", "fuel.ignitionrobotics.org")
_CANONICAL_HOST = "fuel.gazebosim.org"
_DEFAULT_CACHE = Path(
    os.environ.get("ROQSIM_FUEL_CACHE", Path.home() / ".cache" / "roqsim" / "fuel")
)
_TIMEOUT = 60


class FuelError(RuntimeError):
    """A model could not be resolved or fetched. Never swallowed: a missing asset is a finding."""


@dataclass
class Asset:
    """One pinned model. Serialised into assets.lock.json."""

    uri: str  # verbatim, as written in the world file -- the provenance claim
    owner: str
    name: str
    version: str | int | None
    sha256: str  # of the downloaded zip
    licence: str | None
    licence_url: str | None
    fetched_at: str
    local: str  # cache-relative directory holding the unpacked model


def _is_fuel(uri: str) -> bool:
    return any(h in uri for h in _FUEL_HOSTS)


def parse_uri(uri: str) -> tuple[str, str, str | None]:
    """``.../<Owner>/models/<Name>[/<version>]`` -> (owner, name, version). Percent-decoded."""
    if not _is_fuel(uri):
        raise FuelError(f"not a Fuel URI: {uri}")
    path = urllib.parse.urlparse(uri).path
    m = re.search(r"/([^/]+)/models/([^/]+)(?:/(\d+))?/?$", urllib.parse.unquote(path))
    if not m:
        raise FuelError(f"unparseable Fuel URI: {uri}")
    return m.group(1), m.group(2), m.group(3)


def _api(owner: str, name: str, version: str | None = None) -> str:
    base = f"https://{_CANONICAL_HOST}/1.0/{urllib.parse.quote(owner)}/models/{urllib.parse.quote(name)}"
    return f"{base}/{version}" if version else base


def metadata(owner: str, name: str, version: str | None = None) -> dict:
    r = requests.get(_api(owner, name, version), timeout=_TIMEOUT)
    if r.status_code != 200:
        raise FuelError(
            f"Fuel metadata {r.status_code} for {owner}/{name} -- record as a finding, do not substitute"
        )
    return r.json()


def _local_dirname(owner: str, name: str, version: str | int | None) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")
    return f"{owner}__{slug}__v{version if version is not None else 'latest'}"


def fetch(uri: str, cache: Path = _DEFAULT_CACHE, force: bool = False) -> Asset:
    """Download + unpack one Fuel model into the cache. Idempotent; returns the pin record."""
    owner, name, version = parse_uri(uri)
    meta = metadata(owner, name, version)
    version = version or meta.get("version")

    cache.mkdir(parents=True, exist_ok=True)
    dest = cache / _local_dirname(owner, name, version)
    zip_path = dest.with_suffix(".zip")

    if force and dest.exists():
        shutil.rmtree(dest)

    if not zip_path.exists() or force:
        url = _api(owner, name, version) + f"/{urllib.parse.quote(name)}.zip"
        r = requests.get(url, timeout=_TIMEOUT, stream=True)
        if r.status_code != 200:
            raise FuelError(f"Fuel download {r.status_code} for {url}")
        tmp = zip_path.with_suffix(".zip.part")
        with open(tmp, "wb") as fh:
            for chunk in r.iter_content(1 << 16):
                fh.write(chunk)
        tmp.replace(zip_path)

    sha = hashlib.sha256(zip_path.read_bytes()).hexdigest()

    if not dest.exists():
        tmp_dir = dest.with_suffix(".part")
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(tmp_dir)
        # Fuel zips are sometimes single-rooted, sometimes flat; normalise to the dir holding model.sdf
        roots = [p.parent for p in tmp_dir.rglob("model.sdf")]
        if not roots:
            raise FuelError(f"no model.sdf inside {zip_path} -- unexpected Fuel package layout")
        shutil.move(str(sorted(roots, key=lambda p: len(p.parts))[0]), str(dest))
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return Asset(
        uri=uri,
        owner=owner,
        name=name,
        version=version,
        sha256=sha,
        licence=meta.get("license_name"),
        licence_url=meta.get("license_url"),
        fetched_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        local=dest.name,
    )


def resolve_model_uri(uri: str, model_paths: list[Path]) -> Path:
    """``model://<name>[/rel/path]`` -> local path, searched on the given model paths."""
    rest = uri[len("model://") :]
    name, _, rel = rest.partition("/")
    for root in model_paths:
        cand = root / name
        if cand.is_dir():
            return cand / rel if rel else cand
    raise FuelError(
        f"cannot resolve {uri}: not found on model path {[str(p) for p in model_paths]}. "
        "Do not substitute a lookalike -- record the miss as a resolution_attempt in the spec."
    )


def default_model_paths(extra: list[str] | None = None) -> list[Path]:
    out = [Path(p) for p in (extra or [])]
    for var in ("GZ_SIM_RESOURCE_PATH", "IGN_GAZEBO_RESOURCE_PATH", "GAZEBO_MODEL_PATH"):
        out += [Path(p) for p in os.environ.get(var, "").split(os.pathsep) if p]
    return out


def write_lock(path: Path, assets: list[Asset], world: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "world": world,
        "registry": f"https://{_CANONICAL_HOST}",
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "note": (
            "Pinned third-party assets. The world file referenced these by URI and contained no "
            "geometry of its own; this lock is what makes the port reproducible. Re-fetching without "
            "checking these hashes silently changes the experiment."
        ),
        "assets": [asdict(a) for a in sorted(assets, key=lambda a: (a.owner, a.name))],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _world_uris(world: Path) -> list[str]:
    from lxml import etree

    tree = etree.parse(str(world))
    return sorted(
        {e.text.strip() for e in tree.iter() if _tag(e) == "uri" and e.text and _is_fuel(e.text)}
    )


def _tag(el) -> str:
    return (
        etree.QName(el).localname
        if hasattr(el, "tag") and not isinstance(el.tag, str)
        else str(el.tag).split("}")[-1]
    )


def main(argv: list | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Fetch and pin Fuel models referenced by an SDF world."
    )
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--uri", help="a single Fuel model URI")
    g.add_argument("--world", type=Path, help="an SDF world; fetches every Fuel URI it references")
    ap.add_argument("--lock", type=Path, help="write/update an assets.lock.json here")
    ap.add_argument("--cache", type=Path, default=_DEFAULT_CACHE)
    ap.add_argument("--force", action="store_true", help="re-download even if cached")
    args = ap.parse_args(argv)

    uris = [args.uri] if args.uri else _world_uris(args.world)
    if not uris:
        print(
            "no Fuel URIs found -- if the world has no inline geometry either, that IS the finding",
            file=sys.stderr,
        )
        return 1

    assets, failed = [], []
    for u in uris:
        try:
            a = fetch(u, cache=args.cache, force=args.force)
            assets.append(a)
            print(f"  ok    {a.owner}/{a.name} v{a.version} [{a.licence}] -> {a.local}")
        except FuelError as e:  # a miss is data, not a crash: report all, then fail
            failed.append((u, str(e)))
            print(f"  FAIL  {u}: {e}", file=sys.stderr)

    if args.lock and assets:
        write_lock(args.lock, assets, world=str(args.world) if args.world else None)
        print(f"pinned {len(assets)} assets -> {args.lock}")

    if failed:
        print(
            f"\n{len(failed)} asset(s) unresolved. Each is a resolution_attempt for the spec "
            "(status: dead_link_404 / found_but_missing_value) -- not a reason to substitute geometry.",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    from lxml import etree  # noqa: F401  (import here so library use does not require it)

    raise SystemExit(main())
