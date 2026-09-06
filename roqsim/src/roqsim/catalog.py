# Copyright (C) 2026 Frederik Pasch
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""What this installation can spawn and run: the model and world catalogs, as JSON.

The companion to :mod:`roqsim.introspection`, which answers *what plugins are installed*. This
answers the two questions that come before it -- **what can I put in a world, and what worlds are
there already** -- and it answers them from the same registries the loader resolves against, so a
name this prints is a name that resolves.

That mattered enough to write: the catalog existed only as a Sphinx directive
(``docs/_ext/model_docs.py``), which is a rendered HTML page produced at docs-build time. Inside a
container -- which is where a campaign, an agent or anyone with a shell actually stands -- there was
no way to ask, and ``roqsim.world.available_worlds()`` answers with the one *built-in* definition
while saying nothing about the 30-odd worlds a provider package ships. The failure mode is not an
error message: it is guessing a model name, getting ``ModelError``, and guessing again.

Three entry points, each returning plain dicts::

    list_models()          # every model, by the ref that resolves it
    get_model_details(ref) # one model: its file, its manifest's components, its provenance
    list_worlds()          # every runnable world YAML, baked scene, and built-in definition

Runnable as a module, so it works inside a runtime image whose caller parses the JSON on the host::

    python -m roqsim.catalog models
    python -m roqsim.catalog worlds
    python -m roqsim.catalog model turtlebot4

**A listed name is a usable name.** Every model row carries the ``ref`` that
:func:`roqsim.models.resolve_model` takes, and every world row the target ``roqsim sim`` takes or the
``sim.world`` / ``extends`` value it belongs in -- named ``use`` in each row, because "what do I type"
is the actual question. ``tests/test_catalog.py`` resolves every one of them, so a catalog that
drifts from the loader fails there rather than in someone's world file.

**What is listed is what a package OFFERS**, which is not the same as what it ships. A
``roqsim.worlds`` entry point is a package saying "these worlds are an interface": nameable by
``roqsim sim <ref>`` and by ``extends:``, and answerable for. Several packages also ship worlds that
are debugging aids -- a rig that exercises one sensor, a scene that reproduces one bug -- and those
are deliberately unregistered: they are run by path, and advertising them here would invite a world
to ``extends:`` something nobody maintains as an interface. A world missing from this listing is
therefore a decision, not an omission.
"""

from __future__ import annotations

import json
import sys
from importlib import import_module
from pathlib import Path

import yaml

from roqsim.manifest import manifest_license, manifest_path
from roqsim.models import ENTRY_POINT_GROUP as MODELS_GROUP
from roqsim.models import _entry_points, _provider_dirs
from roqsim.world import _world_entry_points, available_worlds

#: Files that ship beside a model to say where it came from. Reported rather than read: a model
#: without one is a redistribution problem, and that is worth being able to ask about in bulk.
_PROVENANCE_GLOBS = ("*LICENSE*", "*License*", "CREDITS.txt")


def _dist_name(ep) -> str:
    dist = getattr(ep, "dist", None)
    return dist.name if dist is not None else ep.name


def _manifest_components(model_file: Path) -> list[str]:
    """The plugin refs a model's manifest brings with it, in order (``[]`` when it has none).

    This is the answer to "what do I get when I spawn this", which is not visible from the MJCF: a
    TurtleBot 4 arrives with its drive, its lidar and its camera because its manifest says so.
    """
    path = manifest_path(model_file)
    if not path.is_file():
        return []
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        # A broken manifest must not sink the catalog -- the same rule list_plugins follows for a
        # plugin whose import fails. The model is still listed, with nothing claimed about it.
        return []
    refs = []
    for entry in data.get("components") or []:
        if isinstance(entry, dict):
            refs += [k for k in entry if k not in ("name", "enabled", "components")]
    return refs


def _provenance(model_file: Path) -> list[str]:
    """Names of the licence/credits files that apply to a model.

    A model that names its licence (``license:`` in its manifest) is reported by that name plus any
    credits beside it; only a model that names none falls back to everything licence-shaped in its
    directory. The fallback is right for the folder-per-model layout, where the sidecar sits alone
    beside the MJCF, and wrong for a flat provider whose one directory holds several vendors' terms
    -- which is what the declaration is for (:func:`roqsim.manifest.manifest_license`).
    """
    declared = [path.name for path in manifest_license(model_file)]
    found: list[str] = list(declared)
    for pattern in _PROVENANCE_GLOBS if not declared else ("CREDITS.txt",):
        found += sorted(
            p.name for p in model_file.parent.glob(pattern) if p.is_file() and p.name not in found
        )
    return found


def _model_files(models_dir: Path) -> list[tuple[str, Path]]:
    """``(name, mjcf)`` for every model a provider dir holds, in both shipped layouts.

    Flat (``<name>.xml``) and one-folder-per-model (``<name>/<name>.xml``) -- the same two the
    resolver accepts (:func:`roqsim.models._find_in_dir`), so the catalog cannot list a layout that
    would not load, or miss one that would.
    """
    found: dict[str, Path] = {}
    for xml in sorted(models_dir.glob("*.xml")):
        found[xml.stem] = xml
    for sub in sorted(p for p in models_dir.iterdir() if p.is_dir()):
        nested = sub / f"{sub.name}.xml"
        if nested.is_file():
            found.setdefault(sub.name, nested)
    return sorted(found.items())


def list_models() -> dict:
    """Every model installed here, one row per model, sorted by ref.

    Returns ``{"items": [{name, ref, use, provider, package, path, components, provenance,
    thumbnail}, ...]}``. ``ref`` is what resolves (``<provider>:<name>``); ``use`` is the line to put
    in a world. ``components`` is what the model's manifest brings with it. A provider that fails to
    import is reported as its own row with an ``error`` rather than dropped, so a broken package is
    visible instead of merely absent.
    """
    items: list[dict] = []
    for ep in sorted(_entry_points(MODELS_GROUP), key=lambda e: e.name):
        package = _dist_name(ep)
        try:
            module = import_module(
                ep.value.split(":")[0] if isinstance(ep.value, str) else ep.value
            )
            models_dir, _mesh, _tex = _provider_dirs(module)
        except Exception as exc:  # noqa: BLE001 - one broken provider must not sink the catalog
            items.append(
                {"provider": ep.name, "package": package, "error": f"{type(exc).__name__}: {exc}"}
            )
            continue
        for name, mjcf in _model_files(Path(models_dir)):
            thumb = mjcf.parent / f"{mjcf.stem}.thumb.png"
            items.append(
                {
                    "name": name,
                    # Qualified, always: two providers may ship a model of one name, and the short
                    # name then resolves to whichever provider is searched first. The qualified ref
                    # is the one that cannot change meaning when a package is installed.
                    "ref": f"{ep.name}:{name}",
                    "use": f"spawn_model: {{model: {ep.name}:{name}}}",
                    "provider": ep.name,
                    "package": package,
                    "path": str(mjcf),
                    "components": _manifest_components(mjcf),
                    "provenance": _provenance(mjcf),
                    "thumbnail": str(thumb) if thumb.is_file() else None,
                }
            )
    items.sort(key=lambda item: item.get("ref") or item.get("provider", ""))
    return {"items": items}


def get_model_details(name: str) -> dict:
    """One model's detail, or ``{"error": ...}`` if *name* resolves to none.

    Accepts anything :func:`roqsim.models.resolve_model` does -- a qualified ref, a bare short name,
    or a path -- so a caller can hand back a row's ``ref`` unchanged. Adds to the list row the
    manifest's full component config (what a spawn actually injects, with its defaults) and the
    ``fov`` block a sensor model publishes for coverage analysis.
    """
    from roqsim.models import ModelError, resolve_model

    try:
        asset = resolve_model(name)
    except ModelError as exc:
        return {"error": str(exc)}
    path = Path(asset.path)
    manifest: dict = {}
    if manifest_path(path).is_file():
        try:
            manifest = yaml.safe_load(manifest_path(path).read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError) as exc:
            manifest = {"error": f"unreadable manifest: {exc}"}
    thumb = path.parent / f"{path.stem}.thumb.png"
    return {
        "name": path.stem,
        "path": str(path),
        # Where its meshes and textures are searched for, in order -- the answer to "why did my
        # model load without its geometry", which is otherwise a silent failure in MuJoCo.
        "mesh_dirs": [str(d) for d in asset.meshdirs],
        "texture_dirs": [str(d) for d in asset.texturedirs],
        "components": manifest.get("components") or [],
        "assets": manifest.get("assets"),
        "fov": manifest.get("fov"),
        "provenance": _provenance(path),
        "thumbnail": str(thumb) if thumb.is_file() else None,
    }


def _yaml_summary(path: Path) -> str | None:
    """A world YAML's first comment line -- what its author wrote at the top of the file.

    The same convention the command tree uses for a tool (its module docstring's first line): the
    description already exists in the file, so nothing has to be written twice to have a catalog
    that says what each world is.
    """
    try:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                if stripped.startswith("#"):
                    return stripped.lstrip("#").strip() or None
                return None
    except OSError:
        return None
    return None


def list_worlds() -> dict:
    """Every world this installation can run or build on, one row each, sorted by ref.

    Three kinds, and the distinction is what a caller needs to know rather than a taxonomy:

    * ``world`` -- a world YAML a provider ships. ``roqsim sim <ref>`` runs it, and another world
      inherits it with ``extends: <ref>``.
    * ``scene`` -- a baked MJCF. It is not runnable on its own; it goes in a world's ``sim.world``.
    * ``builtin`` -- a world definition built in code (``empty_room``), also a ``sim.world`` value.

    Each row carries ``use``, the line that consumes it.
    """
    items: list[dict] = [
        {
            "name": name,
            "ref": name,
            "kind": "builtin",
            "use": f"sim: {{world: {name}}}",
            "package": "roqsim",
            "path": None,
            "summary": "built-in world definition (ground + light)",
        }
        for name in available_worlds()
    ]
    for ep in sorted(_world_entry_points(), key=lambda e: e.name):
        package = _dist_name(ep)
        try:
            worlds_dir = Path(ep.load().WORLDS_DIR)
        except Exception as exc:  # noqa: BLE001 - as in list_models
            items.append(
                {"provider": ep.name, "package": package, "error": f"{type(exc).__name__}: {exc}"}
            )
            continue
        for path in sorted(worlds_dir.glob("*.yaml")) + sorted(worlds_dir.glob("*.yml")):
            items.append(
                {
                    "name": path.stem,
                    "ref": f"{ep.name}:{path.stem}",
                    "kind": "world",
                    "use": f"roqsim sim {ep.name}:{path.stem}",
                    "provider": ep.name,
                    "package": package,
                    "path": str(path),
                    "summary": _yaml_summary(path),
                }
            )
        for sub in sorted(p for p in worlds_dir.iterdir() if p.is_dir()):
            mjcf = sub / f"{sub.name}.xml"
            if not mjcf.is_file():
                continue
            items.append(
                {
                    "name": sub.name,
                    "ref": f"{ep.name}:{sub.name}",
                    "kind": "scene",
                    "use": f"sim: {{world: {ep.name}:{sub.name}}}",
                    "provider": ep.name,
                    "package": package,
                    "path": str(mjcf),
                    "summary": None,
                }
            )
    items.sort(key=lambda item: (item.get("kind", ""), item.get("ref") or item.get("provider", "")))
    return {"items": items}


def main(argv=None):
    import argparse  # pylint: disable=import-outside-toplevel

    parser = argparse.ArgumentParser(
        prog="roqsim catalog",
        description="What this installation can spawn and run: models and worlds, as JSON.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (
        ("models", "Every installed model, as JSON."),
        ("worlds", "Every world, baked scene and built-in world definition, as JSON."),
    ):
        p = sub.add_parser(name, help=help_text)
        # A bare list of refs, for the caller who wants to pipe them: `roqsim catalog models --refs
        # | xargs -n1 roqsim render` is a real thing to want, and jq should not be a prerequisite.
        p.add_argument("--refs", action="store_true", help="print one ref per line instead of JSON")

    p_model = sub.add_parser("model", help="One model's full detail, as JSON.")
    p_model.add_argument("name", help="A ref from `models`, a bare model name, or a path")

    args = parser.parse_args(argv)
    if args.command == "model":
        result = get_model_details(args.name)
        print(json.dumps(result, indent=2))
        return 1 if "error" in result else 0
    result = list_models() if args.command == "models" else list_worlds()
    if args.refs:
        # Deduplicated: one name can be two kinds (`roqsim_scenes:depot` is both a world YAML and
        # the baked scene it loads), and a list meant for `xargs` should not run it twice. The JSON
        # keeps both rows, where `kind` says which is which.
        refs = dict.fromkeys(item["ref"] for item in result["items"] if "ref" in item)
        print("\n".join(refs))
    else:
        print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
