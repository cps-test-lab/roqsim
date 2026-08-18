"""Resolve a walker *blueprint* -- a ``people/<Walker>/`` folder -- into everything the humanoid
builder and the controller need.

Ported from our earlier in-house nav prototype's ``world.py::_resolve_walkers`` / ``_anim_dirs`` / ``_apply_outfit``, minus
the YAML world plumbing (roqsim passes plugin config instead).

A blueprint folder holds a textured OBJ plus a ``*.walker.json`` sidecar describing its materials
(with outfit variants), the per-rig bone table, per-limb collision radii and measured shoe-sole
offsets. Locomotion clips live in ``anims/<set>/<kind>.npz`` and are picked per body type + gender.

Blueprints are discovered across **every** installed ``roqsim.models`` provider that ships a
``people/`` directory, the same way :func:`roqsim.models.resolve_model` finds models -- so a world
names ``walker: MaleVisitorWalk`` and never says which package it came from, and a downstream package
can ship its own characters without this one being edited. This package's own ``models/`` is searched
first, so a bundled name can never be shadowed by a same-named folder elsewhere.
"""

from __future__ import annotations

import glob
import json
import logging
import os
import re

logger = logging.getLogger(__name__)

#: Locomotion clips the controller looks for. ``idle``/``walk``/``run``/``short`` are required
#: (procedural fallbacks exist); the rest flavour the direction axis and are optional.
CLIP_KINDS = ("idle", "walk", "run", "short", "turn_l", "turn_r", "walk_l", "walk_r", "walk_back")

_MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")


def models_dir() -> str:
    """Root of the bundled ``people/`` + ``anims/`` asset tree."""
    return _MODELS_DIR


def _asset_roots(base_dir: str | None = None) -> list[str]:
    """Directories to search for ``people/`` and ``anims/``, most specific first.

    An explicit ``base_dir`` is used alone -- callers that pass one (tests, a world pointing at its
    own tree) mean *that* tree and nothing else. Otherwise: this package first, then every other
    registered ``roqsim.models`` provider, so a bundled blueprint cannot be shadowed by a same-named
    folder in a package that happens to load earlier.

    A provider that ships no ``people/`` is skipped rather than reported: most model providers are
    robots and have no business owning characters.
    """
    if base_dir:
        return [base_dir]
    roots = [_MODELS_DIR]
    try:
        from roqsim.models import providers
    except Exception:  # pragma: no cover - roqsim is a hard dependency; be defensive, not clever
        return roots
    for _name, provider_models_dir, _meshdir, _texturedir in providers():
        root = str(provider_models_dir)
        if root not in roots and os.path.isdir(os.path.join(root, "people")):
            roots.append(root)
    return roots


def available_walkers(base_dir: str | None = None) -> list[str]:
    """Names of every blueprint folder discoverable under ``<root>/people/``."""
    names: set[str] = set()
    for root in _asset_roots(base_dir):
        people = os.path.join(root, "people")
        if not os.path.isdir(people):
            continue
        names.update(d for d in os.listdir(people) if os.path.isdir(os.path.join(people, d)))
    return sorted(names)


def _apply_outfit(materials: dict, outfit) -> dict:
    """Swap each material's texture for the chosen clothing ``variant``.

    ``outfit`` is either a single variant letter applied to every slot, or a ``{slot: letter}`` map
    (e.g. ``{pants: C, jacket: A}``). Materials without variants pass through untouched.
    """
    out = {}
    for mat, info in materials.items():
        info = dict(info)
        variants = info.get("variants")
        if variants and outfit:
            pick = outfit if isinstance(outfit, str) else outfit.get(info.get("slot"))
            if pick and pick in variants:
                info["texture"] = variants[pick].get("texture")
                info["normal"] = variants[pick].get("normal")
        out[mat] = info
    return out


def _infer_gender(name: str) -> str | None:
    """CARLA's name letter before the id: ``M`` man, ``F`` woman, ``G`` girl, ``B`` boy (EuroF01,
    AfroM01B, EuroG02, AfroB01). Drives the anim set. Returns the letter or None."""
    m = re.search(r"[A-Za-z]+?([MFGB])\d", name or "")
    return m.group(1) if m else None


def _anim_dirs(meta: dict, name: str) -> list[str]:
    """Ordered anim subdirs (most specific first) for this walker's body type + gender.

    CARLA's generic adult locomotion *is* the male-bodied set, so ``adult`` serves the ``M`` (and
    unknown) walkers directly; ``female`` and ``kid`` are the dedicated sets, tried before ``adult``
    as a graceful fallback. An explicit ``anim_set`` in the walker.json wins.
    """
    forced = meta.get("anim_set") if isinstance(meta, dict) else None
    if forced:
        return [forced] if forced != "adult" else ["adult"]
    gender = _infer_gender(name)
    if gender in ("G", "B"):  # girl / boy -> child locomotion
        return ["kid", "adult"]
    if gender == "F":  # woman -> female set
        return ["female", "adult"]
    return ["adult"]  # man / unknown -> generic (= male) adult set


class BlueprintError(Exception):
    """Raised when a walker blueprint cannot be resolved."""


def resolve_walker(name: str, outfit=None, motion=None, base_dir: str | None = None) -> dict:
    """Resolve blueprint ``name`` into a spec dict for :func:`~roqsim_walker.humanoid.build_humanoid`
    and :class:`~roqsim_walker.nav.controller.WalkerController`.

    Returns keys: ``mesh``, ``materials``, ``tpose``, ``flip``, ``skeleton``, ``collision``,
    ``sole``, ``motion`` (clip kind -> absolute ``.npz`` path). ``motion`` entries passed in override
    the resolved defaults for that kind.
    """
    roots = _asset_roots(base_dir)
    wdir, sidecars = None, []
    for root in roots:
        candidate = os.path.join(root, "people", name)
        found = sorted(glob.glob(os.path.join(candidate, "*.walker.json")))
        if found:
            wdir, sidecars = candidate, found
            break
    if not sidecars:
        known = available_walkers(base_dir)
        searched = ", ".join(os.path.join(r, "people") for r in roots)
        raise BlueprintError(
            f"walker blueprint {name!r}: no *.walker.json under any of {searched}. "
            f"Known walkers: {known}"
        )
    with open(sidecars[0]) as f:
        meta = json.load(f)

    spec = {
        "mesh": os.path.join(wdir, meta["obj"]),
        "materials": _apply_outfit(meta.get("materials") or {}, outfit),
        # CARLA meshes are exported in a T-pose (bind arms-out); a rigid robot like the G1 is baked
        # arms-down in our frame and sets tpose: false.
        "tpose": bool(meta.get("tpose", True)),
        # Facing is independent of the bind pose, but CARLA happens to be both T-posed and -X-facing,
        # so it defaults to tpose. The Open-RMF actors are T-posed and face +X -> flip: false.
        "flip": bool(meta.get("flip", meta.get("tpose", True))),
        "skeleton": meta.get("skeleton"),  # per-walker bone table (native proportions)
        "collision": meta.get("collision"),  # per-limb collision radii (CARLA Phys)
        "sole": meta.get("sole"),  # measured shoe-sole offsets (exact grounding)
    }

    # CARLA uses a separate locomotion set per body type AND gender. Resolve each clip from the most
    # specific anim subdir that has it, falling back to the shared set ("" = anims/<kind>.npz).
    search = _anim_dirs(meta, name) + [""]
    # The blueprint's own root first: a package shipping characters may ship clips for them too.
    clip_roots = [os.path.dirname(os.path.dirname(wdir))]
    clip_roots += [r for r in roots if r not in clip_roots]
    clips = dict(motion or {})
    for kind in CLIP_KINDS:
        if kind in clips:  # explicit override wins
            continue
        for sub, root in ((s, r) for s in search for r in clip_roots):
            clip = os.path.join(root, "anims", sub, f"{kind}.npz")
            if os.path.isfile(clip):
                clips[kind] = clip
                break
    spec["motion"] = clips
    missing = [k for k in ("idle", "walk", "run", "short") if k not in clips]
    if missing:
        logger.info(
            "walker %s: no clip for %s; using procedural fallback", name, ", ".join(missing)
        )
    return spec
