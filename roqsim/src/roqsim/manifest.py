"""Robot-description manifests: the plugins a spawned model brings with it.

A robot/arm ships a ``<model>.manifest.yaml`` next to its MJCF listing the controller/sensor plugins
that are intrinsic to it (a diff-drive + lidar for a mobile base, an arm_controller for a
manipulator). A spawn plugin implements :meth:`roqsim.plugin.Plugin.expand` by delegating to
:func:`expand_manifest`, so a world that just spawns the model gets those plugins without
re-declaring them -- and any spawn plugin (mobile, arm, ...) reuses the same mechanism, differing
only in the config key its injected plugins use to name their entity (``robot`` vs ``arm``).
"""

from __future__ import annotations

from pathlib import Path

import yaml

from .config import PluginError, PluginSpec, document_entries, parse_plugin_entry
from .models import resolve_model


def manifest_path(model_file: Path) -> Path:
    """``<model-stem>.manifest.yaml`` beside a resolved model file."""
    return model_file.parent / f"{model_file.stem}.manifest.yaml"


def manifest_fov(model_file: Path) -> dict:
    """The ``fov:`` block from a model's manifest (``{}`` when it has no manifest or no block).

    Where a sensor model states its own valid detection range, so neither a world nor an analysis has
    to repeat it: ``near``/``far`` for a camera frustum, plus the angular bounds a camera-less device
    needs (``mid360`` adds ``h_min``/``h_max``/``v_min``/``v_max``). Two callers read it and must
    agree -- ``spawn_sensor`` draws the ``show_fov`` cone from it, and the coverage catalog derives a
    hypothetical placement's range from it -- which is why it is here rather than private to either.

    Deliberately unvalidated: the keys are the reader's business (a frustum and an angular sector want
    different ones), and a model with no ``fov:`` is normal rather than broken.
    """
    path = manifest_path(model_file)
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text()) or {}
    return data.get("fov", {}) or {}


def load_manifest(
    model_file: Path, base_dir: Path | None = None, seen: frozenset[Path] = frozenset()
) -> list[dict]:
    """The manifest's ``components:`` list for a resolved model file, or ``[]`` when it has none.

    A manifest is a document like any other, so it may ``extends:`` another model's manifest and
    inherit its components: ``unitree_g1_dex1`` is ``unitree_g1`` plus hands, and said that by
    repeating the base's ``g1_locomotion`` and ``lidar`` blocks verbatim until it could say it once.
    It inherits **components, not geometry** -- a derived model keeps its own MJCF; ``extends:`` never
    carried geometry, ``sim.world`` did, and a manifest may not carry ``sim:`` at all (see below).

    The base is named the way anything else names a model (a ``roqsim.models`` ref, or a path
    relative to this manifest), and cycles raise rather than recursing forever.
    """
    path = manifest_path(model_file)
    if not path.exists():
        return []
    data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(data, dict):
        raise PluginError(f"manifest {path} must be a mapping at the top level")
    if "sim" in data:
        # A model is a component of a world, and `sim:` belongs to the run and the scene. Letting a
        # manifest set `seed`, `pacing` or `contact_override` would let a robot reach up and change
        # the experiment it is part of -- action at a distance, from a file the world never opened.
        raise PluginError(
            f"manifest {path} has a 'sim:' block. `sim:` belongs to the world being run, not to a "
            f"model included in it: a manifest cannot set the run's seed, pacing or contact "
            f"overrides. Move those keys to the world that spawns this model."
        )
    inherited: list[dict] = []
    ext = data.get("extends")
    if ext is not None:
        if path in seen:
            chain = " -> ".join(str(p) for p in (*seen, path))
            raise PluginError(f"manifest 'extends' cycle detected: {chain}")
        base_model = resolve_model(str(ext), base_dir=base_dir or path.parent).path
        inherited = load_manifest(base_model, base_dir=base_dir, seen=seen | {path})
    return inherited + document_entries(data, str(path))


def expand_manifest(
    spec: PluginSpec,
    world: list[PluginSpec],
    *,
    base_dir: Path | None = None,
) -> list[PluginSpec]:
    """Plugin specs a spawn plugin should inject for its model, wired to its entity.

    The model is resolved via :func:`roqsim.models.resolve_model` (so it may live in any installed
    package), and its ``<model>.manifest.yaml`` manifest is read from beside the resolved file. Each
    (a mobile spawn wires ``robot: <name>``, an arm spawn ``arm: <name>``) and inherits the spawn's
    ``prefix`` -- so a build-time plugin (e.g. ``fiducial_marker`` welding a geom onto a spawned body)
    can form prefixed names like ``<prefix>wrist_3_link`` before the entity registry exists; runtime
    plugins keep reading the prefix from the entity. Distinct entities (two arms) never collide.

    When the owner already declares a component with the same **label**, the manifest default is not
    injected -- the world's entry is the one that runs -- but the manifest's config is **merged
    underneath it** (per key: the world's value always wins, missing keys are filled from the
    manifest). That is what lets a world override *part* of a default: ``diff_drive: {robot: robot,
    test_cmd: [0.5, 0.4]}`` adds a scripted command while keeping the model's wheel geometry, slip
    calibration and actuator names.

    The merge is shallow, deliberately: a nested value the world sets (``topics: {scan: /s}``)
    replaces the manifest's whole mapping rather than being deep-merged into it, so what a world says
    is what a reader gets, without a per-key excavation of two files. The world's ``PluginSpec`` is
    mutated in place; that is safe because every plugin is constructed only after expansion finishes
    (see :func:`roqsim.config.instantiate_plugins`), so it does not matter whether the spawn is
    declared before or after the plugin it fills in.

    Off with ``default_plugins: false`` on the spawn config; a no-op when there is no ``model``.
    """
    cfg = spec.config
    if not cfg.get("default_plugins", True) or not cfg.get("model"):
        return []
    entity = spec.address
    model_file = resolve_model(cfg["model"], base_dir=base_dir).path
    # Keyed on the LABEL, not on the plugin ref: a model may ship two of a kind (tiago_pro's front
    # and rear lidars), and keying on the ref would collapse them onto one entry -- silently losing
    # a sensor. A label is unique among an owner's components by construction, so this cannot.
    declared: dict[str, PluginSpec] = {}
    for s in world:
        if s.entity == entity:
            declared.setdefault(s.label, s)
    out: list[PluginSpec] = []
    prefix = cfg.get("prefix", "")
    for entry in load_manifest(model_file, base_dir=base_dir):
        base = parse_plugin_entry(entry, "manifest plugin")
        pcfg = base.config
        pcfg.setdefault("prefix", prefix)
        base.entity = entity
        declared_spec = declared.get(base.label)
        if declared_spec is not None:
            # The world runs its own entry; give it the manifest's defaults for everything it did
            # not say. Without this, a partial override silently drops the rest of the model's
            # description -- e.g. a husky's diff_drive falling back to the plugin's TurtleBot
            # wheel_radius/actuator names, which then fails to resolve against the husky's MJCF.
            for key, value in pcfg.items():
                declared_spec.config.setdefault(key, value)
            continue
        out.append(PluginSpec(ref=base.ref, name=base.name, config=pcfg, entity=entity))
    return out
