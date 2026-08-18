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

from .config import PluginSpec, parse_plugin_entry
from .models import resolve_model


def manifest_path(model_file: Path) -> Path:
    """``<model-stem>.manifest.yaml`` beside a resolved model file."""
    return model_file.parent / f"{model_file.stem}.manifest.yaml"


def load_manifest(model_file: Path) -> list[dict]:
    """The manifest's ``plugins:`` list for a resolved model file, or ``[]`` when it has none."""
    path = manifest_path(model_file)
    if not path.exists():
        return []
    data = yaml.safe_load(path.read_text()) or {}
    return list(data.get("plugins", []) or [])


def expand_manifest(
    spec: PluginSpec,
    world: list[PluginSpec],
    *,
    target_key: str,
    default_name: str,
    base_dir: Path | None = None,
) -> list[PluginSpec]:
    """Plugin specs a spawn plugin should inject for its model, wired to its entity.

    The model is resolved via :func:`roqsim.models.resolve_model` (so it may live in any installed
    package), and its ``<model>.manifest.yaml`` manifest is read from beside the resolved file. Each
    manifest plugin is wired to this entity by setting ``config[target_key]`` to the spawn's ``name``
    (a mobile spawn wires ``robot: <name>``, an arm spawn ``arm: <name>``) and inherits the spawn's
    ``prefix`` -- so a build-time plugin (e.g. ``fiducial_marker`` welding a geom onto a spawned body)
    can form prefixed names like ``<prefix>wrist_3_link`` before the entity registry exists; runtime
    plugins keep reading the prefix from the entity. Distinct entities (two arms) never collide.

    When ``world`` already declares the same ``(plugin ref, entity)``, the manifest default is not
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
    name = cfg.get("name", default_name)
    model_file = resolve_model(cfg["model"], base_dir=base_dir).path
    # First declaration wins as the merge target; a world that declares the same (plugin, entity)
    # twice is already ambiguous and not made less so here.
    declared: dict[tuple[str, object], PluginSpec] = {}
    for s in world:
        declared.setdefault((s.ref, s.config.get(target_key)), s)
    out: list[PluginSpec] = []
    prefix = cfg.get("prefix", "")
    for entry in load_manifest(model_file):
        base = parse_plugin_entry(entry, "manifest plugin")
        pcfg = base.config
        pcfg.setdefault(target_key, name)
        pcfg.setdefault("prefix", prefix)
        declared_spec = declared.get((base.ref, pcfg.get(target_key)))
        if declared_spec is not None:
            # The world runs its own entry; give it the manifest's defaults for everything it did
            # not say. Without this, a partial override silently drops the rest of the model's
            # description -- e.g. a husky's diff_drive falling back to the plugin's TurtleBot
            # wheel_radius/actuator names, which then fails to resolve against the husky's MJCF.
            for key, value in pcfg.items():
                declared_spec.config.setdefault(key, value)
            continue
        out.append(PluginSpec(ref=base.ref, name=base.name, config=pcfg))
    return out
