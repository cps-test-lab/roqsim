"""Load and validate the single world YAML, and instantiate its plugin pipeline.

The YAML has two top-level sections::

    sim:
      name: OpenSpace        # optional; viewer window title becomes "Roqsim: <name>"
      timestep: 0.004        # optional; else taken from the model
      pacing: realtime       # realtime | {factor: 4.0} | asap
      view:                  # optional initial viewer setup (windowed only; see viewer)
        lookat: [0, 0.4, 0.9]
        distance: 3.2
        azimuth: 130
        elevation: -20
        track: robot         # optional: keep an entity/body centred (MuJoCo tracking camera)
        follow_heading: true # optional: chase cam -- azimuth becomes an offset from the robot's yaw
      sync: {enabled: false} # foreseen lockstep mode (inert in M1)

    plugins:
      - floorplan:                                # a short-name ref is the entry's key
          size: 3.0                               # its config: opaque, validated by the plugin itself
        name: ground                              # reserved sibling key; identifies this instance
      - "my_pkg.mod:MyPlugin": { rays: 90 }       # a ref with a ':' MUST be a quoted key (see below)
      - "./plugins/custom.py:Foo": {}             # empty config; file refs are quoted too

Each entry is a mapping with exactly one plugin-ref key (whose value is the plugin's ``config`` map)
plus an optional reserved ``name:`` sibling that names the instance (any plugin may carry one). An
empty config is written ``<ref>: {}`` (or ``<ref>:``).

A plugin ref containing a colon -- the ``module.path:Class`` and ``path/to/file.py:Class`` forms
(see :func:`roqsim.registry.resolve_plugin`) -- **must be quoted** when used as the key, e.g.
``- "my_pkg.mod:MyPlugin": {rays: 90}``. Unquoted it still parses as long as no space follows the
colon, but a stray ``key: value`` space (``- my_pkg.mod: MyPlugin``) would silently split the ref;
quoting keeps the whole ref intact. Short entry-point names never contain a colon, so they need no
quotes.

A world may inherit another with two optional top-level keys (see :func:`_resolve_inheritance`)::

    extends: roqsim_scenes:depot # a parent world YAML ("<package>:<world>" ref or a path)
    disable: [table_2]      # OPTIONAL: drop inherited plugins by name (needs ``extends``)

The parent's ``sim`` is deep-merged (the child wins per key) and the child's ``plugins`` are
appended after the parent's (minus any ``disable``\\ d). To *modify* an inherited plugin, ``disable``
it and re-add a tweaked copy in the child's ``plugins``.

Validation is delegated to each plugin's ``validate_config``; the engine aggregates all errors and
fails fast with one readable report before the build phase.

A caller may override parts of the world before it is built by passing a nested ``overrides`` dict to
:func:`load_config` (see :func:`apply_overrides`). Plugins are addressed **by name** (their ``name:``,
else their plugin ref) rather than by list index, and the value is merged into that plugin's config::

    load_config("world.yaml", {"sim": {"pacing": "asap"},
                               "plugins": {"floorplan": {"floor": {"reflectance": 0.3}}}})

:func:`overrides_from_dotlist` turns ``["plugins.floorplan.floor.reflectance=0.3"]`` into that dict,
for command-line use.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .plugin import Plugin, PluginError
from .registry import resolve_plugin
from .world import resolve_world_yaml_ref

_logger = logging.getLogger("roqsim.config")


@dataclass
class PluginSpec:
    ref: str
    name: str | None
    config: dict


#: Reserved sibling key on a plugins-list entry: names the instance rather than being its plugin ref.
_NAME_KEY = "name"


def entry_ref(entry: dict) -> str:
    """The plugin ref of a plugins-list ``entry`` -- its single non-``name`` key.

    Raises :class:`PluginError` if the entry is not a mapping with exactly one plugin-ref key.
    """
    if not isinstance(entry, dict):
        raise PluginError(f"plugin entry must be a mapping, got {type(entry).__name__}")
    refs = [k for k in entry if k != _NAME_KEY]
    if len(refs) != 1:
        raise PluginError(
            "plugin entry must have exactly one plugin-ref key (plus an optional 'name'); "
            f"got keys {list(entry)}"
        )
    return refs[0]


def parse_plugin_entry(entry: dict, where: str = "plugin entry") -> PluginSpec:
    """Parse one ``plugins:`` list entry (``<ref>: {config}`` + optional ``name:``) into a spec."""
    try:
        ref = entry_ref(entry)
    except PluginError as exc:
        raise PluginError(f"{where}: {exc}") from None
    config = entry[ref]
    if config is not None and not isinstance(config, dict):
        raise PluginError(
            f"{where}: config for '{ref}' must be a mapping, got {type(config).__name__}"
        )
    return PluginSpec(ref=str(ref), name=entry.get(_NAME_KEY), config=dict(config or {}))


@dataclass
class SimConfig:
    sim: dict = field(default_factory=dict)
    plugins: list[PluginSpec] = field(default_factory=list)
    base_dir: Path = field(default_factory=Path.cwd)
    raw: dict = field(default_factory=dict)

    # -- convenience accessors for the ``sim:`` block -----------------------------------------
    @property
    def name(self) -> str | None:
        """Display name for the viewer window (``Roqsim: <name>``); ``None`` keeps MuJoCo's
        model-derived name. Cosmetic, windowed-only (see :func:`roqsim.window_title.retitle_window_async`)."""
        n = self.sim.get("name")
        return None if n is None else str(n)

    @property
    def timestep(self) -> float | None:
        ts = self.sim.get("timestep")
        return None if ts is None else float(ts)

    @property
    def seed(self) -> int | None:
        """The run's noise seed, or ``None`` to draw one.

        Sensor noise is drawn from :meth:`roqsim.context.SimContext.rng_for`, which is
        keyed on this. Stating it here rather than only on the command line is what lets
        a seed be set the same way as every other simulator setting -- through
        ``--set``/``--override``, and therefore from whatever drives the run.

        ``None`` means "draw one and announce it", which is NOT the same as ``0``: a
        seed of zero is a chosen seed like any other.
        """
        v = self.sim.get("seed")
        return None if v is None else int(v)

    @property
    def pacing(self) -> Any:
        return self.sim.get("pacing", "realtime")

    @property
    def sync(self) -> dict:
        return dict(self.sim.get("sync", {}) or {})

    @property
    def view(self) -> dict:
        """Initial viewer setup; windowed only (ignored in headless mode). See :mod:`roqsim.viewer`.

        Free camera (``lookat``/``distance``/``azimuth``/``elevation``) — any subset; omitted keys
        keep MuJoCo's model-derived default. ``track`` (entity or body name) instead attaches the
        camera to a robot, and ``follow_heading`` makes it a chase cam.

        Camera only: the Simulate side panels are run-level switches (``roqsim --left-ui`` /
        ``--right-ui``), and a world still naming ``left_ui``/``right_ui`` here is rejected.
        """
        return dict(self.sim.get("view", {}) or {})


# -- world inheritance (``extends`` / ``disable``) ---------------------------------------------
#
# A world YAML may inherit another world's ``sim`` block and ``plugins`` list, then add, remove, or
# modify elements::
#
#     extends: roqsim_scenes:depot # parent world YAML: a "<package>:<world>" ref or a path
#     sim: {timestep: 0.001}         # deep-merged over the parent's sim (child wins per key)
#     disable: [table_2]      # drop inherited plugins by identity (name: sibling / config name)
#     plugins: [ ... ]               # child plugins are APPENDED after the (kept) parent plugins
#
# The two primitives -- append (child ``plugins``) and remove (``disable``) -- also cover *modify*:
# disable the inherited element and re-add a tweaked copy in the child's ``plugins``.


def _absolutize_world(world: Any, parent_dir: Path) -> Any:
    """Rewrite a parent's *relative file* ``sim.world`` to an absolute path (so it still resolves
    when the inheriting child lives in another dir). Built-in names and ``<package>:<world>`` refs
    resolve globally and are left untouched."""
    if not isinstance(world, str) or ":" in world:
        return world
    is_path = world.endswith((".xml", ".mjcf")) or "/" in world or os.sep in world
    if not is_path or Path(world).is_absolute():
        return world
    return str((parent_dir / world).resolve())


def _resolve_extends_target(ext: Any, base_dir: Path) -> Path:
    """Resolve an ``extends`` value (a ``<package>:<world>`` ref or a path relative to the child
    YAML's dir) to the parent world YAML's absolute path. Fails loud if it does not exist."""
    if not isinstance(ext, str):
        raise PluginError(
            f"'extends' must be a string ('<package>:<world>' ref or a path), got {type(ext).__name__}"
        )
    if ":" in ext and not Path(ext).exists():
        resolved = resolve_world_yaml_ref(ext)
        if resolved is None:
            raise PluginError(
                f"'extends' ref {ext!r} names no known 'roqsim.worlds' provider/module"
            )
        return Path(resolved)
    path = Path(ext)
    if not path.is_absolute():
        path = base_dir / ext
    path = path.resolve()
    if not path.is_file():
        raise PluginError(f"'extends' world YAML not found: {path}")
    return path


def _entry_identities(entry: dict) -> set[str]:
    """Names by which a ``disable`` selector may address a plugin entry: its reserved ``name:``
    sibling and its config ``name`` field (spawn/shelf entries carry the entity name in the config,
    e.g. ``spawn_model: {name: table_2, ...}``)."""
    ids: set[str] = set()
    reserved = entry.get(_NAME_KEY)
    if isinstance(reserved, str):
        ids.add(reserved)
    cfg = entry.get(entry_ref(entry))
    if isinstance(cfg, dict) and isinstance(cfg.get("name"), str):
        ids.add(cfg["name"])
    return ids


def _apply_disable(plugins: list, selectors: list) -> list:
    """Return ``plugins`` with every entry an inherited ``disable`` selector names removed.

    Selectors are plugin names (see :func:`_entry_identities`). A selector matching **no** inherited
    plugin is a hard error -- a typo must not silently keep something the world meant to drop.
    """
    if not isinstance(selectors, list):
        raise PluginError(
            f"'disable' must be a list of plugin names, got {type(selectors).__name__}"
        )
    sel_set: set[str] = set()
    for sel in selectors:
        if not isinstance(sel, str):
            raise PluginError(
                f"'disable' entries must be strings (plugin names), got {type(sel).__name__}"
            )
        sel_set.add(sel)
    kept, matched = [], set()
    for entry in plugins:
        drop = _entry_identities(entry) & sel_set
        if drop:
            matched |= drop
        else:
            kept.append(entry)
    unmatched = sel_set - matched
    if unmatched:
        raise PluginError(f"'disable' selector(s) matched no inherited plugin: {sorted(unmatched)}")
    return kept


def _resolve_inheritance(raw: dict, base_dir: Path, seen: frozenset[Path] = frozenset()) -> dict:
    """Expand an ``extends``/``disable`` world into a plain ``{sim, plugins}`` dict.

    Recursively merges the parent world (which may itself ``extends``): ``sim`` is deep-merged with
    the child winning, and ``plugins`` becomes ``(parent - disabled) + child``. A no-op when the
    world declares no ``extends``. Cycles raise.
    """
    if not isinstance(raw, dict):
        raise PluginError("world config must be a mapping at the top level")
    ext = raw.get("extends")
    disable = raw.get("disable")
    if ext is None:
        if disable is not None:
            raise PluginError(
                "'disable' requires 'extends' (nothing to disable without a parent world)"
            )
        return raw

    parent_path = _resolve_extends_target(ext, base_dir)
    if parent_path in seen:
        chain = " -> ".join(str(p) for p in (*seen, parent_path))
        raise PluginError(f"'extends' cycle detected: {chain}")
    with parent_path.open() as fh:
        parent_raw = yaml.safe_load(fh) or {}
    if not isinstance(parent_raw, dict):
        raise PluginError(f"extended world {parent_path} must be a mapping at the top level")
    parent_raw = _resolve_inheritance(parent_raw, parent_path.parent, seen | {parent_path})

    parent_sim = dict(parent_raw.get("sim") or {})
    if "world" in parent_sim:
        parent_sim["world"] = _absolutize_world(parent_sim["world"], parent_path.parent)
    merged_sim = deep_merge(parent_sim, raw.get("sim") or {})

    kept = _apply_disable(list(parent_raw.get("plugins") or []), disable or [])
    merged = {k: v for k, v in raw.items() if k not in ("extends", "disable")}
    merged["sim"] = merged_sim
    merged["plugins"] = kept + list(raw.get("plugins") or [])
    return merged


def load_config(
    path: str | Path, overrides: dict | None = None, transport: dict | None = None
) -> SimConfig:
    """Parse a world YAML file into a :class:`SimConfig` (does not instantiate plugins).

    If the YAML declares ``extends`` it is first expanded against its parent world (see
    :func:`_resolve_inheritance`). ``overrides`` is an optional nested dict deep-merged into the
    parsed YAML *before* the :class:`SimConfig` is built -- see :func:`apply_overrides`. It must be
    applied here (not to a built ``SimConfig``) because :func:`_from_dict` copies ``sim`` and each
    plugin's ``config``.

    ``transport`` is :func:`with_transport`'s keyword arguments (e.g. ``{"ros": True}``), applied
    after the overrides. Injected here rather than authored into the world so a checked-in world
    stays ROS-free and standalone-runnable.
    """
    # A `<package>:<world>` ref is accepted here, not just as an `extends:` target. `roqsim sim` and the
    # scenario adapter each resolved one before calling in, so the ref was a property of those two
    # entry points rather than of a world -- and `roqsim-export-web` (which no human types, but which
    # RoboVAST's scene cache runs to build a run view's geometry) got the ref verbatim and reported
    # "world config /abs/cwd/tiago_pick:tiago_pick_ros2 does not exist". Downstream that surfaces
    # as "no 3D geometry" for the whole campaign, so the campaign looks broken rather than one caller.
    #
    # Guarded on the path not existing, the same way `extends` and `sim.world` decide: a real file
    # whose name contains a colon is still a file.
    path = Path(path)
    if ":" in str(path) and not path.exists():
        resolved = resolve_world_yaml_ref(str(path))
        if resolved is None:
            raise PluginError(
                f"world ref {str(path)!r} names no known 'roqsim.worlds' provider/module"
            )
        path = Path(resolved)
    path = path.resolve()
    if not path.exists():
        raise PluginError(f"world config {path} does not exist")
    with path.open() as fh:
        raw = yaml.safe_load(fh) or {}
    raw = _resolve_inheritance(raw, path.parent)
    if overrides:
        raw = apply_overrides(raw, overrides)
    if transport:
        raw = with_transport(raw, **transport)
    return _from_dict(raw, base_dir=path.parent)


def world_sources(path: str | Path) -> list[Path]:
    """Every file a world is *defined by*: the YAML, its ``extends`` ancestors, the MJCF they
    name, that MJCF's mesh/texture assets, and whatever its plugins point at.

    A caller that caches something compiled from a world -- an export, a baked scene, a
    thumbnail -- needs to know when to redo it, and the leaf YAML alone is the wrong answer:
    a world that ``extends: roqsim_scenes:depot`` changes when the *parent* changes, or when a
    mesh under the parent's ``assets/`` is replaced. Asking the config layer, which already
    walks the chain to load it, keeps that answer in one place instead of hand-listed by
    every caller.

    A plugin's own config is the one place the YAML walk cannot see: a floorplan's mesh and the
    json-ld beside it, a trajectory CSV. Those are asked of the plugin (:meth:`roqsim.plugin.
    Plugin.sources`), because only it knows which of its keys are paths and how they resolve.

    Best-effort by design: a world that cannot be parsed yields what was resolved so far
    rather than raising, because a caller asking "what does this depend on?" is usually
    about to report a *different* error and should not be pre-empted by this one.
    """
    found: list[Path] = []
    seen: set[Path] = set()

    def _add(candidate: Path) -> None:
        candidate = candidate.resolve()
        if candidate not in seen and candidate.is_file():
            seen.add(candidate)
            found.append(candidate)

    def _walk(yaml_path: Path) -> dict:
        _add(yaml_path)
        try:
            with yaml_path.open() as fh:
                raw = yaml.safe_load(fh) or {}
        except (OSError, yaml.YAMLError):
            return {}
        if not isinstance(raw, dict):
            return {}
        ext = raw.get("extends")
        if ext is not None:
            try:
                _walk(_resolve_extends_target(ext, yaml_path.parent))
            except PluginError:
                _logger.debug("world_sources: unresolvable 'extends' %r in %s", ext, yaml_path)
        return raw

    leaf = Path(path).resolve()
    raw = _walk(leaf)

    # The MJCF the chain settles on, plus its asset dirs. Resolved through the same helper the
    # engine uses, so a built-in name or a '<package>:<world>' ref behaves identically here.
    cfg = None
    try:
        cfg = load_config(leaf)
        world_name = cfg.sim.get("world")
        base_dir = cfg.base_dir
    except Exception:  # noqa: BLE001 - see docstring: never pre-empt the caller's own error
        world_name, base_dir = (raw.get("sim") or {}).get("world"), leaf.parent
    if world_name:
        from .world import world_file  # noqa: PLC0415 - avoids a config<->world import cycle

        try:
            mjcf = world_file(world_name, base_dir)
        except Exception:  # noqa: BLE001
            mjcf = None
        if mjcf:
            mjcf_path = Path(mjcf)
            _add(mjcf_path)
            for asset_dir in _mjcf_asset_dirs(mjcf_path):
                for asset in sorted(asset_dir.rglob("*")):
                    _add(asset)

    for source in _plugin_sources(cfg, leaf):
        _add(Path(source))
    return found


def _plugin_sources(cfg, leaf: Path) -> list:
    """What the world's plugins say they point at, asked plugin by plugin.

    Resolved **per spec, skipping what does not resolve**, rather than through
    :func:`_expand_plugins` -- that raises on the first unresolvable ref, which would turn any
    ROS world into a hard failure here in a pip-only environment where the bridge is not
    installed. This function's contract is best-effort (see :func:`world_sources`), and the
    same tolerance the ``extends`` walk already has.
    """
    if cfg is None:
        return []
    found: list = []
    for spec in getattr(cfg, "plugins", []) or []:
        try:
            plugin_cls = resolve_plugin(spec.ref, base_dir=cfg.base_dir)
            found.extend(plugin_cls(spec.config, name=spec.name).sources() or [])
        except Exception as exc:  # noqa: BLE001 - see docstring: best-effort by design
            _logger.debug(
                "world_sources: no sources from plugin %r in %s (%s)", spec.ref, leaf, exc
            )
    return found


def _mjcf_asset_dirs(mjcf: Path) -> list[Path]:
    """``meshdir``/``texturedir`` of an MJCF's ``<compiler>``, resolved against its own dir."""
    try:
        import xml.etree.ElementTree as ET  # noqa: PLC0415 - only needed on this path

        compiler = ET.parse(mjcf).getroot().find("compiler")  # noqa: S314 - our own asset
    except Exception:  # noqa: BLE001
        return []
    if compiler is None:
        return []
    dirs = []
    for attr in ("meshdir", "texturedir", "assetdir"):
        value = compiler.get(attr)
        if not value:
            continue
        candidate = (mjcf.parent / value).resolve()
        if candidate.is_dir() and candidate not in dirs:
            dirs.append(candidate)
    return dirs


def load_config_from_dict(raw: dict, base_dir: str | Path | None = None) -> SimConfig:
    """Build a :class:`SimConfig` from an already-parsed dict (used in tests).

    Honours ``extends``/``disable`` (resolved relative to ``base_dir``), like :func:`load_config`.
    """
    base = Path(base_dir) if base_dir else Path.cwd()
    return _from_dict(_resolve_inheritance(raw, base), base_dir=base)


# -- world overrides ---------------------------------------------------------------------------
#
# A caller (e.g. a scenario parameter, or the CLI's ``--set``) may override any part of the world
# before it is built. Overrides are a *nested dict* mirroring the world YAML, deep-merged into it.
# The one wrinkle: ``plugins:`` is a YAML **list**, but overrides address plugins **by name** so a
# caller never has to know list indices; each value is merged into that plugin's config::
#
#     {"sim": {"pacing": "asap"},
#      "plugins": {"floorplan": {"size": 4.0}}}
#
# A plugin key matches its ``name:`` first, then its plugin ref.


def _model_default_hint(raw_plugins: list, key: str) -> str:
    """Extra guidance when the named plugin may be running as a MODEL DEFAULT.

    A spawn plugin pulls its model's manifest plugins in at expansion time, which happens
    *after* overrides are applied here -- so a world that just spawns a robot has no
    ``lidar`` entry to address even though a lidar is very much running. Without this the
    refusal reads as "there is no such plugin", and the honest answer is "it exists, it is
    just not addressable until the world names it".

    Only offered when a model is actually spawned; otherwise the hint would be noise.
    """
    models = []
    for entry in raw_plugins:
        cfg = entry.get(entry_ref(entry))
        if isinstance(cfg, dict) and cfg.get("model"):
            models.append((str(cfg["model"]), cfg.get("name", "robot")))
    if not models:
        return ""
    names = ", ".join(sorted({m for m, _ in models}))
    entity = models[0][1]
    return (
        f". A model's default plugins (from {names}) are merged in after overrides are "
        f"applied, so they cannot be addressed until the world names them. Declare a stub "
        f"-- '{key}: {{robot: {entity}}}' -- and the manifest still fills in the rest; "
        f"only the keys the world or the override sets are changed"
    )


def _resolve_plugin_index(raw_plugins: list, key: str | int) -> int:
    """Map a plugin override key (instance name, plugin ref, or index) to its index in the list."""
    if isinstance(key, int) or (isinstance(key, str) and key.lstrip("-").isdigit()):
        idx = int(key)
        if not -len(raw_plugins) <= idx < len(raw_plugins):
            raise PluginError(f"plugin override index {idx} out of range")
        return idx
    by_name = [i for i, e in enumerate(raw_plugins) if e.get("name") == key]
    matches = by_name or [i for i, e in enumerate(raw_plugins) if entry_ref(e) == key]
    if not matches:
        known = [e.get("name") or entry_ref(e) for e in raw_plugins]
        raise PluginError(
            f"plugin override '{key}' matches no plugin in the world (have: {known})"
            + _model_default_hint(raw_plugins, key)
        )
    if len(matches) > 1:
        raise PluginError(
            f"plugin override '{key}' is ambiguous ({len(matches)} plugins match); "
            "give the plugin an explicit 'name:' in the world"
        )
    return matches[0]


def deep_merge(base: Any, override: Any) -> Any:
    """Recursively merge ``override`` into ``base``. Non-dict values (incl. lists) replace."""
    if isinstance(base, dict) and isinstance(override, dict):
        merged = dict(base)
        for key, value in override.items():
            merged[key] = deep_merge(merged.get(key), value) if key in merged else value
        return merged
    return override


def apply_overrides(raw: dict, overrides: dict) -> dict:
    """Return a copy of the parsed world ``raw`` with the nested ``overrides`` deep-merged in.

    ``overrides['plugins']`` is a mapping keyed by plugin name/ref (or index), which is resolved
    against the world's plugin list; each value is deep-merged into that plugin's config. Everything
    else merges by key.
    """
    if not isinstance(overrides, dict):
        raise PluginError(f"world overrides must be a mapping, got {type(overrides).__name__}")
    result = dict(raw)
    plugin_overrides = overrides.get("plugins")
    for key, value in overrides.items():
        if key != "plugins":
            result[key] = deep_merge(result.get(key), value)

    if plugin_overrides is None:
        return result
    if not isinstance(plugin_overrides, dict):
        raise PluginError(
            "'plugins' override must be a mapping keyed by plugin name/ref, "
            f"got {type(plugin_overrides).__name__}"
        )
    raw_plugins = [dict(e) for e in (raw.get("plugins") or [])]
    for key, value in plugin_overrides.items():
        idx = _resolve_plugin_index(raw_plugins, key)
        entry = raw_plugins[idx]
        ref = entry_ref(entry)
        entry[ref] = deep_merge(entry.get(ref), value)
    result["plugins"] = raw_plugins
    return result


def overrides_from_files(paths) -> dict:
    """Load and merge override documents from *paths*, in order (later wins).

    The **file spelling of** ``--set``: the same nested mapping
    :func:`apply_overrides` takes, kept in a file instead of flattened onto a command
    line. That matters for anything structured -- a list of obstacle instances, a nested
    plugin config -- which survives a file intact and does not survive argv at all once
    quoting and word splitting have had it.

    Repeatable and mergeable so a saved override set can be combined with an ad-hoc one:
    ``roqsim sim world.yaml --override debug.yaml --set sim.pacing=asap``.
    """
    merged: dict = {}
    for path in paths or []:
        try:
            with open(path, encoding="utf-8") as handle:
                loaded = yaml.safe_load(handle)
        except OSError as err:
            raise PluginError(f"override file {path!r} could not be read: {err}") from None
        except yaml.YAMLError as err:
            raise PluginError(f"override file {path!r} is not valid YAML: {err}") from None
        if loaded is None:
            continue
        if not isinstance(loaded, dict):
            raise PluginError(
                f"override file {path!r} must contain a mapping, got {type(loaded).__name__}"
            )
        merged = deep_merge(merged, loaded)
    return merged


def overrides_from_dotlist(dotlist: list[str]) -> dict:
    """Parse ``["a.b.c=4.0", ...]`` into the nested override dict :func:`apply_overrides` expects.

    Values are parsed with ``yaml.safe_load`` (so ``4.0``/``true``/``[1,2]`` get their natural type;
    quote a value to force a string). A convenience for the CLI's ``--set``; scenarios pass the
    nested dict directly.
    """
    overrides: dict = {}
    for item in dotlist or []:
        path, sep, value = str(item).partition("=")
        if not sep:
            raise PluginError(f"override '{item}' is not of the form path.to.key=value")
        keys = [k for k in path.strip().split(".") if k]
        if not keys:
            raise PluginError(f"override '{item}' has an empty path")
        node = overrides
        for key in keys[:-1]:
            node = node.setdefault(key, {})
        node[keys[-1]] = yaml.safe_load(value)
    return overrides


#: The complete ``sim.view`` schema -- the camera, and nothing else. Anything else there is a typo or
#: a run-level switch that does not belong in a world, and is rejected rather than silently dropped.
_VIEW_KEYS = frozenset({"lookat", "distance", "azimuth", "elevation", "track", "follow_heading"})

#: ``sim.view``'s value shapes: how many numbers each numeric key carries. ``track`` is an entity or
#: body name and ``follow_heading`` a flag, so they are checked on their own.
_VIEW_NUMERIC_WIDTHS = {"lookat": 3, "distance": 1, "azimuth": 1, "elevation": 1}


def _validate_view(view) -> None:
    """Reject a malformed ``sim.view`` at load time, not frames later inside the camera.

    The shape that motivated this is ``lookat: "1,2,0"`` -- a string where three numbers belong.
    Nothing rejected it, so it travelled all the way to ``[float(v) for v in view["lookat"]]`` in
    :func:`roqsim.viewer.apply_view`, which iterates the *characters* of the string and dies on
    ``float('.')``: an error naming a decimal point, no key, and no file.
    """
    if view is None:
        return
    if not isinstance(view, dict):
        raise PluginError("sim.view must be a mapping of camera keys -> values")
    for key, width in _VIEW_NUMERIC_WIDTHS.items():
        if (value := view.get(key)) is None:
            continue
        values = value if isinstance(value, (list, tuple)) else [value]
        numbers = len(values) == width and all(
            isinstance(v, (int, float)) and not isinstance(v, bool) for v in values
        )
        if not numbers:
            want = f"{width} numbers, e.g. [1, 2, 0]" if width > 1 else "a number"
            raise PluginError(f"sim.view.{key}: expected {want}, got {value!r}")
    if (track := view.get("track")) is not None and not isinstance(track, str):
        raise PluginError(f"sim.view.track: expected an entity or body name, got {track!r}")


#: The complete ``sim.contact_override`` schema -- MuJoCo's three global contact overrides and their
#: vector widths. Rejected rather than dropped, because a typo here is invisible: an ignored
#: ``o_solref`` produces a model that compiles, runs, and quietly uses the untouched defaults.
_CONTACT_OVERRIDE_WIDTHS = {"solref": 2, "solimp": 5, "friction": 5}


def _validate_seed(seed) -> None:
    """Reject a malformed ``sim.seed`` at load time.

    Loudly, and here rather than at first use: a seed that quietly became ``0`` would
    give every run of a campaign one noise draw while still looking varied, and nothing
    downstream could tell that from a deliberate ``seed: 0``.
    """
    if seed is None:
        return
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise PluginError(f"sim.seed must be a non-negative integer, got {type(seed).__name__}")
    if seed < 0:
        raise PluginError(f"sim.seed must be a non-negative integer, got {seed}")


def _validate_contact_override(override) -> None:
    """Reject a malformed ``sim.contact_override`` at load time, not at compile time."""
    if override is None:
        return
    if not isinstance(override, dict):
        raise PluginError(
            "sim.contact_override must be a mapping of "
            f"{{{', '.join(sorted(_CONTACT_OVERRIDE_WIDTHS))}}} -> value(s)"
        )
    if unknown := sorted(set(override) - set(_CONTACT_OVERRIDE_WIDTHS)):
        raise PluginError(
            f"sim.contact_override: unknown key(s) {', '.join(unknown)}; MuJoCo's global contact "
            f"overrides are {', '.join(sorted(_CONTACT_OVERRIDE_WIDTHS))} "
            "(o_solref / o_solimp / o_friction). Per-geom values belong in the model."
        )
    for key, width in _CONTACT_OVERRIDE_WIDTHS.items():
        if (value := override.get(key)) is None:
            continue
        values = value if isinstance(value, (list, tuple)) else [value]
        if not values or len(values) > width:
            raise PluginError(
                f"sim.contact_override.{key}: expected 1..{width} numbers, got {len(values)}. "
                "A short vector varies its leading elements and leaves the rest at MuJoCo's values."
            )
        if any(not isinstance(v, (int, float)) or isinstance(v, bool) for v in values):
            raise PluginError(f"sim.contact_override.{key}: all entries must be numbers")


def _from_dict(raw: dict, base_dir: Path) -> SimConfig:
    if not isinstance(raw, dict):
        raise PluginError("world config must be a mapping at the top level")
    if "headless" in (raw.get("sim") or {}):
        _logger.warning(
            "\u26a0\ufe0f  sim.headless is IGNORED: the viewer is windowed by default; run with "
            "--headless (standalone) or the headless scenario parameter to suppress the window. "
            "Remove the key from the world YAML to silence this."
        )
    unknown = sorted(set((raw.get("sim") or {}).get("view") or {}) - _VIEW_KEYS)
    if unknown:
        raise PluginError(
            f"sim.view: unknown key(s) {', '.join(unknown)}; sim.view configures the camera only "
            f"({', '.join(sorted(_VIEW_KEYS))}). The viewer's side panels are run-level switches: "
            "see 'roqsim --help'."
        )
    _validate_view((raw.get("sim") or {}).get("view"))
    _validate_seed((raw.get("sim") or {}).get("seed"))
    _validate_contact_override((raw.get("sim") or {}).get("contact_override"))
    plugins = [
        parse_plugin_entry(entry, f"plugins[{i}]")
        for i, entry in enumerate(raw.get("plugins", []) or [])
    ]
    return SimConfig(
        sim=dict(raw.get("sim", {}) or {}), plugins=plugins, base_dir=base_dir, raw=raw
    )


#: Plugin refs :func:`with_transport` appends, in order. Names rather than classes: they
#: are registered by a colcon package (``roqsim_ros_bridge``) that a pip-only environment
#: does not have, and resolution happens later, where a missing one is a loud failure.
_ROS_TRANSPORT = "ros2_bridge"
_ROS_CONTROL = "sim_interfaces"


def with_transport(
    raw: dict, *, ros: bool = True, tf_namespace: str | None = None, control: bool = False
) -> dict:
    """Return *raw* with a transport plugin appended, if it has none.

    The exact inverse of :func:`drop_transport_plugins`, and the reason a checked-in
    world can stay **ROS-free**: a world that declares ``ros2_bridge`` cannot be run by
    ``roqsim sim`` in a pip-only environment, so keeping the bridge out of the file is what
    makes it standalone-runnable. Transport is a property of *how a run is deployed*,
    not of the experiment, so it is added here rather than authored into the world.

    This replaces hand-rolled rewrites that appended the same plugin to a temporary copy
    of the world -- one of them implemented twice, in bash and in Python, in a single
    experiment. Those also had to re-resolve the scene's relative path because the copy
    lived elsewhere; nothing is copied here, so that problem does not arise.

    ``control`` additionally serves the ``simulation_interfaces`` control plane, which is
    what a scenario's ``osc.sim`` actions are clients of. Off by default: it is a handful
    of extra services that only a scenario touching entities needs.

    Idempotent -- a world that already declares a transport is returned unchanged, so a
    caller never has to know whether the author put one there.
    """
    if not ros:
        return raw
    plugins = list(raw.get("plugins") or [])
    if any(
        entry_ref(entry) in (_ROS_TRANSPORT, _ROS_CONTROL)
        for entry in plugins
        if isinstance(entry, dict)
    ):
        return raw
    bridge: dict = {}
    if tf_namespace:
        # Topic-only: /<ns>/tf and /<ns>/tf_static, frames unchanged. Nav2 launches with
        # the standard /tf -> tf remap, and scenario-execution's transform listener is
        # namespaced too; without this both look under /<ns>/tf while the bridge
        # publishes globally, and init_nav2 hangs on "Waiting for transform".
        bridge["tf_namespace"] = tf_namespace
    result = dict(raw)
    result["plugins"] = (
        plugins + [{_ROS_TRANSPORT: bridge}] + ([{_ROS_CONTROL: {}}] if control else [])
    )
    return result


def drop_transport_plugins(cfg: SimConfig) -> tuple[list[str], list[str]]:
    """Keep only the plugins that build the scene; return ``(transport, unavailable)`` labels dropped.

    For a consumer that wants the *world* rather than a running simulation -- ``roqsim render``, the
    scene-review window, the exporters -- a transport plugin is dead weight: it publishes what the
    others built and adds nothing to the model. Two kinds go:

    * a plugin whose class declares :attr:`roqsim.plugin.Plugin.transport_only` (``BridgeBase`` and any
      subclass, so an out-of-tree transport is covered without anyone listing its name here);
    * a ref that does not resolve at all -- typically the ROS bridge, registered by a colcon package
      and so absent from a pip-only environment. It cannot contribute geometry either way, so a
      picture of the world is still exactly right; the caller warns and names it, because the other
      way to get here is a typo.

    Returned separately so the caller can log the expected case and the surprising one at different
    levels. A simulation must not use this: dropping *any* unresolvable ref would silently swallow a
    misspelt geometry plugin, which changes the run. ``roqsim sim --no-communication`` uses :func:`drop_transport`
    instead, which only removes what it can identify as transport.
    """
    transport: list[str] = []
    unavailable: list[str] = []
    kept: list[PluginSpec] = []
    for spec in cfg.plugins:
        label = f"{spec.name} ({spec.ref})" if spec.name else spec.ref
        try:
            cls = resolve_plugin(spec.ref, base_dir=cfg.base_dir)
        except PluginError:
            unavailable.append(label)
            continue
        if cls.transport_only:
            transport.append(label)
            continue
        kept.append(spec)
    cfg.plugins = kept
    return transport, unavailable


def drop_transport(cfg: SimConfig) -> list[str]:
    """Drop the transport plugins from *cfg* and return their labels -- for a deliberately offline run.

    The inverse of :func:`with_transport` for a world whose author put the bridge in the file, and the
    narrow counterpart of :func:`drop_transport_plugins`: a plugin goes only when it can be *identified*
    as transport, never merely because this environment cannot load it. An unresolvable ref that is not
    a bridge stays in the list, so a misspelt geometry plugin remains the loud failure a simulation
    needs it to be.

    Two ways to be identified, one per environment:

    * the ref resolves and the class declares :attr:`roqsim.plugin.Plugin.transport_only` -- the middleware
      is installed and the caller is simply choosing not to publish;
    * the ref is one of the bridges this module already names for :func:`with_transport`, in a pip-only
      environment where it does not resolve at all. Nothing else can identify it there: the class that
      would declare the capability is precisely what is missing.

    **The caller must say out loud what the run gives up** (see ``roqsim sim --no-communication``). A simulation
    stripped of its transport is not a quieter run of the same experiment -- it publishes nothing and
    receives nothing, so every consumer outside the process is looking at a simulator that, as far as
    it can tell, was never started.
    """
    dropped: list[str] = []
    kept: list[PluginSpec] = []
    for spec in cfg.plugins:
        try:
            is_transport = resolve_plugin(spec.ref, base_dir=cfg.base_dir).transport_only
        except PluginError:
            is_transport = spec.ref in (_ROS_TRANSPORT, _ROS_CONTROL)
        if is_transport:
            dropped.append(f"{spec.name} ({spec.ref})" if spec.name else spec.ref)
        else:
            kept.append(spec)
    cfg.plugins = kept
    return dropped


def _expand_plugins(cfg: SimConfig) -> list[tuple[PluginSpec, type[Plugin]]]:
    """Resolve each plugin class and splice in any specs a plugin's ``expand`` implies.

    Expansion-injected specs (e.g. a spawn plugin's model manifest) land right after the plugin that
    produced them. Whether to skip a default the world already declares is the producing plugin's
    call -- it gets the world's explicit specs and dedupes on its own entity key (see
    :func:`roqsim.manifest.expand_manifest`). Classes are resolved once here and handed back.

    Resolution failures are collected rather than raised at the first one, like the validation errors
    in :func:`instantiate_plugins`: one report listing every bad ref beats three edit-and-rerun cycles,
    and -- the reason it was worth changing -- only the full list can tell whether the world failed
    *solely* on its transport, which is a fixable situation rather than a mistake (see
    :func:`_unresolved_message`).
    """
    resolved: list[tuple[PluginSpec, type[Plugin]]] = []
    unresolved: list[tuple[str, str]] = []
    for spec in cfg.plugins:
        try:
            cls = resolve_plugin(spec.ref, base_dir=cfg.base_dir)
        except PluginError as exc:
            unresolved.append((spec.ref, str(exc)))
            continue
        resolved.append((spec, cls))
        for sub in cls.expand(spec, cfg.plugins, cfg.base_dir):
            try:
                resolved.append((sub, resolve_plugin(sub.ref, base_dir=cfg.base_dir)))
            except PluginError as exc:
                unresolved.append((sub.ref, str(exc)))
    if unresolved:
        raise PluginError(_unresolved_message(unresolved))
    return resolved


def _unresolved_message(unresolved: list[tuple[str, str]]) -> str:
    """The report for refs that did not resolve -- with the way out when they are all transport.

    A world whose *only* unresolvable plugins are bridges is not misspelt; it is a ROS world being run
    somewhere the middleware is not installed, and that has two real answers. Saying so beats the bare
    "unknown plugin" that sent the reader looking for a typo that was never there.

    The distinction has to be made from the ref alone: the class that would declare
    :attr:`roqsim.plugin.Plugin.transport_only` is exactly what could not be imported. So it rests on the
    same two names :func:`with_transport` injects -- an out-of-tree transport is indistinguishable from
    a typo here and correctly gets the plain report.
    """
    detail = (
        unresolved[0][1]
        if len(unresolved) == 1
        else "unresolved plugins:\n  - " + "\n  - ".join(msg for _ref, msg in unresolved)
    )
    if not all(ref in (_ROS_TRANSPORT, _ROS_CONTROL) for ref, _msg in unresolved):
        return detail
    names = ", ".join(ref for ref, _msg in unresolved)
    return (
        f"{detail}\n"
        f"\n"
        f"{names}: transport, not scene -- this world publishes over ROS 2, and the bridge ships in "
        f"the colcon package roqsim_ros_bridge rather than on PyPI, so a pip-only environment cannot "
        f"resolve it. Two ways on:\n"
        f"  - source the ROS 2 overlay (ros2_ws/install/setup.bash) and run the world as authored;\n"
        f"  - pass --no-communication to drop the transport and run the world mute -- it then "
        f"publishes nothing and receives nothing, so it is good for looking at the scene and not for "
        f"running the experiment."
    )


def instantiate_plugins(cfg: SimConfig) -> list[Plugin]:
    """Resolve and construct every plugin in order, then run aggregated config validation.

    Raises :class:`PluginError` listing *all* validation errors if any plugin rejects its config.
    """
    resolved = _expand_plugins(cfg)
    instances: list[Plugin] = [cls(spec.config, name=spec.name) for spec, cls in resolved]

    errors: list[str] = []
    for inst, (spec, _cls) in zip(instances, resolved, strict=True):
        try:
            found = inst.validate_config(inst.config) or []
        except Exception as exc:  # a plugin's validator itself blew up
            found = [f"validate_config raised: {exc}"]
        errors.extend(f"[{inst.name} ({spec.ref})] {msg}" for msg in found)

    if errors:
        raise PluginError("plugin config validation failed:\n  - " + "\n  - ".join(errors))
    return instances
