"""Model discovery: resolve a spawn plugin's ``model:`` string to an MJCF file + its asset dirs.

Mirrors plugin resolution (see :mod:`roqsim.registry`), so a spawn plugin can place a model that
lives in *any* installed package -- not just its own:

1. **Short name** -> looked up across every model provider registered in the ``roqsim.models``
   entry-point group. e.g. ``ur10e`` (from ``roqsim_manipulation``) or ``conveyor`` (from
   ``roqsim_assets``) -- the caller need not know which package ships it.
2. **Package-qualified ``<package>:<model>``** -> ``model`` resolved within that provider's directory.
   e.g. ``roqsim_manipulation:ur10e``. Use this to disambiguate when two packages ship a model
   of the same name, or to self-document which package a world depends on. ``<package>`` is a
   ``roqsim.models`` provider's entry-point name; a dotted importable module exposing ``MODELS_DIR``
   also works (e.g. ``my_pkg.models:foo``) for models outside the entry-point group.
3. **Filesystem path** -> loaded directly. Absolute, else relative to exactly one anchor:
   ``base_dir`` -- the directory of the document that named it -- or the working directory
   for a caller that has no document, such as a path typed on a command line. There is no
   second attempt: a fallback would let one document naming one string resolve to different
   files depending on where the process was started, and neither the world nor the run
   records which one it got.
   e.g. ``./models/my_arm.xml``.

Crucially, each resolved model carries **its own** ``meshdir``/``texturedir`` (its provider's, or
the model file's own directory), so a model bundled in one package uses that package's meshes -- a
spawn plugin from another package no longer forces its own asset dirs onto it (which is what made a
cross-package model impossible before).

**Borrowing another package's assets.** A model that reuses another package's meshes (e.g. a custom
arm variant that keeps the stock arm meshes) need not copy them: add an ``assets:`` key to its
``<model>.manifest.yaml`` naming the provider(s) to borrow mesh/texture dirs from -- a single name or
a list::

    assets: roqsim_manipulation                    # one provider
    assets: [roqsim_manipulation, roqsim_sensors]  # e.g. arm meshes + a d435 camera mesh

Only the ``.xml`` and ``.manifest.yaml`` then live in the new package; the meshes stay put. Because
MuJoCo allows a single ``meshdir`` per model, borrowed meshes/textures are resolved to absolute paths
and written onto the child spec by :func:`apply_assets` -- so meshes from *several* packages can
coexist in one model. Search order: the model's **own** declared ``meshdir`` first (its assets can
never be shadowed by same-named provider files), then the listed providers in order, then the
provider default and the model file's dir.

A **provider** is a module exposing a ``MODELS_DIR`` path (and optionally ``MESHES_DIR`` /
``TEXTUREDIR``; they default to ``MODELS_DIR/"meshes"`` and ``MODELS_DIR``). A package registers one
in its ``pyproject.toml``::

    [project.entry-points."roqsim.models"]
    manipulation = "roqsim_manipulation.models"
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from importlib import import_module, metadata
from pathlib import Path

import yaml

ENTRY_POINT_GROUP = "roqsim.models"


class ModelError(Exception):
    """Raised when a ``model:`` reference cannot be resolved to a model file."""


@dataclass(frozen=True)
class ModelAsset:
    """A resolved model: the MJCF file plus the (ordered) dirs its meshes/textures resolve against.

    ``meshdirs``/``texturedirs`` are searched in order (borrowed providers first, then the model's
    own dir); :func:`apply_assets` uses them to rewrite the child spec's file refs to absolute paths.
    """

    path: Path
    meshdirs: tuple[Path, ...]
    texturedirs: tuple[Path, ...]

    @property
    def meshdir(self) -> Path:
        """Primary mesh dir (first searched) -- convenience for single-provider models."""
        return self.meshdirs[0]

    @property
    def texturedir(self) -> Path:
        return self.texturedirs[0]


def apply_assets(spec, asset: ModelAsset) -> None:
    """Rewrite a child spec's mesh/texture file refs to absolute paths found across ``asset`` dirs.

    Called by spawn plugins after ``MjSpec.from_file`` instead of setting a single ``meshdir`` -- so a
    model may draw meshes/textures from more than one package (see ``assets:`` in the module docstring).

    The spec's **own** ``meshdir``/``texturedir`` (as declared in its ``<compiler>``, joined with the
    model file's dir) is searched **first**: a model's own assets must never be shadowed by
    same-named files in a provider dir. This bit for real -- the Oli's ``left_hip_yaw_link.STL``
    (in ``meshes/oli/``) collides with the G1's Menagerie mesh of the same name in the provider's
    flat ``meshes/`` dir, and resolving provider-first silently dressed the Oli in G1 limb shells.
    """

    def _own_dir(subdir: str) -> tuple[Path, ...]:
        # spec.modelfiledir is set by MjSpec.from_file; in-memory specs have neither dir.
        base = getattr(spec, "modelfiledir", "") or ""
        sub = subdir or ""
        if not base and not sub:
            return ()
        return (Path(base) / sub,)

    def _resolve(fname: str, dirs: tuple[Path, ...]) -> Path | None:
        p = Path(fname)
        if p.is_absolute():
            return p if p.exists() else None
        for d in dirs:
            cand = d / fname
            if cand.exists():
                return cand.resolve()
        return None

    meshdirs = _own_dir(getattr(spec, "meshdir", "")) + asset.meshdirs
    texturedirs = _own_dir(getattr(spec, "texturedir", "")) + asset.texturedirs
    for mesh in spec.meshes:
        if mesh.file:
            resolved = _resolve(mesh.file, meshdirs)
            if resolved is not None:
                mesh.file = str(resolved)
    for tex in getattr(spec, "textures", []):
        fname = getattr(tex, "file", "") or ""
        if fname:
            resolved = _resolve(fname, texturedirs)
            if resolved is not None:
                tex.file = str(resolved)


@functools.cache
def _entry_points(group: str):
    """Return entry points for ``group`` across Python versions (see registry._entry_points).

    Cached for the same reason as there: the scan is ~30 ms and spawn-heavy worlds trigger it
    once per spawn plugin (validate + build), which dominated world loading.
    """
    eps = metadata.entry_points()
    if hasattr(eps, "select"):  # Python 3.10+
        return tuple(eps.select(group=group))
    return tuple(eps.get(group, ()))  # pragma: no cover - legacy


def _provider_dirs(module) -> tuple[Path, Path, Path]:
    """(models_dir, meshdir, texturedir) for a provider module (latter two default off MODELS_DIR)."""
    models_dir = Path(module.MODELS_DIR)
    meshdir = Path(getattr(module, "MESHES_DIR", models_dir / "meshes"))
    texturedir = Path(getattr(module, "TEXTUREDIR", models_dir))
    return models_dir, meshdir, texturedir


def _find_in_dir(models_dir: Path, name: str) -> Path | None:
    """A model file for ``name`` under ``models_dir``.

    Accepts a bare stem or filename (``<models_dir>/<name>.xml``) and the nested
    ``<models_dir>/<name>/<name>.xml`` layout used by props and baked scenes (mirrors
    :func:`roqsim.world._find_world`). Only files are returned, never a directory.
    """
    for candidate in (
        models_dir / name,
        models_dir / f"{name}.xml",
        models_dir / name / f"{name}.xml",
        # Filename form against the nested layout ("d435.xml" -> d435/d435.xml). Without this the two
        # forms this function documents do not compose, and a provider that migrates from flat to
        # one-folder-per-model silently stops answering to `<name>.xml`.
        models_dir / Path(name).stem / name,
    ):
        if candidate.is_file():
            return candidate
    return None


def _finalize(model_file: Path, meshdir: Path, texturedir: Path) -> ModelAsset:
    """Wrap a resolved model, honoring an ``assets:`` key in its ``<model>.manifest.yaml``.

    ``assets:`` (a provider name or list) borrows those providers' mesh/texture dirs, searched before
    the model's own dir -- so a model can reuse other packages' meshes without copying them.
    """
    meshdirs: list[Path] = []
    texturedirs: list[Path] = []
    manifest = model_file.parent / f"{model_file.stem}.manifest.yaml"
    if manifest.exists():
        data = yaml.safe_load(manifest.read_text()) or {}
        want = data.get("assets")
        if want:
            names = [want] if isinstance(want, str) else list(want)
            provs = {name: (mesh, tex) for name, _md, mesh, tex in providers()}
            for name in names:
                if name not in provs:
                    raise ModelError(
                        f"{manifest.name} (beside {model_file.name}) 'assets' names unknown "
                        f"provider {name!r}"
                    )
                mesh, tex = provs[name]
                meshdirs.append(mesh)
                texturedirs.append(tex)
    meshdirs.append(meshdir)
    texturedirs.append(texturedir)
    # The model file's own dir is the final fallback -- this is where a nested prop keeps its mesh +
    # PNGs (``<name>/<name>.xml`` beside ``<name>.obj``), and it is harmless for flat models.
    if model_file.parent not in meshdirs:
        meshdirs.append(model_file.parent)
    if model_file.parent not in texturedirs:
        texturedirs.append(model_file.parent)
    return ModelAsset(model_file, tuple(meshdirs), tuple(texturedirs))


def _asset_for_file(path: Path) -> ModelAsset:
    """Asset for a standalone model file: meshes from a sibling ``meshes/`` dir, else the file's dir."""
    parent = path.parent
    meshes = parent / "meshes"
    return _finalize(path, meshes if meshes.is_dir() else parent, parent)


def providers() -> list[tuple[str, Path, Path, Path]]:
    """(name, models_dir, meshdir, texturedir) for every registered ``roqsim.models`` provider."""
    out: list[tuple[str, Path, Path, Path]] = []
    for ep in _entry_points(ENTRY_POINT_GROUP):
        module = ep.load()
        try:
            out.append((ep.name, *_provider_dirs(module)))
        except AttributeError as exc:  # provider module missing MODELS_DIR
            raise ModelError(
                f"'{ENTRY_POINT_GROUP}' provider {ep.name!r} ({ep.value}) has no MODELS_DIR"
            ) from exc
    return out


@functools.cache
def resolve_model(model: str, base_dir: Path | None = None) -> ModelAsset:
    """Resolve a ``model:`` reference to a :class:`ModelAsset`. See the module docstring for forms.

    Results are cached per (model, base_dir): resolution is deterministic within a process (the
    provider set is fixed and the CWD stable), ``ModelAsset`` is frozen, and every spawn plugin
    resolves twice (validate_config + build) — a world spawning N copies of a prop went from
    2N provider searches to one. Errors are not cached (lru_cache does not memoize raises).
    """
    # 1) filesystem path: absolute, else relative to ONE anchor -- `base_dir`, or the process's
    #    working directory for a caller that has no document to be relative to (a CLI argument
    #    typed in a shell). One anchor, not a chain: a fallback lets the same reference resolve to
    #    different files depending on where the caller was standing, which is a difference nothing
    #    downstream can see and nothing records.
    p = Path(model)
    if p.is_absolute():
        if p.exists():
            return _asset_for_file(p)
        raise ModelError(f"model path {model!r} does not exist")
    rel = (Path(base_dir) if base_dir is not None else Path.cwd()) / model
    if rel.exists():
        return _asset_for_file(rel)

    # 2) package-qualified ref "<package>:<model>": a registered provider by name, else a module path.
    if ":" in model:
        left, _, modelname = model.partition(":")
        for name, models_dir, meshdir, texturedir in providers():
            if name == left:
                found = _find_in_dir(models_dir, modelname)
                if found is None:
                    raise ModelError(
                        f"model {modelname!r} not found in provider {left!r} ({models_dir})"
                    )
                return _finalize(found, meshdir, texturedir)
        # Fall back to treating the left side as an importable module exposing MODELS_DIR.
        try:
            module = import_module(left)
        except ImportError as exc:
            raise ModelError(
                f"{left!r} (from {model!r}) is neither a registered '{ENTRY_POINT_GROUP}' provider "
                f"nor an importable module: {exc}"
            ) from exc
        try:
            models_dir, meshdir, texturedir = _provider_dirs(module)
        except AttributeError as exc:
            raise ModelError(f"module {left!r} is not a model provider (no MODELS_DIR)") from exc
        found = _find_in_dir(models_dir, modelname)
        if found is None:
            raise ModelError(f"model {modelname!r} not found in provider {left!r} ({models_dir})")
        return _finalize(found, meshdir, texturedir)

    # 3) short name -> first match across all registered providers.
    searched: list[str] = []
    for _name, models_dir, meshdir, texturedir in providers():
        searched.append(str(models_dir))
        found = _find_in_dir(models_dir, model)
        if found is not None:
            return _finalize(found, meshdir, texturedir)
    raise ModelError(
        f"model {model!r} not found in any '{ENTRY_POINT_GROUP}' provider; "
        f"searched: {searched or '(no providers registered)'}"
    )
