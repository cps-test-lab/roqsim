"""Resolving a named implementation to a class, for this package's two registries.

Both the embodiment (``roqsim_nav.outputs``) and the local-avoidance model
(``roqsim_nav.avoidance``) are chosen by name in a world YAML, and both must be extensible from
outside this package -- a robot family shipping its own way of moving, an experiment trying a
different local planner. So both accept the same three forms :mod:`roqsim.registry` already
established for plugins, and neither is a name list in this package:

1. a short name registered in the entry-point group;
2. ``package.module:Class`` off ``PYTHONPATH``;
3. ``path/to/file.py:Class``, relative to the world's directory.

Keeping this in one place is what stops the two registries drifting apart, which would make "how do
I add one" a different answer depending on which one you are adding.
"""

from __future__ import annotations

import functools
import importlib
import importlib.util
import sys
from importlib import metadata
from pathlib import Path


class RegistryError(Exception):
    """A name in a world could not be resolved to an implementation."""


@functools.cache
def entry_points(group: str) -> tuple:
    """Entry points in ``group``, scanned once per process (the scan walks every distribution)."""
    eps = metadata.entry_points()
    return tuple(eps.select(group=group) if hasattr(eps, "select") else eps.get(group, ()))


def registered(group: str) -> list[str]:
    """The short names available in ``group`` -- for an error message that helps."""
    return sorted(ep.name for ep in entry_points(group))


def resolve(ref: str, *, group: str, base: type, kind: str, base_dir: Path | None = None) -> type:
    """Resolve ``ref`` to a subclass of ``base``. ``kind`` names the thing, for error messages."""
    if ":" not in ref:
        for ep in entry_points(group):
            if ep.name == ref:
                try:
                    obj = ep.load()
                except ImportError as err:
                    # Registered but not importable: its distribution is installed while one of its
                    # own dependencies is not. `orca` is the standing example -- the entry point is
                    # always there, `rvo2` is an optional extra -- and the caller needs to tell that
                    # apart from a typo, so it must not surface as a bare ImportError.
                    raise RegistryError(
                        f"{kind} {ref!r} is registered but could not be imported: {err}. "
                        f"Its package is installed; one of its own dependencies is not."
                    ) from err
                _check(obj, base, f"{kind} {ref!r}")
                return obj
        raise RegistryError(
            f"unknown {kind} {ref!r}: not a registered '{group}' entry point. "
            f"Available: {', '.join(registered(group)) or '(none)'}. "
            f"Use 'module.path:Class' or 'path/to/file.py:Class' for one outside this package."
        )

    left, _, cls_name = ref.partition(":")
    if not cls_name:
        raise RegistryError(
            f"{kind} reference {ref!r} must be an entry-point name, 'module.path:Class', "
            f"or 'path/to/file.py:Class'"
        )
    path = Path(left) if Path(left).is_absolute() else Path(base_dir or ".") / left
    if path.exists():
        module = _import_file(path, ref, kind)
    else:
        try:
            module = importlib.import_module(left)
        except Exception as exc:  # noqa: BLE001 - report the ref, whatever import raised
            raise RegistryError(
                f"could not import module {left!r} (from {kind} {ref!r}); is it on PYTHONPATH? {exc}"
            ) from exc
    try:
        obj = getattr(module, cls_name)
    except AttributeError as exc:
        raise RegistryError(f"class {cls_name!r} not found in {kind} {ref!r}") from exc
    _check(obj, base, f"{kind} {ref!r}")
    return obj


def _import_file(path: Path, ref: str, kind: str):
    module_name = f"roqsim_nav_ext_{path.stem}_{abs(hash(str(path.resolve())))}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RegistryError(f"could not load {kind} file {path} (from {ref!r})")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:  # noqa: BLE001
        raise RegistryError(f"error importing {kind} file {path} (from {ref!r}): {exc}") from exc
    return module


def _check(obj, base: type, where: str) -> None:
    if not (isinstance(obj, type) and issubclass(obj, base)):
        raise RegistryError(f"{where} does not resolve to a {base.__name__} subclass")
