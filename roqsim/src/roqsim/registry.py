"""Plugin resolution: turn a YAML ``plugin:`` string into a :class:`Plugin` subclass.

Three forms are supported (mirroring scenario-execution's own simulation loader):

1. **Short name** -> Python entry-point in the ``roqsim.plugins`` group.
   e.g. ``ros2_bridge``
2. **Importable ``package.module:Class``** -> imported off ``PYTHONPATH``.
   e.g. ``my_company.sensors.radar:RadarPlugin``
3. **Filesystem ``path/to/file.py:Class``** -> loaded from a file (relative to ``base_dir``).
   e.g. ``./plugins/custom_ctrl.py:CustomController``

Resolution order: an entry-point name wins; otherwise, if the string contains ``:`` the left side
is treated as a file path when it exists on disk, else as a module path. Failures raise
:class:`PluginError` naming the form attempted and why it failed.
"""

from __future__ import annotations

import functools
import importlib
import importlib.util
import sys
from importlib import metadata
from pathlib import Path

from .plugin import Plugin, PluginError

ENTRY_POINT_GROUP = "roqsim.plugins"


@functools.cache
def _entry_points(group: str):
    """Return entry points for ``group`` across Python versions.

    Cached for the process lifetime: one scan walks every installed distribution (~30 ms in a
    system-site-packages venv) and a large world resolves hundreds of refs, which made the scan
    the dominant load cost. Packages installed mid-run are not a supported scenario.
    """
    eps = metadata.entry_points()
    if hasattr(eps, "select"):  # Python 3.10+
        return tuple(eps.select(group=group))
    return tuple(eps.get(group, ()))  # pragma: no cover - legacy


def _load_from_entry_point(name: str) -> type[Plugin] | None:
    for ep in _entry_points(ENTRY_POINT_GROUP):
        if ep.name == name:
            try:
                obj = ep.load()
            except ImportError as err:
                # Registered but not importable — its distribution is installed while one of *its*
                # dependencies is not. The ROS bridge is the standing example: a wheel puts the entry
                # point on the path, and importing it needs rclpy, which a ROS-free process has not
                # sourced. That must be a PluginError like every other resolution failure, because
                # callers distinguish "cannot resolve" from "resolved to something wrong" by this type
                # alone: `drop_transport_plugins` drops an unresolvable ref (a geometry-only export
                # needs no transport), and a raw ImportError sailed past it and killed the export.
                raise PluginError(
                    f"plugin {name!r} is registered by an installed package but could not be "
                    f"imported: {err}"
                ) from err
            _check_is_plugin(obj, f"entry-point {name!r}")
            return obj
    return None


def _split_ref(ref: str) -> tuple[str, str]:
    left, _, cls = ref.partition(":")
    if not cls:
        raise PluginError(
            f"plugin reference {ref!r} must be an entry-point name or of the form "
            f"'module.path:Class' or 'path/to/file.py:Class'"
        )
    return left, cls


def _load_from_file(path: Path, cls_name: str, ref: str) -> type[Plugin]:
    if not path.exists():
        raise PluginError(f"plugin file {path} (from {ref!r}) does not exist")
    module_name = f"roqsim_ext_{path.stem}_{abs(hash(str(path.resolve())))}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise PluginError(f"could not load plugin file {path} (from {ref!r})")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:  # noqa: BLE001
        raise PluginError(f"error importing plugin file {path} (from {ref!r}): {exc}") from exc
    return _get_class(module, cls_name, ref)


def _load_from_module(module_path: str, cls_name: str, ref: str) -> type[Plugin]:
    try:
        module = importlib.import_module(module_path)
    except Exception as exc:  # noqa: BLE001
        raise PluginError(
            f"could not import module {module_path!r} (from {ref!r}); is it on PYTHONPATH? {exc}"
        ) from exc
    return _get_class(module, cls_name, ref)


def _get_class(module, cls_name: str, ref: str) -> type[Plugin]:
    try:
        obj = getattr(module, cls_name)
    except AttributeError as exc:
        raise PluginError(f"class {cls_name!r} not found in {ref!r}") from exc
    _check_is_plugin(obj, ref)
    return obj


def _check_is_plugin(obj, where: str) -> None:
    if not (isinstance(obj, type) and issubclass(obj, Plugin)):
        raise PluginError(f"{where} does not resolve to a roqsim.plugin.Plugin subclass")


def resolve_plugin(ref: str, base_dir: Path | None = None) -> type[Plugin]:
    """Resolve ``ref`` to a :class:`Plugin` subclass. See module docstring for the three forms."""
    # 1) entry-point short name (only when there is no ':' — names never contain a colon)
    if ":" not in ref:
        cls = _load_from_entry_point(ref)
        if cls is not None:
            return cls
        raise PluginError(
            f"unknown plugin {ref!r}: not a registered '{ENTRY_POINT_GROUP}' entry-point. "
            f"Use 'module.path:Class' or 'path/to/file.py:Class' for external plugins."
        )

    # 2/3) has ':' -> file path if it exists, else module path
    left, cls_name = _split_ref(ref)
    candidate = Path(left)
    if not candidate.is_absolute() and base_dir is not None:
        candidate = base_dir / candidate
    if left.endswith(".py") or candidate.exists():
        return _load_from_file(candidate, cls_name, ref)
    return _load_from_module(left, cls_name, ref)
