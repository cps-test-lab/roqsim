"""Importing the modules that teach the bridge new types, once at start-up.

Handlers, converters and decoders live in module-level registries populated by import, and entry
points are loaded lazily by name -- so a handler in a *different* package would never be imported at
all. Any package therefore advertises its extension module in the ``roqsim_ros_bridge.extensions``
entry-point group::

    # <your package>/setup.py
    entry_points={
        "roqsim_ros_bridge.extensions": [
            "roqsim_nav = roqsim_nav_ros.actions",
        ],
    }

Importing the module runs its ``@action_handler`` / ``@service_handler`` / ``@converter`` /
``@decoder`` decorators. This is how ``roqsim_nav_ros`` teaches the bridge
``nav2_msgs/NavigateThroughPoses`` without the core bridge ever depending on nav2.

Its own module, and free of ROS imports, so that a registry which needs no ROS types (see
:mod:`roqsim_ros_bridge.services`) does not have to import one that does just to reach the loader.
"""

from __future__ import annotations

import logging

from roqsim.entry_points import entry_points

logger = logging.getLogger(__name__)

#: Entry-point group whose modules are imported once at bridge start-up so their decorators run.
EXTENSION_GROUP = "roqsim_ros_bridge.extensions"

_loaded = False


class ExtensionError(RuntimeError):
    """A module registered in :data:`EXTENSION_GROUP` failed to import."""


def load_extensions() -> None:
    """Import every module registered in :data:`EXTENSION_GROUP` (idempotent once it succeeds).

    A module that fails to import raises :class:`ExtensionError`, naming the entry point and chained
    to the original exception, and the bridge does not start: a bridge without the extension would
    run without the handlers and converters it registers.
    """
    global _loaded
    if _loaded:
        return
    for ep in entry_points(EXTENSION_GROUP):
        try:
            ep.load()
        except Exception as exc:
            raise ExtensionError(
                f"bridge extension {ep.name!r} ({ep.value}) in the {EXTENSION_GROUP!r} entry-point "
                f"group failed to load: {type(exc).__name__}: {exc}"
            ) from exc
        logger.info("bridge extension loaded: %s (%s)", ep.name, ep.value)
    _loaded = True
