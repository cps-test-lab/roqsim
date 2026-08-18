"""Importing the modules that teach the bridge new types, once at start-up.

Handlers, converters and decoders live in module-level registries populated by import, and entry
points are loaded lazily by name -- so a handler in a *different* package would never be imported at
all. Any package therefore advertises its extension module in the ``roqsim_ros_bridge.extensions``
entry-point group::

    # <your package>/setup.py
    entry_points={
        "roqsim_ros_bridge.extensions": [
            "walker_nav = roqsim_walker_ros.actions",
        ],
    }

Importing the module runs its ``@action_handler`` / ``@service_handler`` / ``@converter`` /
``@decoder`` decorators. This is how ``roqsim_walker_ros`` teaches the bridge
``nav2_msgs/NavigateThroughPoses`` without the core bridge ever depending on nav2.

Its own module, and free of ROS imports, so that a registry which needs no ROS types (see
:mod:`roqsim_ros_bridge.services`) does not have to import one that does just to reach the loader.
"""

from __future__ import annotations

import logging
from importlib import metadata

logger = logging.getLogger(__name__)

#: Entry-point group whose modules are imported once at bridge start-up so their decorators run.
EXTENSION_GROUP = "roqsim_ros_bridge.extensions"

_loaded = False


def load_extensions() -> None:
    """Import every module registered in :data:`EXTENSION_GROUP` (idempotent).

    A broken extension is logged and skipped rather than taking the whole bridge down with it.
    """
    global _loaded
    if _loaded:
        return
    _loaded = True
    eps = metadata.entry_points()
    found = (
        eps.select(group=EXTENSION_GROUP)
        if hasattr(eps, "select")
        else eps.get(EXTENSION_GROUP, [])
    )
    for ep in found:
        try:
            ep.load()
            logger.info("bridge extension loaded: %s (%s)", ep.name, ep.value)
        except Exception:  # noqa: BLE001 - one bad extension must not kill the bridge
            logger.exception("failed to load bridge extension %r (%s)", ep.name, ep.value)
