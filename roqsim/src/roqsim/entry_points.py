"""The one scan of the installed entry points, shared by every registry in the tree.

Plugins, models, worlds, render overlays and ``roqsim_nav``'s outputs and avoidance models are all
chosen by a short name that an installed package registered in its own entry-point group. Listing a
group walks every installed distribution's metadata -- tens of milliseconds in a
``--system-site-packages`` venv -- and a spawn-heavy world asks for a group once per plugin it
resolves, so the scan is cached for the life of the process. A package installed mid-run is not a
supported scenario.
"""

from __future__ import annotations

import functools
from importlib import metadata


@functools.cache
def entry_points(group: str) -> tuple:
    """Every entry point registered in ``group``, in the order the metadata lists them."""
    eps = metadata.entry_points()
    if hasattr(eps, "select"):  # Python 3.10+
        return tuple(eps.select(group=group))
    return tuple(eps.get(group, ()))  # pragma: no cover - legacy
