"""roqsim: a lightweight, plugin-driven MuJoCo simulation framework."""

from __future__ import annotations

# BEFORE every other import, because the one below reaches mujoco through .config/.engine and
# `import mujoco` binds its GL backend from MUJOCO_GL once and for all. Setting the variable after
# that point is silently ineffective -- which is the bug this call exists to make impossible, for
# every entry point at once: the CLI, the ROS bridge, the stepped scenario adapter (which has no
# command line at all), the tools, and pytest. See roqsim.gl for why unset is the dangerous value.
#
# Yes, this mutates os.environ at import. It is the only point that can act in time, it never
# overrides an explicit MUJOCO_GL, it is idempotent, and ROQSIM_NO_GL_SELECT=1 opts out.
from .gl import select_offscreen_gl

select_offscreen_gl()

from .config import (
    SimConfig,
    assignments_from_mapping,
    deep_merge,
    drop_transport,
    drop_transport_plugins,
    load_config,
    load_config_from_dict,
    overrides_from_dotlist,
)
from .context import Entity, RobotHandle, SimContext
from .engine import Engine
from .plugin import Plugin, PluginError
from .registry import resolve_plugin
from .rendering import WALK_KEYS, FrameRenderer, default_free_camera, walk_delta
from .runner import config_for_input

__all__ = [
    "WALK_KEYS",
    "Engine",
    "Entity",
    "FrameRenderer",
    "Plugin",
    "PluginError",
    "RobotHandle",
    "SimConfig",
    "SimContext",
    "config_for_input",
    "assignments_from_mapping",
    "deep_merge",
    "default_free_camera",
    "drop_transport",
    "drop_transport_plugins",
    "load_config",
    "load_config_from_dict",
    "overrides_from_dotlist",
    "resolve_plugin",
    "walk_delta",
]

__version__ = "0.1.0"
