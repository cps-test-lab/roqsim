"""roqsim_webctrl -- robotics-site-ops (`rso_web`) integration for roqsim.

The sim-control web plugin is generic and lives in rso_web; nothing scene-specific is needed on the
sim side because ``roqsim_ros_bridge`` already exposes the ``simulation_interfaces`` services
(``set_simulation_state`` / ``reset_simulation`` / ``get_simulation_state``) it calls over rosbridge.

This package's job is to ship the reusable **web.yaml fragment** that enables that plugin, so any
roqsim scene can include it without hand-authoring. Merge :func:`sim_control_fragment_path` into a
scene's web.yaml (top-level ``plugins:`` list).
"""

from __future__ import annotations

from importlib import resources
from pathlib import Path


def sim_control_fragment_path() -> Path:
    """Filesystem path to the reusable sim-control web.yaml fragment."""
    return Path(resources.files("roqsim_webctrl") / "web" / "sim_control.web.yaml")


__all__ = ["sim_control_fragment_path"]
