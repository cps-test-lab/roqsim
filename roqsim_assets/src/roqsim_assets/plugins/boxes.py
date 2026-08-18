"""Scene plugin: **many** parametric boxes from one config value.

The same geometry as :mod:`roqsim_assets.plugins.box`, declared as a *list* rather than as one plugin
entry per box::

    boxes:
      name: obstacles      # prefix for the generated entity names (default 'boxes')
      instances:
      - {pos: [2.1, -3.4], size: [0.5, 0.5, 1.0]}
      - {pos: [5.8, -1.2], size: [0.5, 0.5, 1.0], yaw: 0.4}

Each entry accepts every key ``box`` does (``pos``, ``size``, ``yaw``, ``color``, ``collide``,
``friction``, ``free``, and an optional ``name``), because each one *is* a box: this plugin owns the
list, not the geometry.

Why a list and not a count
--------------------------

The point is **how many** becomes the length of one config value instead of the number of plugin
entries. That is what lets a campaign vary it: ``roqsim.apply_overrides`` resolves a plugin by name and
deep-merges into its config, and it *refuses an override that matches no plugin* -- so a campaign can
replace ``boxes.instances`` wholesale, but could never have appended a fourth ``box:`` entry. Before
this, "8 obstacles instead of 4" was a structural edit to the world file and therefore not a factor at
all.

A list rather than a ``count:`` scalar because real populations are **heterogeneous**: an obstacle
generator scales its count by path length and gives each obstacle its own pose and size. A count plus
in-plugin layout sampling would serve only the uniform case, and would move pose selection into the
substrate -- which does not know the map, the robot's path, or the clearance rule the experiment
cares about. The generator does.

Entity names are ``<name>_<index>`` unless an entry names itself, so ``SetEntityState`` can address a
single box out of the population and ``on_reset`` restores each to its own declared pose.

Config::

    boxes:
      name: obstacles     # prefix for the generated entity names (default: 'boxes')
      instances: []       # list of box configs, each accepting every key `box` does (required)
"""

from __future__ import annotations

import mujoco

from roqsim.context import SimContext
from roqsim.plugin import Plugin

from .box import BoxPlugin

_DEFAULT_NAME = "boxes"


class BoxesPlugin(Plugin):
    """A population of :class:`~roqsim_assets.plugins.box.BoxPlugin` instances.

    Delegates rather than reimplements: every instance is a real ``BoxPlugin``, so the two cannot
    drift and a fix to box geometry reaches both. This plugin's whole job is owning the list and
    keeping the names distinct.
    """

    def __init__(self, config=None, *, name=None):
        super().__init__(config, name=name)
        self.entity_name = self.config.get("name", _DEFAULT_NAME)
        self._children: list[BoxPlugin] = [
            BoxPlugin(self._child_config(entry, index), name=self.name)
            for index, entry in enumerate(self.config.get("instances") or [])
        ]

    def _child_config(self, entry: dict, index: int) -> dict:
        """One instance's config, with a name and MJCF prefix that cannot collide.

        A distinct ``prefix`` per instance is not cosmetic: MuJoCo names must be unique, so without
        it a second box fails model compilation.
        """
        child = dict(entry or {})
        child.setdefault("name", f"{self.entity_name}_{index}")
        child.setdefault("prefix", f"{child['name']}_")
        return child

    def validate_config(self, config: dict) -> list[str]:
        """Validate the list, then let each box validate itself.

        Errors are prefixed with the instance index, because "size must have three elements" is not
        actionable when the world declares twelve boxes.
        """
        errors: list[str] = []
        instances = config.get("instances")
        if instances is None:
            return ["'instances' is required (a list of box configs)"]
        if not isinstance(instances, list):
            return [f"'instances' must be a list, got {type(instances).__name__}"]

        for index, entry in enumerate(instances):
            if not isinstance(entry, dict):
                errors.append(f"instances[{index}]: must be a mapping, got {type(entry).__name__}")
                continue
            child = BoxPlugin(self._child_config(entry, index), name=self.name)
            errors.extend(
                f"instances[{index}]: {err}" for err in child.validate_config(child.config)
            )
        return errors

    def build(self, spec: mujoco.MjSpec, ctx: SimContext) -> None:
        for child in self._children:
            child.build(spec, ctx)

    def configure(self, ctx: SimContext) -> None:
        for child in self._children:
            child.configure(ctx)

    def on_reset(self, ctx: SimContext) -> None:
        for child in self._children:
            child.on_reset(ctx)
