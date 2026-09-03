"""Scene plugin: **many** parametric cylinders from one config value.

The round sibling of :mod:`roqsim_assets.plugins.boxes`, and the same geometry as
:mod:`roqsim_assets.plugins.cylinder`, declared as a *list* rather than as one plugin entry per
cylinder::

    cylinders:
      name: clutter        # prefix for the generated entity names (default 'cylinders')
      instances:
      - {pos: [0.12, -0.30], radius: 0.035, height: 0.15, free: true, mass: 0.3}
      - {pos: [0.31,  0.08], radius: 0.042, height: 0.15, free: true, mass: 0.3}

Each entry accepts every key ``cylinder`` does (``pos``, ``radius``, ``height``, ``color``,
``collide``, ``friction``, ``free``, ``mass``, and an optional ``name``), because each one *is* a
cylinder: this plugin owns the list, not the geometry.

Why a list and not a count
--------------------------

Same reason as ``boxes``: **how many** becomes the length of one config value instead of the number
of plugin entries, and ``roqsim.apply_overrides`` resolves a plugin by name and deep-merges into its
config while *refusing an override that matches no plugin*. So a campaign can replace
``cylinders.instances`` wholesale, but could never have appended a twenty-first ``cylinder:`` entry.
Without that, "20 objects instead of 12" is a structural edit to the world file and therefore not a
factor at all.

And a list rather than a ``count:`` scalar because a population is **heterogeneous** in the property
that matters here: a family of round workpieces differs in diameter at a common height, and one
modelled asset cannot supply that (``spawn_model`` scales uniformly). A count plus in-plugin layout
sampling would move pose selection into the substrate, which does not know the clearance rule, the
reachability predicate or the admissibility test the experiment cares about. The generator does.

Entity names are ``<name>_<index>`` unless an entry names itself, so ``SetEntityState`` can address a
single cylinder out of the population and ``on_reset`` restores each to its own declared pose --
which is what makes a per-trial layout a reset rather than a reload.
"""

from __future__ import annotations

import mujoco

from roqsim.context import SimContext
from roqsim.plugin import Plugin

from .cylinder import CylinderPlugin

_DEFAULT_NAME = "cylinders"


class CylindersPlugin(Plugin):
    """A population of :class:`~roqsim_assets.plugins.cylinder.CylinderPlugin` instances.

    Delegates rather than reimplements: every instance is a real ``CylinderPlugin``, so the two
    cannot drift and a fix to cylinder geometry reaches both. This plugin's whole job is owning the
    list and keeping the names distinct.
    """

    def __init__(self, config=None, *, name=None, entity=None, label=None):
        super().__init__(config, name=name, entity=entity, label=label)
        self.entity_name = self.label or _DEFAULT_NAME
        self._children: list[CylinderPlugin] = [
            # Each instance is a cylinder in its own right, so it gets its own LABEL -- which is
            # what names its entity, exactly as it would if the world had declared it. An instance
            # may name itself; otherwise it is this entry's label plus its index.
            CylinderPlugin(
                self._child_config(entry, index),
                name=self.name,
                label=self._child_label(entry, index),
                # Its children are its SIBLINGS, not its components: `cylinders` makes cylinders,
                # it does not own them, so each takes this entry's own owner.
                entity=self.entity,
            )
            for index, entry in enumerate(self.config.get("instances") or [])
        ]

    def _child_label(self, entry: dict, index: int) -> str:
        return str((entry or {}).get("name") or f"{self.entity_name}_{index}")

    def _child_config(self, entry: dict, index: int) -> dict:
        """One instance's config, with an MJCF prefix that cannot collide.

        A distinct ``prefix`` per instance is not cosmetic: MuJoCo names must be unique, so without
        it a second cylinder fails model compilation.
        """
        child = {k: v for k, v in (entry or {}).items() if k != "name"}
        child.setdefault("prefix", f"{self._child_label(entry, index)}_")
        return child

    def validate_config(self, config: dict) -> list[str]:
        """Validate the list, then let each cylinder validate itself.

        Errors are prefixed with the instance index, because "'radius' is required" is not
        actionable when the world declares twenty cylinders.
        """
        errors: list[str] = []
        instances = config.get("instances")
        if instances is None:
            return ["'instances' is required (a list of cylinder configs)"]
        if not isinstance(instances, list):
            return [f"'instances' must be a list, got {type(instances).__name__}"]

        for index, entry in enumerate(instances):
            if not isinstance(entry, dict):
                errors.append(f"instances[{index}]: must be a mapping, got {type(entry).__name__}")
                continue
            child = CylinderPlugin(self._child_config(entry, index), name=self.name)
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
