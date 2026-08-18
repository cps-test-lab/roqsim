"""A trivial plugin used to validate the framework end-to-end without any assets.

It adds a single free-floating box to the scene (so the compiled model is non-empty) and counts its
own hook invocations on the blackboard, which the test suite asserts against.

Config::

    dummy:
      size: 0.1   # half-extent (m) of the free-floating box this plugin adds; must be > 0
"""

from __future__ import annotations

import mujoco

from ..context import Entity, SimContext
from ..plugin import Plugin


class DummyPlugin(Plugin):
    parallel_safe = True

    def validate_config(self, config: dict) -> list[str]:
        errors = []
        if "size" in config and float(config["size"]) <= 0:
            errors.append("'size' must be > 0")
        return errors

    def build(self, spec: mujoco.MjSpec, ctx: SimContext) -> None:
        size = float(self.config.get("size", 0.1))
        body = spec.worldbody.add_body(name=f"{self.name}_box", pos=[0, 0, size])
        body.add_freejoint()
        body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[size, size, size])

    def configure(self, ctx: SimContext) -> None:
        ctx.entities.add(Entity(name=self.name, kind="object", body=f"{self.name}_box"))
        self._bump(ctx, "configure")

    def on_reset(self, ctx: SimContext) -> None:
        self._bump(ctx, "on_reset")

    def pre_step(self, ctx: SimContext) -> None:
        self._bump(ctx, "pre_step")

    def post_step(self, ctx: SimContext) -> None:
        self._bump(ctx, "post_step")

    def shutdown(self, ctx: SimContext) -> None:
        self._bump(ctx, "shutdown")

    def _bump(self, ctx: SimContext, hook: str) -> None:
        key = f"dummy_counts::{self.name}"
        counts = ctx.blackboard.get(key, {})
        counts[hook] = counts.get(hook, 0) + 1
        ctx.blackboard.set(key, counts)
