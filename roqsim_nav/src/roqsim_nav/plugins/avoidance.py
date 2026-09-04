"""World-level plugin: which local-avoidance model this world's navigators share, and its parameters.

Declared once, in one place::

    components:
      - avoidance: {model: orca, neighbor_dist: 4.0, time_horizon: 3.0}

A plugin rather than something the first navigator creates on demand, for four reasons that all bite
in practice: the choice of model is then visible in the world file instead of implied by whichever
mover happens to load first; it lands in the run's provenance, so a campaign records which local
planner produced its trajectories; two movers cannot disagree about it; and the per-step solve has an
unambiguous owner rather than a "first instance wins" rule to remember.

``model`` resolves the same three ways plugins do -- a short name from the ``roqsim_nav.avoidance``
entry-point group, ``package.module:Class``, or ``file.py:Class`` beside the world. Everything else
in the block is the model's own, checked against the model's ``params_schema`` so a key left behind
after switching models is refused rather than ignored.

**Who yields is derived, not configured.** An entity with a ``navigator`` is apparatus and joins as a
yielding agent; an entity without one is externally controlled -- the robot under test -- and joins
non-yielding, its state overwritten from ground truth every step so the others go round it and it is
never pushed by them. That replaces a heuristic with a fact: the pedestrian stack used to pick *the
first robot entity* in the world, so in a two-robot world the second was invisible to every walker.
"""

from __future__ import annotations

import logging

import mujoco

from roqsim.context import SimContext
from roqsim.plugin import Plugin

from ..avoidance import AvoidanceService, RegistryError, available, resolve_model
from ..obstacles import wall_polygons

logger = logging.getLogger(__name__)

#: Blackboard key the navigators look the model up under.
MODEL_KEY = "nav:avoidance"

#: Reserved keys of the plugin's own config; everything else belongs to the model.
_OWN_KEYS = frozenset({"model", "obstacle_height"})


class AvoidancePlugin(Plugin):
    #: It builds no geometry and owns no entity: it is a world-level service.
    requires_owner = False

    def __init__(self, config=None, *, name=None, entity=None, label=None):
        super().__init__(config, name=name, entity=entity, label=label)
        self.model_ref = str(self.config.get("model", "orca"))
        self._model = None

    def validate_config(self, config: dict) -> list[str]:
        errors: list[str] = []
        ref = str(config.get("model", "orca"))
        try:
            cls = resolve_model(ref, self.base_dir)
        except RegistryError as exc:
            return [str(exc)]
        schema = getattr(cls, "params_schema", ())
        if schema:
            unknown = sorted(set(config) - _OWN_KEYS - set(schema))
            if unknown:
                errors.append(
                    f"avoidance model {ref!r} does not accept {', '.join(unknown)}. "
                    f"It accepts: {', '.join(schema)}. (A key left over from another model is "
                    f"refused rather than ignored.)"
                )
        band = config.get("obstacle_height")
        if band is not None and not (
            isinstance(band, (list, tuple)) and len(band) == 2 and float(band[0]) < float(band[1])
        ):
            errors.append("'obstacle_height' must be [z_lo, z_hi] with z_lo < z_hi")
        return errors

    def configure(self, ctx: SimContext) -> None:
        cls = resolve_model(self.model_ref, self.base_dir)
        model = cls()
        params = {k: v for k, v in self.config.items() if k not in _OWN_KEYS}
        model.configure(ctx, params)
        self._model = AvoidanceService(model)
        ctx.blackboard.set(MODEL_KEY, self._model)

        band = self.config.get("obstacle_height") or (0.1, 1.8)
        mujoco.mj_forward(ctx.model, ctx.data)  # geom_xpos must be valid to read wall footprints
        polys = wall_polygons(ctx.model, ctx.data, z_lo=float(band[0]), z_hi=float(band[1]))
        # The planner's own polygons, so the two layers cannot disagree about where a wall is.
        self._model.add_static(polys)
        logger.info(
            "avoidance: %s, %d wall footprint(s); registered models: %s",
            self.model_ref,
            len(polys),
            ", ".join(available()) or "(none)",
        )

    def pre_step(self, ctx: SimContext) -> None:
        """Solve once per step, before any navigator reads its result.

        Ordering in the world file is irrelevant because the solve is stamped with the step it ran
        for: this call and a navigator's own ``ensure_solved`` race harmlessly, and whichever arrives
        first does the work. The lag between submitting and using is one physics step -- about 2 ms,
        far below the navigators' own tick.
        """
        if self._model is not None:
            self._model.ensure_solved(ctx)

    def on_reset(self, ctx: SimContext) -> None:
        if self._model is not None:
            self._model.reset()
