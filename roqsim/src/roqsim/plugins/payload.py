"""Component plugin: a carried payload, as added mass on a robot's body.

Carried mass is a first-class experiment factor: it changes what a mobile base can accelerate, what
an arm can hold at reach, and -- where thrust is bounded -- whether a vehicle flies at all. This
plugin states it in the world, so it is swept like any other factor and recorded with the run.

Config::

    payload:
      robot: robot        # entity whose body carries it (default: the owning entity)
      mass: 2.5           # kg, REQUIRED -- added to the body's own mass
      body: tray          # body to load (default: the entity's root body)

The payload is a **point mass at the body's centre of mass**: mass adds, and the inertia a point
mass contributes about its own centre is zero. An *offset* payload is a different physical object --
it shifts the centre of mass and adds a parallel-axis inertia term, changing the attitude dynamics
rather than the load alone -- and representing it means adding a child body before compile. An
``offset`` key is therefore refused rather than approximated.

The mass is applied in ``configure``, after compile: the entity registry is what maps ``robot:`` to
a *prefixed* MJCF body name, and it is populated by the spawn plugins during ``configure``. Writing
``model.body_mass`` and re-deriving the cached constants with ``mj_setConst`` is the same mechanism
:mod:`roqsim.plugins.model_override` uses for ``body_mass``.

The mass is stated in the world and therefore recorded with it, so a run's provenance carries the
payload even though the compiled MJCF's own inertial does not.
"""

from __future__ import annotations

import logging

import mujoco

from roqsim.context import SimContext
from roqsim.plugin import Plugin
from roqsim.schema import Field

logger = logging.getLogger(__name__)


class PayloadPlugin(Plugin):
    #: Loads an entity's body, so it belongs inside that entity's ``components:`` block.
    requires_owner = True

    def __init__(self, config=None, *, name=None, entity=None, label=None):
        super().__init__(config, name=name, entity=entity, label=label)
        self.robot = self.config.get("robot") or self.entity
        self._bid = -1

    #: Declared once, so `roqsim plugins describe payload` publishes the same keys the checks run
    #: on -- with their types, units and bounds, which a docstring cannot give a caller.
    CONFIG_SCHEMA = {
        "mass": Field(
            float,
            required=True,
            minimum=0.0,
            unit="kg",
            doc="added to the body's own mass; a payload plugin with no mass is not a payload",
        ),
        "body": Field(
            str, default="", static=True, doc="body to load (default: the entity's root body)"
        ),
        "robot": Field(
            str, default="", static=True, doc="entity that carries it (default: the owning entity)"
        ),
    }

    def validate_config(self, config: dict) -> list[str]:
        # The mechanical half from the declaration; the rest is what only this plugin knows.
        errors = self.validate_schema(config)
        if "offset" in config:
            # Loud rather than approximate: an offset payload moves the centre of mass and adds a
            # parallel-axis term, which this plugin does not model. Ignoring the key would report a
            # result for a body nobody configured.
            errors.append(
                "'offset' is not supported: this plugin models a point mass at the body's centre "
                "of mass. An offset payload changes the inertia tensor and needs a body added "
                "before compile."
            )
        return errors

    def _resolve_body(self, ctx: SimContext, entity) -> int:
        """The body the payload is carried on, by the first rule that resolves.

        A named ``body:`` wins. Otherwise the body owning the entity's base joint, which is the one
        rule that holds across families: whatever a robot calls its root, its base joint is on it.
        Otherwise the entity's registered body.
        """
        model = ctx.model
        prefix = entity.meta.get("prefix", "")

        named = self.config.get("body")
        if named:
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, prefix + named)
            if bid < 0:
                raise RuntimeError(
                    f"payload: no such body {prefix + named!r} for robot {self.robot!r}"
                )
            return bid

        base_joint = entity.meta.get("base_joint")
        if base_joint:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, base_joint)
            if jid >= 0:
                return int(model.jnt_bodyid[jid])

        if entity.body:
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, entity.body)
            if bid >= 0:
                return bid

        raise RuntimeError(
            f"payload: could not resolve a body for robot {self.robot!r} (tried its base joint "
            f"{base_joint!r} and body {entity.body!r}); name one with 'body:'"
        )

    def configure(self, ctx: SimContext) -> None:
        mass = float(self.config["mass"])
        entity = ctx.entities.get(self.robot)
        if entity is None:
            raise RuntimeError(
                f"payload: no entity named {self.robot!r} -- declare this plugin inside the "
                f"robot's components: block, or name the robot with 'robot:'"
            )
        self._bid = self._resolve_body(ctx, entity)

        if mass == 0.0:
            # The zero level of a sweep: leave the model untouched so the unloaded cell is
            # bit-identical to a world that never declared a payload at all.
            return

        model = ctx.model
        own_mass = float(model.body_mass[self._bid])
        model.body_mass[self._bid] = own_mass + mass
        # body_mass feeds cached quantities (subtree masses, gravity compensation); mj_setConst is
        # what makes a mass write take effect rather than sit in an array nothing re-reads.
        mujoco.mj_setConst(model, ctx.data)
        logger.info(
            "payload (%s): %.4f kg on %r -- body %.4f kg, total %.4f kg",
            self.robot,
            mass,
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, self._bid),
            own_mass,
            own_mass + mass,
        )
