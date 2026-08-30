"""Component plugin: a carried payload, as added mass on a robot's body.

Thrust margin is the first thing an aerial experiment varies, and until now there was no way to
state it. A Crazyflie's airframe is 27 g against 0.35 N of collective thrust -- a thrust-to-weight
ratio of 1.32 -- so a few grams of payload is not a detail, it is the whole flight envelope: 5 g
takes T/W to 1.11, and somewhere under 10 g the drone cannot hold altitude at all. A campaign that
cannot set payload mass cannot ask the question every aerial developer asks first.

Config::

    payload:
      robot: drone        # entity whose body carries it (default: the owning entity)
      mass: 0.005         # kg, REQUIRED -- added to the body's own mass
      body: cf2           # body to load (default: the entity's root body)

Deliberately a **point mass at the body's centre of mass**, and nothing else. That is what a payload
strapped to an airframe's belly is to within the precision anyone has about it, and it is the case
where "added mass" has an unambiguous meaning: mass adds, and the inertia a point mass contributes
about its own centre is zero. An *offset* payload is a different physical object -- it shifts the
centre of mass and adds a parallel-axis inertia term, which changes the attitude dynamics rather than
just the thrust margin -- and getting that right means building a child body before compile. Rather
than quietly approximate it, an ``offset`` is refused with a message saying so.

**Why this runs after compile rather than as a build hook.** Both are defensible; this one is chosen
because the entity registry is what maps ``robot: drone`` to a *prefixed* MJCF body name, and that
registry does not exist until ``configure`` (``spawn_robot`` registers its entity there). A build
hook would have to be told the prefix a second time, in a second place, with nothing checking the two
agree. Writing ``model.body_mass`` and re-deriving the cached constants is the same mechanism
``model_override`` already uses for ``body_mass``, so this is an established write, not a new one.

The mass is stated in the world and therefore recorded with it, so a run's provenance carries the
payload even though the compiled MJCF's own inertial does not.
"""

from __future__ import annotations

import logging

import mujoco

from roqsim.context import SimContext
from roqsim.plugin import Plugin

logger = logging.getLogger(__name__)


class PayloadPlugin(Plugin):
    #: Loads an entity's body, so it belongs inside that entity's ``components:`` block.
    requires_owner = True

    def __init__(self, config=None, *, name=None, entity=None, label=None):
        super().__init__(config, name=name, entity=entity, label=label)
        self.robot = self.config.get("robot") or self.entity
        self._bid = -1

    def validate_config(self, config: dict) -> list[str]:
        errors = []
        if "mass" not in config:
            errors.append("'mass' is required (kg): a payload plugin with no mass is not a payload")
        else:
            try:
                mass = float(config["mass"])
            except (TypeError, ValueError):
                errors.append("'mass' must be a number (kg)")
            else:
                if mass < 0:
                    errors.append("'mass' must be >= 0 kg")
        if "offset" in config:
            # Loud rather than approximate: an offset payload moves the centre of mass and adds a
            # parallel-axis term, which this plugin does not model. Silently ignoring the key would
            # report an attitude result for an airframe nobody configured.
            errors.append(
                "'offset' is not supported: this plugin models a point mass at the body's centre "
                "of mass. An offset payload changes the inertia tensor and needs a body added "
                "before compile."
            )
        return errors

    def _resolve_body(self, ctx: SimContext, entity) -> int:
        """The body the payload is strapped to, by the first rule that resolves.

        Three rules rather than one because the families disagree about naming. ``spawn_robot``
        registers ``entity.body`` as ``<prefix>base_link``, which every wheeled base has and the
        Crazyflie does not -- its root body is ``cf2`` (``quadrotor_controller`` carries a fallback
        for the same reason). The body owning the base *joint* is the one general answer: whatever a
        robot calls its root, the free joint is on it.
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
            # The zero level of a sweep, not a mistake: leave the model untouched so the unloaded
            # cell is bit-identical to a world that never declared a payload at all.
            return

        model = ctx.model
        airframe = float(model.body_mass[self._bid])
        model.body_mass[self._bid] = airframe + mass
        # body_mass feeds cached quantities (subtree masses, gravity compensation); mj_setConst is
        # what makes a mass write take effect rather than sit in an array nothing re-reads.
        mujoco.mj_setConst(model, ctx.data)
        logger.info(
            "payload (%s): %.1f g on %r -- airframe %.1f g, total %.1f g",
            self.robot,
            mass * 1e3,
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, self._bid),
            airframe * 1e3,
            (airframe + mass) * 1e3,
        )
