"""Scene plugin: override the contact between ONE pair of things.

Named for what it does relative to the two things beside it: ``sim.contact_override`` sets the same
three parameters for EVERY contact in the world, and this sets them for one pair. (``model_override``
is the third of the family, and changes named model fields mid-run on a trigger.)

Per-geom friction cannot express a pair. MuJoCo combines two geoms' values by taking the **maximum**
of each component, so a geom's friction is a floor on every contact it takes part in and never a
statement about one of them. Two consequences follow, and the second is why this plugin exists:

* A pair cannot be made *less* frictional than either side already is. Lowering one geom does
  nothing while the other stays high.
* Two different pair frictions involving one shared object are not expressible at all. An object
  sliding on a floor at 0.3 while being pushed by a robot at 0.35 needs the two pairs set
  independently, and no assignment of per-geom values produces it.

MuJoCo's own answer is an explicit ``<pair>``, which carries its own friction and wins over the
combination rule. This plugin declares one from world config::

    contact_pair_override:
      a: {entity: robot}     # each side is named ONE of: entity, body, geom
      b: {entity: crate}
      friction: 0.35         # sliding, or the full [slide, slide, spin, roll, roll]
      condim: 3              # contact dimensionality (1 frictionless, 3 sliding, 4 +spin, 6 +roll)
      solref: null           # optional, MuJoCo's two-element contact solver reference
      solimp: null           # optional, five-element solver impedance

The two sides are named separately, and independently, because a real pair often mixes kinds: the
floor is a geom while the thing sliding on it is an entity, and requiring both to be the same kind
would make the commonest pair in a pushing experiment inexpressible.

**A declared pair also FORCES the contact to exist**, which is the part to be careful with. MuJoCo
adds explicit pairs to its contact list without consulting ``contype``/``conaffinity``, so overriding
the friction between two geoms that were deliberately non-colliding *makes them collide*. Measured:
two boxes with ``contype=0 conaffinity=0`` generate no contacts, and four the moment a pair between
them is declared. So this plugin overrides two things at once -- whether the pair touches, and how --
and a world using it on a decorative or sensor-only geom will find that geom has become solid.

Every geom of side A is paired with every geom of side B, so a robot base with twenty collision
geoms touching a five-geom object declares a hundred pairs -- MuJoCo handles that as a list, and the
alternative (asking a world to name every geom) is how a pair silently misses the one that matters.

**Ownership is a pair, so this plugin is NOT nested under an entity.** It names both sides, unlike
the observation plugins that watch one thing and read it from where they sit.

Declaring the same two geoms twice is refused rather than resolved: MuJoCo keeps both pairs and uses
one of them, and which one is not something a world should have to know.
"""

from __future__ import annotations

import logging

import mujoco

from ..context import SimContext
from ..plugin import Plugin

_log = logging.getLogger(__name__)

#: MuJoCo's friction row is (slide, slide, spin, roll, roll); a scalar sets both slide terms and
#: leaves the spin/roll defaults, which is what a world means by "friction 0.35" almost always.
_DEFAULT_SPIN_ROLL = (0.005, 0.005, 0.0001, 0.0001)


class ContactPairOverridePlugin(Plugin):
    #: It names both sides itself, so it sits at the top of a document rather than under an entity.
    requires_owner = False

    def __init__(self, config=None, *, name=None, entity=None, label=None):
        super().__init__(config, name=name, entity=entity, label=label)
        self.a = self.config.get("a")
        self.b = self.config.get("b")
        self.friction = self.config.get("friction")
        self.condim = self.config.get("condim")
        self.solref = self.config.get("solref")
        self.solimp = self.config.get("solimp")

    # -- validation ----------------------------------------------------------------------------
    def validate_config(self, config: dict) -> list[str]:
        errors: list[str] = []
        for side in ("a", "b"):
            sel = config.get(side)
            if sel is None:
                errors.append(f"'{side}' is required: a pair has two sides")
                continue
            if not isinstance(sel, dict):
                errors.append(f"'{side}' must be a mapping, e.g. {{entity: robot}}; got {sel!r}")
                continue
            kinds = [k for k in ("entity", "body", "geom") if sel.get(k)]
            if len(kinds) != 1:
                errors.append(
                    f"'{side}' names exactly one of 'entity', 'body' or 'geom'"
                    + (f"; got {kinds}" if kinds else "")
                )
        if "friction" in config:
            f = config["friction"]
            if isinstance(f, (list, tuple)):
                if len(f) != 5:
                    errors.append(
                        "'friction' is one number or MuJoCo's full 5-element row "
                        f"(slide, slide, spin, roll, roll); got {len(f)} values"
                    )
                elif any(float(v) < 0 for v in f):
                    errors.append("'friction' components must be >= 0")
            elif float(f) < 0:
                errors.append("'friction' must be >= 0")
        if "condim" in config and int(config["condim"]) not in (1, 3, 4, 6):
            errors.append(f"'condim' must be 1, 3, 4 or 6; got {config['condim']}")
        for key, width in (("solref", 2), ("solimp", 5)):
            v = config.get(key)
            if v is not None and (not isinstance(v, (list, tuple)) or len(v) != width):
                errors.append(f"'{key}' must be a list of {width} numbers")
        return errors

    # -- lifecycle -----------------------------------------------------------------------------
    def _friction_row(self) -> list[float] | None:
        if self.friction is None:
            return None
        if isinstance(self.friction, (list, tuple)):
            return [float(v) for v in self.friction]
        f = float(self.friction)
        return [f, f, *(_DEFAULT_SPIN_ROLL[1:])]

    @staticmethod
    def _subtree_geoms(spec: mujoco.MjSpec, body_name: str) -> list[str]:
        """Every geom of `body_name` and everything under it, by name.

        Walks the spec's bodies rather than the compiled model, because pairs must be declared
        before compile -- MuJoCo builds its pair list at compile time and a pair added afterwards
        would be ignored, silently.
        """
        root = spec.body(body_name)
        if root is None:
            raise RuntimeError(f"contact_pair_override: body {body_name!r} not found")
        out, stack = [], [root]
        while stack:
            b = stack.pop()
            out.extend(g.name for g in b.geoms if g.name)
            stack.extend(b.bodies)
        return out

    def _side(self, sel: dict, spec: mujoco.MjSpec, ctx: SimContext) -> list[str]:
        """One side's geom names, from whichever of entity / body / geom it was given by."""
        if sel.get("geom"):
            return [sel["geom"]]
        if sel.get("body"):
            return self._subtree_geoms(spec, sel["body"])
        name = sel["entity"]
        entity = ctx.entities.get(name)
        if entity is None or not entity.body:
            raise RuntimeError(
                f"contact_pair_override: no entity named {name!r} with a base body. Entities are registered "
                "by the plugin that spawns them, so this pair must be declared AFTER both -- one "
                "naming a robot listed below it finds nothing."
            )
        return self._subtree_geoms(spec, entity.body)

    def _sides(self, spec: mujoco.MjSpec, ctx: SimContext) -> tuple[list[str], list[str]]:
        return self._side(self.a, spec, ctx), self._side(self.b, spec, ctx)

    def build(self, spec: mujoco.MjSpec, ctx: SimContext) -> None:
        a, b = self._sides(spec, ctx)
        if not a or not b:
            raise RuntimeError(
                f"contact_pair_override: one side carries no named geoms (got {len(a)} and {len(b)}). "
                "An unnamed geom cannot be paired, because a pair is declared by name."
            )
        seen = {(p.geomname1, p.geomname2) for p in spec.pairs}
        friction = self._friction_row()
        added = 0
        for g1 in a:
            for g2 in b:
                if g1 == g2:
                    continue  # a geom cannot be paired with itself
                if (g1, g2) in seen or (g2, g1) in seen:
                    raise RuntimeError(
                        f"contact_pair_override: {g1!r} <-> {g2!r} is already an explicit pair. MuJoCo keeps "
                        "both and uses one, and which is not something a world should have to know."
                    )
                pair = spec.add_pair()
                pair.geomname1, pair.geomname2 = g1, g2
                if friction is not None:
                    pair.friction = friction
                if self.condim is not None:
                    pair.condim = int(self.condim)
                if self.solref is not None:
                    pair.solref = [float(v) for v in self.solref]
                if self.solimp is not None:
                    pair.solimp = [float(v) for v in self.solimp]
                seen.add((g1, g2))
                added += 1
        _log.info(
            "contact_pair_override %r: %d pair(s) over %d x %d geoms%s",
            self.address,
            added,
            len(a),
            len(b),
            f", friction {friction[0]}" if friction else "",
        )
