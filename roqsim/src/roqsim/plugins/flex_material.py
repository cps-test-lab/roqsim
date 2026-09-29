"""Scene plugin: set a flex's material from the world, whatever model declared the flex.

A flex is written in MuJoCo's own ``<flexcomp>``, inside whichever MJCF owns it: a spawned asset, an
arm's ``end_effector``, the world MJCF, or a plugin's build. Its material is what an experiment on it
varies, and a value that lives only in that MJCF cannot be a campaign factor. This plugin states it
in the world, where ``--set`` and a campaign reach it like any other component key.

Config::

    flex_material:
      flex: block             # REQUIRED: the flex's name in the model (an attached model's prefix
                              # included), or a list of names that all take this material
      young: 5.0e+5           # Pa, Young's modulus (write 5.0e+5, not 5e5 -- see below)
      poisson: 0.45           # Poisson's ratio, [0, 0.5)
      damping: 0.002          # s, stiffness-proportional damping
      friction: 1.5           # sliding, or [slide, spin, roll]
      solref: [0.004, 1]      # contact solver reference; a scalar sets the time constant
      solimp: [0.95, 0.99, 0.001, 0.5, 2]
      priority: 1             # contact priority: the higher side's friction/solref/solimp win
      radius: 0.002           # m, collision radius around vertices and elements
      thickness: 0.002        # m, dim=2 (shell) only
      elastic2d: both         # dim=2 only: none | bend | stretch | both

Every key but ``flex`` is optional, and one left out keeps the model's own value, so a world states
exactly the factors it varies. A vector given short keeps the flex's own values for the rest. Each key
is a sweep axis as ``components.<address>.<key>`` -- ``--set components.flex_material.young=2.0e+5``
for an unnamed instance, ``components.<name>.young`` for one with a ``name:``.

**It applies to the spec, before compile, because MuJoCo keeps no Young's modulus.** ``young`` and
``poisson`` are baked into the compiled element stiffness and ``mjModel`` has neither, so no
post-compile write can change them (:mod:`roqsim.plugins._flex_material`, "baked at compile").
The other keys are compiled fields and are set the same way, so one block states the whole material
and the run's provenance records it.
Four of them can ALSO change during a run, on a trigger: ``flex_damping``, ``flex_friction``,
``flex_solref`` and ``flex_solimp`` are rows of the ``model_override`` plugin.

**Declare it after the component that brings the flex in** -- the spawned asset, the arm whose
``end_effector`` carries it -- since plugins build in YAML order; a flex that comes from the world
MJCF is there before any plugin builds. A name that matches no flex is refused, listing the flexes
the model has and naming that order; a key the flex would not
read is refused too -- elasticity on a rigid flex, on a ``dim=1`` flex, or on a shell left at
``elastic2d: none``, and a shell key on a flex that is not one. The integrator is chosen after this
runs, so a flex made elastic here gets ``discrete`` under ``sim.integrator: auto``.

**Write an exponent with a dot and a sign.** The world is YAML 1.1, which reads ``5e5`` as the
string ``"5e5"``; ``5.0e+5`` is a number. A string here is refused with that fix rather than
converted.

Why a plugin rather than a key of ``model_override``: that plugin changes values of the COMPILED
model on a trigger, and restores them; this sets the spec once, before there is a model. Folding a
build phase into it would give one component two lifetimes, and a material that cannot be restored
would sit beside faults that must be. ``sim.contact_override`` is not the place either -- it is global
contact tuning, and a flex's material is a property of one named object.
"""

from __future__ import annotations

import logging

import mujoco

from ..context import SimContext
from ..plugin import Plugin
from ..schema import INJECTED_KEYS, Field
from ._flex_material import ELASTIC2D, MATERIAL_KEYS, MATERIAL_VECTORS, apply_material, find_flex

_log = logging.getLogger(__name__)


class FlexMaterialPlugin(Plugin):
    #: The flex it edits was added by another plugin's build, or by the world MJCF.

    CONFIG_SCHEMA = {
        "young": Field(float, minimum=0.0, unit="Pa", doc="Young's modulus"),
        "poisson": Field(float, minimum=0.0, doc="Poisson's ratio, below 0.5"),
        "damping": Field(float, minimum=0.0, unit="s", doc="stiffness-proportional damping"),
        "priority": Field(int, doc="contact priority; the higher side's contact parameters win"),
        "radius": Field(float, minimum=0.0, unit="m", doc="collision radius"),
        "thickness": Field(float, minimum=0.0, unit="m", doc="shell thickness (dim=2 only)"),
        "elastic2d": Field(str, choices=ELASTIC2D, doc="shell elasticity mode (dim=2 only)"),
    }

    def __init__(self, config=None, *, name=None, entity=None, label=None):
        super().__init__(config, name=name, entity=entity, label=label)
        flex = self.config.get("flex")
        self.flexes: list[str] = [flex] if isinstance(flex, str) else list(flex or [])

    def validate_config(self, config: dict) -> list[str]:
        errors: list[str] = []
        flex = config.get("flex")
        if isinstance(flex, str):
            flex = [flex]
        if (
            not flex
            or not isinstance(flex, list)
            or not all(isinstance(f, str) and f for f in flex)
        ):
            errors.append("'flex' is required: the name of the flex (or a list of names) to set")
        known = {"flex", *MATERIAL_KEYS, *INJECTED_KEYS}
        for key in config:
            if key not in known:
                errors.append(
                    f"'{key}' is not a flex material key. Known: {', '.join(MATERIAL_KEYS)}"
                )
        if not any(key in config for key in MATERIAL_KEYS):
            errors.append(f"sets nothing: give at least one of {', '.join(MATERIAL_KEYS)}")
        for key, value in config.items():
            if key in MATERIAL_KEYS and isinstance(value, str) and _is_number(value):
                errors.append(
                    f"'{key}' is the string {value!r}: YAML 1.1 reads an exponent without a dot "
                    f"and a sign as text. Write it as {float(value):.1e} (e.g. 5.0e+5)."
                )
        poisson = config.get("poisson")
        if _is_real(poisson) and poisson >= 0.5:
            errors.append(
                f"'poisson' must be below 0.5 (incompressible is singular), got {poisson}"
            )
        for key, width in MATERIAL_VECTORS.items():
            if key not in config:
                continue
            value = config[key]
            values = value if isinstance(value, (list, tuple)) else [value]
            if not 1 <= len(values) <= width or not all(_is_real(v) for v in values):
                errors.append(f"'{key}' must be a number or a list of 1 to {width} numbers")
            elif key == "friction" and any(float(v) < 0 for v in values):
                errors.append("'friction' components must be >= 0")
        return errors

    def build(self, spec: mujoco.MjSpec, ctx: SimContext) -> None:
        values = {key: self.config[key] for key in MATERIAL_KEYS if key in self.config}
        where = f"flex_material {self.address!r}"
        for name in self.flexes:
            flex = find_flex(spec, name, where, _ORDER_HINT)
            applied = apply_material(flex, values, where)
            _log.info("%s: flex %r material %s", where, name, applied)


#: Plugins build in YAML order, so the component that brings a flex in has to come first.
_ORDER_HINT = (
    " Plugins build in YAML order: declare flex_material after the component that brings the flex "
    "in (the spawned model, the arm whose end_effector carries it)."
)


def _is_real(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_number(text: str) -> bool:
    try:
        float(text)
    except ValueError:
        return False
    return True
