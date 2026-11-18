"""A flex's material, set on the spec: the rules and helpers behind ``flex_material``.

A flex's material is a campaign factor, so it has to be settable from a world rather than only in
the MJCF that declares the flex -- whichever asset, end effector or world MJCF that is. Where it is
set is decided by what MuJoCo does with each value at compile. The rules here concern only the
``flex_material`` plugin and ``model_override``'s flex rows, which is why they live beside them
rather than among :mod:`roqsim.flex`'s numbered ones. Each was measured on MuJoCo 4.0.0 and is
pinned by ``tests/test_flex_material.py``:

- **Baked at compile.** ``young`` and ``poisson`` do not survive compile: they are baked into
  ``flex_stiffness`` (the per-element stiffness matrices) and ``mjModel`` keeps neither. A change to
  either has to reach the ``MjSpec`` before compile, which is what :func:`apply_material` does -- a
  flex compiled with ``young="0"`` and given one here is exactly the flex compiled with that value.
  ``damping``, ``friction``, ``solref``, ``solimp``, ``priority`` and ``radius`` are compiled fields
  (``flex_damping``, ...), set here the same way so one block states a whole material.
- **Elasticity only where it is integrated.** Elasticity is read only where :mod:`roqsim.flex`'s
  rule 1 says MuJoCo integrates it: on a solid, on a rope (bending, and torsion through
  ``poisson``), and on a shell whose ``elastic2d`` is not ``none``. ``young`` on a shell left at
  ``elastic2d: none`` changes nothing, and a rigid flex has no degrees of freedom for it to act on.
  A rope is elastic in full only if its MJCF makes it so: MuJoCo gives it a twist joint per edge
  while parsing the ``<flexcomp>``, so a modulus set on the spec of a rope written without one bends
  it but never twists it (measured: the same tip sag, and no twist joints). :func:`apply_material`
  refuses all three rather than recording a material that did not run.
- **No damping written over a compiled zero.** At run time ``flex_damping``, ``flex_friction``,
  ``flex_solref`` and ``flex_solimp`` are live, with one exception: MuJoCo builds the flex's edge
  Jacobian at compile only for a flex whose damping is non-zero, so a damping written into a flex
  COMPILED with ``damping=0`` acts only in part (measured: a cantilever's oscillation barely
  changes, where the same value compiled in, or written over a compiled 1e-9, reproduces the
  compiled run exactly). :func:`refuse_damping_from_zero` refuses that write.
"""

from __future__ import annotations

import mujoco

from ..flex import _flex_label, _is_rigid
from ..plugin import PluginError

#: ``elastic2d`` by name, as MuJoCo's XML spells it; the spec stores the index.
ELASTIC2D = ("none", "bend", "stretch", "both")

#: The material keys :func:`apply_material` sets, and the width of each vector-valued one.
MATERIAL_VECTORS = {"friction": 3, "solref": 2, "solimp": 5}
MATERIAL_SCALARS = ("young", "poisson", "damping", "priority", "radius", "thickness", "elastic2d")
MATERIAL_KEYS = MATERIAL_SCALARS + tuple(MATERIAL_VECTORS)

#: Keys that set elasticity, which a flex reads only where it is integrated.
_ELASTIC_KEYS = ("young", "poisson")
#: Keys only a ``dim=2`` flex has.
_SHELL_KEYS = ("thickness", "elastic2d")


def flex_names(spec: mujoco.MjSpec) -> list[str]:
    """Every flex in *spec*, by the name the compiled model will give it (``#i`` for an unnamed one)."""
    return [_flex_label(flex, index) for index, flex in enumerate(spec.flexes)]


def find_flex(spec: mujoco.MjSpec, name: str, where: str, hint: str = ""):
    """The flex named *name* in *spec*, or a refusal that lists the flexes there are.

    *name* is the flex's name as it compiles -- an attached model's flex carries the prefix it was
    attached under, which is why the refusal lists the names rather than suggesting one.
    """
    for flex in spec.flexes:
        if flex.name == name:
            return flex
    known = ", ".join(repr(n) for n in flex_names(spec)) or "none"
    raise PluginError(
        f"{where}: no flex named {name!r} in this model (flexes: {known}). A flex from an attached "
        "model -- a spawned asset, an end effector -- carries that model's prefix." + hint
    )


def _vector(key: str, value, current) -> list[float]:
    """A vector material value, padded from the flex's own where it is given short."""
    values = [float(v) for v in (value if isinstance(value, (list, tuple)) else [value])]
    return values + [float(v) for v in list(current)[len(values) :]]


def apply_material(flex, values: dict, where: str) -> dict:
    """Set *values* (a subset of :data:`MATERIAL_KEYS`) on the spec *flex*; return what was set.

    A vector key given short (``friction: 1.5``, ``solref: [0.004]``) keeps the flex's own values for
    the rest, so a sweep over one element leaves the others as the model states them. Refuses a key
    the flex would not read (elasticity only where it is integrated): a shell key on a flex that is
    not ``dim=2``, and elasticity on a rigid flex, on a ``dim=2`` flex left at ``elastic2d: none``,
    or on a rope its MJCF leaves without elasticity.
    """
    label = flex.name or "unnamed flex"
    shell = [k for k in _SHELL_KEYS if k in values]
    if shell and flex.dim != 2:
        raise PluginError(
            f"{where}: {', '.join(shell)} set on flex {label!r}, which is dim={flex.dim}; only a "
            "dim=2 flex (a shell) has a thickness and an elastic2d mode."
        )
    applied: dict = {}
    if "elastic2d" in values:
        flex.elastic2d = ELASTIC2D.index(str(values["elastic2d"]))
        applied["elastic2d"] = str(values["elastic2d"])
    elastic = [k for k in _ELASTIC_KEYS if k in values]
    if elastic:
        if _is_rigid(flex):
            raise PluginError(
                f"{where}: {', '.join(elastic)} set on flex {label!r}, which is rigid (every "
                "vertex on one body), so it has no degrees of freedom for elasticity to act on."
            )
        if flex.dim == 2 and int(flex.elastic2d) == 0:
            raise PluginError(
                f"{where}: {', '.join(elastic)} set on flex {label!r}, where MuJoCo does not read "
                "it (dim=2, elastic2d none) -- the flex would run exactly as without it. To make "
                "it elastic, set elastic2d (bend, stretch or both) as well."
            )
        if flex.dim == 1 and flex.young <= 0:
            raise PluginError(
                f"{where}: {', '.join(elastic)} set on rope {label!r}, which its MJCF leaves "
                "without elasticity. MuJoCo gives an elastic rope a twist joint per edge while it "
                "parses the <flexcomp>, so a modulus set afterwards would bend the rope but never "
                "twist or roll it. Write <elasticity young=...> in the rope's <flexcomp>; this "
                "plugin then sets the value a run uses."
            )
    for key in ("young", "poisson", "damping", "radius", "thickness"):
        if key in values:
            setattr(flex, key, float(values[key]))
            applied[key] = float(values[key])
    if "priority" in values:
        flex.priority = int(values["priority"])
        applied["priority"] = int(values["priority"])
    for key, width in MATERIAL_VECTORS.items():
        if key in values:
            vector = _vector(key, values[key], getattr(flex, key))[:width]
            setattr(flex, key, vector)
            applied[key] = vector
    return applied


def refuse_damping_from_zero(model: mujoco.MjModel, flex_ids, target, where: str) -> None:
    """Refuse a run-time ``flex_damping`` write onto a flex compiled with none (no damping written
    over a compiled zero)."""
    for fid in flex_ids:
        if float(model.flex_damping[fid]) == 0.0 and float(target) != 0.0:
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_FLEX, int(fid)) or f"#{fid}"
            raise PluginError(
                f"{where}: flex {name!r} was compiled with damping 0, and MuJoCo builds the edge "
                "Jacobian that damping acts through only for a flex compiled with some -- written "
                "at run time it would act only in part. Compile the flex with a non-zero damping "
                "(in its MJCF, or with a flex_material component) and override from there."
            )
