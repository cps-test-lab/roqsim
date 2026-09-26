"""What MuJoCo requires of a model with a flex, in the one place roqsim knows it.

A world writes MuJoCo's own ``<flexcomp>``, and MuJoCo decides at compile time which integrator and
solver options such a model accepts. roqsim reads the same rules off the ``MjSpec`` before compile,
for two reasons: to choose the integrator when a world leaves ``sim.integrator`` at ``auto``, and to
refuse a combination with a message that names the world key to change -- MuJoCo's own error names
an ``<option>`` attribute the world may never have written, because roqsim set it.

**These rules are MuJoCo-version-sensitive.** Each was measured on MuJoCo 3.14.0, the range the
packages pin (``>=3.14,<3.15``), and ``tests/test_flex_rules.py`` compiles every case and checks
the verdict here against MuJoCo's own, so a MuJoCo that changes one fails that test rather than a
run. Nothing else in roqsim encodes one of these rules; a change to one belongs here. A rule that
only one consumer reads is stated in that consumer's module, named there rather than numbered, and
pinned by that consumer's tests the same way.

The rules, as measured on 3.14.0:

1. **A discrete flex.** MuJoCo integrates some flexes through the ``discrete`` integrator's
   effective metric and refuses to compile them under any integrator that lacks it. A flex is one
   when it is *not rigid* -- its vertices (its nodes, under ``dof="trilinear"`` or
   ``"quadratic"``) sit on more than one body -- and either

   - it has **passive contact** (``<contact passive="true">``). Every other integrator is refused
     ("passive flex contact requires an integrator with the effective metric"); or
   - it has **elasticity MuJoCo integrates**: ``young > 0`` on a ``dim=3`` flex, or on a ``dim=2``
     flex whose ``elastic2d`` is not ``none``. ``implicit`` and ``implicitfast`` are refused ("flex
     elasticity is no longer integrated implicitly"); ``euler`` and ``rk4`` compile it and
     integrate it explicitly.

   Not on this path, and accepted by every integrator: a ``dim=1`` flex's ``young``, a ``dim=2``
   flex with ``elastic2d="none"`` (the default), edge stiffness, damping alone, and any rigid flex
   (all vertices on one body, e.g. ``rigid="true"`` or every vertex pinned).
2. **Solver options under ``discrete``.** A model with a discrete flex refuses the PGS solver and
   any ``noslip_iterations > 0`` ("PGS and noslip not yet supported with flex"). CG and Newton
   work. A model whose flexes are all outside rule 1 accepts both.
3. **No flex under a mocap body.** A flex's vertex DOFs are relative to the body it is declared in,
   and a mocap body moves by having its frame replaced, which applies no acceleration. The flex is
   therefore carried rigidly and never deforms however the mocap moves -- measured: zero
   deformation under a 2 Hz, 5 cm oscillation that bends the same flex by millimetres on a
   position-servoed slide joint. Pinned or not, trilinear or full, it compiles and runs, which is
   why roqsim refuses it: the result would be a flex that silently does not respond to the motion
   the experiment is about.

``sim.integrator: auto`` (the default) resolves to ``discrete`` for a model with a discrete flex and
to ``implicitfast`` otherwise, so a world without one runs exactly as it did before flexes existed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import mujoco

from .plugin import PluginError

_log = logging.getLogger(__name__)

#: The ``sim.integrator`` value that lets the model decide. The default.
AUTO = "auto"

#: What ``auto`` resolves to for a model with no discrete flex. The velocity-servo wheel drives need
#: an implicit integrator for stability (Euler blows them up), and ``implicitfast`` was roqsim's
#: unconditional integrator before ``auto`` existed, so every rigid world keeps it.
RIGID_DEFAULT = "implicitfast"

#: The integrator a discrete flex requires (rule 1).
DISCRETE = "discrete"

#: Integrators that refuse flex elasticity (rule 1, second case).
_IMPLICIT = ("implicit", "implicitfast")


@dataclass(frozen=True)
class DiscreteFlex:
    """One flex MuJoCo integrates under ``discrete`` only, and why."""

    name: str
    #: ``"passive contact"`` or ``"elasticity"``.
    reason: str

    def __str__(self) -> str:
        return f"flex {self.name!r} ({self.reason})"


@dataclass(frozen=True)
class IntegratorChoice:
    """The integrator a model runs under, and how it was arrived at."""

    #: What ``sim.integrator`` said: ``"auto"`` or an integrator name.
    requested: str
    #: The integrator the model is compiled with.
    resolved: str
    #: One line saying why, for the log and ``roqsim check``.
    reason: str


def _flex_label(flex, index: int) -> str:
    return flex.name or f"#{index}"


def _anchor_bodies(flex) -> list[str]:
    """The bodies a flex's degrees of freedom live on: its nodes where it has any, else its vertices."""
    return list(flex.nodebody) if len(flex.nodebody) else list(flex.vertbody)


def _is_rigid(flex) -> bool:
    return len(set(_anchor_bodies(flex))) <= 1


def discrete_flexes(spec: mujoco.MjSpec) -> list[DiscreteFlex]:
    """Every flex in *spec* that MuJoCo integrates under ``discrete`` only (rule 1)."""
    found = []
    for index, flex in enumerate(spec.flexes):
        if _is_rigid(flex):
            continue
        label = _flex_label(flex, index)
        if flex.passive:
            found.append(DiscreteFlex(label, "passive contact"))
            continue
        elastic_dim = flex.dim == 3 or (flex.dim == 2 and int(flex.elastic2d) != 0)
        if flex.young > 0 and elastic_dim:
            found.append(DiscreteFlex(label, "elasticity"))
    return found


def needs_discrete(spec: mujoco.MjSpec) -> bool:
    """Whether MuJoCo requires ``integrator="discrete"`` for *spec* (rule 1)."""
    return bool(discrete_flexes(spec))


def resolve_integrator(requested: str, spec: mujoco.MjSpec) -> IntegratorChoice:
    """The integrator *spec* runs under when ``sim.integrator`` is *requested*.

    ``auto`` resolves to ``discrete`` for a model with a discrete flex and ``implicitfast``
    otherwise; any other value is taken as stated -- :func:`check_flex_options` refuses one the
    model cannot run under. Called after every plugin has built, since a plugin may add the flex.
    """
    if requested != AUTO:
        return IntegratorChoice(requested, requested, "sim.integrator")
    flexes = discrete_flexes(spec)
    if flexes:
        return IntegratorChoice(AUTO, DISCRETE, "auto: " + ", ".join(map(str, flexes)))
    return IntegratorChoice(AUTO, RIGID_DEFAULT, "auto: no flex that needs discrete")


def check_flex_options(spec: mujoco.MjSpec, choice: IntegratorChoice) -> None:
    """Refuse, before compile, a model MuJoCo would reject or run wrongly because of a flex.

    *spec* must carry the options it will compile with (``sim.solver`` and
    ``sim.noslip_iterations`` applied), since the solver rule reads them there -- a world MJCF's own
    ``<option>`` counts as much as a ``sim`` key. Each refusal names the ``sim`` key that fixes it.
    """
    flexes = discrete_flexes(spec)
    named = ", ".join(map(str, flexes))
    integrator = choice.resolved
    if flexes and integrator != DISCRETE:
        passive = [f for f in flexes if f.reason == "passive contact"]
        if integrator in _IMPLICIT or passive:
            raise PluginError(
                f"sim.integrator: {integrator} cannot run {named}: MuJoCo integrates "
                "flex elasticity and passive flex contact only under the discrete integrator. Set "
                "sim.integrator: auto (or discrete)."
            )
    if flexes and integrator == DISCRETE:
        if spec.option.solver == mujoco.mjtSolver.mjSOL_PGS:
            raise PluginError(
                f"sim.solver: pgs cannot solve {named} under the discrete integrator (MuJoCo "
                "supports neither PGS nor noslip with such a flex). Set sim.solver: newton or cg."
            )
        if spec.option.noslip_iterations > 0:
            raise PluginError(
                f"sim.noslip_iterations: {spec.option.noslip_iterations} cannot be used with "
                f"{named} under the discrete integrator (MuJoCo supports neither PGS nor noslip "
                "with such a flex). Set sim.noslip_iterations: 0, or remove the key."
            )
    _refuse_mocap_parent(spec)


def _refuse_mocap_parent(spec: mujoco.MjSpec) -> None:
    """Rule 3: a flex declared in a mocap body is carried rigidly and never deforms."""
    for index, flex in enumerate(spec.flexes):
        for body_name in dict.fromkeys(_anchor_bodies(flex)):
            body = spec.body(body_name)
            while body is not None and body.name != "world":
                if body.mocap:
                    raise PluginError(
                        f"flex {_flex_label(flex, index)!r} is attached to the mocap body "
                        f"{body.name!r}. A mocap body moves by having its frame replaced, which "
                        "applies no acceleration, so the flex would be carried rigidly and never "
                        "deform. Declare the flex in a dynamic body instead -- one that follows the "
                        "mocap target through a weld equality, or an arm's end effector."
                    )
                body = body.parent


# -- which entity a flex belongs to -----------------------------------------------------------------
#
# Ownership rather than an integrator rule, and read off the COMPILED model: a contact observable
# resolves its entity after compile, when the flex's vertices have become bodies with ids.


def flex_dof_body_ids(model: mujoco.MjModel, flex: int) -> list[int]:
    """The bodies flex *flex*'s degrees of freedom live on: its nodes where it has any, else its
    vertices -- the compiled counterpart of the anchor bodies rule 1 reads off the spec.

    A pinned vertex lives on the body the flex is declared in, so that body is among them.
    """
    nodes = int(model.flex_nodenum[flex])
    if nodes:
        start = int(model.flex_nodeadr[flex])
        return [int(b) for b in model.flex_nodebodyid[start : start + nodes]]
    start, count = int(model.flex_vertadr[flex]), int(model.flex_vertnum[flex])
    return [int(b) for b in model.flex_vertbodyid[start : start + count]]


def entity_flex_ids(model: mujoco.MjModel, body_name: str) -> list[int]:
    """Every flex that belongs to the subtree of *body_name*: all of its DOF bodies lie in it.

    **All**, not any: a flex strung between two entities -- a cable from a robot to a wall -- is
    neither one's, because a contact on it is not a contact of either. A flex declared in an
    entity's body, or in an arm's end effector, is that entity's: its vertex bodies are children of
    the body it is declared in, and its pinned vertices sit on that body.

    A flex that only partly lies in the subtree is logged rather than counted, so the entity it
    straddles says so when it is looked at: hiding a cloth pinned between two robots with one of
    them would leave the other holding a flex nothing can see.
    """
    # Imported here so that presence, which hides an entity's flexes, may import this module.
    from .presence import entity_body_ids

    bodies = set(entity_body_ids(model, body_name))
    if not bodies:
        return []
    flexes = []
    for flex in range(model.nflex):
        on = set(flex_dof_body_ids(model, flex))
        inside = on & bodies
        if not inside:
            continue
        if inside == on:
            flexes.append(flex)
        else:
            _log.warning(
                "flex %r sits partly in %r's subtree (%d of its %d bodies) and is not counted as "
                "part of it: presence and anything else that works per entity leave it alone",
                flex_label(model, flex),
                body_name,
                len(inside),
                len(on),
            )
    return flexes


def spec_flex_body_names(flex) -> list[str]:
    """The body names an ``MjSpec`` flex's degrees of freedom live on (rule 1's anchor bodies),
    for a check that must run before compile."""
    return _anchor_bodies(flex)


# -------------------------------------------------------------------------------------------------
# A flex inside a model a plugin attaches: a prop, an arm's tool
# -------------------------------------------------------------------------------------------------
#
# A ``<flexcomp>`` expands, while MuJoCo parses the file, into one body per vertex (per node, under
# ``dof="trilinear"``/``"quadratic"``) carrying three unnamed slide joints and an explicit mass, and
# the flex itself, which names those bodies. A pinned vertex gets no body: the flex names the body
# the ``<flexcomp>`` was declared in -- its *pin body* -- and keeps the vertex's position in that
# body's frame. Three more MuJoCo 3.14.0 behaviours decide how a plugin that attaches such a model
# treats it, each measured in ``tests/test_flex_models.py`` (6: ``roqsim_sensors``'
# ``tests/test_force_torque_flex.py``):
#
# 4. **``MjSpec.attach`` silently drops a flex pinned to the attached model's world body.** A
#    top-level ``<flexcomp>`` with a ``<pin>`` names ``world`` as its pin body, and the spec it is
#    attached into has no flex at all. :func:`lift_top_level_flexes` gives it a body to be pinned to.
# 5. **A mesh or gmsh ``<flexcomp>`` reads its ``file`` while the model is parsed**, against the
#    model's own directory and ``<compiler meshdir>``, and leaves no mesh asset behind. So
#    :func:`roqsim.models.apply_assets` never sees it, and the directories a manifest borrows
#    through ``assets:`` are not searched: the file has to live beside the model.
# 6. **A site force/torque sensor does not see a contact with a flex.** The contact acts on the
#    flex's vertices and is not among the external forces MuJoCo's sensor pass
#    (``mj_rnePostConstraint``) accounts for, so the sensor reads as though it were not there --
#    whether the flex hangs below the sensor or the sensed tool presses into one. Everything else a
#    flex does reaches the sensor: its weight, its elastic reaction (at rest, and in motion, where
#    the reading matches the vertices' momentum balance step by step) and a force applied to a
#    vertex body.


def pin_bodies(spec: mujoco.MjSpec, flex) -> set[str]:
    """The bodies *flex*'s pinned vertices sit on.

    ``<flexcomp>`` declares every vertex body inside the body it is written in, which is also where
    it puts a pinned vertex, so a pin body is an anchor that is the parent of another anchor. A flex
    whose vertices all sit on one body is rigid, and that body is its pin body.
    """
    anchors = set(_anchor_bodies(flex))
    if len(anchors) <= 1:
        return anchors
    pins = set()
    for name in anchors:
        body = spec.body(name)
        parent = body.parent if body is not None else None
        if parent is not None and parent.name in anchors:
            pins.add(parent.name)
    return pins


def owned_bodies(spec: mujoco.MjSpec, flex) -> list[str]:
    """The bodies that carry *flex*'s own degrees of freedom: its vertex (or node) bodies.

    Not its pin bodies, which carry pinned vertices but belong to the model around the flex.
    """
    pins = pin_bodies(spec, flex)
    return [name for name in dict.fromkeys(_anchor_bodies(flex)) if name not in pins]


def flex_is_free(spec: mujoco.MjSpec, flex) -> bool:
    """Whether *flex* is held by nothing: it has vertex bodies and no vertex pinned to a body."""
    return bool(owned_bodies(spec, flex)) and not pin_bodies(spec, flex)


def _declared_at_top(spec: mujoco.MjSpec, flex) -> bool:
    if "world" in _anchor_bodies(flex):
        return True
    for name in owned_bodies(spec, flex):
        body = spec.body(name)
        if body is not None and body.parent is not None and body.parent.name == "world":
            return True
    return False


def lift_top_level_flexes(child: mujoco.MjSpec, root_name: str) -> mujoco.MjSpec:
    """*child*, with everything under its world body moved into one static body, if it has a flex there.

    A ``<flexcomp>`` written directly under ``<worldbody>`` leaves a model with no body of its own:
    its vertex bodies are top-level, so the first of them would pass for the model's root, and a
    pinned one names ``world``, which ``MjSpec.attach`` drops with the flex (rule 4 above). The
    returned spec holds a body named *root_name* at the origin, with no joint, carrying what the
    world body carried; a ``world`` pin now names that body, which is where the pinned vertices
    were. A model with no such flex is returned unchanged, so nothing else gains a body.

    Call it after :func:`roqsim.models.apply_assets` -- the returned spec is built in memory and
    knows no model directory.
    """
    if not any(_declared_at_top(child, flex) for flex in child.flexes):
        return child
    if child.body(root_name) is not None:
        raise PluginError(
            f"a <flexcomp> under <worldbody> needs a body to be attached in, and {root_name!r}, "
            "the name it would get, is already a body of this model. Declare the flexcomp inside a "
            "<body> of its own."
        )
    outer = mujoco.MjSpec()
    root = outer.worldbody.add_body(name=root_name)
    outer.attach(child, prefix="", frame=root.add_frame())
    for flex in outer.flexes:
        flex.vertbody = [root_name if name == "world" else name for name in flex.vertbody]
        flex.nodebody = [root_name if name == "world" else name for name in flex.nodebody]
    return outer


def flex_collides(model: mujoco.MjModel, flex_id: int) -> bool:
    """Whether flex *flex_id* takes part in contacts at all (a nonzero ``contype`` or ``conaffinity``)."""
    return bool(int(model.flex_contype[flex_id]) or int(model.flex_conaffinity[flex_id]))


def flex_label(model: mujoco.MjModel, flex_id: int) -> str:
    """Flex *flex_id*'s name, or ``#<id>`` for an unnamed one."""
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_FLEX, flex_id) or f"#{flex_id}"
