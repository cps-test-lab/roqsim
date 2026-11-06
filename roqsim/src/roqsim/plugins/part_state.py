# Copyright (C) 2026 Frederik Pasch
#
# SPDX-License-Identifier: Apache-2.0

"""World plugin: an entity's discrete states -- what a part looks like after a machine worked on it.

A press punches a hole, a bender bends a strip, a labeller labels a carton; a machine's own lamp is
on or off, its film fed or not. The entity stays the same body with the same mass; what changes is
which of its states is shown. The model cannot change at run time (no recompile), so an entity
carries every state it can have from the start, as child bodies compiled in and absent until
something shows them. This plugin finds them, keeps them, and shows and hides them -- for a
machine, a scenario, a trigger or a test, through one handle. It decides nothing: *when* a part
changes is the machine's (the stroke's bottom) or the scenario's (see ``docs/architecture.rst``,
"Machines and scenarios").

**A state is a child body named** ``look_<name>`` **anywhere under an entity's root body**, after the
entity's prefix (``s1_look_hole`` on an entity spawned with ``prefix: s1_``), or ``<body>_look_<name>``
for one a plugin added to the entity at build. A state's bodies are massless -- one with mass is
refused when the states are collected, naming the body -- so showing or hiding a state never changes
what the entity weighs. Its geoms are usually visual only; a state that changes the shape -- a flat
strip against the bent one -- carries its own collision geoms, and they come and go with it. The mass
is the root body's and does not change.

**A state that changes the shape can change where the mass is.** A strip bent into an L has its
centre of mass elsewhere than the flat strip it was. Such a state names its inertia in the model as
a custom numeric ``look_<name>_inertia`` of ten values, the root body's ``<inertial>`` with the
state shown: ``pos`` (3), ``quat`` (4, w x y z) and ``diaginertia`` (3), in the root body's frame.
While the state is shown the root body has that inertia; hidden, the root's own comes back. Two
shown states that both name an inertia are an error, since the entity cannot have both: showing the
second is refused before anything changes.

**Shown and hidden through** :mod:`roqsim.presence`. Each state is registered as an entity of its
own, named after its owner (``s1.look_hole``, kind ``look``), and made absent or present as presence
makes any entity so: its geoms leave the picture, the raycasts and the contacts together. It is not
*listed* (``Entity.listed``): the control plane does not list a state, give it a pose endpoint, or
spawn or delete one -- only this plugin changes a state. A recording notes every change of the
absent set and a replay re-applies it, so ``roqsim render --state`` shows a strip punched from the
step it was punched. What a replay does not restore is the inertia a shown state names: a replay
draws the recorded poses and needs no mass, so the picture is the same. Every change is also an
event (``shown``/``hidden``, with ``entity`` and ``look``), so a run's timeline says when.

Presence composes (:mod:`roqsim.presence`), so a state's own presence and its owner's are independent:
while the owner is absent a state shows nothing, and a state changed meanwhile is as it was left when
the owner returns.

**Limits.** A state body nested inside another state body is refused. Of an exclusive group at most one state is shown: ``show`` of a member
while another is shown is refused (``set`` swaps them), and so is a ``shown:`` naming two.

Config::

    part_state:
      shown:                 # shown at the start of every episode, by entity name or fnmatch
        strip_1: [plain]     #   pattern ("strip_*"); every other state of every entity starts hidden
        press_1: [lamp_off]
      exclusive:             # by entity name or pattern: groups of which at most one is shown;
        strip_1: [[plain, hole, bent]]   # set() shows one and hides the rest of its group

Every entity is searched; an entity needs no declaration beyond having ``look_*`` bodies. An entity
is found wherever its entry stands: at ``configure`` if it was spawned before this entry, at the
first reset otherwise.

Use: a :class:`PartStates` handle on the blackboard under ``part_state`` (``show``, ``hide``,
``set``, ``shown``, ``is_shown``, ``of``, ``entities``), and the same ``show``/``hide``/``set``
commands as endpoints. Naming an entity or a state that does not exist raises, and so does a
``shown`` or ``exclusive`` pattern that matches no entity with states, or a group naming a state an
entity it matches lacks: a machine configured with the wrong name would otherwise run a whole line
and change nothing.
"""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass, field

import mujoco
import numpy as np

from roqsim import endpoint
from roqsim.context import Entity, SimContext
from roqsim.plugin import Plugin
from roqsim.presence import set_present, subtree_body_ids

LOOK_PREFIX = "look_"


class PartStateError(RuntimeError):
    pass


@dataclass
class LookInertia:
    """An entity whose states move its mass: its root body, the root's own inertia, and each state's."""

    root: int
    #: (ipos, iquat, inertia, sameframe) the root body compiled with
    own: tuple
    #: state name -> (ipos, iquat, inertia) with that state shown
    looks: dict[str, tuple] = field(default_factory=dict)


@dataclass
class PartStates:
    """The states of every entity in the world. Physics thread only."""

    #: entity -> state name -> the state's entity
    looks: dict[str, dict[str, Entity]] = field(default_factory=dict)
    #: entity -> where its states put its mass, for the entities with a state that names one
    inertia: dict[str, LookInertia] = field(default_factory=dict)
    #: entity -> its exclusive groups (each a list of state names)
    groups: dict[str, list[list[str]]] = field(default_factory=dict)
    ctx: SimContext | None = None
    #: the source of the events a change emits
    source: str = "part_state"

    def entities(self) -> list[str]:
        """The entities that have states."""
        return sorted(self.looks)

    def of(self, entity: str) -> list[str]:
        """The names of ``entity``'s states."""
        return sorted(self._entity(entity))

    def shown(self, entity: str) -> list[str]:
        """The names of ``entity``'s states that are shown now."""
        return sorted(name for name, e in self._entity(entity).items() if e.present)

    def is_shown(self, entity: str, look: str) -> bool:
        return self._look(entity, look).present

    def show(self, entity: str, look: str) -> bool:
        """Show ``look`` on ``entity``; whether it was hidden before. A state of an exclusive
        group is refused while another of the group is shown: :meth:`set` swaps them."""
        others = [o for o in self._group_of(entity, look) if o != look and self.is_shown(entity, o)]
        if others and not self.is_shown(entity, look):
            raise PartStateError(
                f"part_state: {entity!r} shows {others}, of the exclusive group of {look!r}; "
                f"set({entity!r}, {look!r}) shows it and hides the rest"
            )
        return self._change(entity, look, True)

    def hide(self, entity: str, look: str) -> bool:
        """Hide ``look`` on ``entity``; whether it was shown before."""
        return self._change(entity, look, False)

    def set(self, entity: str, look: str) -> bool:
        """Show ``look`` and hide the other states of its exclusive group, as one change; whether
        anything changed. A state in no group is shown alone."""
        self._look(entity, look)
        hiding = [o for o in self._group_of(entity, look) if o != look]
        self._check_inertia(entity, look, hiding=hiding)  # refused before anything changes
        changed = False
        for other in hiding:
            changed |= self._change(entity, other, False, place=False)
        changed |= self._change(entity, look, True)
        return changed

    def _group_of(self, entity: str, look: str) -> list[str]:
        """The exclusive group ``look`` belongs to on ``entity``, or ``[]``."""
        for group in self.groups.get(entity, []):
            if look in group:
                return group
        return []

    def _change(self, entity: str, look: str, present: bool, place: bool = True) -> bool:
        if present:
            self._check_inertia(entity, look)  # before anything changes
        changed = set_present(self.ctx, self._look(entity, look), present)
        if place:
            self.place_mass(entity)
        if changed:
            self.ctx.events.emit(
                self.source, "shown" if present else "hidden", entity=entity, look=look
            )
        return changed

    def _check_inertia(self, entity: str, look: str, hiding: list[str] = ()) -> None:
        """Refuse showing ``look`` while another shown state of ``entity`` names an inertia too --
        one the same change does not hide."""
        record = self.inertia.get(entity)
        if record is None or look not in record.looks:
            return
        others = [
            name
            for name in self.shown(entity)
            if name in record.looks and name != look and name not in hiding
        ]
        if others:
            raise PartStateError(
                f"part_state: {entity!r} shows {others}, which names an inertia, and {look!r} names "
                "one too; a part has one -- hide the other first"
            )

    def place_mass(self, entity: str) -> None:
        """Give ``entity``'s root body the inertia of its shown state that names one, else its own."""
        record = self.inertia.get(entity)
        if record is None:
            return
        shown = [name for name in self.shown(entity) if name in record.looks]
        if len(shown) > 1:
            raise PartStateError(
                f"part_state: {entity!r} shows {shown}, which each name an inertia; it has one"
            )
        model = self.ctx.model
        if shown:
            ipos, iquat, inertia = record.looks[shown[0]]
            # The compiled root may have its inertial frame at its body frame (sameframe), which
            # makes MuJoCo skip ipos and iquat; a state's inertia sits elsewhere, so it is not.
            sameframe = int(mujoco.mjtSameFrame.mjSAMEFRAME_NONE)
        else:
            ipos, iquat, inertia, sameframe = record.own
        model.body_ipos[record.root] = ipos
        model.body_iquat[record.root] = iquat
        model.body_inertia[record.root] = inertia
        model.body_sameframe[record.root] = sameframe

    def _entity(self, entity: str) -> dict[str, Entity]:
        if entity not in self.looks:
            raise PartStateError(
                f"part_state: {entity!r} has no states; entities with states: "
                f"{self.entities() or 'none'}"
            )
        return self.looks[entity]

    def _look(self, entity: str, look: str) -> Entity:
        looks = self._entity(entity)
        if look not in looks:
            raise PartStateError(
                f"part_state: {entity!r} has no state {look!r}; it has {sorted(looks)}"
            )
        return looks[look]


def find_looks(model: mujoco.MjModel, entity: Entity) -> dict[str, str]:
    """``{state name: body name}`` for the ``look_*`` bodies under ``entity``'s root body."""
    if not entity.body:
        return {}
    root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, entity.body)
    if root < 0:
        return {}
    prefix = entity.meta.get("prefix", "") if isinstance(entity.meta, dict) else ""
    out = {}
    ids = {}
    for bid in subtree_body_ids(model, root):
        if bid == root:
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid) or ""
        local = name[len(prefix) :] if prefix and name.startswith(prefix) else name
        if local.startswith(LOOK_PREFIX):
            look = local[len(LOOK_PREFIX) :]
        elif f"_{LOOK_PREFIX}" in local:
            # A state another plugin added to the entity, named after the entity's body.
            look = local.rsplit(f"_{LOOK_PREFIX}", 1)[1]
        else:
            continue
        if look in out:
            raise PartStateError(
                f"part_state: bodies {out[look]!r} and {name!r} of {entity.name!r} both name the "
                f"state {look!r}; a state is one body"
            )
        out[look] = name
        ids[bid] = name
    # A state inside another would be shown and hidden with it as well as on its own.
    for bid, name in ids.items():
        parent = int(model.body_parentid[bid])
        while parent != root and parent > 0:
            if parent in ids:
                raise PartStateError(
                    f"part_state: state body {name!r} of {entity.name!r} is inside the state body "
                    f"{ids[parent]!r}; a state is a body of its own under the entity, not one "
                    "nested in another"
                )
            parent = int(model.body_parentid[parent])
    return out


def _refuse_mass(model: mujoco.MjModel, entity: str, body: str) -> None:
    """Refuse a state whose bodies carry mass: showing or hiding it must not change the part's weight
    (and an absent body's gravity is compensated, so its weight would come and go with it)."""
    root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
    mass = sum(float(model.body_mass[b]) for b in subtree_body_ids(model, root))
    if mass > 0.0:
        raise PartStateError(
            f"part_state: state body {body!r} of {entity!r} has mass {mass:g} kg; a state's bodies "
            'are massless (mass="0" on their geoms), so that a state changes how the part looks and '
            "collides, never what it weighs -- a state that moves the mass names an inertia instead"
        )


def _names_list(value) -> bool:
    return isinstance(value, list) and all(isinstance(x, str) for x in value)


class PartStatePlugin(Plugin):
    def __init__(self, config=None, *, name=None, entity=None, label=None):
        super().__init__(config, name=name, entity=entity, label=label)
        self.handle = PartStates(source=self.address)
        self._owner_of: dict[str, str] = {}

    def validate_config(self, config: dict) -> list[str]:
        errors = []
        unknown = set(config) - {"shown", "exclusive"}
        if unknown:
            errors.append(
                f"part_state: unknown key(s) {sorted(unknown)}; it takes 'shown', 'exclusive'"
            )
        shown = config.get("shown", {})
        if not isinstance(shown, dict) or not all(_names_list(v) for v in shown.values()):
            errors.append(
                "part_state: 'shown' maps an entity name pattern to a list of state names"
            )
        exclusive = config.get("exclusive", {})
        if not isinstance(exclusive, dict) or not all(
            isinstance(groups, list) and all(_names_list(g) and len(g) > 1 for g in groups)
            for groups in exclusive.values()
        ):
            errors.append(
                "part_state: 'exclusive' maps an entity name pattern to a list of groups, each a "
                "list of two or more state names"
            )
        return errors

    def configure(self, ctx: SimContext) -> None:
        if "part_state" in ctx.blackboard:
            raise PartStateError(
                f"part_state: {self.address!r} is a second entry; one serves the whole world"
            )
        self.handle.ctx = ctx
        ctx.blackboard.set("part_state", self.handle)
        self._refuse_stray_inertia(ctx.model)
        self._collect(ctx)

    @staticmethod
    def _refuse_stray_inertia(model: mujoco.MjModel) -> None:
        """A numeric meant for a state whose body is not there would never move any mass."""
        for nid in range(model.nnumeric):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_NUMERIC, nid) or ""
            if LOOK_PREFIX in name and name.endswith("_inertia"):
                body = name[: -len("_inertia")]
                if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body) < 0:
                    raise PartStateError(
                        f"part_state: numeric {name!r} names an inertia for a state body "
                        f"{body!r} the model does not have"
                    )

    def _collect(self, ctx: SimContext) -> None:
        """Find every entity's states and register each as an entity.

        At ``configure`` for the entities spawned before this entry, and again at every reset for
        those spawned after it; a state already registered is kept, so the second pass only adds."""
        for entity in list(ctx.entities.all()):
            if entity.kind == "look":
                continue
            found = find_looks(ctx.model, entity)
            if not found:
                continue
            looks = {}
            for look, body in found.items():
                owner = self._owner_of.setdefault(body, entity.name)
                if owner != entity.name:
                    # Each would hide it and keep what it hid, so the second's show changes nothing.
                    raise PartStateError(
                        f"part_state: state body {body!r} is under both {owner!r} and "
                        f"{entity.name!r}; a state belongs to one entity"
                    )
                _refuse_mass(ctx.model, entity.name, body)
                name = f"{entity.name}.{LOOK_PREFIX}{look}"
                look_entity = ctx.entities.get(name)
                if look_entity is None:
                    # Registered, so a recording carries and replays its presence, but not listed:
                    # the control plane neither lists a state nor spawns or deletes one.
                    look_entity = Entity(
                        name=name,
                        kind="look",
                        body=body,
                        meta={"part": entity.name, "look": look, "owner": self.address},
                        listed=False,
                    )
                    ctx.entities.add(look_entity)
                looks[look] = look_entity
            self.handle.looks[entity.name] = looks
            self._collect_inertia(ctx, entity, found)

    def _matching(self, pattern: str, what: str) -> list[str]:
        names = [e for e in self.handle.entities() if fnmatch.fnmatchcase(e, pattern)]
        if not names:
            raise PartStateError(
                f"part_state: {what} pattern {pattern!r} matches no entity with states; entities "
                f"with states: {self.handle.entities() or 'none'}"
            )
        return names

    def _resolve(self) -> dict[str, set[str]]:
        """The states shown at the start, by entity; the exclusive groups, checked."""
        shown: dict[str, set[str]] = {}
        for pattern, looks in (self.config.get("shown") or {}).items():
            for entity in self._matching(pattern, "a 'shown'"):
                for look in looks:
                    self.handle._look(entity, look)  # raises on a state that is not there
                    shown.setdefault(entity, set()).add(look)
        self.handle.groups = {}
        for pattern, groups in (self.config.get("exclusive") or {}).items():
            for entity in self._matching(pattern, "an 'exclusive'"):
                for group in groups:
                    for look in group:
                        self.handle._look(entity, look)
                    others = self.handle.groups.setdefault(entity, [])
                    twice = sorted({look for other in others for look in other} & set(group))
                    if twice:
                        raise PartStateError(
                            f"part_state: {twice} on {entity!r} are in two exclusive groups; a "
                            "state is in one"
                        )
                    others.append(list(group))
        for entity, looks in shown.items():
            for group in self.handle.groups.get(entity, []):
                both = sorted(set(group) & looks)
                if len(both) > 1:
                    raise PartStateError(
                        f"part_state: 'shown' shows {both} on {entity!r}, of one exclusive group; "
                        "at most one of a group is shown"
                    )
        return shown

    def _collect_inertia(self, ctx: SimContext, entity: Entity, found: dict[str, str]) -> None:
        """Note the states of ``entity`` that name an inertia (``<state body>_inertia`` numerics)."""
        if entity.name in self.handle.inertia:
            return
        model = ctx.model
        looks = {}
        for look, body in found.items():
            nid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_NUMERIC, f"{body}_inertia")
            if nid < 0:
                continue
            adr, size = model.numeric_adr[nid], model.numeric_size[nid]
            if size != 10:
                raise PartStateError(
                    f"part_state: {body}_inertia has {size} values; it takes pos (3), quat (4) and "
                    "diaginertia (3)"
                )
            values = model.numeric_data[adr : adr + 10].copy()
            norm = float(np.linalg.norm(values[3:7]))
            a, b, c = sorted(values[7:])
            if not np.isfinite(values).all() or norm == 0.0 or a <= 0.0 or a + b < c:
                raise PartStateError(
                    f"part_state: {body}_inertia is no inertia: it takes finite numbers, a quat "
                    "that is not zero, and a positive diaginertia no value of which exceeds the "
                    "sum of the other two"
                )
            looks[look] = (values[:3].copy(), values[3:7] / norm, values[7:].copy())
        if not looks:
            return
        root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, entity.body)
        own = (
            model.body_ipos[root].copy(),
            model.body_iquat[root].copy(),
            model.body_inertia[root].copy(),
            int(model.body_sameframe[root]),
        )
        self.handle.inertia[entity.name] = LookInertia(root=root, own=own, looks=looks)

    def on_reset(self, ctx: SimContext) -> None:
        self._collect(ctx)
        shown = self._resolve()
        for entity, looks in self.handle.looks.items():
            for look, look_entity in looks.items():
                set_present(ctx, look_entity, look in shown.get(entity, ()))
        for entity in self.handle.inertia:
            self.handle.place_mass(entity)
        mujoco.mj_forward(ctx.model, ctx.data)

    @endpoint.command(ros2=None)
    def show(self, entity: str, look: str) -> bool:
        """Show a state on an entity; true if it was hidden.

        Args:
            entity: the entity's name
            look: the state's name (its body is look_<name>)
        """
        return self.handle.show(entity, look)

    @endpoint.command(ros2=None)
    def hide(self, entity: str, look: str) -> bool:
        """Hide a state on an entity; true if it was shown.

        Args:
            entity: the entity's name
            look: the state's name (its body is look_<name>)
        """
        return self.handle.hide(entity, look)

    @endpoint.command(ros2=None)
    def set(self, entity: str, look: str) -> bool:
        """Show a state and hide the others of its exclusive group; true if anything changed.

        Args:
            entity: the entity's name
            look: the state's name (its body is look_<name>)
        """
        return self.handle.set(entity, look)
