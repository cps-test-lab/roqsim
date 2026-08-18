"""Fault plugin: change named MuJoCo model values while a run is in progress, on an external trigger.

The runtime twin of ``--set``. A world can be overridden by name *before* it is built, and
:mod:`roqsim.presence` can make a compiled entity absent, but nothing could change a model *parameter*
mid-run -- so a grasp that has to fail at a chosen instant, a wheel that has to lose traction, a
gripper that has to lose clamping force, all had to be scripted by whatever was driving the robot.
This plugin makes them properties of the world instead: name a field, name the objects, name the
target, and let a scenario switch it on and off.

**It never decides when to fire.** There is no time trigger and no condition trigger: a fault's
timing is the experiment's independent variable, and a plugin that owned it would make the trigger
un-sweepable and would put trial logic in the substrate. Severity is not on the wire either -- it is
the configured ``to:`` value, so sweeping how slippery a grasp gets is an ordinary campaign factor and
is deterministic per cell. What crosses the wire is one bit: apply, or restore.

This changes the *physics*, not a report. A perturbation of a reported value is sensor noise and
belongs in a sensor's own config (architecture.rst §9); no perturbation of a report can make a
gripper stop holding.

Not every model field can be written at runtime, and the ones that cannot fail SILENTLY -- so this
ships a curated allowlist, one row per field, and refuses everything else by name with the reason.
Measured against MuJoCo 3.11.0:

* ``body_mass`` written alone is *ignored by the dynamics* (the mass matrix does not move, and
  ``body_invweight0``/``body_subtreemass`` stay at their compile-time values). It needs
  ``mj_setConst``, which this plugin calls for that write class.
* ``geom_size`` corrupts collision: ``geom_rbound`` is cached at compile, so a box grown 0.05 -> 0.4 m
  kept ``rbound`` at 0.0866 and, while overlapping the floor, produced ``ncon = 0`` -- geometry that
  renders big and collides as if small. Refused; change size in ``build``.
* ``geom_friction`` and the two contact masks are live and reverse exactly.

Three refusals are *decisions*, not safety, and are listed so they read as chosen:

``geom_priority``
    It would let a selected geom win the friction argument, but it also governs ``condim``,
    ``solref`` and ``solimp`` -- so raising it swaps the contact's stiffness model at the instant of
    the fault, which is an uncontrolled confound in the one variable being measured.
``sensor_noise`` / ``sensor_cutoff``
    MuJoCo's native sensor noise is a second path competing with the per-sensor config §9 chose.
``opt.*``
    ``sim.contact_override`` and the rest of the ``sim:`` block own the globals, and they own them
    *before compile*, so they are in the compiled model and in the run's provenance. A runtime
    plugin writing the same values would make the recorded value differ from the one that ran.

Two ways an override can silently do nothing, and what happens instead:

1. **The write is ignored.** Handled by the write class above -- refused, or followed by
   ``mj_setConst``.
2. **The write lands and changes nothing.** MuJoCo takes a contact's friction from the geom with the
   higher ``priority``, and at *equal* priority it takes the element-wise **maximum** of the two -- so
   lowering one side cannot bring a pair below the other side's value, and an explicit ``<pair>``
   overrides both. ``configure`` raises for what is certain (an unknown name, an empty selection, a
   ``<pair>`` covering a selected geom, a geom made absent by :mod:`roqsim.presence`, a selection that is
   entirely visual-only geoms, a ``geom_contype`` override without its ``geom_conaffinity`` partner).
   The rest is caught at runtime by reading the *applied* contact rather than predicting it: one step
   after a change, ``verified`` becomes ``landed``, ``no_effect`` (with a warning) or ``untested``.

Config::

    model_override:
      overrides:                       # one or more; each names a field, a selection and a target
        - field: geom_friction         # must be on the allowlist (see `field_catalog`)
          select: [pad_left, pad_right]  # names in the field's own namespace (geom/body/actuator/joint)
          bodies: []                   # ...or every geom of these bodies' subtrees (geom fields only)
          entity: ""                   # ...or an entity's body subtree (geom fields only)
          to: 0.0                      # scalar (broadcast) or the field's full row
      active: false                    # initial state; false = nominal, i.e. the plugin is inert
      namespace: ""                    # transport scope; defaults to this instance's `name:`, so two
                                       # faults in one world do not both serve `/override`
      rate_hz: 10.0                    # publish rate of the two out endpoints

Endpoint ``override`` (in) is a **service**, ``std_srvs/SetBool``: apply or restore, replying
``success`` plus a ``message`` carrying the verdict -- so a scenario's ``service_call()`` can fail the
trial when a fault did not land, instead of a warning nobody reads. Endpoints ``override_state``
(``Bool`` of ``active``) and ``override_verified`` (``String`` of ``verified``) publish continuously,
because a service call leaves no trace in a rosbag and ``mjModel`` is in neither the bag nor the state
recording -- without them an injected fault is invisible to every downstream analysis.

All three are scoped by the instance's ``name:``, so the world above serves
``/grip_fault/override``, ``/grip_fault/override_state`` and ``/grip_fault/override_verified``.

In-process, ``ctx.blackboard`` carries a :class:`ModelOverrideHandle` under
``model_override:<name>``, which is how a ROS-free stepped run (an ``.osc`` action, a test) fires it --
and the plugin imports no ROS at all, so a world using it runs in a plain venv with no middleware
installed. The service is what the same fault looks like when a bridge *is* present.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

import mujoco
import numpy as np

from ..context import Endpoint, SimContext
from ..plugin import Plugin
from ..presence import ABSENT_GEOM_GROUP, entity_geom_ids

_log = logging.getLogger(__name__)

#: MuJoCo clamps a contact's friction to this, so a target of 0 never reads back as 0.
MJMINMU = 1e-5

#: Write classes. ``LIVE`` takes effect on the next step; ``SETCONST`` needs derived quantities
#: recomputed or the dynamics ignore it entirely (measured for ``body_mass``).
LIVE = "live"
SETCONST = "needs_setconst"

#: How an override of this field is verified after it is applied. ``contact`` reads the applied
#: contact -- the only honest check for a field MuJoCo mixes between two geoms; ``model`` reads the
#: row back, which is all that can be checked cheaply for a field with no contact of its own.
_BY_CONTACT = "contact"
_BY_MODEL = "model"


@dataclass(frozen=True)
class FieldSpec:
    """One allowlisted ``mjModel`` field. Rows are data, so adding a field is a row plus a measurement.

    ``does`` / ``caveats`` / ``measured`` are written for a *reader deciding what to override* -- the
    physical effect and the failure it models, not the field's C semantics. They are the plugin's
    documentation and, through ``roqsim scenes describe``, what an agent is handed.
    """

    field: str
    namespace: str  # "geom" | "body" | "actuator" | "joint"; also the id namespace, see _OBJ
    write: str  # LIVE | SETCONST
    verify: str  # _BY_CONTACT | _BY_MODEL
    does: str
    caveats: str
    measured: str


#: The field's prefix declares its id namespace, so no field needs resolution code of its own.
_OBJ = {
    "geom": mujoco.mjtObj.mjOBJ_GEOM,
    "body": mujoco.mjtObj.mjOBJ_BODY,
    "actuator": mujoco.mjtObj.mjOBJ_ACTUATOR,
    "joint": mujoco.mjtObj.mjOBJ_JOINT,
}

#: v1 is exactly what has been measured. Everything else is refused, so no row here is a guess.
_ALLOWED: dict[str, FieldSpec] = {
    spec.field: spec
    for spec in (
        FieldSpec(
            "geom_friction",
            "geom",
            LIVE,
            _BY_CONTACT,
            does=(
                "sliding, torsional and rolling friction of a contact. Lowering it makes a grasp "
                "slip or a wheel spin; 0 is the most a contact can be weakened without removing it."
            ),
            caveats=(
                "MuJoCo takes friction from the geom with the higher `priority`, and at EQUAL "
                "priority the element-wise MAXIMUM of the two -- so lowering one side cannot bring "
                "a pair below the other side's value. Select the geom that owns the contact (for a "
                "parallel gripper, the pads), or select both sides. An explicit <pair> overrides "
                "per-geom friction entirely. A target of 0 reads back as 1e-5 (mjMINMU)."
            ),
            measured="pad friction 0.7 -> 0 drops a held 0.2 kg carton",
        ),
        FieldSpec(
            "geom_contype",
            "geom",
            LIVE,
            _BY_CONTACT,
            does=(
                "collision bitmask. Zeroing it with `geom_conaffinity` stops these geoms colliding "
                "with anything, which releases a grasped object instantly and straight down."
            ),
            caveats=(
                "BOTH masks must be zeroed: the filter is (contype_a & conaffinity_b) || "
                "(contype_b & conaffinity_a), and zeroing one alone was measured to leave the "
                "contact count unchanged. While faulted these geoms also pass through everything "
                "else, so this models 'the object was released', not 'the pads got slippery'."
            ),
            measured="contype alone left ncon unchanged; both fields zeroed took it to 0",
        ),
        FieldSpec(
            "geom_conaffinity",
            "geom",
            LIVE,
            _BY_CONTACT,
            does="the other half of the collision bitmask; see `geom_contype`.",
            caveats="Never override this without `geom_contype` on the same selection.",
            measured="see geom_contype",
        ),
        FieldSpec(
            "actuator_forcerange",
            "actuator",
            LIVE,
            _BY_MODEL,
            does=(
                "the force or torque an actuator may exert. Lowering it models a motor or "
                "air-pressure fault: a gripper that still commands the same position but can no "
                "longer hold its clamp, so a held object creeps out rather than being dropped."
            ),
            caveats=(
                "A position servo keeps commanding its target; only the achievable force changes. "
                "The row is (min, max) and both are usually needed."
            ),
            measured="a saturating position servo went from 50.0 N to 0.5 N on the next step",
        ),
        FieldSpec(
            "body_mass",
            "body",
            SETCONST,
            _BY_MODEL,
            does="a body's mass. Raising it mid-carry models a payload that changes under load.",
            caveats=(
                "The dynamics IGNORE this write until mj_setConst recomputes the derived inertia; "
                "this plugin calls it. Body inertia is not scaled with it, so a large change makes "
                "a body dense rather than bigger."
            ),
            measured=(
                "1 -> 20 kg left the mass matrix at 1.0 until mj_setConst, after which "
                "acceleration under 100 N was -4.81 m/s^2, i.e. correct for 20 kg"
            ),
        ),
    )
}

#: Refused by name, each with the reason. A refusal that explains itself is the difference between
#: "the substrate cannot" and "the substrate chose not to, here is what to use".
_REFUSED: dict[str, str] = {
    "geom_size": (
        "geom_rbound is cached at compile and does not follow, so collision culling silently misses "
        "contacts (measured: a box grown 0.05 -> 0.4 m overlapping the floor produced ncon = 0). "
        "Change geometry in build(), not at runtime."
    ),
    "geom_priority": (
        "priority also governs condim, solref and solimp, so raising it to win a friction argument "
        "silently swaps the contact's stiffness model at the instant of the fault. Select the geom "
        "that already owns the contact instead."
    ),
    "sensor_noise": (
        "sensor noise is per-sensor config (see a sensor's own range_stddev / dropout_percent), "
        "deliberately not a shared mechanism -- and it perturbs a reported value, which is a "
        "different thing from changing the physics."
    ),
    "sensor_cutoff": "see sensor_noise: a sensor owns its own error model.",
}

#: Verification verdicts, published as a string rather than a bool because "nothing was touching the
#: selected geoms, so there was nothing to check" and "the write had no effect" are opposite facts.
LANDED = "landed"
NO_EFFECT = "no_effect"
UNTESTED = "untested"


def field_catalog() -> list[dict]:
    """The allowlist as plain data: what may be overridden, and what each field does.

    World-independent -- ``mjModel``'s field set is a property of MuJoCo, not of a compiled model --
    so a caller can answer "what can be overridden at all?" without building anything. ``roqsim scenes
    describe`` reports exactly this, which is how the table reaches a caller that is not a roqsim
    process. One copy: the runtime guard, the plugin's own documentation and that JSON all read it.
    """
    return [
        {
            "field": spec.field,
            "namespace": spec.namespace,
            "write": spec.write,
            "does": spec.does,
            "caveats": spec.caveats,
            "measured": spec.measured,
        }
        for spec in _ALLOWED.values()
    ]


def refusal_reasons() -> dict[str, str]:
    """Fields refused by name, with the reason -- so a caller is redirected rather than stuck."""
    return dict(_REFUSED)


@dataclass
class OverrideReport:
    """Neutral payload for the two out endpoints."""

    active: bool
    since: float  # sim time the override last became active; -1.0 if it never has
    changes: int  # how many times the state actually changed since reset
    verified: str  # LANDED | NO_EFFECT | UNTESTED


@dataclass
class ModelOverrideHandle:
    """Published by :class:`ModelOverridePlugin`; consumed by in-process drivers (``.osc``, tests)."""

    name: str
    set_active: Callable[[bool], None]
    is_active: Callable[[], bool]
    read_state: Callable[[], OverrideReport]


class _Target:
    """One resolved override: which rows of which field, their nominal values and their targets."""

    def __init__(self, spec: FieldSpec, ids: list[int], nominal: np.ndarray, target: np.ndarray):
        self.spec = spec
        self.ids = ids
        self.nominal = nominal  # saved from the compiled model, so restoring is exact
        self.target = target

    def write(self, model, active: bool) -> None:
        getattr(model, self.spec.field)[self.ids] = self.target if active else self.nominal


class ModelOverridePlugin(Plugin):
    parallel_safe = True  # post_step only reads data.contact and writes its own state

    def __init__(self, config=None, *, name=None):
        super().__init__(config, name=name)
        self.overrides = list(self.config.get("overrides") or [])
        self.initial_active = bool(self.config.get("active", False))
        self.rate_hz = float(self.config.get("rate_hz", 10.0))
        self._ctx: SimContext | None = None
        self._targets: list[_Target] = []
        self._active = self.initial_active
        self._report = OverrideReport(self.initial_active, -1.0, 0, UNTESTED)
        #: Contacts involving the selected geoms at the moment of the last change. A contact-verified
        #: field cannot be judged without it: "no contact now" means nothing if there was none before.
        self._contacts_before = 0
        self._verify_pending = False

    # -- validation ----------------------------------------------------------------------------
    def validate_config(self, config: dict) -> list[str]:
        errors = self.validate_topics(config)
        if float(config.get("rate_hz", 10.0)) <= 0:
            errors.append("'rate_hz' must be > 0")
        entries = config.get("overrides")
        if not entries or not isinstance(entries, list):
            errors.append(
                "'overrides' must be a non-empty list of {field, select/bodies/entity, to}"
            )
            return errors
        for i, entry in enumerate(entries):
            where = f"overrides[{i}]"
            if not isinstance(entry, dict):
                errors.append(f"{where} must be a mapping")
                continue
            field = str(entry.get("field", ""))
            if not field:
                errors.append(f"{where} needs a 'field'")
            elif field in _REFUSED:
                errors.append(f"{where}: field {field!r} is refused -- {_REFUSED[field]}")
            elif field.split(".")[0] == "opt" or field.startswith("opt_"):
                errors.append(
                    f"{where}: field {field!r} is a global option, not a per-object field. The "
                    "sim: block owns those before compile -- use sim.contact_override (or sim.cone / "
                    "sim.gravity), which is in the compiled model and so in the run's provenance."
                )
            elif field not in _ALLOWED:
                errors.append(
                    f"{where}: field {field!r} is not on the allowlist "
                    f"({', '.join(sorted(_ALLOWED))}). Add a row to roqsim.plugins.model_override "
                    "once its write class has been measured."
                )
            if "to" not in entry:
                errors.append(f"{where} needs a 'to' (a scalar, or the field's full row)")
            for key in ("select", "bodies"):
                if key in entry and not isinstance(entry[key], list):
                    errors.append(f"{where}: '{key}' must be a list of names")
            if not any(entry.get(k) for k in ("select", "bodies", "entity")):
                errors.append(f"{where} selects nothing: give 'select', 'bodies' or 'entity'")
            spec = _ALLOWED.get(field)
            if (
                spec
                and spec.namespace != "geom"
                and any(entry.get(k) for k in ("bodies", "entity"))
            ):
                errors.append(
                    f"{where}: 'bodies'/'entity' expand to GEOMS, so they cannot select "
                    f"{spec.namespace}s -- use 'select' with {spec.namespace} names"
                )
        return errors

    # -- lifecycle -----------------------------------------------------------------------------
    def configure(self, ctx: SimContext) -> None:
        self._ctx = ctx
        # Scope the endpoints by this instance's NAME unless the world says otherwise. Two faults in
        # one world -- a grip fault and a traction fault, which is the ordinary case -- would
        # otherwise both serve `/override`, and two services on one name is a collision rather than
        # redundancy. An explicit `namespace:` still wins; an instance the world never named stays
        # unscoped, since `self.name` is then just the class name.
        named = self.name != type(self).__name__
        ns = self.config.get("namespace") or (self.name if named else "")

        self._targets = [self._resolve(ctx, i, entry) for i, entry in enumerate(self.overrides)]
        self._check_mask_pairs()

        # The initial state is applied here, so a world that ships `active: true` (a whole campaign
        # cell that runs degraded) is degraded from the first step rather than from the first trigger.
        if self._active:
            self._apply(ctx, True)

        ctx.blackboard.set(
            f"model_override:{self.name}",
            ModelOverrideHandle(
                name=self.name,
                set_active=self.set_active,
                is_active=lambda: self._active,
                read_state=self.read_state,
            ),
        )
        ctx.interface.add(
            Endpoint(
                name="override",
                direction="in",
                owner=self.name,
                namespace=ns,
                write=lambda payload: self.set_active(bool(payload)),
                backend={
                    "ros2": {
                        # A service, not a topic: apply/restore is a command with an outcome, and the
                        # reply is what lets a scenario fail the trial when a fault did not land.
                        "service": "std_srvs.srv.SetBool",
                        "name": self.topic_override("override") or "override",
                        # Where the handler reads the verdict it replies with.
                        "state_key": f"model_override:{self.name}",
                    }
                },
            )
        )
        for endpoint_name, field, msg in (
            ("override_state", "active", "std_msgs.msg.Bool"),
            ("override_verified", "verified", "std_msgs.msg.String"),
        ):
            ctx.interface.add(
                Endpoint(
                    name=endpoint_name,
                    direction="out",
                    owner=self.name,
                    namespace=ns,
                    read=lambda: self._report,
                    rate_hz=self.rate_hz,
                    backend={
                        "ros2": {
                            "type": msg,
                            # The report is a structure and these types carry one value, so the
                            # endpoint says WHICH field rather than the bridge holding a converter
                            # that knows this plugin's attribute names.
                            "field": field,
                            "topic": self.topic_override(endpoint_name) or endpoint_name,
                        }
                    },
                )
            )
        _log.info(
            "model_override %r: %d override(s) over %d row(s), active=%s",
            self.name,
            len(self._targets),
            sum(len(t.ids) for t in self._targets),
            self._active,
        )

    def _resolve(self, ctx: SimContext, index: int, entry: dict) -> _Target:
        model = ctx.model
        where = f"model_override {self.name!r}: overrides[{index}]"
        spec = _ALLOWED[str(entry["field"])]
        ids = self._select(ctx, where, spec, entry)
        if spec.namespace == "geom":
            self._check_geoms(where, model, ids)

        rows = getattr(model, spec.field)
        nominal = np.array(rows[ids], copy=True)
        target = np.array(nominal, copy=True)
        value = np.asarray(entry["to"], dtype=rows.dtype)
        if value.ndim == 0:
            target[...] = value  # a scalar broadcasts across the row, which is what `to: 0` means
        elif rows.ndim == 2 and value.shape == (rows.shape[1],):
            target[...] = value
        else:
            raise RuntimeError(
                f"{where}: 'to' has shape {tuple(value.shape)} but {spec.field} rows are "
                f"{'scalar' if rows.ndim == 1 else f'length {rows.shape[1]}'}"
            )
        return _Target(spec, ids, nominal, target)

    def _select(self, ctx: SimContext, where: str, spec: FieldSpec, entry: dict) -> list[int]:
        model, ids = ctx.model, set()
        for name in entry.get("select") or []:
            oid = mujoco.mj_name2id(model, _OBJ[spec.namespace], str(name))
            if oid < 0:
                # Fail loudly: a typo'd selection never fires, and a trial then reports the
                # UNFAULTED outcome as if the fault had been injected and survived.
                raise RuntimeError(f"{where}: no {spec.namespace} named {name!r} in this world")
            ids.add(int(oid))
        for body in entry.get("bodies") or []:
            found = entity_geom_ids(model, str(body))
            if not found:
                raise RuntimeError(f"{where}: body {body!r} is unknown or carries no geoms")
            ids.update(found)
        wanted = str(entry.get("entity") or "")
        if wanted:
            # An entity is NOT its body: `spawn_model: {model: graspable_carton, name: parcel}` is
            # entity `parcel` on body `graspable_carton`, so `bodies: [parcel]` resolves to nothing.
            entity = ctx.entities.get(wanted)
            if entity is None or not entity.body:
                raise RuntimeError(
                    f"{where}: entity {wanted!r} is not registered, or has no body "
                    f"(known: {', '.join(sorted(ctx.entities.names())) or 'none'})"
                )
            found = entity_geom_ids(model, entity.body)
            if not found:
                raise RuntimeError(
                    f"{where}: entity {wanted!r} (body {entity.body!r}) has no geoms"
                )
            ids.update(found)
        if not ids:
            raise RuntimeError(f"{where}: selects no {spec.namespace}s")
        return sorted(ids)

    def _check_geoms(self, where: str, model, ids: list[int]) -> None:
        def gname(gid: int) -> str:
            return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or f"geom{gid}"

        absent = [gname(g) for g in ids if int(model.geom_group[g]) == ABSENT_GEOM_GROUP]
        if absent:
            # presence writes the same two mask fields with its own save/restore, and interleaving
            # the two corrupts both. Otherwise the mechanisms stay disjoint: this never touches
            # geom_group or rgba.
            raise RuntimeError(
                f"{where}: {', '.join(absent)} belong to an entity roqsim.presence has made absent; "
                "make it present before overriding its geoms"
            )
        # An explicit <pair> carries its own friction/solref/solimp, which win over the per-geom
        # values -- so the write would land in the model and change nothing about the contact.
        for p in range(model.npair):
            g1, g2 = int(model.pair_geom1[p]), int(model.pair_geom2[p])
            if g1 in ids or g2 in ids:
                raise RuntimeError(
                    f"{where}: {gname(g1)} <-> {gname(g2)} is an explicit <pair>, whose own "
                    "parameters override the per-geom ones; overriding these geoms would do nothing"
                )
        visual = [
            g for g in ids if not int(model.geom_contype[g]) and not int(model.geom_conaffinity[g])
        ]
        if len(visual) == len(ids):
            raise RuntimeError(
                f"{where}: every selected geom is visual-only (contype and conaffinity are 0), so "
                "none of them can affect a contact: " + ", ".join(gname(g) for g in visual)
            )
        if visual:
            # Common with `bodies:`/`entity:`, which sweep up render meshes alongside the colliders.
            _log.warning(
                "%s: %d of %d selected geoms are visual-only and cannot affect a contact: %s",
                where,
                len(visual),
                len(ids),
                ", ".join(gname(g) for g in visual),
            )

    def _check_mask_pairs(self) -> None:
        """``geom_contype`` without ``geom_conaffinity`` over the same rows is a measured no-op."""
        by_field = {"geom_contype": set(), "geom_conaffinity": set()}
        for target in self._targets:
            if target.spec.field in by_field:
                by_field[target.spec.field].update(target.ids)
        for field, partner in (
            ("geom_contype", "geom_conaffinity"),
            ("geom_conaffinity", "geom_contype"),
        ):
            missing = by_field[field] - by_field[partner]
            if missing:
                raise RuntimeError(
                    f"model_override {self.name!r}: {len(missing)} geom(s) override {field} but not "
                    f"{partner}. The collision filter is (contype_a & conaffinity_b) || "
                    "(contype_b & conaffinity_a), so zeroing one alone leaves the contact in place "
                    "-- measured. Override both fields over the same selection."
                )

    # -- the trigger ---------------------------------------------------------------------------
    def set_active(self, on: bool) -> None:
        """Apply the configured targets, or restore nominal. Physics thread only.

        The ROS side reaches this through the bridge, which marshals every inbound payload through
        ``ctx.post`` -- so this always runs on the physics thread at the start of a step, and the
        single-writer rule holds without this plugin doing anything about it.
        """
        ctx = self._ctx
        if ctx is None or bool(on) == self._active:
            return
        self._apply(ctx, bool(on))

    def _apply(self, ctx: SimContext, on: bool) -> None:
        for target in self._targets:
            target.write(ctx.model, on)
        if any(t.spec.write == SETCONST for t in self._targets):
            # Without this the dynamics ignore a mass write entirely -- measured, the mass matrix
            # does not move. Cheap, and only run when a SETCONST-class row is in play.
            mujoco.mj_setConst(ctx.model, ctx.data)

        self._active = on
        self._contacts_before = self._count_selected_contacts(ctx)
        self._verify_pending = on  # a restore writes back saved values; there is nothing to verify
        self._report = OverrideReport(
            active=on,
            since=float(ctx.sim_time) if on else self._report.since,
            changes=self._report.changes + 1,
            verified=UNTESTED,
        )
        _log.info(
            "model_override %r: %s at t=%.3f s (%s)",
            self.name,
            "applied" if on else "restored",
            ctx.sim_time,
            ", ".join(t.spec.field for t in self._targets),
        )

    def read_state(self) -> OverrideReport:
        return self._report

    def on_reset(self, ctx: SimContext) -> None:
        """Back to the CONFIGURED state, not to nominal.

        ``Engine.reset`` resets ``MjData`` and never touches ``MjModel``, so without this an override
        survives into the next repetition of the same process: trial 1 faulted, trials 2..N inherit
        it, nothing crashes, and the nominal control cell silently becomes a faulted one. Restoring
        to the configured value rather than to ``false`` is what keeps ``active: true`` usable as a
        static campaign factor.
        """
        self._ctx = ctx
        for target in self._targets:
            target.write(ctx.model, self.initial_active)
        if any(t.spec.write == SETCONST for t in self._targets):
            mujoco.mj_setConst(ctx.model, ctx.data)
        self._active = self.initial_active
        self._contacts_before = 0
        self._verify_pending = False
        self._report = OverrideReport(self.initial_active, -1.0, 0, UNTESTED)

    def shutdown(self, ctx: SimContext) -> None:
        for target in self._targets:
            target.write(ctx.model, self.initial_active)

    # -- did it land? --------------------------------------------------------------------------
    def post_step(self, ctx: SimContext) -> None:
        if not self._verify_pending:
            return
        self._verify_pending = False
        verdict, detail = self._verify(ctx)
        self._report = OverrideReport(
            self._report.active, self._report.since, self._report.changes, verdict
        )
        if verdict == NO_EFFECT:
            # The failure that would otherwise produce plausible wrong data: a run that believes it
            # injected a fault, with the unfaulted outcome in its metrics.
            _log.warning("model_override %r: the override did not land -- %s", self.name, detail)

    def _verify(self, ctx: SimContext) -> tuple[str, str]:
        """Read what MuJoCo actually applied, rather than predicting what it should have."""
        verdicts = []
        for target in self._targets:
            if target.spec.verify == _BY_MODEL:
                rows = getattr(ctx.model, target.spec.field)[target.ids]
                landed = np.allclose(np.asarray(rows, dtype=float), target.target.astype(float))
                verdicts.append(
                    (LANDED if landed else NO_EFFECT, f"{target.spec.field} read back as {rows!r}")
                )
                continue
            if self._contacts_before == 0:
                verdicts.append((UNTESTED, f"{target.spec.field}: nothing was touching"))
                continue
            if target.spec.field in ("geom_contype", "geom_conaffinity"):
                remaining = self._count_selected_contacts(ctx, target.ids)
                verdicts.append(
                    (LANDED, "contacts gone")
                    if remaining == 0
                    else (NO_EFFECT, f"{remaining} contact(s) still involve the selected geoms")
                )
                continue
            verdicts.append(self._verify_friction(ctx, target))

        if any(v == NO_EFFECT for v, _ in verdicts):
            return NO_EFFECT, "; ".join(d for v, d in verdicts if v == NO_EFFECT)
        if all(v == UNTESTED for v, _ in verdicts):
            return UNTESTED, "; ".join(d for _, d in verdicts)
        return LANDED, "; ".join(d for _, d in verdicts)

    def _verify_friction(self, ctx: SimContext, target: _Target) -> tuple[str, str]:
        want = max(float(target.target[0][0]), MJMINMU)
        selected = set(target.ids)
        for i in range(ctx.data.ncon):
            c = ctx.data.contact[i]
            if int(c.geom1) not in selected and int(c.geom2) not in selected:
                continue
            applied = float(c.friction[0])
            if not np.isclose(applied, want, rtol=0.05, atol=2 * MJMINMU):
                other = int(c.geom2) if int(c.geom1) in selected else int(c.geom1)
                name = (
                    mujoco.mj_id2name(ctx.model, mujoco.mjtObj.mjOBJ_GEOM, other) or f"geom{other}"
                )
                return (
                    NO_EFFECT,
                    f"contact friction is {applied:.4g}, not {want:.4g}: {name} governs this pair "
                    "(higher priority, or equal priority and higher friction -- MuJoCo takes the "
                    "element-wise maximum). Select it too, or select it instead",
                )
        return LANDED, f"contact friction is {want:.4g}"

    def _count_selected_contacts(self, ctx: SimContext, ids: list[int] | None = None) -> int:
        selected = (
            set(ids)
            if ids is not None
            else {gid for t in self._targets if t.spec.namespace == "geom" for gid in t.ids}
        )
        if not selected or ctx.data is None:
            return 0
        return sum(
            1
            for i in range(ctx.data.ncon)
            if int(ctx.data.contact[i].geom1) in selected
            or int(ctx.data.contact[i].geom2) in selected
        )
