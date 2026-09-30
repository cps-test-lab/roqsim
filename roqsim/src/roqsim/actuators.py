# Copyright (C) 2026 Frederik Pasch
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""What law a joint runs under, stated by the world rather than baked into the shared model.

A model ships one set of actuator gains, and they are that model's own sizing -- the ur5e's servo is
softer than the ur10e's because it is a 5 kg-payload arm. An experiment reproducing a published
controller needs *its* gains, which are a property of the experiment and not of the robot. Without
this, saying so meant editing the shared MJCF: every other world that spawns the model silently
inherits the edit, and the value that ran is recorded nowhere.

``actuators:`` on a spawn plugin states the law and the gains, and this module rewrites the model's
own actuators to match **on the world's copy of the spec, before it is compiled**. The file on disk
is never touched, a world that declares nothing compiles byte-identically, and the resolved table
travels into the run's provenance so a reader can see what the joints actually ran under.

The vocabulary is the robot's, not MuJoCo's: ``control`` names a ros2_control command interface, and
the gains are the ones a real controller's yaml carries. That is deliberate -- a port transcribing a
paper reads its numbers out of a controller config, and a key it has to translate is a key it can get
wrong. ``effort_limit`` is URDF's ``<limit effort=>``, which :mod:`roqsim.export_urdf` already emits
from ``actuator_forcerange``, so the key a world writes is the key roqsim exports.

Four laws, and what each compiles to::

    control      commands        gains                 MuJoCo
    position     joint position  p, d                  gaintype fixed, biastype affine
    velocity     joint velocity  d                     gaintype fixed, biastype affine
    effort       joint torque    --                    gaintype fixed, biastype none
    impedance    joint position  stiffness, damping    the same affine law, in stiffness terms

``impedance`` is not a second spelling of ``position``: it states the joint in the terms a
compliance controller is specified in (Franka's ``joint_impedance``, a UR in force mode), and a
paper that gives a stiffness should be transcribed without first converting it into somebody's
servo gain.

**All three of these carry their own weight**, because the hardware they name does: a drive
commanded to a pose holds it, and its gain says how hard the joint resists a *disturbance*, not how
much of the arm it can lift. That is :func:`servo_holds_against_gravity`, and
:func:`apply_gravity_compensation` is how a spawn plugin honours it. ``effort`` is the exception --
a torque-commanded joint applies the torque it is handed, and supplying the gravity term is the
controller's job, which is frequently the very thing under test.

**The compensation is a joint torque, supplied by the drive.** A servo holding a link up pushes
against the body it is mounted on, so the weight it holds still reaches the ground through that body.
MuJoCo's ``body_gravcomp`` alone is an *external* force at each body's centre of mass, which on a
robot standing on a free joint lifts the whole robot. Two things make it internal:
:func:`apply_gravity_compensation` routes the term through each holding joint's actuator
(``actuatorgravcomp``, so it counts against the drive's force limit), and :class:`GravityReaction`
takes it off every degree of freedom above the mechanism -- a mobile base's free joint -- each step.

A drive that genuinely has no gravity term -- a hobby servo, a backdrivable joint -- is a real
machine too, and a spawn plugin's ``gravity_compensation: false`` says so.

**Shared keys sit on the block; per-actuator entries nest under** ``each:``. Nothing a world writes
can then collide with an actuator name, and the common case -- one law for the whole arm, which is
what a paper states -- needs no nesting at all::

    actuators: {control: impedance, stiffness: 2.0, damping: 0.02}

Keys under ``each:`` are **actuator** names as the model declares them, unprefixed: this runs on the
child spec before it is attached, so no prefix has been applied yet. A joint name is refused naming
the actuator that drives it, rather than resolved silently -- two namespaces in one mapping would
mean a reader of someone else's world could not tell which a key was without opening the MJCF.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import mujoco
import numpy as np

from .plugin import PluginError
from .schema import Field

#: The control laws a world may name. Ordered as the docstring's table.
CONTROLS = ("position", "velocity", "effort", "impedance")

#: What a control's ``ctrl`` value MEANS. Two laws share the position unit, which is the whole point
#: of this table: moving between them leaves the model's ``ctrlrange`` correct, so it is only a move
#: ACROSS units that turns a joint-range ctrlrange into a clamp on a torque command.
_COMMAND_UNIT = {
    "position": "position",
    "impedance": "position",
    "velocity": "velocity",
    "effort": "effort",
}

#: How a command unit reads in an error. Both spellings, because a slide joint is metres.
_UNIT_LABEL = {
    "position": "rad (m on a slide joint)",
    "velocity": "rad/s (m/s on a slide joint)",
    "effort": "N*m (N on a slide joint)",
}

#: The gains each control reads. A gain named for a control that does not read it is refused: it
#: would otherwise be accepted, ignored, and the joint would run under something else entirely.
_GAINS: dict[str, tuple[str, ...]] = {
    "position": ("p", "d"),
    "velocity": ("d",),
    "effort": (),
    "impedance": ("stiffness", "damping"),
}

#: MuJoCo's spellings of a gain block's keys, each with what this vocabulary writes instead. They are
#: refused as unknown keys, with this as the hint (:attr:`roqsim.schema.Field.hints`), and nothing
#: reads them: a key merely not read would be accepted in silence and the world would run under
#: gains nobody chose.
MUJOCO_KEYS = {
    key: f"that is MuJoCo's spelling; use {replacement}"
    for key, replacement in {
        "kp": "'p' (control: position) or 'stiffness' (control: impedance)",
        "kv": "'d' (control: position/velocity) or 'damping' (control: impedance)",
        "kd": "'d' (control: position) or 'damping' (control: impedance)",
        "forcerange": "'effort_limit', a single positive magnitude",
        "gainprm": "'p'/'stiffness' -- state the gain, not MuJoCo's parameter vector",
        "biasprm": "'d'/'damping' -- state the gain, not MuJoCo's parameter vector",
        "mode": "'control'",
    }.items()
}

#: MuJoCo's actuator types, which a ``control`` names a robot's command interface in place of.
MUJOCO_CONTROLS = {
    control: f"that is a MuJoCo actuator type; the command interface is {interface}"
    for control, interface in {
        "motor": "effort",
        "pd": "impedance",
        "general": "position, velocity, effort or impedance",
    }.items()
}

#: One gain block: the shared keys, and the same set again inside every ``each:`` entry. Units are
#: declared because a paper states "kp = 2.0" with none, and a value wrong by a factor of a thousand
#: is indistinguishable from a right one -- which is the failure this whole module exists to prevent.
#: The rotational unit is named; a slide joint's is the linear equivalent (N/m, N*s/m, N).
GAIN_SCHEMA: dict[str, Field] = {
    "control": Field(
        str,
        choices=CONTROLS,
        hints=MUJOCO_CONTROLS,
        doc="the command interface this joint runs under, as a robot's driver exposes it",
    ),
    "p": Field(
        float,
        minimum=0.0,
        unit="N*m/rad",
        doc="position-loop proportional gain (control: position)",
    ),
    "d": Field(
        float,
        minimum=0.0,
        unit="N*m*s/rad",
        doc="damping/derivative gain (control: position, velocity)",
    ),
    "stiffness": Field(
        float, minimum=0.0, unit="N*m/rad", doc="joint stiffness (control: impedance)"
    ),
    "damping": Field(
        float, minimum=0.0, unit="N*m*s/rad", doc="joint damping (control: impedance)"
    ),
    "effort_limit": Field(
        float,
        minimum=0.0,
        unit="N*m",
        doc="torque magnitude cap; URDF's <limit effort=>. A one-sided forcerange keeps its side",
    ),
    "ctrlrange": Field(
        list,
        length=2,
        doc="command limits, in the unit `control` commands; needed on a unit change",
    ),
}

#: The ``actuators:`` block as a spawn plugin declares it: the gains, shared, plus ``each:``, whose
#: keys are the model's actuator names. The schema check covers its shape at ``roqsim check``'s
#: config stage; the rule between two keys is :func:`unread_gain_errors`, and whatever needs the
#: model -- whether it has those names, what unit they command -- is :func:`resolve`'s, at build.
ACTUATORS = Field(
    dict,
    schema={
        **GAIN_SCHEMA,
        "each": Field(
            dict,
            values=Field(
                dict,
                schema=GAIN_SCHEMA,
                hints=MUJOCO_KEYS,
                doc="this actuator's settings, on top of the shared keys",
            ),
            doc="per actuator, keyed by its name in the model (unprefixed)",
        ),
    },
    hints=MUJOCO_KEYS,
    doc="the control law and gains the model's actuators run under (roqsim.actuators)",
)


@dataclass(frozen=True)
class ResolvedActuator:
    """One actuator's final law and gains, and where each came from.

    This is what the run records. It carries every actuator, not only the overridden ones, because
    "what did this joint run under" is a question about the run and not about the diff -- an answer
    that listed only the changes would need the model to be read to be understood.
    """

    name: str
    joint: str
    control: str
    #: ``model`` -- nothing said otherwise; ``shared`` -- the block's own keys; ``each`` -- a named
    #: entry. Reported per actuator rather than per key: an entry that names one gain still leaves
    #: that actuator described by the block as a whole.
    source: str
    p: float | None = None
    d: float | None = None
    stiffness: float | None = None
    damping: float | None = None
    effort_limit: float | None = None
    ctrlrange: tuple[float, float] | None = None

    def as_record(self) -> dict:
        """Plain data for the run's provenance: the keys that apply, in a stable order."""
        record = {
            "name": self.name,
            "joint": self.joint,
            "control": self.control,
            "source": self.source,
        }
        for key in ("p", "d", "stiffness", "damping", "effort_limit"):
            value = getattr(self, key)
            if value is not None:
                record[key] = float(value)
        if self.ctrlrange is not None:
            record["ctrlrange"] = [float(self.ctrlrange[0]), float(self.ctrlrange[1])]
        return record


def unread_gain_errors(override, *, where: str = "actuators") -> list[str]:
    """Gains an ``actuators:`` block states that the law in force does not read.

    The one rule of the block its declaration (:data:`ACTUATORS`) cannot state, because it relates
    two keys. Called from a spawn plugin's ``validate_config``, beside the schema check, so it lands
    at ``roqsim check``'s config stage too; a shape the schema refuses is left to the schema.
    """
    if not isinstance(override, dict):
        return []
    shared_control = override.get("control")
    errors = _unread_gains(override, shared_control, where)
    each = override.get("each")
    for name, entry in each.items() if isinstance(each, dict) else ():
        if isinstance(entry, dict):
            # An entry inherits the shared law unless it names its own, so that is what its gains
            # are judged against -- both are known here, with no model needed.
            control = entry.get("control") or shared_control
            errors += _unread_gains(entry, control, f"{where}.each.{name}")
    return errors


def _unread_gains(block: dict, control, where: str) -> list[str]:
    """Gains this block states that its law does not read -- accepted, then never applied.

    *control* is the law this block's actuator ends up under -- its own ``control`` if it states
    one, else the shared block's. One the block does not know (unset, or refused by the schema)
    leaves the gains to :func:`resolve`, which knows the model's.

    Only gains stated in *this* block: one inherited from the shared keys is not a mistake but the
    ordinary case of an ``each:`` entry choosing a different law, where the shared gains simply do
    not apply to it.
    """
    if not isinstance(control, str) or control not in _GAINS:
        return []
    return [
        f"'{where}.{gain}' is not a gain of control: {control}, which reads "
        f"{', '.join(_GAINS[control]) or 'no gains'}. Stated here it would never be read."
        for gain in ("p", "d", "stiffness", "damping")
        if gain in block and gain not in _GAINS[control]
    ]


def resolve(spec, override, *, model_name: str, where: str = "actuators") -> list[ResolvedActuator]:
    """Rewrite *spec*'s actuators to match *override*, and return what every actuator ended up as.

    Mutates the child ``MjSpec`` in place and must run **before** the model is attached into the
    world and before anything is grafted onto it -- see the callers. Raises :class:`PluginError`
    once, naming every problem, so a world with several mistakes in this block takes one run to find
    them all rather than one run each.

    An override of ``None`` rewrites nothing and reports the model as it stands, which is what makes
    the recorded table answerable for every world rather than only for one that overrides something.
    """
    actuators = list(spec.actuators)
    by_name = {a.name: a for a in actuators}
    rows = [_model_row(a) for a in actuators]

    if not override:
        return rows

    shared = {k: v for k, v in override.items() if k != "each"}
    each = override.get("each") or {}
    errors: list[str] = []

    unknown = [name for name in each if name not in by_name]
    if unknown:
        joints = {a.target: a.name for a in actuators if _is_joint(a)}
        for name in unknown:
            if name in joints:
                errors.append(
                    f"'{where}.each.{name}' is a joint, not an actuator. The actuator driving it is "
                    f"'{joints[name]}' -- key on that."
                )
            else:
                errors.append(
                    f"'{where}.each.{name}': {model_name} has no actuator of that name. It actuates "
                    f"{', '.join(a.name for a in actuators)}. "
                    f"See `roqsim catalog model {model_name}`."
                )

    resolved: list[ResolvedActuator] = []
    for act, row in zip(actuators, rows, strict=True):
        entry = each.get(act.name)
        if not shared and entry is None:
            resolved.append(row)
            continue
        merged = {**shared, **(entry or {})}
        source = "each" if entry else "shared"
        # A shared law falling on a transmission that is not a joint. Because this runs before an
        # end effector is grafted on, it can only be the model's own (a Franka's tendon-driven
        # fingers), never a gripper someone bolted to an arm -- so the fix really is to name it.
        if not _is_joint(act):
            if entry is None:
                errors.append(
                    f"'{where}' states a joint law, but {model_name}'s '{act.name}' drives a "
                    f"{_TRN_LABEL.get(act.trntype, 'non-joint transmission')}, which has no joint "
                    f"stiffness or position. Name it under '{where}.each' to say what it should do."
                )
            else:
                errors.append(
                    f"'{where}.each.{act.name}' drives a "
                    f"{_TRN_LABEL.get(act.trntype, 'non-joint transmission')}, not a joint."
                )
            resolved.append(row)
            continue
        control = merged.get("control", row.control)
        errors += _model_dependent_errors(shared, entry, merged, control, row, act.name, where)
        resolved.append(_apply(act, merged, control, row, source))

    if errors:
        raise PluginError("; ".join(errors))
    return resolved


def _model_dependent_errors(shared, entry, merged, control, row, name, where) -> list[str]:
    """The two checks the config stage could not make, because both need the model.

    A gain is judged here only when NEITHER the shared keys nor the entry named a law: the law is
    then whatever the model already runs, which is the one thing the config stage cannot know. And
    the command unit is compared against the model's, because that is what decides whether the
    ``ctrlrange`` it shipped still means anything.
    """
    errors: list[str] = []
    if "control" not in merged:
        for block, at in ((shared, where), (entry or {}, f"{where}.each.{name}")):
            for gain in ("p", "d", "stiffness", "damping"):
                if gain in block and gain not in _GAINS[control]:
                    errors.append(
                        f"'{at}.{gain}' would never be read: nothing states a control for "
                        f"'{name}', so it keeps the model's control: {control}, which reads "
                        f"{', '.join(_GAINS[control]) or 'no gains'}. Name the control you mean."
                    )
    was, now = _COMMAND_UNIT[row.control], _COMMAND_UNIT[control]
    if was != now and "ctrlrange" not in merged:
        errors.append(
            f"'{where}.each.{name}' switches control: {row.control} -> {control}, so its command "
            f"changes from {_UNIT_LABEL[was]} to {_UNIT_LABEL[now]} -- but it keeps the ctrlrange "
            f"{_fmt_range(row.ctrlrange)} the model gave the old unit, which would clamp the new "
            f"command to it. Give 'ctrlrange' in {_UNIT_LABEL[now]}."
        )
    return errors


def _apply(act, merged: dict, control: str, row: ResolvedActuator, source: str) -> ResolvedActuator:
    """Write one actuator's law and gains, completely.

    Every parameter the law uses is written, never only the ones that changed: MuJoCo's ``set_to_*``
    helpers leave the previous law's parameters in place (``set_to_motor`` after ``set_to_velocity``
    keeps the old ``biasprm``), so a partial write leaves a stale term acting on the joint.
    """
    gains = {k: merged.get(k, getattr(row, k)) for k in _GAINS[control]}
    gains = {k: (0.0 if v is None else float(v)) for k, v in gains.items()}

    act.gaintype = mujoco.mjtGain.mjGAIN_FIXED
    if control == "effort":
        act.biastype = mujoco.mjtBias.mjBIAS_NONE
        act.gainprm = _prm(1.0)
        act.biasprm = _prm()
    elif control == "velocity":
        act.biastype = mujoco.mjtBias.mjBIAS_AFFINE
        act.gainprm = _prm(gains["d"])
        act.biasprm = _prm(0.0, 0.0, -gains["d"])
    else:  # position, impedance -- one affine joint law, its gains named for the controller
        k = gains["p" if control == "position" else "stiffness"]
        c = gains["d" if control == "position" else "damping"]
        act.biastype = mujoco.mjtBias.mjBIAS_AFFINE
        act.gainprm = _prm(k)
        act.biasprm = _prm(0.0, -k, -c)

    # A limit the override does not state stays exactly as the model has it, flag included: a
    # single-acting drive's one-sided forcerange is the model's to state, not the table's.
    effort_limit = row.effort_limit
    if "effort_limit" in merged:
        effort_limit = float(merged["effort_limit"])
        act.forcerange = _scaled_forcerange(act, effort_limit)
        act.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE
    ctrlrange = row.ctrlrange
    if "ctrlrange" in merged:
        ctrlrange = merged["ctrlrange"]
        act.ctrlrange = [float(ctrlrange[0]), float(ctrlrange[1])]
        act.ctrllimited = mujoco.mjtLimited.mjLIMITED_TRUE

    return replace(
        row,
        control=control,
        source=source,
        p=gains.get("p"),
        d=gains.get("d"),
        stiffness=gains.get("stiffness"),
        damping=gains.get("damping"),
        effort_limit=None if effort_limit is None else float(effort_limit),
        ctrlrange=None if ctrlrange is None else (float(ctrlrange[0]), float(ctrlrange[1])),
    )


#: How a non-joint transmission reads in an error, so the message says what the thing IS.
_TRN_LABEL = {
    mujoco.mjtTrn.mjTRN_TENDON: "tendon",
    mujoco.mjtTrn.mjTRN_SITE: "site",
    mujoco.mjtTrn.mjTRN_SLIDERCRANK: "slider-crank",
    mujoco.mjtTrn.mjTRN_BODY: "body (adhesion)",
}


def _scaled_forcerange(act, effort_limit: float) -> list[float]:
    """The model's ``forcerange`` with its larger bound's magnitude set to *effort_limit*.

    ``effort_limit`` is a magnitude, as URDF's effort is, so it rescales the range and keeps its
    shape: a single-acting ``[0, F]`` becomes ``[0, effort_limit]``, never a drive that also pulls.
    An actuator the model leaves unlimited has no side to keep and gets the symmetric range.
    """
    if _is_limited(act.forcelimited, act.forcerange):
        lo, hi = float(act.forcerange[0]), float(act.forcerange[1])
        magnitude = max(abs(lo), abs(hi))
        if magnitude > 0.0:
            return [effort_limit * (lo / magnitude), effort_limit * (hi / magnitude)]
    return [-effort_limit, effort_limit]


def _is_joint(act) -> bool:
    return act.trntype in (mujoco.mjtTrn.mjTRN_JOINT, mujoco.mjtTrn.mjTRN_JOINTINPARENT)


def _prm(*values: float) -> list[float]:
    """A ``gainprm``/``biasprm`` vector: the values given, zero for the rest of MuJoCo's ten."""
    return list(values) + [0.0] * (10 - len(values))


def _model_row(act) -> ResolvedActuator:
    """What an actuator is before anyone overrides it, read from the spec rather than a compile.

    A spec actuator already carries the values its ``<default class>`` gives it -- they are identical
    to what the compiler will produce -- so the model's own half of the table costs no extra compile.
    """
    control = _model_control(act)
    gain = float(act.gainprm[0])
    damping = -float(act.biasprm[2])
    row = ResolvedActuator(
        name=act.name,
        joint=act.target if _is_joint(act) else "",
        control=control,
        source="model",
        effort_limit=_limit(act.forcerange)
        if _is_limited(act.forcelimited, act.forcerange)
        else None,
        ctrlrange=_range(act.ctrlrange) if _is_limited(act.ctrllimited, act.ctrlrange) else None,
    )
    if control == "position":
        return replace(row, p=gain, d=damping)
    if control == "velocity":
        return replace(row, d=gain)
    return row


def _model_control(act) -> str:
    """Which of the four laws the model already runs this actuator under.

    ``impedance`` is never reported: the two compile to the same affine law and differ only in
    what their gains are called, so a model that declares one is declaring a position servo as far
    as anything readable from the actuator goes.
    """
    if act.biastype == mujoco.mjtBias.mjBIAS_AFFINE:
        if act.biasprm[1] != 0.0:
            return "position"
        if act.biasprm[2] != 0.0:
            return "velocity"
    return "effort"


def _limit(forcerange) -> float | None:
    """The larger bound's magnitude of a ``forcerange``, or ``None`` for MuJoCo's "unlimited" zeros.

    A one-sided range reads as the force it delivers in the direction it can push.
    """
    lo, hi = float(forcerange[0]), float(forcerange[1])
    if lo == 0.0 and hi == 0.0:
        return None
    return max(abs(lo), abs(hi))


def _range(ctrlrange) -> tuple[float, float] | None:
    lo, hi = float(ctrlrange[0]), float(ctrlrange[1])
    return None if lo == 0.0 and hi == 0.0 else (lo, hi)


def _fmt_range(ctrlrange) -> str:
    return "it has" if ctrlrange is None else f"[{ctrlrange[0]:g}, {ctrlrange[1]:g}]"


def uses_impedance(rows: list[ResolvedActuator]) -> bool:
    """Whether any actuator ended up under the one law that needs a body-level term."""
    return any(row.control == "impedance" for row in rows)


#: Laws whose real hardware holds its own weight inside the joint's own servo loop, so a model of
#: one that does not is a model of a different machine. ``effort`` is deliberately absent: a
#: torque-commanded joint applies exactly the torque it is given, and supplying the gravity term is
#: the *controller's* job -- compensating it here would quietly answer the question an experiment on
#: gravity compensation is asking.
_SELF_SUPPORTING = frozenset({"position", "velocity", "impedance"})

#: The subset :func:`apply_gravity_compensation` acts on, per joint. ``velocity`` is out of it
#: because that is how a WHEEL is driven, and a wheel carries the robot rather than being carried
#: by it. No bundled arm ships velocity actuators, so nothing that holds a pose loses by it.
_HELD_BY_A_DRIVE = frozenset({"position", "impedance"})


def servo_holds_against_gravity(rows: list[ResolvedActuator]) -> bool:
    """Whether these actuators model hardware that holds a pose without external help.

    A position or velocity servo commanded to stand still does stand still: the drive's own loop
    supplies whatever torque the load demands, and the joint's gain describes how hard it resists a
    *disturbance*. Modelled without :func:`apply_gravity_compensation` the same gain has to carry
    the mechanism as well, so the servo trades position error for holding torque and the joint
    stands somewhere it was never sent -- which is not a soft arm, it is a different arm.

    **Only half the question, and the other half is not about actuators at all**: see
    :func:`apply_gravity_compensation` before acting on this. A law that holds a pose says nothing
    about whether the drives are what carry the weight, and on a legged or wheeled machine they
    are not.
    """
    return any(row.control in _SELF_SUPPORTING for row in rows)


def apply_gravity_compensation(spec, rows: list[ResolvedActuator] | None = None) -> int:
    """Compensate the weight this mechanism's own drives carry, and report how many bodies.

    What it models: a drive whose own loop carries what hangs off it, so its gain sets how hard
    the joint resists a DISTURBANCE rather than how much weight it can hold -- without this a
    stiffness of 2 N*m/rad does not hold a UR5e up, it folds it, and even a UR5e's shipped
    2000 N*m/rad leaves the flange 9 mm low.

    **Which bodies, and why not all of them.** Given *rows*, a body is compensated when the chain
    from the world down to it passes a joint driven by a ``position`` or ``impedance`` actuator --
    everything, that is, whose weight some drive is holding up. What that leaves out is the load
    path to the ground: a mobile base hangs off nothing and its wheels are driven by ``velocity``,
    so neither is compensated. The arm bolted to that base IS compensated, because its links really
    are held up by its motors.

    ``velocity`` is excluded for that reason and no other: it is how a wheel is driven. No bundled
    arm ships it, so nothing that holds a pose loses anything by its absence here.

    Without *rows* every body is compensated. That is the whole-mechanism form, for a caller that
    asks for it explicitly -- a torque controller handed its own gravity term -- and it is wrong
    for a model that stands on the ground itself, whose base it would compensate too.

    **The drive supplies the term, so the weight stays on the ground.** Each hinge or slide of a
    compensated body that a holding actuator drives through a joint transmission (every such
    actuator, in the whole-mechanism form) gets ``actuatorgravcomp``. MuJoCo 3.14 then adds that
    joint's rows of ``qfrc_gravcomp`` to ``qfrc_actuator`` in ``mj_fwdActuation`` instead of to
    ``qfrc_passive``, *after* each actuator's own ``forcerange`` has clamped ``actuator_force``, and
    clamps the sum by the joint's ``actuatorfrcrange`` -- the only limit that sees the gravity term.
    So a joint with no ``actuatorfrcrange`` of its own gets its drive's: the actuator's
    ``forcerange`` times its gear, summed over the actuators driving it, or none when any of them is
    unlimited. A drive too weak for its load then sags or stalls, as the hardware does. The limit is
    copied when the model is built; an ``actuator_forcerange`` written at run time clamps the servo's
    own share only.

    What that leaves is the part of ``body_gravcomp`` MuJoCo projects onto the degrees of freedom
    ABOVE the mechanism -- a mobile base's free joint, where it lifts the whole robot. The engine
    removes it every step with :class:`GravityReaction`, which is what keeps a mobile manipulator's
    full weight on its wheels. On a fixed base there is nothing above the mechanism and nothing to
    remove.

    Called with the **whole entity's** spec, after anything is grafted onto it and before it is
    attached into the world. That timing is load-bearing and differs from :func:`resolve`'s on
    purpose: ``body_gravcomp`` is per body and does not cascade to children, so an arm compensated
    before its gripper was attached would sag by exactly the tool's weight. Compensating the tool
    is also the right physics -- a real controller is told its payload and holds that too.

    **Where compensation stops, in either form: a free-swinging joint.** A hinge, slide or ball
    joint that nothing acts on -- no actuator, directly or through a tendon, no joint or tendon
    equality, and no ``connect`` or ``weld`` closing a loop through it -- hands nothing below it a
    torque, so no drive holds that part's pose and it hangs under its own weight: a pendulum on the
    flange, a cable, a swinging tool. Those bodies keep their weight, and the arm's drives carry it
    as a load the way they carry any other. A gripper's linkage is not one of them: its joints are
    coupled to its actuator or close a loop with the ones that are.
    """
    drives: dict[str, list] = {}
    for actuator in spec.actuators:
        if _is_joint(actuator):
            drives.setdefault(actuator.target, []).append(actuator)
    if rows is None:
        held = None
        supplied_by_drive = set(drives)
    else:
        held = {row.joint for row in rows if row.joint and row.control in _HELD_BY_A_DRIVE}
        supplied_by_drive = held
    acted_on, looped_bodies = _joints_acted_on(spec)

    compensated = 0

    def _subtree_names(body) -> set[str]:
        names = {body.name}
        for child in body.bodies:
            names |= _subtree_names(child)
        return names

    def _swings_freely(body) -> bool:
        passive = [
            j
            for j in body.joints
            if j.type != mujoco.mjtJoint.mjJNT_FREE and j.name not in acted_on
        ]
        return bool(passive) and not (_subtree_names(body) & looped_bodies)

    def _walk(body, carried: bool, swinging: bool) -> None:
        nonlocal compensated
        for child in body.bodies:
            # Everything below a free-swinging joint swings with it, a drive further down included:
            # that drive holds its links against the swinging part, not against gravity.
            child_swinging = swinging or _swings_freely(child)
            below = not child_swinging and (
                carried or held is None or any(j.name in held for j in child.joints)
            )
            if below:
                child.gravcomp = 1.0
                compensated += 1
                for joint in child.joints:
                    if joint.name in supplied_by_drive and joint.type in _SCALAR_JOINTS:
                        _drive_supplies_gravity(joint, drives[joint.name])
            _walk(child, below, child_swinging)

    _walk(spec.worldbody, False, False)
    return compensated


#: The joints ``actuatorfrcrange`` bounds: MuJoCo clamps one entry of ``qfrc_actuator`` per joint.
_SCALAR_JOINTS = (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE)


def _is_limited(limited, value_range) -> bool:
    """MuJoCo's reading of a ``*limited`` flag: ``auto`` means limited exactly when a range is set."""
    if limited == mujoco.mjtLimited.mjLIMITED_TRUE:
        return True
    if limited == mujoco.mjtLimited.mjLIMITED_FALSE:
        return False
    return not (float(value_range[0]) == 0.0 and float(value_range[1]) == 0.0)


def _drive_supplies_gravity(joint, actuators) -> None:
    """Route *joint*'s gravity term through its actuators, bounded by what they can deliver."""
    joint.actgravcomp = True
    if _is_limited(joint.actfrclimited, joint.actfrcrange):
        return  # The model states the joint's limit itself, and it is the one MuJoCo applies.
    lo = hi = 0.0
    for actuator in actuators:
        if not _is_limited(actuator.forcelimited, actuator.forcerange):
            return  # One unlimited drive on the joint leaves the joint unlimited.
        gear = float(actuator.gear[0])
        ends = sorted((gear * float(actuator.forcerange[0]), gear * float(actuator.forcerange[1])))
        lo, hi = lo + ends[0], hi + ends[1]
    if lo < hi:
        joint.actfrclimited = mujoco.mjtLimited.mjLIMITED_TRUE
        joint.actfrcrange = [lo, hi]


def joint_effort(model, data, dof: int) -> float:
    """The generalised force a joint's drives deliver at *dof*: what its torque sensor reads.

    ``qfrc_actuator`` already holds the gravity term of a joint whose drive supplies it
    (``actuatorgravcomp``, set by :func:`apply_gravity_compensation`). Anywhere else a nonzero
    ``qfrc_gravcomp`` row is the passive route -- a model that declares ``gravcomp`` itself, a
    gripper finger held through a tendon -- and the joint carries it all the same, so it is added.
    """
    force = float(data.qfrc_actuator[dof])
    if not model.jnt_actgravcomp[model.dof_jntid[dof]]:
        force += float(data.qfrc_gravcomp[dof])
    return force


class GravityReaction:
    """Hands the weight a drive holds back to the body the drive is mounted on, every step.

    MuJoCo implements ``body_gravcomp`` as an external force, ``-m*g`` at each compensated body's
    centre of mass, and projects it onto every degree of freedom of the chain above that body. The
    rows of a holding joint are the torque its drive supplies (:func:`apply_gravity_compensation`
    routes them through the actuator). The rows above the mechanism are not a torque anything
    supplies: on a robot standing on a free joint they are a skyhook that carries the arm, the mast
    or the steered wheel, and the floor carries the rest -- wheel loads, traction and tipping all
    come out wrong. MuJoCo has no per-DOF switch for them.

    **What this removes, exactly.** A *mechanism* is a connected set of compensated bodies whose top
    body hangs from an uncompensated body that can move -- the *root*, a mobile base. For each, the
    compensated weight ``W`` at its combined centre of mass ``c`` is applied to the root with
    :func:`mujoco.mj_applyFT` and subtracted from ``qfrc_gravcomp`` and from ``qfrc_passive``. The
    root's point Jacobian at ``c`` equals any member's on every DOF above the root, so that removes
    precisely those rows and leaves every row inside the mechanism as MuJoCo computed it. It is the
    reaction a real drive puts into its mount.

    **When it runs.** Between ``mj_step1`` and ``mj_step2``: ``mj_passive`` has computed
    ``qfrc_gravcomp`` at the state being stepped, and ``mj_fwdActuation``, which reads it for
    ``actuatorgravcomp`` joints, has not run yet. Nothing global is touched -- ``mjcb_passive`` would
    do the same from inside ``mj_passive`` but is one slot per process, called for every model any
    thread steps -- and ``qfrc_applied`` is left to whoever uses it for perturbations.
    ``mj_step2`` integrates RK4 as Euler, so the engine refuses a world that needs this under
    ``rk4``. An ``mj_forward`` outside a step (a reset, a plugin's own) leaves the rows in its
    accelerations; the next step computes without them.

    Per step it reads model fields, so an entity made absent (:mod:`roqsim.presence` compensates
    its root too, to freeze it in place) keeps its full compensation while absent and gets the
    correction back on return, and nothing needs resetting between trials.
    """

    def __init__(self, model, mechanisms: list[tuple[int, np.ndarray]]):
        self._mechanisms = mechanisms
        #: The top body of each robot carrying such a mechanism, for a refusal to name the robot by.
        self.robots = sorted(
            {
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.body_rootid[root]))
                for root, _members in mechanisms
            }
        )
        above: set[int] = set()
        for root, _members in mechanisms:
            body = root
            while body > 0:
                adr, num = int(model.body_dofadr[body]), int(model.body_dofnum[body])
                above.update(range(adr, adr + num))
                body = int(model.body_parentid[body])
        self._dofs = np.array(sorted(above), dtype=int)
        via_actuator = model.jnt_actgravcomp[model.dof_jntid[self._dofs]].astype(bool)
        self._passive_dofs = self._dofs[~via_actuator]
        self._scratch = np.zeros(model.nv)
        self._torque = np.zeros(3)

    @classmethod
    def of(cls, model) -> GravityReaction | None:
        """The reaction *model* needs, or ``None`` when no drive-held mechanism hangs off a moving
        body -- every fixed-base arm, every robot without a holding drive.

        A mechanism counts when one of its joints routes its gravity term through a drive
        (``actuatorgravcomp``): ``gravcomp`` a model declares on its own, such as a buoyant body,
        is an external force by intent and stays one.
        """
        compensated = np.asarray(model.body_gravcomp) > 0.0
        parent = model.body_parentid
        supplied = {int(model.jnt_bodyid[j]) for j in range(model.njnt) if model.jnt_actgravcomp[j]}
        mechanisms = []
        for top in range(1, model.nbody):
            root = int(parent[top])
            if not compensated[top] or compensated[root] or int(model.body_weldid[root]) == 0:
                continue
            members = [top]
            inside = {top}
            # Bodies are numbered parents first, so one forward pass collects the connected set.
            for body in range(top + 1, model.nbody):
                if compensated[body] and int(parent[body]) in inside:
                    members.append(body)
                    inside.add(body)
            if inside & supplied:
                mechanisms.append((root, np.array(members, dtype=int)))
        if not mechanisms:
            return None
        reaction = cls(model, mechanisms)
        if model.opt.enableflags & mujoco.mjtEnableBit.mjENBL_SLEEP:
            raise PluginError(
                f'the world MJCF enables sleep (<option><flag sleep="enable"/>), but drives on '
                f"the moving robot(s) rooted at {reaction.robots} hold a mechanism up, whose weight "
                "is handed back to the base every step from qfrc_gravcomp, which MuJoCo leaves stale "
                "on a sleeping tree -- remove the flag, or set gravity_compensation: false on that "
                "robot's spawn"
            )
        return reaction

    def apply(self, model, data) -> None:
        """Remove the mechanisms' weight from the DOFs above them. Physics thread, after
        ``mj_step1`` and before ``mj_step2``."""
        if not _gravcomp_live(model):
            return
        scratch = self._scratch
        scratch[:] = 0.0
        applied = False
        for root, members in self._mechanisms:
            if model.body_gravcomp[root] != 0.0:
                # The root floats too (an absent entity, frozen by presence): nothing stands on
                # anything, so the whole entity keeps MuJoCo's external compensation.
                continue
            weights = model.body_mass[members] * model.body_gravcomp[members]
            total = float(weights.sum())
            if total <= 0.0:
                continue
            point = weights @ data.xipos[members] / total
            force = -model.opt.gravity * total
            mujoco.mj_applyFT(model, data, force, self._torque, point, root, scratch)
            applied = True
        if not applied:
            return
        data.qfrc_gravcomp[self._dofs] -= scratch[self._dofs]
        data.qfrc_passive[self._passive_dofs] -= scratch[self._passive_dofs]


def _gravcomp_live(model) -> bool:
    """Whether ``mj_passive`` computed ``qfrc_gravcomp`` at all, under the same gates it uses."""
    disabled = int(model.opt.disableflags)
    flags = mujoco.mjtDisableBit
    if disabled & flags.mjDSBL_GRAVITY or not np.any(model.opt.gravity):
        return False
    return not (disabled & flags.mjDSBL_SPRING and disabled & flags.mjDSBL_DAMPER)


def _joints_acted_on(spec) -> tuple[set[str], set[str]]:
    """The joints something applies a force through, and the bodies a loop constraint names.

    A joint is acted on when an actuator drives it or a tendon that includes it, or an equality
    couples it or such a tendon. The bodies of ``connect`` and ``weld`` equalities are returned
    separately: every joint on the chain above one of them is part of a closed loop.
    """
    tendon_joints: dict[str, set[str]] = {}
    for tendon in spec.tendons:
        tendon_joints[tendon.name] = {
            wrap.target.name
            for wrap in tendon.path
            if wrap.type == mujoco.mjtWrap.mjWRAP_JOINT and wrap.target is not None
        }

    acted_on: set[str] = set()
    for actuator in spec.actuators:
        if actuator.trntype in (mujoco.mjtTrn.mjTRN_JOINT, mujoco.mjtTrn.mjTRN_JOINTINPARENT):
            acted_on.add(actuator.target)
        elif actuator.trntype == mujoco.mjtTrn.mjTRN_TENDON:
            acted_on |= tendon_joints.get(actuator.target, set())

    looped_bodies: set[str] = set()
    for equality in spec.equalities:
        if equality.type == mujoco.mjtEq.mjEQ_JOINT:
            acted_on |= {equality.name1, equality.name2}
        elif equality.type == mujoco.mjtEq.mjEQ_TENDON:
            for name in (equality.name1, equality.name2):
                acted_on |= tendon_joints.get(name, set())
        elif equality.type in (mujoco.mjtEq.mjEQ_CONNECT, mujoco.mjtEq.mjEQ_WELD):
            looped_bodies |= {equality.name1, equality.name2}
    acted_on.discard("")
    looped_bodies.discard("")
    return acted_on, looped_bodies
