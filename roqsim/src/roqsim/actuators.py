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
    impedance    joint position  stiffness, damping    affine, plus body_gravcomp on the subtree

``impedance`` is not a second spelling of ``position``. A real joint-impedance controller (Franka's
``joint_impedance``, a UR in force mode) compensates the arm's own weight, which is what lets a
stiffness of 2 N*m/rad hold a pose at all rather than folding under gravity. That gravity term is the
difference, and it is why the mode earns its own name -- in a world at zero gravity the two do
coincide, and a reader should not conclude the mode did nothing.

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

from .plugin import PluginError
from .schema import Field, validate

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

#: MuJoCo's own spellings, and what replaces each. Refused rather than translated, and rather than
#: ignored: these plugins take config maps they do not fully own, so a key merely not read would be
#: accepted in silence and the world would run under gains nobody chose.
_GONE = {
    "kp": "'p' (control: position) or 'stiffness' (control: impedance)",
    "kv": "'d' (control: position/velocity) or 'damping' (control: impedance)",
    "kd": "'d' (control: position) or 'damping' (control: impedance)",
    "forcerange": "'effort_limit', a single positive magnitude",
    "gainprm": "'p'/'stiffness' -- state the gain, not MuJoCo's parameter vector",
    "biasprm": "'d'/'damping' -- state the gain, not MuJoCo's parameter vector",
    "mode": "'control'",
}

#: Values that were MuJoCo's actuator types rather than a robot's command interface.
_GONE_CONTROLS = {
    "motor": "effort",
    "pd": "impedance",
    "general": "position, velocity, effort or impedance",
}

#: One gain block: the shared keys, and the same set again inside every ``each:`` entry. Units are
#: declared because a paper states "kp = 2.0" with none, and a value wrong by a factor of a thousand
#: is indistinguishable from a right one -- which is the failure this whole module exists to prevent.
#: The rotational unit is named; a slide joint's is the linear equivalent (N/m, N*s/m, N).
GAIN_SCHEMA: dict[str, Field] = {
    "control": Field(
        str,
        choices=CONTROLS,
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
        float, minimum=0.0, unit="N*m", doc="torque magnitude cap; URDF's <limit effort=>"
    ),
    "ctrlrange": Field(
        list,
        length=2,
        doc="command limits, in the unit `control` commands; needed on a unit change",
    ),
}

#: The block's own keys: the gains, shared, plus the one that nests.
_BLOCK_KEYS = frozenset(GAIN_SCHEMA) | {"each"}


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


def validate_override(override, *, where: str = "actuators") -> list[str]:
    """Shape errors for an ``actuators:`` block, without needing the model.

    Called from a spawn plugin's ``validate_config``, so these land at ``roqsim check``'s **config**
    stage -- before anything is compiled, which is where a key typo belongs. What cannot be checked
    here is everything about the names under ``each:``: whether the model has them, and what unit its
    actuators command. Those need the MJCF and are checked in :func:`resolve`, at the build stage.

    Errors accumulate rather than raising at the first, the way :func:`roqsim.schema.validate` does:
    a world with three mistakes should take one run to find them.
    """
    if override is None:
        return []
    if not isinstance(override, dict):
        return [f"'{where}' must be a mapping of shared settings plus an optional 'each'"]

    errors: list[str] = []
    shared = {k: v for k, v in override.items() if k != "each"}
    shared_control = shared.get("control")
    errors += _gain_errors(shared, where, shared_control)

    each = override.get("each")
    if each is not None and not isinstance(each, dict):
        errors.append(f"'{where}.each' must be a mapping of actuator name to its settings")
    elif each:
        for name, entry in each.items():
            if not isinstance(entry, dict):
                errors.append(f"'{where}.each.{name}' must be a mapping of settings")
                continue
            # An entry inherits the shared law unless it names its own, so that is what its gains
            # are judged against -- both are known here, with no model needed.
            errors += _gain_errors(
                entry, f"{where}.each.{name}", entry.get("control") or shared_control
            )
    return errors


def _gain_errors(block: dict, where: str, control: str | None) -> list[str]:
    """One gain block: the gone spellings, the schema, and the gains against the law in force.

    *control* is the law this block's actuator ends up under -- its own ``control`` if it states one,
    else the shared block's. ``None`` means neither said, so the law is whatever the model already
    runs and only :func:`resolve` can judge the gains.
    """
    errors: list[str] = []
    for key, replacement in _GONE.items():
        if key in block:
            errors.append(
                f"'{where}.{key}' is gone -- use {replacement}. '{where.split('.')[0]}' states a "
                f"joint's control law the way a robot's driver does, not the way MuJoCo stores it."
            )
    if isinstance(block.get("control"), str) and block["control"] in _GONE_CONTROLS:
        errors.append(
            f"'{where}.control: {block['control']}' is gone -- "
            f"use {_GONE_CONTROLS[block['control']]}. "
            f"These name a command interface, not a MuJoCo actuator type."
        )
        control = None
    known = {k: v for k, v in block.items() if k not in _GONE}
    errors += validate(GAIN_SCHEMA, known, strict_keys=True)
    errors += _unread_gains(block, control, where)
    return errors


def _unread_gains(block: dict, control: str | None, where: str) -> list[str]:
    """Gains this block states that its law does not read -- accepted, then never applied.

    Only gains stated in *this* block: one inherited from the shared keys is not a mistake but the
    ordinary case of an ``each:`` entry choosing a different law, where the shared gains simply do
    not apply to it.
    """
    if control not in _GAINS:
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
    """The two checks :func:`validate_override` could not make, because both need the model.

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
    else:  # position, impedance -- the same affine joint law, differing in gravity compensation
        k = gains["p" if control == "position" else "stiffness"]
        c = gains["d" if control == "position" else "damping"]
        act.biastype = mujoco.mjtBias.mjBIAS_AFFINE
        act.gainprm = _prm(k)
        act.biasprm = _prm(0.0, -k, -c)

    effort_limit = merged.get("effort_limit", row.effort_limit)
    if effort_limit is not None:
        act.forcerange = [-float(effort_limit), float(effort_limit)]
        act.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE
    ctrlrange = merged.get("ctrlrange", row.ctrlrange)
    if ctrlrange is not None:
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
        effort_limit=_limit(act.forcerange),
        ctrlrange=_range(act.ctrlrange),
    )
    if control == "position":
        return replace(row, p=gain, d=damping)
    if control == "velocity":
        return replace(row, d=gain)
    return row


def _model_control(act) -> str:
    """Which of the four laws the model already runs this actuator under.

    ``impedance`` is never reported: gravity compensation is a property of a body, not of an
    actuator, so a model cannot declare it here and a position servo is what this honestly is.
    """
    if act.biastype == mujoco.mjtBias.mjBIAS_AFFINE:
        if act.biasprm[1] != 0.0:
            return "position"
        if act.biasprm[2] != 0.0:
            return "velocity"
    return "effort"


def _limit(forcerange) -> float | None:
    """The magnitude of a symmetric ``forcerange``, or ``None`` for MuJoCo's "unlimited" zeros."""
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


def apply_gravity_compensation(spec) -> int:
    """Compensate the weight of every body in *spec*, and report how many.

    The half of ``control: impedance`` that is not an actuator parameter. A real joint-impedance
    controller holds a pose against gravity so that its stiffness sets how hard the joint resists a
    DISTURBANCE, not how much of the arm's own weight it can carry -- without this a stiffness of
    2 N*m/rad does not hold a UR5e up, it folds it.

    Called with the **whole entity's** spec, after anything is grafted onto the model and before it
    is attached into the world. That timing is load-bearing and differs from :func:`resolve`'s on
    purpose: ``body_gravcomp`` is per body and does not cascade to children, so an arm compensated
    before its gripper was attached would sag by exactly the tool's weight. Compensating the tool is
    also the right physics -- a real controller is told its payload and holds that too.
    """
    compensated = 0
    for body in spec.bodies:
        if body.name == "world":
            continue
        body.gravcomp = 1.0
        compensated += 1
    return compensated
