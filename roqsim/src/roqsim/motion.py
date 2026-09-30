"""When does a recording start moving?

A run recorded from a live stack opens with the robot standing still: nodes come up, a map is
received, a plan is computed, and only then does anything move. That dead air is most of what a
person watching a replay sits through, so ``roqsim render --from onset`` clips it off::

    from roqsim.motion import motion_onset
    from roqsim.recording import open_recording

    rec = open_recording("run.mcap")
    onset = motion_onset(rec)
    if onset.moved:
        print(onset.time)          # sim seconds -- where a clip should start

**The signal is chosen by what the robot is, not by a tuned threshold**, because of one failure mode:
a robot is spawned a little above the floor and falls onto it. That fall is real motion and it happens
at t=0, so any detector reading "the robot's velocity" starts every clip at the beginning. Measured
across recorded runs, the drop reaches 0.8 m/s in some worlds and 0.02 m/s in others -- a factor of 40
-- so no threshold separates a fall from a drive reliably.

What does separate them is *direction*. A robot that falls moves along ``z`` and rocks about ``x``/``y``;
a robot that drives moves along ``x``/``y`` and turns about ``z``. So for a mobile base the signal is
planar motion alone, and the fall is not in it at all. A fixed base cannot fall, so its signal is the
velocity of its actuated joints.

Which of the two a recording gets is decided from the model rather than from names:

    a robot is *mobile* iff the kinematic root of its **actuated** joints carries a free joint.

Reaching the base through the actuators is what makes a scene with props work. A conveyor's parcel is
a free body too, but it is the root of no actuated joint, so it is not mistaken for a base -- and a
fixed arm beside it is correctly read as fixed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from .kinematics import joint_dof_indices

#: Speed below which a channel is treated as noise rather than motion, whatever the run's own peak.
#: Per channel because they are not in the same unit -- ``lin`` is m/s, ``yaw`` and ``joint`` rad/s --
#: which is also why the channels are thresholded separately instead of through one ``max()``.
FLOORS = {"lin": 0.02, "yaw": 0.05, "joint": 1e-3}

#: Fraction of a channel's own peak that counts as moving, added to the floor. Relative because the
#: peaks span two orders of magnitude across platforms -- half a radian per second for an arm, metres
#: per second for a base -- so one absolute number cannot serve both.
FRACTION = 0.02

#: Seconds a channel must stay over its threshold before the crossing counts, which is what keeps a
#: single-sample spike (a contact tick, a teleport) from being read as the run starting.
HOLD_S = 0.2

#: Seconds of stillness kept before the crossing, so a clip opens just before the motion rather than
#: exactly on it.
PRE_ROLL_S = 0.5


class MotionError(RuntimeError):
    """The onset of motion cannot be determined for this recording (see the message)."""


@dataclass(frozen=True)
class Onset:
    """When a recording's motion starts, and the evidence for it.

    ``moved`` is the field to branch on. It is *"a sustained crossing was found"*, **not** "the peak
    was above the floor": a run can carry a single sample of motion -- a robot nudged by a teleport,
    a contact tick -- and still be a run in which nothing ever drove. Reading the peak alone calls
    that one moved and produces a clip of a stationary robot.
    """

    moved: bool
    #: Where a clip should start: the crossing, less ``pre_roll``, clamped to the recording's span.
    time: float
    #: The crossing itself, before ``pre_roll`` is taken off. ``None`` when nothing moved.
    detected: float | None
    #: Index of the sample at ``detected``. ``None`` when nothing moved.
    index: int | None
    #: ``"mobile"`` or ``"fixed"`` -- which signal was used, and why.
    kind: str
    #: How the signal was chosen, as passed in (``"auto"`` resolves to what it picked).
    select: str
    #: The channel that was first over its threshold at the crossing. ``None`` when nothing moved.
    channel: str | None
    #: Peak of each channel over the whole recording.
    peaks: dict = field(default_factory=dict)
    #: The threshold each channel had to cross.
    thresholds: dict = field(default_factory=dict)
    #: Joints the signal was read from, empty for a planar signal (which reads the base's free joint).
    joints: tuple = ()
    #: The robot's own root body -- what to frame a picture of this run on. ``None`` for ``"any"``,
    #: which never identifies a robot.
    body: str | None = None

    def as_record(self) -> dict:
        """The flat dict a CLI prints, with numbers rounded to what a sim clock can mean."""
        return {
            "moved": self.moved,
            "time": round(self.time, 6),
            "detected": None if self.detected is None else round(self.detected, 6),
            "index": self.index,
            "kind": self.kind,
            "select": self.select,
            "channel": self.channel,
            "peaks": {k: round(float(v), 6) for k, v in self.peaks.items()},
            "thresholds": {k: round(float(v), 6) for k, v in self.thresholds.items()},
            "joints": list(self.joints),
            "body": self.body,
        }


def actuated_joints(rec) -> tuple[str, ...]:
    """The joints this recording's actuators drive, from its provenance alone.

    ``meta["actuators"]`` is written per component as the run resolved them, so this is the robot's
    own joints and not the scenery's. An actuator that drives something other than a joint (a tendon,
    a site) contributes nothing and is skipped.
    """
    names = []
    for rows in (rec.meta.get("actuators") or {}).values():
        for row in rows or ():
            if joint := (row or {}).get("joint"):
                names.append(str(joint))
    return tuple(dict.fromkeys(names))


def robot_roots(model, joints) -> set:
    """The kinematic root bodies of ``joints`` -- the robot those joints belong to.

    Walking *out* from the driven joints is what identifies the robot among everything else in a
    world: a parcel on a conveyor, a ball, any prop dropped in is the root of no actuated joint.
    """
    roots = set()
    for name in joints:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, str(name))
        if jid >= 0:
            roots.add(int(model.body_rootid[int(model.jnt_bodyid[jid])]))
    return roots


def base_free_joint(model, joints) -> int | None:
    """The free joint at the kinematic root of ``joints``, or ``None`` if that root is fixed.

    This is the mobile/fixed test, and :func:`robot_roots` is what keeps it from answering about a
    prop instead of about the robot.
    """
    roots = robot_roots(model, joints)
    for jid in range(model.njnt):
        if (
            int(model.jnt_bodyid[jid]) in roots
            and model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE
        ):
            return jid
    return None


def motion_onset(
    rec,
    *,
    select: str = "auto",
    model=None,
    floors: dict | None = None,
    fraction: float = FRACTION,
    hold: float = HOLD_S,
    pre_roll: float = PRE_ROLL_S,
) -> Onset:
    """When ``rec`` first moves, by the rule this module's docstring states.

    ``select`` is ``"auto"`` (decide from the model), ``"planar"``, ``"actuated"``, ``"any"``, or a
    sequence of joint names. Every one of them except ``"any"`` needs the compiled model: pass
    ``model`` when one has already been built -- a renderer has -- and it is built here otherwise.

    ``"any"`` reads every velocity in the state and needs no model at all, which is what lets it answer
    for a recording whose world can no longer be rebuilt. It is a fallback and it *does* see the spawn
    drop, so :attr:`Onset.kind` reports ``"any"`` to say the answer is the weaker one.
    """
    floors = {**FLOORS, **(floors or {})}
    times = np.asarray(rec.times, dtype=float)
    qvel = np.asarray(rec.qvel, dtype=float)
    if len(times) == 0:
        raise MotionError(f"{rec.path} holds no samples")

    signals, kind, joints, body = _signals(rec, qvel, select, model)

    peaks = {name: float(np.abs(sig).max()) for name, sig in signals.items()}
    thresholds = {
        name: max(float(floors.get(name, FLOORS["joint"])), fraction * peaks[name])
        for name in signals
    }

    over = {name: np.abs(sig) > thresholds[name] for name, sig in signals.items()}
    ok = np.logical_or.reduce(list(over.values()))
    window = max(1, min(int(round(hold * float(rec.fps))), len(ok)))
    sustained = np.convolve(ok.astype(np.int64), np.ones(window, dtype=np.int64), "valid") == window

    if not sustained.any():
        return Onset(
            moved=False,
            time=float(times[0]),
            detected=None,
            index=None,
            kind=kind,
            select=_select_name(select),
            channel=None,
            peaks=peaks,
            thresholds=thresholds,
            joints=joints,
            body=body,
        )

    index = int(np.argmax(sustained))
    detected = float(times[index])
    channel = next((name for name in signals if over[name][index]), None)
    return Onset(
        moved=True,
        time=max(float(times[0]), detected - float(pre_roll)),
        detected=detected,
        index=index,
        kind=kind,
        select=_select_name(select),
        channel=channel,
        peaks=peaks,
        thresholds=thresholds,
        joints=joints,
        body=body,
    )


def _select_name(select) -> str:
    return select if isinstance(select, str) else "joints"


def _signals(rec, qvel: np.ndarray, select, model) -> tuple[dict, str, tuple, str | None]:
    """The channels to threshold, the kind of robot, the joints read, and the robot's root body."""
    if select == "any":
        return {"joint": np.abs(qvel).max(axis=1)}, "any", (), None

    if model is None:
        model, _ = rec.build()

    if isinstance(select, str) and select not in ("auto", "planar", "actuated"):
        raise MotionError(
            f"select={select!r} is not one of 'auto', 'planar', 'actuated', 'any', or a list of "
            "joint names."
        )

    named = None if isinstance(select, str) else tuple(str(n) for n in select)
    driven = named if named is not None else actuated_joints(rec)

    free = None
    if select in ("auto", "planar"):
        free = base_free_joint(model, driven)
        if free is None and select == "planar":
            raise MotionError(
                "select='planar' needs a mobile base, but the root of this recording's actuated "
                f"joints ({', '.join(driven) or 'none'}) carries no free joint. Use "
                "select='actuated' for a fixed base."
            )

    body = _root_name(model, driven)
    if free is not None:
        adr = int(model.jnt_dofadr[free])
        return (
            {
                "lin": np.linalg.norm(qvel[:, adr : adr + 2], axis=1),
                "yaw": np.abs(qvel[:, adr + 5]),
            },
            "mobile",
            (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, free),),
            body,
        )

    dofs = joint_dof_indices(model, driven)
    if not dofs:
        raise MotionError(
            "no joint to read motion from: this recording's provenance names no actuated joint the "
            "rebuilt model has. Use select='any' to read every velocity instead."
        )
    return {"joint": np.abs(qvel[:, dofs]).max(axis=1)}, "fixed", tuple(driven), body


def _root_name(model, driven) -> str | None:
    """The name of the robot's root body, for a caller that wants to frame a picture on it."""
    roots = robot_roots(model, driven)
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, r) for r in sorted(roots)]
    return next((n for n in names if n), None)
