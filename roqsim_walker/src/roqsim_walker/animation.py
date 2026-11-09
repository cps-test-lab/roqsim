"""A walker's body: the motion-clip blendspace that poses its mocap skeleton as it moves.

Each walker is a kinematic articulated humanoid (a flat set of mocap bodies, see
:mod:`roqsim_walker.humanoid`). Where it goes is the navigator's (``roqsim_nav``), which moves the
nav root through the ``walker`` output (:mod:`roqsim_walker.output`). This module makes the body
follow: a motion :class:`~roqsim_walker.motion.Clip` blendspace (idle/short/walk/run on a speed
axis; turn + strafe on a direction axis), phased by *distance travelled* so feet don't slide, then
written to every mocap body via forward kinematics.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import mujoco
import numpy as np

from roqsim.pose import yaw_to_quat
from roqsim_walker.humanoid import (
    JOINT_NAMES,
    forward_kinematics,
    quat_rotate,
    to_skeleton,
)
from roqsim_walker.motion import Clip, ClipError, blend_quats, smoothstep
from roqsim_walker.reach import MAX_BEND, SIDES, needed_bend, reach_pose

logger = logging.getLogger(__name__)

# Animation smoothing (CARLA-style): ease the body, never snap it.
_SPEED_TAU = 0.15  # s, low-pass on the speed that drives the gait/blend
_MAX_TURN_RATE = 4.0  # rad/s cap on how fast the body re-faces its heading
# Speed-axis blend windows (m/s) for CARLA's idle->short->walk->run order.
_IDLE_SHORT = (0.05, 0.25)  # idle -> short shuffle
_SHORT_WALK = (0.35, 0.90)  # short -> walk
_WALK_RUN = (1.70, 2.80)  # walk -> run
# Direction axis -- two parts of CARLA's BS_GEN3 ``Direction`` parameter:
#  (1) turn-in-place / lean: heading-rate (rad/s) for a full turn blend; when standing the in-place
#      turn clip is played through and fully takes over, when walking it is capped (``_TURN_MAX``)
#      so it only flavours (leans) the gait.
#  (2) strafe: when travel is not aligned with facing, blend the matching side / back walk clip by
#      the travel-vs-facing angle (optional clips; inert if absent).
_TURN_REF = 2.5
_TURN_MAX = 0.45
_TURN_STRIDE = math.radians(60.0)  # body heading change per turn-clip cycle (sets step cadence)
# Strafe blend windows on |travel - facing| (rad).
_STRAFE_FWD = math.radians(25.0)
_STRAFE_SIDE = math.radians(90.0)
_STRAFE_BACK = math.radians(150.0)
# A hand's reach (the hand on a point, :mod:`roqsim_walker.reach`) eases in and out at most
# _REACH_RATE (1/s), and the body's bend follows what the points need at most _BEND_RATE (rad/s);
# each speeds up and slows down at its _ACCEL, so neither jerks the body.
# With a body lag the foot grounding's shift eases over this (s) (:func:`_foot_ground`).
_GROUND_TAU = 0.03
_REACH_RATE = 2.5
_REACH_ACCEL = 8.0
_BEND_RATE = 1.2
_BEND_ACCEL = 4.0
# The joints a reach bends the trunk with, and each arm's (suffixed by its side).
_TRUNK = ("pelvis", "spine", "hip_l", "hip_r")
_ARM = ("shoulder", "elbow", "wrist")


def _rest_foot_z(skel):
    """Lower-ankle height at this walker's rest pose -- the target a grounded clip's lowest foot
    should reach so the planted sole touches the floor."""
    ident = {n: np.array([1.0, 0.0, 0.0, 0.0]) for n in JOINT_NAMES}
    poses = forward_kinematics([0.0, 0.0, skel.root_height], 0.0, ident, skeleton=skel)
    return min(poses["ankle_l"][0][2], poses["ankle_r"][0][2])


def _ground_clip(clip, skel):
    """Lower a clip's ``root_z`` so its lowest foot over the whole cycle just reaches the rest stance
    height -- the planted foot touches the floor instead of hovering. Some retargeted clips (notably
    run) never fully plant, leaving the character floating several cm."""
    rest = _rest_foot_z(skel)
    lows = [
        min(
            forward_kinematics(
                [0.0, 0.0, skel.root_height + clip.root_z[i]],
                0.0,
                {n: clip.joint_rot[i, j] for j, n in enumerate(JOINT_NAMES)},
                skeleton=skel,
            )[a][0][2]
            for a in ("ankle_l", "ankle_r")
        )
        for i in range(clip.num_frames)
    ]
    clip.root_z = clip.root_z - (float(min(lows)) - rest)
    return clip


def _foot_rest(skel):
    """Rest-pose heel(ankle) and toe-tip heights per foot -- the z's at which that foot's sole
    touches the floor, used to ground each frame to the real contact."""
    ident = {n: np.array([1.0, 0.0, 0.0, 0.0]) for n in JOINT_NAMES}
    poses = forward_kinematics([0.0, 0.0, skel.root_height], 0.0, ident, skeleton=skel)
    tip = np.array(skel.foot_tip)
    rest = {}
    for s in ("l", "r"):
        tp, tq = poses[f"toe_{s}"]
        rest[s] = (float(poses[f"ankle_{s}"][0][2]), float((tp + quat_rotate(tq, tip))[2]))
    return rest


def _foot_ground(poses, st, dt=0.0):
    """Vertical-only foot grounding: shift the whole body so the **lowest shoe-sole point** of either
    foot (:func:`_lowest_sole`) sits exactly on the floor. Tracking the real sole (not the ankle/toe
    joint, which is several cm above it) removes the residual hover.

    With ``st.body_lag`` set the shift follows that as a spring of ``_GROUND_TAU`` over ``dt``: the
    lowest point changes from one foot to the other as a toe lifts, and the body would drop the
    difference in one tick."""
    c = _lowest_sole(poses, st)
    if st.body_lag > 0.0 and dt > 0.0 and st.ground is not None:
        st.ground, st.ground_vel = _ease(st.ground, st.ground_vel, c, _GROUND_TAU, dt)
        c = st.ground
    else:
        st.ground, st.ground_vel = c, 0.0
    if abs(c) > 1e-6:
        for nm, (p, q) in poses.items():
            poses[nm] = (p - np.array([0.0, 0.0, c]), q)
    return poses


def _sole_hull(spec_sole, skel):
    """Per foot, a shoe's hull points ready to skin: ``(joints, weights (K, J), offsets (J, K, 3))``.

    ``spec_sole[s]["hull"]`` (written by the blueprint tools) gives each point's bind position in
    the walker's frame (``at``) and its skin weights over that leg's ``joints``. The skin binds the
    legs at the rest pose (a T-pose only turns the shoulders), so each point's offset from each of
    its joints is taken at the rest pose of this walker's skeleton, in that joint's frame. ``None``
    when the blueprint has no hull."""
    if not spec_sole or not all("hull" in (spec_sole.get(s) or {}) for s in ("l", "r")):
        return None
    ident = {n: np.array([1.0, 0.0, 0.0, 0.0]) for n in JOINT_NAMES}
    rest = forward_kinematics([0.0, 0.0, skel.root_height], 0.0, ident, skeleton=skel)
    out = {}
    for s in ("l", "r"):
        hull = spec_sole[s]["hull"]
        joints = list(hull["joints"])
        at = np.asarray(hull["at"], float)
        w = np.asarray(hull["w"], float)
        if at.ndim != 2 or at.shape[1] != 3 or w.shape != (len(at), len(joints)):
            raise ValueError(f"sole.{s}.hull: 'at' must be (K, 3) and 'w' (K, {len(joints)})")
        offsets = np.stack(
            [_rotate_rows(_conj(rest[j][1]), at - rest[j][0]) for j in joints]
        )  # (J, K, 3)
        out[s] = (joints, w, offsets)
    return out


def _conj(q) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]])


def _rotate_rows(q, v) -> np.ndarray:
    """Rotate each row of ``v`` (n, 3) by quaternion ``q`` (w, x, y, z)."""
    u = np.asarray(q[1:], float)
    return v + 2.0 * np.cross(u, np.cross(u, v) + q[0] * v)


def _lowest_sole(poses, st) -> float:
    """Height of the lowest shoe-sole point of ``poses`` above the floor.

    With a hull (:func:`_sole_hull`) every hull point is skinned as the mesh is -- its joints' poses
    blended by its weights -- so the lowest is the skin's own, whichever way the foot is turned.
    Without one, the measured heel and toe sole points, each riding one joint; failing those, the
    ankle and toe-tip joints against their rest heights."""
    if st.sole_hull:
        low = np.inf
        for joints, w, offsets in st.sole_hull.values():
            z = np.stack(
                [
                    poses[j][0][2] + _rotate_rows(poses[j][1], offsets[k])[:, 2]
                    for k, j in enumerate(joints)
                ],
                axis=1,
            )  # (K, J)
            low = min(low, float((w * z).sum(axis=1).min()))
        return low
    tip = np.array(st.skeleton.foot_tip)
    lows = []
    for s in ("l", "r"):
        ap, aq = poses[f"ankle_{s}"]
        tp, tq = poses[f"toe_{s}"]
        if st.sole:  # measured sole offsets
            heel = float((ap + quat_rotate(aq, np.array(st.sole[s]["heel"])))[2])
            toe = float((tp + quat_rotate(tq, np.array(st.sole[s]["toe"])))[2])
            lows.append(min(heel, toe))
        else:  # fallback: ankle/toe-tip joints
            toetip = float((tp + quat_rotate(tq, tip))[2])
            ra, rt = st.foot_rest[s]
            lows.append(min(ap[2] - ra, toetip - rt))
    return min(lows)


def _approach_angle(cur, target, max_step) -> float:
    """Move ``cur`` toward ``target`` by at most ``max_step`` (shortest way)."""
    d = math.atan2(math.sin(target - cur), math.cos(target - cur))
    if abs(d) <= max_step:
        return target
    return cur + math.copysign(max_step, d)


def animate(data, st, new_pos, dt):
    """Ease the body toward the new nav position along CARLA's 2D blendspace: a speed axis
    (idle->short->walk->run) and a direction axis (turn-in-place when stopped / lean when moving,
    plus strafe) -- then FK + write the pose.

    With ``st.body_lag`` set, the body is drawn on a point that follows the nav root as a critically
    damped spring of that time constant: it starts, turns and stops over a fraction of a second where
    the nav root changes velocity at once, and comes to rest where the root is. The gait is phased
    by the body's own travel, so the feet stay planted; its turns (:func:`_turn`) and its foot
    grounding (:func:`_foot_ground`) ease too.
    """
    body = _follow(st, new_pos, dt)
    # How far the drawn body moved: with a lag that is the spring's point, not the nav root.
    delta = body - st.body_pos if st.body_lag > 0.0 else new_pos - st.pos
    dist = float(np.linalg.norm(delta))
    st.body_pos = body
    raw_speed = dist / max(dt, 1e-9)
    st.disp_speed += (1.0 - math.exp(-dt / _SPEED_TAU)) * (raw_speed - st.disp_speed)
    # Face where the walker *wants* to go (nav preferred velocity), not the instantaneous
    # avoidance push: side-steps then become a strafe/lean (direction axis) rather than the body
    # whipping around, and a walker that is blocked (~0 displacement) still turns in place to
    # face its goal.
    pref = st.pref_vel
    if float(np.hypot(pref[0], pref[1])) > 1e-3:
        target = math.atan2(float(pref[1]), float(pref[0]))
    elif st.face is not None:
        target = st.face  # a move turned the standing body (hands on only)
    elif dist > 1e-4:
        target = math.atan2(float(delta[1]), float(delta[0]))
    else:
        target = st.yaw
    new_yaw = _turn(st, target, dt)
    d_yaw = math.atan2(math.sin(new_yaw - st.yaw), math.cos(new_yaw - st.yaw))
    yaw_rate = d_yaw / max(dt, 1e-9)
    st.yaw = new_yaw
    st.phase += dist / st.walk.stride_len  # gait phases by distance (no slide)
    st.phase_run += dist / st.run.stride_len
    st.phase_short += dist / st.short.stride_len
    st.t_idle += dt  # idle by time
    # -- speed axis: idle -> short -> walk -> run (each takes over in turn) --
    qi, zi = st.idle.sample_array(st.t_idle / st.idle.duration)
    qs, zs = st.short.sample_array(st.phase_short)
    qw, zw = st.walk.sample_array(st.phase)
    qr, zr = st.run.sample_array(st.phase_run)
    w_short = smoothstep(st.disp_speed, *_IDLE_SHORT)
    w_walk = smoothstep(st.disp_speed, *_SHORT_WALK)
    w_run = smoothstep(st.disp_speed, *_WALK_RUN)
    q = blend_quats(qi, qs, w_short)
    root_z = (1 - w_short) * zi + w_short * zs
    q = blend_quats(q, qw, w_walk)
    root_z = (1 - w_walk) * root_z + w_walk * zw
    q = blend_quats(q, qr, w_run)
    root_z = (1 - w_run) * root_z + w_run * zr
    q = _direction_axis(st, q, d_yaw, yaw_rate, w_short, w_walk, delta, dist)
    # Over the gait, in this order, each over what came before: the clips played on some joints
    # (motion layers), a hand's reach, the motion core's hands and posture (hands on), the look.
    root_z = _layers(st, q, root_z, dt)
    _reach(st, q, body, root_z, dt)
    drop = st.rig.pose(st, q, body, root_z, dt) if st.rig is not None else 0.0
    root = [body[0], body[1], st.skeleton.root_height + root_z - drop]
    _look(st, q, root, dt)
    joint_rot = {name: q[j] for j, name in enumerate(JOINT_NAMES)}
    poses = forward_kinematics(root, st.yaw, joint_rot, skeleton=st.skeleton)
    write_pose(data, st, _foot_ground(poses, st, dt))
    if st.rig is not None:
        st.rig.write_fingers(data)
    st.pos = new_pos


def _turn(st, target, dt) -> float:
    """The heading after ``dt`` turning toward ``target`` at up to ``_MAX_TURN_RATE``: at once, or
    with ``st.body_lag`` set, speeding up and slowing down over twice the lag, so a body bent over
    something in its hands does not swing round from one frame to the next."""
    cap = getattr(st, "turn_rate", None)  # a move's cap while it carries
    rate = _MAX_TURN_RATE if cap is None else cap
    if st.body_lag <= 0.0 or dt <= 0.0:
        return _approach_angle(st.yaw, target, rate * dt)
    err = math.atan2(math.sin(target - st.yaw), math.cos(target - st.yaw))
    yaw, st.yaw_rate = _slew(
        st.yaw, st.yaw_rate, st.yaw + err, rate, rate / (2.0 * st.body_lag), dt
    )
    return yaw


def _slew(x, v, target, vmax, accel, dt):
    """``(x, v)`` after ``dt`` moving ``x`` (at rate ``v``) to ``target`` as fast as a rate of at most
    ``vmax``, changed by at most ``accel`` a second, allows: it speeds up, cruises and slows down to
    stop on the target."""
    err = target - x
    # The fastest rate that can still stop on the target in steps of dt, slowing by accel * dt a
    # step (k steps cover accel * dt^2 * k (k + 1) / 2).
    unit = accel * dt
    k = (math.sqrt(1.0 + 8.0 * abs(err) / (unit * dt)) - 1.0) / 2.0
    want = math.copysign(min(vmax, k * unit), err)
    v += max(-unit, min(unit, want - v))
    step = v * dt
    if abs(err) < 1e-9 or (step * err > 0.0 and abs(step) >= abs(err)):
        return target, 0.0
    return x + step, v


def _follow(st, target, dt) -> np.ndarray:
    """Where the body is drawn: the nav root itself, or with ``st.body_lag`` a critically damped
    spring's point following it."""
    target = np.asarray(target, float)
    if st.body_lag <= 0.0 or st.body_pos is None:
        st.body_vel = np.zeros(2)
        return target.copy()
    if dt <= 0.0:
        return st.body_pos.copy()
    pos, st.body_vel = _ease(st.body_pos, st.body_vel, target, st.body_lag, dt)
    return pos


def _ease(x, v, target, tau, dt):
    """``(x, v)`` after ``dt`` of a critically damped spring of time constant ``tau`` pulling ``x``
    (moving at ``v``) to ``target``; exact over the step, so a coarse tick cannot overshoot."""
    w = 1.0 / tau
    e = x - target
    decay = math.exp(-w * dt)
    c = v + w * e
    return target + (e + c * dt) * decay, (v - c * w * dt) * decay


def _reach(st, q, new_pos, root_z, dt) -> None:
    """Blend each hand's reach (its wrist on ``st.reach[side].target``) over the gait's, in place.

    Each hand eases in and out at its own weight and moves only its own arm's joints; a hand that
    is not reaching keeps the gait's arm. The trunk -- pelvis, spine, hips -- is blended by the
    stronger hand's weight, and bends only as far as the hands that are reaching need. A hand let go
    fades from the arm pose it last had, in the body's frame, to the gait's: it does not reach back
    to its point."""
    for hand in st.reach.values():
        if dt > 0.0:
            aim = 1.0 if hand.target is not None else 0.0
            hand.w, hand.w_rate = _slew(hand.w, hand.w_rate, aim, _REACH_RATE, _REACH_ACCEL, dt)
        if hand.target is None and hand.w < 1e-3:
            hand.pose = None
    weight = max(hand.w for hand in st.reach.values())
    if weight < 1e-3:
        # Nothing in either hand: the next reach bends down from upright.
        st.reach_bend = st.reach_bend_rate = 0.0
        st.reach_need = (frozenset(), 0.0)
        return
    # The body is shifted down onto its feet after this (_foot_ground); solved with the last
    # tick's shift, the wrists land on the points rather than that far under them.
    shift = st.ground if st.ground is not None else 0.0
    root = [new_pos[0], new_pos[1], st.skeleton.root_height + root_z - shift]
    local = {name: q[JOINT_NAMES.index(name)] for name in _TRUNK}
    heading = yaw_to_quat(st.yaw)
    targets = {side: hand.target for side, hand in st.reach.items()}
    reaching = frozenset(side for side, t in targets.items() if t is not None)
    # The bend eases toward what the reaching hands' points need, at a bounded rate: the body bends
    # down to them, and straightens after letting go, without a jerk. As far as the points come up
    # in the hands it follows them at once, never staying bent over them, so a box lifted does not
    # rise into the body; a hand let go is not a point coming up, so straightening from it eases.
    need = needed_bend(st.skeleton, root, heading, local, targets)
    if dt > 0.0:
        bend, rate = _slew(st.reach_bend, st.reach_bend_rate, need, _BEND_RATE, _BEND_ACCEL, dt)
        last_hands, last_need = st.reach_need
        if reaching and reaching == last_hands and need < last_need:
            lifted = max(need, st.reach_bend - (last_need - need))
            if lifted < bend:
                # Followed at once, the bend starts from rest: a rate of the lift's size would
                # carry it on past the need, and below upright, if the points then came down.
                bend, rate = lifted, 0.0
        st.reach_bend, st.reach_bend_rate = min(max(bend, 0.0), MAX_BEND), rate
    st.reach_need = (reaching, need)
    pose = reach_pose(st.skeleton, root, heading, local, targets, bend=st.reach_bend)
    for side, hand in st.reach.items():
        if hand.target is not None:
            hand.pose = {f"{j}_{side}": pose[f"{j}_{side}"] for j in _ARM}
    over = [(name, pose[name], weight) for name in _TRUNK]
    for hand in st.reach.values():
        if hand.pose is not None:
            over += [(name, rot, hand.w) for name, rot in hand.pose.items()]
    for name, rot, w in over:
        j = JOINT_NAMES.index(name)
        q[j] = blend_quats(q[j : j + 1], np.asarray(rot)[None, :], w)[0]


@dataclass
class MotionLayer:
    """A clip played over the gait on some of the joints: started by ``play``, it fades in to its
    weight, plays (once, or round and round with ``loop``) at ``speed``, and fades out at its end
    or once stopped. A layer that includes the pelvis carries the clip's height too."""

    id: int
    name: str
    clip: Clip
    joints: tuple  # indices into JOINT_NAMES
    weight: float = 1.0
    loop: bool = False
    fade: float = 0.3
    speed: float = 1.0
    t: float = 0.0  # s into the clip
    w: float = 0.0  # its weight now
    stopping: bool = False

    @property
    def end(self) -> float:
        """When a clip played once reaches its last frame (s)."""
        return (self.clip.num_frames - 1) / self.clip.fps

    def describe(self) -> dict:
        return {
            "id": self.id,
            "clip": self.name,
            "joints": [JOINT_NAMES[j] for j in self.joints],
            "loop": self.loop,
            "t": self.t,
            "weight": self.w,
            "stopping": self.stopping,
        }


def _layers(st, q, root_z, dt) -> float:
    """Blend each playing layer's clip over ``q`` on its joints, in place, in the order they were
    started; the root's height after them."""
    for layer in list(st.layers):
        if dt > 0.0:
            layer.t += dt * layer.speed
            if not layer.loop and layer.t >= layer.end - layer.fade * layer.speed:
                layer.stopping = (
                    True  # it fades out over its last `fade` and ends on its last frame
                )
            aim = 0.0 if layer.stopping else layer.weight
            step = math.inf if layer.fade <= 0.0 else dt * layer.weight / layer.fade
            layer.w += max(-step, min(step, aim - layer.w))
        if layer.stopping and layer.w <= 1e-6:
            st.layers.remove(layer)
            st.layers_ended.add(layer.id)
            continue
        qc, zc = layer.clip.sample_at(layer.t, layer.loop)
        for j in layer.joints:
            q[j] = blend_quats(q[j : j + 1], qc[j : j + 1], layer.w)[0]
        if 0 in layer.joints:  # the pelvis: the clip's height comes with it
            root_z = (1.0 - layer.w) * root_z + layer.w * zc
    return root_z


@dataclass
class Look:
    """Where the head looks: a world point, or a callable giving one as it moves (``None``: where the
    gait turns it); how much of the head's turn is the look's, and the head's rotation now."""

    point: object = None
    w: float = 0.0
    rot: np.ndarray | None = None
    settled: bool = False  # the head is on the point, within a degree
    desc: object = None  # the walker's description for the head's limits (made on first use)


#: How fast a look turns the head (rad/s), and fades in and out over the gait's (1/s).
_LOOK_RATE = 3.0
_LOOK_FADE = 2.5


def look_point(point) -> np.ndarray | None:
    p = point() if callable(point) else point
    return None if p is None else np.asarray(p, float)


def _look(st, q, root, dt) -> None:
    """Turn the head toward ``st.look``'s point within its limits, in place, at a bounded rate."""
    lk = st.look
    head = JOINT_NAMES.index("head")
    if dt > 0.0:
        aim = 1.0 if lk.point is not None else 0.0
        step = _LOOK_FADE * dt
        lk.w = min(max(lk.w + max(-step, min(step, aim - lk.w)), 0.0), 1.0)
    if lk.point is None and lk.w <= 1e-6:
        lk.rot = None
        return
    if lk.rot is None:
        lk.rot = np.asarray(q[head], float).copy()
    point = look_point(lk.point) if lk.point is not None else None
    if point is not None:
        from roqsim_motion.ik import head_toward

        from roqsim_motion import quat as Q

        if lk.desc is None:
            from roqsim_walker.handling import description_for

            lk.desc = description_for(st.skeleton)
        poses = forward_kinematics(
            root,
            st.yaw,
            {name: q[j] for j, name in enumerate(JOINT_NAMES)},
            skeleton=st.skeleton,
        )
        want = head_toward(lk.desc, *poses["spine"], point)
        angle = Q.angle_between(lk.rot, want)
        turn = _LOOK_RATE * dt
        lk.rot = want if angle <= turn else Q.slerp(lk.rot, want, turn / angle)
        lk.settled = angle <= max(turn, math.radians(1.0))
    q[head] = blend_quats(q[head : head + 1], lk.rot[None, :], lk.w)[0]


def _direction_axis(st, q, d_yaw, yaw_rate, w_short, w_walk, delta, dist):
    """CARLA's BS_GEN3 ``Direction`` parameter, in two parts.

    **Turn:** the in-place turn clip, played through (its cycle advanced by the actual heading
    change so the steps don't slide). When the walker is standing (``w_short`` ~ 0) it fully
    takes over -> a real turn-in-place; when walking it is capped to ``_TURN_MAX`` so it only
    leans the gait into the turn.

    **Strafe:** when travel is not aligned with facing (the walker re-faced toward its goal while
    avoidance pushes it sideways), blend the matching side / back walk clip by the travel-vs-facing
    angle. These clips are optional; absent, the body simply translates along the small residual
    deviation.
    """
    turning = min(abs(yaw_rate) / _TURN_REF, 1.0)
    turn = st.turn_l if d_yaw > 0 else st.turn_r
    if turn is not None and turning > 1e-3:
        st.phase_turn += abs(d_yaw) / _TURN_STRIDE
        w_turn = turning * ((1.0 - w_short) + w_short * _TURN_MAX)
        q = blend_quats(q, turn.sample_array(st.phase_turn)[0], w_turn)
    if dist > 1e-4 and w_walk > 1e-3:
        travel = math.atan2(float(delta[1]), float(delta[0]))
        direction = math.atan2(math.sin(travel - st.yaw), math.cos(travel - st.yaw))
        a = abs(direction)
        side = st.walk_l if direction > 0 else st.walk_r
        if side is not None:
            w_side = smoothstep(a, _STRAFE_FWD, _STRAFE_SIDE) * w_walk
            if w_side > 1e-3:
                q = blend_quats(q, side.sample_array(st.phase)[0], w_side)
        if st.walk_back is not None:
            w_back = smoothstep(a, _STRAFE_SIDE, _STRAFE_BACK) * w_walk
            if w_back > 1e-3:
                q = blend_quats(q, st.walk_back.sample_array(st.phase)[0], w_back)
    return q


def write_pose(data, st, poses):
    d = data
    for part, (pos, quat) in poses.items():  # per-limb collision rides the joints
        mid = st.mocap[part]
        d.mocap_pos[mid] = pos
        d.mocap_quat[mid] = quat


@dataclass
class HandReach:
    """One hand's reach: the world point it is on (``None``: back to the gait), its pose's weight
    over the gait's and how fast that changes, and the arm's local rotations it was last posed with,
    which a hand let go fades out from."""

    target: np.ndarray | None = None
    w: float = 0.0
    w_rate: float = 0.0
    pose: dict | None = None


def _hands() -> dict:
    return {side: HandReach() for side in SIDES}


@dataclass
class AnimState:
    """One walker's skeleton, clips and blendspace state; the nav root is ``pos`` and ``yaw``."""

    name: str
    walk: Clip
    idle: Clip
    run: Clip
    mocap: dict  # part name -> mocap id
    start: np.ndarray  # (x, y, yaw) the walker stands at after a reset: its configured pose
    skeleton: object = None  # this walker's per-rig bone table (humanoid.Skeleton)
    short: Clip = None  # slow-shuffle gait (idle->walk transition)
    turn_l: Clip = None  # in-place turn-left / turn-right (lean into turns)
    turn_r: Clip = None
    walk_l: Clip = None  # strafe-left / strafe-right / back-pedal (direction axis);
    walk_r: Clip = None  # optional -- inert when the clips are absent
    walk_back: Clip = None
    foot_rest: dict = None  # per-foot rest heel/toe-tip z (grounding fallback)
    sole: dict = None  # per-foot measured shoe-sole offsets (heel, toe; and the hull)
    sole_hull: dict = None  # per-foot hull points ready to skin (:func:`_sole_hull`)
    # -- runtime ---------------------------------------------------------------------------------
    pos: np.ndarray = field(default=None)
    yaw: float = 0.0
    phase: float = 0.0  # walk-clip phase (cycles, advanced by distance)
    phase_run: float = 0.0
    phase_short: float = 0.0
    phase_turn: float = 0.0  # turn-in-place phase (cycles, advanced by |heading change|)
    t_idle: float = 0.0  # idle-clip clock (s, advanced by time)
    disp_speed: float = 0.0  # low-passed body speed (drives gait + blend)
    pref_vel: np.ndarray = field(default_factory=lambda: np.zeros(2))
    #: each hand's reach, by arm (``"l"``, ``"r"``; :class:`HandReach`), and the body's forward bend
    #: for them (rad), how fast it changes, and which hands' points needed how much bend last tick
    reach: dict = field(default_factory=_hands)
    reach_bend: float = 0.0
    reach_bend_rate: float = 0.0
    reach_need: tuple = (frozenset(), 0.0)
    #: the body's lag behind the nav root (s; 0: drawn on it), and where it is drawn and how fast it
    #: moves (:func:`_follow`)
    body_lag: float = 0.0
    body_pos: np.ndarray | None = None
    body_vel: np.ndarray = field(default_factory=lambda: np.zeros(2))
    ground: float | None = None  # the foot grounding's shift (m) and how fast it changes
    ground_vel: float = 0.0
    yaw_rate: float = 0.0  # rad/s the heading turns at, eased with a body lag (:func:`_turn`)
    #: hands on: the motion core's body (:class:`roqsim_walker.handling.WalkerBody`), and the heading
    #: a move turned the standing body to; both ``None`` with hands off
    rig: object = None
    face: float | None = None
    #: the body's turn rate cap (rad/s) a move sets while it carries; ``None``: ``_MAX_TURN_RATE``
    turn_rate: float | None = None
    #: the clips playing over the gait (:class:`MotionLayer`), in the order they were started, and
    #: the ids of those that have ended
    layers: list = field(default_factory=list)
    layers_ended: set = field(default_factory=set)
    #: where the head looks (:class:`Look`)
    look: Look = field(default_factory=Look)


def make_anim_state(model, spec) -> AnimState:
    start = np.array(spec.get("start") or (0.0, 0.0, 0.0), dtype=float)
    skel = to_skeleton(spec.get("skeleton"))
    st = AnimState(
        name=spec["name"],
        walk=_load_clip(spec, "walk", skel),
        idle=_load_clip(spec, "idle", skel),
        run=_load_clip(spec, "run", skel),
        short=_load_clip(spec, "short", skel),
        turn_l=_opt_clip(spec, "turn_l", skel),
        turn_r=_opt_clip(spec, "turn_r", skel),
        walk_l=_opt_clip(spec, "walk_l", skel),
        walk_r=_opt_clip(spec, "walk_r", skel),
        walk_back=_opt_clip(spec, "walk_back", skel),
        mocap=_mocap_ids(model, spec["name"]),
        skeleton=skel,
        foot_rest=_foot_rest(skel),
        sole=spec.get("sole"),
        sole_hull=_sole_hull(spec.get("sole"), skel),
        start=start,
        body_lag=float(spec.get("body_lag", 0.0)),
    )
    st.pos = start[:2].copy()
    st.body_pos = start[:2].copy()
    st.yaw = float(start[2])
    return st


def _load_clip(spec, kind, skel) -> Clip:
    """The ``kind`` locomotion clip the blueprint resolved, grounded on this walker's feet."""
    path = (spec.get("motion") or {}).get(kind)
    if not path:
        raise ClipError(f"walker {spec.get('name')}: no {kind} clip resolved")
    return _ground_clip(Clip.load(path), skel)


def _opt_clip(spec, kind, skel):
    """An optional locomotion clip (a turn, a strafe), grounded; ``None`` when there is none."""
    path = (spec.get("motion") or {}).get(kind)
    return _ground_clip(Clip.load(path), skel) if path else None


def load_layer_clip(path, skel) -> Clip:
    """A clip a motion layer plays, grounded as the locomotion clips are."""
    return _ground_clip(Clip.load(path), skel)


def _mocap_ids(model, name) -> dict:
    ids = {}
    for part in JOINT_NAMES:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{name}/{part}")
        if bid < 0:
            raise ValueError(
                f"humanoid body {name}/{part} not in model (was build_humanoid run before compile?)"
            )
        ids[part] = int(model.body_mocapid[bid])
    return ids
