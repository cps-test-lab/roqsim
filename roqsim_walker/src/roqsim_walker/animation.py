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

from roqsim_walker.humanoid import (
    JOINT_NAMES,
    forward_kinematics,
    quat_rotate,
    to_skeleton,
)
from roqsim_walker.motion import (
    Clip,
    blend_quats,
    procedural_idle,
    procedural_walk,
    smoothstep,
)

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


def _foot_ground(poses, st):
    """Vertical-only foot grounding: shift the whole body so the **lowest shoe-sole point** -- the
    measured heel/toe of either foot -- sits exactly on the floor. Tracking the real sole (not the
    ankle/toe joint, which is several cm above it) removes the residual hover."""
    tip = np.array(st.skeleton.foot_tip)
    lows = []
    for s in ("l", "r"):
        ap, aq = poses[f"ankle_{s}"]
        tp, tq = poses[f"toe_{s}"]
        if st.sole:  # exact: measured sole offsets
            heel = float((ap + quat_rotate(aq, np.array(st.sole[s]["heel"])))[2])
            toe = float((tp + quat_rotate(tq, np.array(st.sole[s]["toe"])))[2])
            lows.append(min(heel, toe))
        else:  # fallback: ankle/toe-tip joints
            toetip = float((tp + quat_rotate(tq, tip))[2])
            ra, rt = st.foot_rest[s]
            lows.append(min(ap[2] - ra, toetip - rt))
    c = min(lows)
    if abs(c) > 1e-6:
        for nm, (p, q) in poses.items():
            poses[nm] = (p - np.array([0.0, 0.0, c]), q)
    return poses


def _wp_xy(p) -> tuple[float, float]:
    """World (x, y) of a waypoint entry: ``[x, y]``, ``[x, y, dwell]``, or ``{pos: [x, y], dwell:
    ...}``."""
    if isinstance(p, dict):
        return float(p["pos"][0]), float(p["pos"][1])
    return float(p[0]), float(p[1])


def _heading(a, b) -> float:
    return math.atan2(float(b[1]) - float(a[1]), float(b[0]) - float(a[0]))


def _approach_angle(cur, target, max_step) -> float:
    """Move ``cur`` toward ``target`` by at most ``max_step`` (shortest way)."""
    d = math.atan2(math.sin(target - cur), math.cos(target - cur))
    if abs(d) <= max_step:
        return target
    return cur + math.copysign(max_step, d)


def animate(data, st, new_pos, dt):
    """Ease the body toward the new nav position along CARLA's 2D blendspace: a speed axis
    (idle->short->walk->run) and a direction axis (turn-in-place when stopped / lean when moving,
    plus strafe) -- then FK + write the pose."""
    delta = new_pos - st.pos
    dist = float(np.linalg.norm(delta))
    raw_speed = dist / max(dt, 1e-9)
    st.disp_speed += (1.0 - math.exp(-dt / _SPEED_TAU)) * (raw_speed - st.disp_speed)
    # Face where the walker *wants* to go (nav preferred velocity), not the instantaneous
    # avoidance push: side-steps then become a strafe/lean (direction axis) rather than the body
    # whipping around, and a walker that is blocked (~0 displacement) still turns in place to
    # face its goal.
    pref = st.pref_vel
    if float(np.hypot(pref[0], pref[1])) > 1e-3:
        target = math.atan2(float(pref[1]), float(pref[0]))
    elif dist > 1e-4:
        target = math.atan2(float(delta[1]), float(delta[0]))
    else:
        target = st.yaw
    new_yaw = _approach_angle(st.yaw, target, _MAX_TURN_RATE * dt)
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
    joint_rot = {name: q[j] for j, name in enumerate(JOINT_NAMES)}
    poses = forward_kinematics(
        [new_pos[0], new_pos[1], st.skeleton.root_height + root_z],
        st.yaw,
        joint_rot,
        skeleton=st.skeleton,
    )
    write_pose(data, st, _foot_ground(poses, st))
    st.pos = new_pos


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
class AnimState:
    """One walker's skeleton, clips and blendspace state; the nav root is ``pos`` and ``yaw``."""

    name: str
    walk: Clip
    idle: Clip
    run: Clip
    mocap: dict  # part name -> mocap id
    patrol_wps: np.ndarray  # (N, 2) configured waypoints; the walker starts at the first
    skeleton: object = None  # this walker's per-rig bone table (humanoid.Skeleton)
    short: Clip = None  # slow-shuffle gait (idle->walk transition)
    turn_l: Clip = None  # in-place turn-left / turn-right (lean into turns)
    turn_r: Clip = None
    walk_l: Clip = None  # strafe-left / strafe-right / back-pedal (direction axis);
    walk_r: Clip = None  # optional -- inert when the clips are absent
    walk_back: Clip = None
    foot_rest: dict = None  # per-foot rest heel/toe-tip z (grounding fallback)
    sole: dict = None  # per-foot measured shoe-sole offsets -> exact grounding
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


def spec_waypoints(spec) -> list:
    """The spec's patrol waypoints, or a single spawn point when it is goal-driven only."""
    wps = list(spec.get("waypoints") or [])
    if wps:
        return wps
    return [list(spec.get("pos") or (0.0, 0.0))]


def make_anim_state(model, spec) -> AnimState:
    wps = np.array([_wp_xy(p) for p in spec_waypoints(spec)], dtype=float)
    skel = to_skeleton(spec.get("skeleton"))
    st = AnimState(
        name=spec["name"],
        walk=_load_clip(spec, "walk", procedural_walk, skel),
        idle=_load_clip(spec, "idle", procedural_idle, skel),
        run=_load_clip(spec, "run", procedural_walk, skel),
        short=_load_clip(spec, "short", procedural_walk, skel),
        turn_l=_opt_clip(spec, "turn_l", skel),
        turn_r=_opt_clip(spec, "turn_r", skel),
        walk_l=_opt_clip(spec, "walk_l", skel),
        walk_r=_opt_clip(spec, "walk_r", skel),
        walk_back=_opt_clip(spec, "walk_back", skel),
        mocap=_mocap_ids(model, spec["name"]),
        skeleton=skel,
        foot_rest=_foot_rest(skel),
        sole=spec.get("sole"),
        patrol_wps=wps.copy(),
    )
    st.pos = wps[0].copy()
    st.yaw = _heading(wps[0], wps[1]) if len(wps) > 1 else 0.0
    return st


def _load_clip(spec, kind, fallback, skel) -> Clip:
    path = (spec.get("motion") or {}).get(kind)
    if path:
        try:
            return _ground_clip(Clip.load(path), skel)
        except Exception as e:  # noqa: BLE001 -- fall back, log why
            logger.warning(
                "walker %s: could not load %s clip %s (%s); using procedural",
                spec.get("name"),
                kind,
                path,
                e,
            )
    return _ground_clip(fallback(), skel)


def _opt_clip(spec, kind, skel):
    """Load an optional clip (e.g. a turn) -> grounded Clip or None."""
    path = (spec.get("motion") or {}).get(kind)
    if not path:
        return None
    try:
        return _ground_clip(Clip.load(path), skel)
    except Exception as e:  # noqa: BLE001
        logger.warning("walker %s: could not load %s clip %s (%s)", spec.get("name"), kind, path, e)
        return None


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
