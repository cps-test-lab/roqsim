#!/usr/bin/env python3
"""Convert a BVH motion capture into a walker clip (``.npz``) that a walker plays.

    roqsim walker import-clip wave.bvh --out my_anims/adult/wave.npz
    roqsim walker import-clip walk.bvh --out walk.npz --loop --map spine=Chest

The clip is the walker's own format (:class:`roqsim_walker.motion.Clip`): each of the seventeen
joints' local rotation per frame, and the root's height. The capture's travel and heading are
dropped -- where a walker goes is its navigator's -- and its turns relative to the first frame are
kept. Play it with ``play`` (a motion layer, on the joints you name) or as a locomotion clip with a
walker's ``motion:``; ``--loop`` marks a cycle, and its stride is measured from the capture's travel.

**Retargeting by bone direction.** Each of the walker's bones points where the source's bone does,
frame by frame: the source's rest pose (a T-pose or an A-pose, as BVH captures come) is matched to the
walker's arms-down rest by turning each walker bone onto its source bone at rest, so a capture moves
the walker's limbs as it moves its own, whatever the two rigs' proportions. A bone's twist about its
own line follows the source's. The walker's trunk has fewer bones than most rigs': its pelvis turns
as the source's hips do and its spine as the source's upper spine, so the bend of the spine bones
between them and of a neck is not drawn; the head turns as the source's does.

**Which bone is which.** Joints are found by name -- the Mixamo, CMU and most game rigs' names
(``Hips``, ``LeftUpLeg``, ``LeftArm``, ``LeftForeArm``, ``thigh_l``, ...), any ``prefix:`` dropped --
and the walker's ``spine`` is the source bone the neck and both arms leave from. A joint not found
is an error naming the source's bones; ``--map joint=Bone`` names it (``--map toe_l=LeftFoot`` for
a rig without toes: the toe then follows the foot).

**Axes.** BVH is Y up; ``--up z`` reads a Z-up capture. ``--forward`` is the direction the character
faces in the source's rest pose (default ``+z``, as Mixamo and CMU face).
"""

from __future__ import annotations

import argparse
import math
import os
import re
import sys

import numpy as np

from roqsim_walker.humanoid import DEFAULT_SKELETON, JOINT_NAMES, SKELETON
from roqsim_walker.motion import Clip

# -- the BVH file -------------------------------------------------------------------------------------


class BvhError(ValueError):
    """A BVH file that cannot be read, or mapped onto the walker."""


class Bone:
    def __init__(self, name: str, parent: int, offset):
        self.name = name
        self.parent = parent
        self.offset = np.asarray(offset, float)
        self.channels: list[str] = []
        self.end: np.ndarray | None = None  # an End Site's offset, for a leaf


def parse_bvh(text: str):
    """``(bones, frames (F, C), frame_time)`` of a BVH document."""
    tokens = text.split()
    i = 0
    bones: list[Bone] = []
    stack: list[int] = []

    def take() -> str:
        nonlocal i
        if i >= len(tokens):
            raise BvhError("the file ends inside its HIERARCHY")
        i += 1
        return tokens[i - 1]

    if take() != "HIERARCHY":
        raise BvhError("not a BVH file: it does not start with HIERARCHY")
    while i < len(tokens) and tokens[i] != "MOTION":
        word = take()
        if word in ("ROOT", "JOINT"):
            bones.append(Bone(take(), stack[-1] if stack else -1, (0.0, 0.0, 0.0)))
            if take() != "{":
                raise BvhError(f"bone {bones[-1].name}: '{{' expected")
            stack.append(len(bones) - 1)
        elif word == "End":
            take()  # "Site"
            if take() != "{" or take() != "OFFSET":
                raise BvhError("an End Site without its OFFSET")
            bones[stack[-1]].end = np.array([float(take()) for _ in range(3)])
            if take() != "}":
                raise BvhError("an End Site not closed")
        elif word == "OFFSET":
            bones[stack[-1]].offset = np.array([float(take()) for _ in range(3)])
        elif word == "CHANNELS":
            n = int(take())
            bones[stack[-1]].channels = [take() for _ in range(n)]
        elif word == "}":
            stack.pop()
        else:
            raise BvhError(f"unexpected {word!r} in the HIERARCHY")
    if not bones:
        raise BvhError("the HIERARCHY has no bones")
    take()  # MOTION
    if take() != "Frames:":
        raise BvhError("MOTION without 'Frames:'")
    n_frames = int(take())
    if take() != "Frame" or take() != "Time:":
        raise BvhError("MOTION without 'Frame Time:'")
    frame_time = float(take())
    n_ch = sum(len(b.channels) for b in bones)
    values = np.array([float(v) for v in tokens[i:]], float)
    if values.size != n_frames * n_ch:
        raise BvhError(
            f"MOTION holds {values.size} values, not {n_frames} frames of {n_ch} channels"
        )
    if n_frames < 1 or frame_time <= 0.0:
        raise BvhError(f"{n_frames} frames at {frame_time} s a frame")
    return bones, values.reshape(n_frames, n_ch), frame_time


def _axis_matrix(axis: str, deg: float) -> np.ndarray:
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    if axis == "X":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == "Y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def pose_frames(bones, frames):
    """Each bone's global rotation (F, B, 3, 3) and position (F, B, 3), in the source's axes."""
    n_f, n_b = frames.shape[0], len(bones)
    rot = np.zeros((n_f, n_b, 3, 3))
    pos = np.zeros((n_f, n_b, 3))
    col = 0
    cols = []
    for b in bones:
        cols.append(col)
        col += len(b.channels)
    for f in range(n_f):
        for k, b in enumerate(bones):
            local = np.eye(3)
            trans = b.offset.copy()
            for c, name in enumerate(b.channels):
                v = frames[f, cols[k] + c]
                if name.endswith("rotation"):
                    local = local @ _axis_matrix(name[0].upper(), v)
                elif name.endswith("position"):
                    trans["XYZ".index(name[0].upper())] = v
            if b.parent < 0:
                rot[f, k], pos[f, k] = local, trans
            else:
                rot[f, k] = rot[f, b.parent] @ local
                pos[f, k] = pos[f, b.parent] + rot[f, b.parent] @ trans
    return rot, pos


def rest_positions(bones) -> np.ndarray:
    """Each bone's position at rest (every rotation zero), the root at its OFFSET."""
    pos = np.zeros((len(bones), 3))
    for k, b in enumerate(bones):
        pos[k] = b.offset if b.parent < 0 else pos[b.parent] + b.offset
    return pos


# -- which bone is which -------------------------------------------------------------------------------

#: Name stems per joint, most specific first; each side's names are built from them.
_STEMS = {
    "hip": ("upleg", "upperleg", "thigh", "hip"),
    "knee": ("leg", "lowerleg", "shin", "calf", "knee"),
    "ankle": ("foot", "ankle"),
    "toe": ("toebase", "toe", "toes", "ball"),
    "shoulder": ("arm", "upperarm"),
    "elbow": ("forearm", "lowerarm", "elbow"),
    "wrist": ("hand", "wrist"),
}
_SIDES = {"l": ("left", "l"), "r": ("right", "r")}


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.rsplit(":", 1)[-1].lower())


def _candidates(joint: str) -> list[str]:
    if joint == "pelvis":
        return ["hips", "pelvis", "hip", "root"]
    if joint == "head":
        return ["head"]
    stem, side = joint.rsplit("_", 1)
    out = []
    for s in _STEMS[stem]:
        for word in _SIDES[side]:
            out += [word + s, s + word]
    return out


def _ancestors(bones, k: int) -> list[int]:
    chain = []
    while k >= 0:
        chain.append(k)
        k = bones[k].parent
    return chain


def map_bones(bones, overrides: dict[str, str] | None = None) -> dict[str, int]:
    """Each walker joint's source bone (an index into ``bones``). Raises :class:`BvhError` naming
    the joints it cannot find and the source's bones."""
    by_norm: dict[str, int] = {}
    by_name = {b.name: k for k, b in enumerate(bones)}
    for k, b in enumerate(bones):
        by_norm.setdefault(_norm(b.name), k)
    found: dict[str, int] = {}
    for joint, bone in (overrides or {}).items():
        if joint not in JOINT_NAMES:
            raise BvhError(f"--map {joint}=...: the walker has no joint {joint!r}")
        if bone not in by_name:
            raise BvhError(f"--map {joint}={bone}: the capture has no bone {bone!r}")
        found[joint] = by_name[bone]
    for joint in JOINT_NAMES:
        if joint in found or joint == "spine":
            continue
        for cand in _candidates(joint):
            if cand in by_norm:
                found[joint] = by_norm[cand]
                break
    if "spine" not in found and all(j in found for j in ("head", "shoulder_l", "shoulder_r")):
        # The bone the neck and both arms leave from: their deepest common ancestor.
        common = set(_ancestors(bones, found["head"]))
        for j in ("shoulder_l", "shoulder_r"):
            common &= set(_ancestors(bones, found[j]))
        common -= {found["head"], found["shoulder_l"], found["shoulder_r"]}
        if common:
            found["spine"] = max(common, key=lambda k: len(_ancestors(bones, k)))
    missing = [j for j in JOINT_NAMES if j not in found]
    if missing:
        raise BvhError(
            f"no bone found for {', '.join(missing)}; the capture's bones are "
            f"{', '.join(b.name for b in bones)} -- name each with --map joint=Bone"
        )
    return found


# -- retargeting -------------------------------------------------------------------------------------

#: The joint each walker joint's bone points to; a leaf turns as its parent's bone does.
_CHILD = {
    "pelvis": "spine",
    "spine": "head",
    "hip_l": "knee_l",
    "knee_l": "ankle_l",
    "ankle_l": "toe_l",
    "hip_r": "knee_r",
    "knee_r": "ankle_r",
    "ankle_r": "toe_r",
    "shoulder_l": "elbow_l",
    "elbow_l": "wrist_l",
    "shoulder_r": "elbow_r",
    "elbow_r": "wrist_r",
}


def basis(up: str = "y", forward: str = "+z") -> np.ndarray:
    """The rotation taking the source's axes to the walker's (x forward, y left, z up)."""
    axes = {"x": np.eye(3)[0], "y": np.eye(3)[1], "z": np.eye(3)[2]}
    sign = -1.0 if forward.startswith("-") else 1.0
    f = sign * axes[forward.lstrip("+-")]
    u = axes[up]
    if abs(float(f @ u)) > 1e-9:
        raise BvhError(f"--forward {forward} lies along --up {up}")
    left = np.cross(u, f)
    return np.stack([f, left, u])  # rows: the source's forward, left and up; walker = this @ source


def _between(a, b) -> np.ndarray:
    """The smallest rotation (3x3) turning direction ``a`` onto ``b``."""
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = float(a @ b)
    if c < -1.0 + 1e-9:  # opposite: half a turn about any axis across a
        axis = np.cross(a, [1.0, 0.0, 0.0])
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(a, [0.0, 1.0, 0.0])
        axis /= np.linalg.norm(axis)
        return 2.0 * np.outer(axis, axis) - np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / (1.0 + c)


def _quat(m) -> np.ndarray:
    from roqsim_motion.quat import from_matrix

    return from_matrix(m)


def convert(text: str, overrides=None, up="y", forward="+z", loop=False, start=0, end=None) -> Clip:
    """The walker clip of a BVH document (see the module docstring)."""
    bones, frames, frame_time = parse_bvh(text)
    frames = frames[start:end]
    if frames.shape[0] < 1:
        raise BvhError(f"frames {start}:{end} select none")
    src = map_bones(bones, overrides)
    c = basis(up, forward)
    rot, pos = pose_frames(bones, frames)
    rest = rest_positions(bones)
    skel = DEFAULT_SKELETON
    # Each walker bone turned onto its source bone at rest (the walker's rest: every joint's frame
    # the world's, its bones along their offsets).
    align: dict[str, np.ndarray] = {}
    for j in SKELETON:
        child = _CHILD.get(j.name)
        if child is None:
            align[j.name] = align[j.parent]
            continue
        theirs = c @ (rest[src[child]] - rest[src[j.name]])
        if np.linalg.norm(theirs) < 1e-9:
            raise BvhError(f"{j.name} and {child} are one point in the capture's rest pose")
        align[j.name] = _between(np.asarray(skel.offset(child), float), theirs)
    n = frames.shape[0]
    joint_rot = np.zeros((n, len(JOINT_NAMES), 4))
    hips = np.array([c @ pos[f, src["pelvis"]] for f in range(n)])
    unturn = None  # takes the capture's heading at its first frame off every frame
    for f in range(n):
        world = {j.name: c @ rot[f, src[j.name]] @ c.T @ align[j.name] for j in SKELETON}
        if unturn is None:
            fwd = world["pelvis"] @ [1.0, 0.0, 0.0]
            yaw = math.atan2(fwd[1], fwd[0])
            unturn = np.array(
                [
                    [math.cos(yaw), math.sin(yaw), 0.0],
                    [-math.sin(yaw), math.cos(yaw), 0.0],
                    [0, 0, 1],
                ]
            )
        world = {name: unturn @ g for name, g in world.items()}
        for k, j in enumerate(SKELETON):
            local = world[j.name] if j.parent is None else world[j.parent].T @ world[j.name]
            joint_rot[f, k] = _quat(local)
    # The capture's scale onto the walker's: leg length to leg length.
    theirs = sum(
        np.linalg.norm(rest[src[b]] - rest[src[a]])
        for a, b in (("hip_l", "knee_l"), ("knee_l", "ankle_l"))
    )
    ours = sum(np.linalg.norm(skel.offset(x)) for x in ("knee_l", "ankle_l"))
    scale = ours / theirs
    root_z = (hips[:, 2] - hips[0, 2]) * scale
    travel = float(np.linalg.norm(hips[-1, :2] - hips[0, :2])) * scale
    return Clip(
        joint_rot,
        root_z,
        fps=1.0 / frame_time,
        stride_len=travel if (loop and travel > 0.05) else 1.0,
        loop=loop,
    )


def main(argv: list | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("bvh", help="the motion capture (.bvh)")
    ap.add_argument("--out", required=True, help="the clip to write (.npz)")
    ap.add_argument(
        "--map",
        action="append",
        default=[],
        metavar="JOINT=BONE",
        help="the source bone of a walker joint, where its name is not found; repeatable",
    )
    ap.add_argument("--up", default="y", choices=["x", "y", "z"], help="the source's up axis")
    ap.add_argument(
        "--forward",
        default="+z",
        choices=["+x", "-x", "+y", "-y", "+z", "-z"],
        help="where the source's character faces at rest",
    )
    ap.add_argument(
        "--loop", action="store_true", help="a cycle: played round, its stride measured"
    )
    ap.add_argument("--start", type=int, default=0, help="first frame")
    ap.add_argument("--end", type=int, default=None, help="frame after the last")
    args = ap.parse_args(argv)
    overrides = {}
    for item in args.map:
        joint, sep, bone = item.partition("=")
        if not sep:
            ap.error(f"--map {item}: give JOINT=BONE")
        overrides[joint] = bone
    try:
        with open(args.bvh) as f:
            clip = convert(
                f.read(), overrides, args.up, args.forward, args.loop, args.start, args.end
            )
    except (OSError, BvhError) as exc:
        sys.exit(f"import-clip: {exc}")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    clip.save(args.out)
    Clip.load(args.out)  # what a walker reads back
    print(
        f"wrote {args.out}: {clip.num_frames} frames at {clip.fps:g} fps "
        f"({clip.duration:.2f} s){', a loop' if clip.loop else ''}"
    )


if __name__ == "__main__":
    main()
