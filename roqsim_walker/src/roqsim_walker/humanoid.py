"""A kinematic, mocap-driven articulated human (capsule skeleton).

Ported from our earlier in-house nav prototype's ``mujoco_nav.humanoid``.

The walker's *body pose* comes from a motion clip while its *nav root* (x, y, heading) comes from
the nav stack (see :mod:`roqsim_walker.nav`). To stay perfectly kinematic and add zero DOFs to
the physics solver, the human is **not** a joint tree: it is a flat set of MuJoCo **mocap bodies**
(one per skeleton joint), each posed directly in the world frame every step by forward kinematics
from the clip's per-joint rotations. Per-limb collision capsules ride those bodies and are what the
robot's lidar/contacts see.

This module owns the single source of truth for the skeleton (joint hierarchy + rest bone offsets):
:func:`build_humanoid` injects matching mocap bodies/geoms into an ``MjSpec`` before compile, and
:func:`forward_kinematics` composes the same offsets at runtime.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass

import mujoco
import numpy as np

# Pelvis standing height (m): with the leg offsets below the feet rest near z=0.
ROOT_HEIGHT = 0.93
# Nav collision proxy (r, half-height) -- the silhouette a lidar sees.
COLLISION_RADIUS = 0.26
COLLISION_HALF_H = 0.83


@dataclass(frozen=True)
class Joint:
    name: str
    parent: str | None
    offset: tuple  # rest translation from parent (parent frame, Z-up), m
    radius: float  # capsule radius of the bone(s) leaving this joint


# Topological order (parents before children). Z up, X forward, Y left.
SKELETON = [
    Joint("pelvis", None, (0.0, 0.0, 0.0), 0.10),
    Joint("spine", "pelvis", (0.0, 0.0, 0.22), 0.09),
    Joint("head", "spine", (0.0, 0.0, 0.25), 0.06),
    Joint("hip_l", "pelvis", (0.0, 0.11, -0.05), 0.07),
    Joint("knee_l", "hip_l", (0.0, 0.0, -0.42), 0.06),
    Joint("ankle_l", "knee_l", (0.0, 0.0, -0.42), 0.05),
    Joint("toe_l", "ankle_l", (0.10, 0.0, -0.03), 0.04),
    Joint("hip_r", "pelvis", (0.0, -0.11, -0.05), 0.07),
    Joint("knee_r", "hip_r", (0.0, 0.0, -0.42), 0.06),
    Joint("ankle_r", "knee_r", (0.0, 0.0, -0.42), 0.05),
    Joint("toe_r", "ankle_r", (0.10, 0.0, -0.03), 0.04),
    Joint("shoulder_l", "spine", (0.0, 0.19, 0.12), 0.045),
    Joint("elbow_l", "shoulder_l", (0.0, 0.0, -0.26), 0.04),
    Joint("wrist_l", "elbow_l", (0.0, 0.0, -0.24), 0.04),
    Joint("shoulder_r", "spine", (0.0, -0.19, 0.12), 0.045),
    Joint("elbow_r", "shoulder_r", (0.0, 0.0, -0.26), 0.04),
    Joint("wrist_r", "elbow_r", (0.0, 0.0, -0.24), 0.04),
]

JOINT_NAMES = [j.name for j in SKELETON]
CHILDREN: dict[str, list[str]] = defaultdict(list)
for _j in SKELETON:
    if _j.parent:
        CHILDREN[_j.parent].append(_j.name)

# The shared joint *topology* above is fixed; the per-joint *offsets* and the pelvis height are
# per-walker (extracted from each CARLA SK rig into people/<W>/*.walker.json, so a child binds to a
# child-sized skeleton). A `Skeleton` bundles those; `DEFAULT_SKELETON` (the table above) is the
# fallback.
_DEFAULT_OFFSETS = {j.name: tuple(j.offset) for j in SKELETON}


@dataclass(frozen=True)
class Skeleton:
    offsets: dict | None = None  # joint name -> parent-relative offset (m); None -> default
    root_height: float = ROOT_HEIGHT
    foot_tip: tuple = (0.16, 0.0, -0.04)

    def offset(self, name: str) -> tuple:
        return (self.offsets or _DEFAULT_OFFSETS).get(name, _DEFAULT_OFFSETS[name])


DEFAULT_SKELETON = Skeleton()


def to_skeleton(spec) -> Skeleton:
    """Coerce a walker.json ``skeleton`` dict (or None/Skeleton) to a :class:`Skeleton`."""
    if spec is None or isinstance(spec, Skeleton):
        return spec or DEFAULT_SKELETON
    return Skeleton(
        offsets={k: tuple(v) for k, v in (spec.get("offsets") or {}).items()},
        root_height=float(spec.get("root_height", ROOT_HEIGHT)),
        foot_tip=tuple(spec.get("foot_tip", (0.16, 0.0, -0.04))),
    )


# Terminal geoms for leaf joints (no child bone): (kind, params).
_TERMINALS = {
    "head": ("sphere", (0.10, (0.0, 0.0, 0.06))),
    "toe_l": ("foot", (0.045, (0.05, 0.0, -0.01))),  # toe tip beyond the ball joint
    "toe_r": ("foot", (0.045, (0.05, 0.0, -0.01))),
    "wrist_l": ("sphere", (0.045, (0.0, 0.0, -0.02))),
    "wrist_r": ("sphere", (0.045, (0.0, 0.0, -0.02))),
}


# -- quaternion helpers (w, x, y, z) ----------------------------------------------------------
def quat_yaw(yaw: float) -> np.ndarray:
    h = yaw / 2.0
    return np.array([math.cos(h), 0.0, 0.0, math.sin(h)])


def quat_mul(a, b) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ]
    )


def quat_rotate(q, v) -> np.ndarray:
    """Rotate 3-vector ``v`` by quaternion ``q`` (w, x, y, z)."""
    w, x, y, z = q
    u = np.array([x, y, z])
    vv = np.asarray(v, dtype=float)
    return vv + 2.0 * np.cross(u, np.cross(u, vv) + w * vv)


# -- build (before compile) -------------------------------------------------------------------
# Arm joints: their collision capsules are lidar-visible but non-colliding (contype/conaffinity 0),
# so the robot isn't shoved by a kinematic swinging arm; the torso + legs physically block.
_ARM_JOINTS = {"shoulder_l", "shoulder_r", "elbow_l", "elbow_r", "wrist_l", "wrist_r"}


def build_humanoid(
    spec,
    name: str,
    rgba=(0.80, 0.66, 0.52, 1.0),
    mesh=None,
    materials=None,
    tpose: bool = False,
    skeleton=None,
    collision=None,
    flip=None,
) -> list[str]:
    """Inject one mocap body per skeleton joint, each carrying a **per-limb collision capsule**
    (the robot's lidar/contact silhouette -- a thick torso, thinner limbs; arms sensing-only).

    The visual is either capsule bones (``mesh=None`` fallback) or a deformable ``<skin>``.
    ``skeleton`` is the per-walker bone table; ``collision`` maps joint -> capsule radius (m, from
    the CARLA Phys asset). ``flip`` turns a -X-facing mesh (defaults to ``tpose``; see
    :func:`roqsim_walker.skin.add_skin`). Returns the joint names (body suffixes).
    """
    skel = to_skeleton(skeleton)
    park = [0.0, 0.0, -50.0]  # parked below the floor until the first pose is written
    for j in SKELETON:
        b = spec.worldbody.add_body(name=f"{name}/{j.name}", pos=list(park), mocap=True)
        if mesh is None:  # capsule visuals (fallback)
            for child in CHILDREN[j.name]:  # bones leaving this joint
                _add_capsule(b, (0.0, 0.0, 0.0), skel.offset(child), j.radius, rgba)
            term = _TERMINALS.get(j.name)
            if term:
                _add_terminal(b, term, rgba)
        _add_collision(b, j, skel, collision)  # per-limb hit-box (always)
    if mesh is not None:  # deformable character skin
        from roqsim_walker import skin

        skin.add_skin(
            spec, name, mesh, materials=materials, rgba=rgba, tpose=tpose, skeleton=skel, flip=flip
        )
    return list(JOINT_NAMES)


def _collision_flags(g, physical: bool) -> None:
    g.rgba = [0.0, 0.0, 0.0, 0.0]  # invisible
    g.group = 3  # hidden by renderer, hit by mj_ray
    # Physical limbs block the robot (its geoms are contype/conaffinity 1) but, via contype bit 2,
    # do NOT collide with each other (no self-contact on the kinematic skeleton). Arms are
    # sensing-only (0) -- lidar still rays them.
    g.contype = 2 if physical else 0
    g.conaffinity = 1 if physical else 0


def _add_collision(body, joint: Joint, skel: Skeleton, collision) -> None:
    """Collision capsule(s) along the bones leaving ``joint`` (+ a leaf cap), sized from the CARLA
    per-bone radius (fallback: the joint's visual radius)."""
    physical = joint.name not in _ARM_JOINTS
    radius = (collision or {}).get(joint.name) or joint.radius
    for child in CHILDREN[joint.name]:
        g = body.add_geom()
        g.type = mujoco.mjtGeom.mjGEOM_CAPSULE
        off = skel.offset(child)
        g.fromto = [0.0, 0.0, 0.0, off[0], off[1], off[2]]
        g.size = [radius, 0.0, 0.0]
        _collision_flags(g, physical)
    term = _TERMINALS.get(joint.name)  # leaf: head sphere / foot tip
    if term:
        kind, (r0, p1) = term
        r = (collision or {}).get(joint.name) or r0
        g = body.add_geom()
        if kind == "sphere":
            g.type = mujoco.mjtGeom.mjGEOM_SPHERE
            g.size = [r, 0.0, 0.0]
            g.pos = list(p1)
        else:  # foot tip capsule
            g.type = mujoco.mjtGeom.mjGEOM_CAPSULE
            g.fromto = [0.0, 0.0, 0.0, p1[0], p1[1], p1[2]]
            g.size = [r, 0.0, 0.0]
        _collision_flags(g, physical)


def _visual(g, rgba) -> None:
    g.rgba = list(rgba)
    g.contype = 0
    g.conaffinity = 0
    g.group = 2


def _add_capsule(body, p0, p1, radius: float, rgba) -> None:
    g = body.add_geom()
    g.type = mujoco.mjtGeom.mjGEOM_CAPSULE
    g.fromto = [p0[0], p0[1], p0[2], p1[0], p1[1], p1[2]]
    g.size = [radius, 0.0, 0.0]
    _visual(g, rgba)


def _add_terminal(body, term, rgba) -> None:
    kind, params = term
    if kind == "sphere":
        radius, pos = params
        g = body.add_geom()
        g.type = mujoco.mjtGeom.mjGEOM_SPHERE
        g.size = [radius, 0.0, 0.0]
        g.pos = list(pos)
        _visual(g, rgba)
    elif kind == "foot":
        radius, tip = params
        _add_capsule(body, (0.0, 0.0, 0.0), tip, radius, rgba)


# -- runtime forward kinematics ---------------------------------------------------------------
def forward_kinematics(root_xyz, yaw: float, joint_rot: dict, skeleton=None) -> dict:
    """World ``(pos, quat)`` for every joint body.

    ``root_xyz`` is the pelvis world position (x, y, root_height + bob); ``yaw`` is the nav heading;
    ``joint_rot`` maps joint name -> local quat (w, x, y, z) from the clip; ``skeleton`` supplies the
    per-walker bone offsets (default = the shared table). Returns ``{name: (pos[3], quat[4])}``.
    """
    skel = skeleton or DEFAULT_SKELETON
    out: dict = {}
    root_xyz = np.asarray(root_xyz, dtype=float)
    pelvis_q = quat_mul(quat_yaw(yaw), joint_rot["pelvis"])
    out["pelvis"] = (root_xyz, pelvis_q)
    for j in SKELETON[1:]:  # parents precede children
        pp, pq = out[j.parent]
        pos = pp + quat_rotate(pq, np.array(skel.offset(j.name)))
        out[j.name] = (pos, quat_mul(pq, joint_rot[j.name]))
    return out
