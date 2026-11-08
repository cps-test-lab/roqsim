# SPDX-License-Identifier: Apache-2.0
"""A humanoid description: the joints, their rest offsets and limits, and its forward kinematics.

The motion core works on one **canonical humanoid**, whatever body it drives in the end: the
walker's seventeen joints (``pelvis`` the root, ``spine``, ``head``, ``shoulder_*``, ``elbow_*``,
``wrist_*``, ``hip_*``, ``knee_*``, ``ankle_*``, ``toe_*``), Z up, X forward, Y left, the arms
hanging down at rest. Each joint has a rest translation from its parent and a rotation the solver
sets; the hands' fingers are posed by :mod:`roqsim_motion.hands`, not by the solver. A body is
mapped onto it by a retargeting table (:mod:`roqsim_motion.retarget`), so the joint names here are
the vocabulary every table speaks.

Limits are per joint, as ``(axis order, {axis: (lo, hi)})`` on the joint's local rotation decomposed
in that order (rad). The solver clamps the joints it sets to them; the rest follow the motion clip.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from . import quat as Q

#: The canonical joints, parents before children, with the default rest offsets (m).
CANONICAL = (
    ("pelvis", None, (0.0, 0.0, 0.0)),
    ("spine", "pelvis", (0.0, 0.0, 0.22)),
    ("head", "spine", (0.0, 0.0, 0.25)),
    ("hip_l", "pelvis", (0.0, 0.11, -0.05)),
    ("knee_l", "hip_l", (0.0, 0.0, -0.42)),
    ("ankle_l", "knee_l", (0.0, 0.0, -0.42)),
    ("toe_l", "ankle_l", (0.10, 0.0, -0.03)),
    ("hip_r", "pelvis", (0.0, -0.11, -0.05)),
    ("knee_r", "hip_r", (0.0, 0.0, -0.42)),
    ("ankle_r", "knee_r", (0.0, 0.0, -0.42)),
    ("toe_r", "ankle_r", (0.10, 0.0, -0.03)),
    ("shoulder_l", "spine", (0.0, 0.19, 0.12)),
    ("elbow_l", "shoulder_l", (0.0, 0.0, -0.26)),
    ("wrist_l", "elbow_l", (0.0, 0.0, -0.24)),
    ("shoulder_r", "spine", (0.0, -0.19, 0.12)),
    ("elbow_r", "shoulder_r", (0.0, 0.0, -0.26)),
    ("wrist_r", "elbow_r", (0.0, 0.0, -0.24)),
)
JOINTS = tuple(j[0] for j in CANONICAL)
SIDES = {"l": 1.0, "r": -1.0}

_D = math.radians
#: Limits of the joints the solver sets, on the local rotation decomposed in the given order. The
#: wrist's z is the forearm's twist (pronation), y its flexion, x its deviation.
LIMITS = {
    "spine": ("zxy", {"z": (_D(-60), _D(60)), "x": (_D(-35), _D(35)), "y": (_D(-30), _D(90))}),
    "head": ("zyx", {"z": (_D(-70), _D(70)), "y": (_D(-40), _D(50)), "x": (_D(-30), _D(30))}),
    "wrist_l": ("zyx", {"z": (_D(-100), _D(100)), "y": (_D(-80), _D(80)), "x": (_D(-45), _D(45))}),
    "wrist_r": ("zyx", {"z": (_D(-100), _D(100)), "y": (_D(-80), _D(80)), "x": (_D(-45), _D(45))}),
}


@dataclass
class Description:
    """A humanoid's joints and rest offsets (``offsets`` over the canonical defaults)."""

    offsets: dict = field(default_factory=dict)
    root_height: float = 0.93
    limits: dict = field(default_factory=lambda: dict(LIMITS))

    def __post_init__(self):
        self.parent = {n: p for n, p, _ in CANONICAL}
        self.children: dict[str, list[str]] = defaultdict(list)
        for n, p, _ in CANONICAL:
            if p:
                self.children[p].append(n)
        self._off = {n: np.asarray(self.offsets.get(n, o), float) for n, _, o in CANONICAL}

    def offset(self, name: str) -> np.ndarray:
        return self._off[name]

    def bone(self, name: str) -> float:
        """The length of the bone ending at ``name`` (m)."""
        return float(np.linalg.norm(self._off[name]))

    def arm_length(self, side: str) -> float:
        return self.bone(f"elbow_{side}") + self.bone(f"wrist_{side}")

    def fk(self, root_pos, root_quat, local: dict) -> dict:
        """World ``(pos, quat)`` of every joint; ``local`` maps a joint to its local rotation (a
        missing one is the rest rotation). ``root_quat`` is the pelvis's frame before its own local
        rotation (the heading)."""
        out = {}
        for name, parent, _ in CANONICAL:
            rot = local.get(name, Q.IDENTITY)
            if parent is None:
                out[name] = (np.asarray(root_pos, float), Q.mul(root_quat, rot))
                continue
            pp, pq = out[parent]
            out[name] = (pp + Q.rotate(pq, self._off[name]), Q.mul(pq, rot))
        return out

    def clamp(self, name: str, rot) -> np.ndarray:
        """``rot`` with each of its angles brought inside ``name``'s limits; unchanged for a joint
        without limits."""
        if name not in self.limits:
            return np.asarray(rot, float)
        order, bounds = self.limits[name]
        angles = Q.euler(rot, order)
        clamped = [
            min(max(a, bounds.get(ax, (-math.inf, math.inf))[0]), bounds.get(ax, (-math.inf, math.inf))[1])
            for ax, a in zip(order, angles, strict=True)
        ]
        return Q.from_euler(order, clamped)

    def within(self, name: str, rot, tol: float = 1e-6) -> bool:
        if name not in self.limits:
            return True
        order, bounds = self.limits[name]
        return all(
            bounds.get(ax, (-math.inf, math.inf))[0] - tol <= a <= bounds.get(ax, (-math.inf, math.inf))[1] + tol
            for ax, a in zip(order, Q.euler(rot, order), strict=True)
        )
