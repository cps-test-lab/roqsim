# SPDX-License-Identifier: Apache-2.0
"""A hand: five finger chains of three bones on a wrist, named poses, blended, and a grip that
closes to the shape of what it holds.

The fingers hang from the wrist in its frame (the forearm's line is the wrist's -z, the arms hanging
at rest): the four fingers side by side down from the palm, the thumb in front of the palm and
turned toward it. Each finger flexes toward the palm about one axis; the thumb about its own. A pose
is a curl (rad) per finger bone; :class:`HandState` blends the current curls toward a pose's at a
bounded rate, and :func:`finger_fk` turns curls into world poses for all fifteen bones of a hand in
one batched pass.

**The grip frame** is where a hand holds: at the palm's centre, +x out of the palm (toward what it
holds), +z along the fingers, +y = z × x. An object's grip site states the same frame, so holding is
making the two frames one. For the right hand +y is the thumb's side, for the left hand the little
finger's.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from . import quat as Q

FINGERS = ("thumb", "index", "middle", "ring", "little")
BONES = 3
#: The bone names of one hand, in the order :func:`finger_fk` returns them.
def bone_names(side: str) -> list[str]:
    return [f"f_{side}_{finger}_{k + 1}" for finger in FINGERS for k in range(BONES)]


#: The named poses: the curl of each finger's three bones (rad), thumb first. ``open`` is a relaxed
#: flat hand; ``power`` wraps a handle; ``pinch`` meets thumb and index; ``flat`` presses with the
#: palm; ``point`` leaves the index straight to press with.
_C = math.radians
POSES = {
    "open": [[_C(5), _C(5), _C(5)]] + [[_C(5), _C(5), _C(5)]] * 4,
    "power": [[_C(35), _C(30), _C(25)]] + [[_C(70), _C(85), _C(50)]] * 4,
    "pinch": [[_C(40), _C(30), _C(20)], [_C(45), _C(40), _C(25)], [_C(15), _C(15), _C(10)],
              [_C(15), _C(15), _C(10)], [_C(15), _C(15), _C(10)]],
    "flat": [[_C(0), _C(0), _C(0)]] * 5,
    "point": [[_C(40), _C(35), _C(20)], [_C(0), _C(0), _C(0)], [_C(80), _C(95), _C(60)],
              [_C(80), _C(95), _C(60)], [_C(80), _C(95), _C(60)]],
}
#: How fast a hand changes pose: rad/s of curl, per bone.
CURL_RATE = 6.0


@dataclass(frozen=True)
class HandLayout:
    """Where one hand's fingers are, in the wrist's frame: each finger's base, its bones' lengths
    and its flex axis; and the grip frame (position, rotation) in the wrist's frame."""

    side: str
    bases: np.ndarray  # (5, 3)
    lengths: np.ndarray  # (5, 3)
    directions: np.ndarray  # (5, 3) unit, a finger's rest direction
    axes: np.ndarray  # (5, 3) unit, a finger's flex axis
    grip_pos: np.ndarray
    grip_quat: np.ndarray


def layout(side: str, scale: float = 1.0, block: dict | None = None) -> HandLayout:
    """The default hand for ``side`` (``l``/``r``), its lengths scaled by ``scale`` (the forearm's
    length over 0.24 m). ``block`` is a blueprint's ``hands`` block, overriding ``bases``,
    ``lengths`` and ``grip`` (``[x, y, z]`` in the wrist's frame) as it states them."""
    s = 1.0 if side == "l" else -1.0
    palm = -s  # the palm faces the body: -y for the left hand, +y for the right
    block = block or {}
    bases = np.array(
        block.get(
            "bases",
            [[0.030, palm * 0.020, -0.035], [0.028, 0.0, -0.095], [0.008, 0.0, -0.100],
             [-0.012, 0.0, -0.095], [-0.030, 0.0, -0.085]],
        ),
        float,
    ) * (1.0 if "bases" in block else scale)
    lengths = np.array(
        block.get("lengths", [[0.035, 0.030, 0.025], [0.042, 0.026, 0.020], [0.046, 0.028, 0.021],
                              [0.043, 0.026, 0.020], [0.034, 0.020, 0.018]]),
        float,
    ) * (1.0 if "lengths" in block else scale)
    thumb_dir = np.array([0.55, palm * 0.35, -0.76])
    directions = np.array([thumb_dir / np.linalg.norm(thumb_dir)] + [[0.0, 0.0, -1.0]] * 4)
    # A finger curls toward the palm: about +x for the right hand (-z turns toward +y), -x for the
    # left; the thumb about the axis across its own direction and the palm's.
    finger_axis = np.array([1.0, 0.0, 0.0]) if palm > 0 else np.array([-1.0, 0.0, 0.0])
    thumb_axis = np.cross(directions[0], [0.0, palm, 0.0])
    thumb_axis /= np.linalg.norm(thumb_axis)
    axes = np.array([thumb_axis] + [finger_axis] * 4)
    grip = np.array(block.get("grip", [0.0, palm * 0.035, -0.09]), float) * (1.0 if "grip" in block else scale)
    # The grip frame: +z along the fingers (the wrist's -z), +x out of the palm (palm * y).
    z = np.array([0.0, 0.0, -1.0])
    x = np.array([0.0, palm, 0.0])
    y = np.cross(z, x)
    grip_quat = Q.from_matrix(np.column_stack([x, y, z]))
    return HandLayout(side, bases, lengths, directions, axes, grip, grip_quat)


def _axis_angles(axes: np.ndarray, angles: np.ndarray) -> np.ndarray:
    half = angles[:, None] / 2.0
    return np.hstack([np.cos(half), axes * np.sin(half)])


def _qmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a.T
    bw, bx, by, bz = b.T
    return np.stack(
        [aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
         aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw], axis=1
    )


def _qrot(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    u = q[:, 1:]
    w = q[:, :1]
    return v + 2.0 * np.cross(u, np.cross(u, v) + w * v)


def finger_fk(lay: HandLayout, wrist_pos, wrist_quat, curls: np.ndarray):
    """World positions (15, 3) and rotations (15, 4) of a hand's bones, finger-major as
    :func:`bone_names`, for ``curls`` (5, 3) rad. Each bone's body sits at its joint; its rotation
    carries the curls of its finger up to it."""
    wq = np.asarray(wrist_quat, float)
    wp = np.asarray(wrist_pos, float)
    pos = np.empty((5, BONES, 3))
    rot = np.empty((5, BONES, 4))
    q = np.repeat(wq[None], 5, axis=0)
    p = wp + _qrot(q, lay.bases)
    for k in range(BONES):
        q = _qmul(q, _axis_angles(lay.axes, curls[:, k]))
        pos[:, k] = p
        rot[:, k] = q
        p = p + _qrot(q, lay.directions * lay.lengths[:, k : k + 1])
    return pos.reshape(15, 3), rot.reshape(15, 4)


def _matrix(q) -> np.ndarray:
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


class FingerFrames:
    """A hand's fifteen bones in its wrist's frame, recomputed only when the curls change, and
    placed on a wrist with one matrix product and one batched quaternion product: what a walker
    pays per update for its fingers while their pose holds."""

    def __init__(self, lay: HandLayout):
        self.lay = lay
        self._key = None
        self._pos = self._rot = self._tips = None

    def _update(self, curls: np.ndarray) -> None:
        key = curls.tobytes()
        if key != self._key:
            ident = np.array([1.0, 0.0, 0.0, 0.0])
            self._pos, self._rot = finger_fk(self.lay, np.zeros(3), ident, curls)
            self._tips = fingertips(self.lay, np.zeros(3), ident, curls)
            self._key = key

    def tip(self, wrist_pos, wrist_quat, curls: np.ndarray, finger: int = 1) -> np.ndarray:
        """One fingertip's world position (the index, by default)."""
        self._update(curls)
        return _matrix(np.asarray(wrist_quat, float)) @ self._tips[finger] + np.asarray(wrist_pos, float)

    def world(self, wrist_pos, wrist_quat, curls: np.ndarray):
        self._update(curls)
        wq = np.asarray(wrist_quat, float)
        pos = self._pos @ _matrix(wq).T + np.asarray(wrist_pos, float)
        return pos, _qmul(np.broadcast_to(wq, self._rot.shape), self._rot)


def fingertips(lay: HandLayout, wrist_pos, wrist_quat, curls: np.ndarray) -> np.ndarray:
    """World positions (5, 3) of the five fingertips."""
    pos, rot = finger_fk(lay, wrist_pos, wrist_quat, curls)
    last = np.arange(5) * BONES + BONES - 1
    return pos[last] + _qrot(rot[last], lay.directions * lay.lengths[:, -1:])


@dataclass
class HandState:
    """One hand's pose: the curls now, the pose it is going to, and, while a grip closes, which
    fingers have met what they hold and stopped."""

    curls: np.ndarray = field(default_factory=lambda: np.array(POSES["open"], float))
    pose: str = "open"
    stopped: np.ndarray = field(default_factory=lambda: np.zeros(5, bool))

    def _goal(self) -> np.ndarray:
        if getattr(self, "_goal_for", None) != self.pose:
            self._goal_arr = np.array(POSES[self.pose], float)
            self._goal_for = self.pose
        return self._goal_arr

    def set_pose(self, name: str) -> None:
        if name not in POSES:
            raise ValueError(f"no hand pose {name!r}: the poses are {sorted(POSES)}")
        self.pose = name
        self.stopped[:] = False

    def advance(self, dt: float) -> bool:
        """Move the curls toward the pose by at most ``CURL_RATE * dt`` per bone, stopped fingers
        held; whether every finger has arrived or stopped."""
        goal = self._goal()
        if self.curls is not None and np.array_equal(self.curls, goal):
            return True
        step = CURL_RATE * max(dt, 0.0)
        delta = np.clip(goal - self.curls, -step, step)
        delta[self.stopped] = 0.0
        self.curls = self.curls + delta
        return bool(np.all(self.stopped | np.all(np.abs(goal - self.curls) < 1e-6, axis=1)))

    def close_on(self, touching) -> None:
        """Stop each finger whose tip ``touching(finger_index)`` reports at what it holds."""
        for i in range(5):
            if not self.stopped[i] and touching(i):
                self.stopped[i] = True
