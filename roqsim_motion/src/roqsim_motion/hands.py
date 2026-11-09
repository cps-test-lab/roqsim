# SPDX-License-Identifier: Apache-2.0
"""A hand: five finger chains of three bones on a wrist, named poses, blended, and a grip that
closes to the shape of what it holds.

The fingers hang from the wrist in its frame (the forearm's line is the wrist's -z, the arms hanging
at rest): the four fingers side by side down from the palm, the thumb on the +x side of the palm.
Each bone turns about its own axis: a finger's three bones flex toward the palm; the thumb's first
bone opposes it -- turns it from beside the palm to in front of it -- and its other two flex. A pose
is an angle (rad) per finger bone; :class:`HandState` eases the current angles toward a pose's, each
finger over its own time and a grip's finger by finger, and :func:`finger_fk` turns angles into world
poses for all fifteen bones of a hand in one batched pass.

A hand is either the default one (:func:`layout`, scaled to the forearm) or one measured on a
character's mesh (:func:`make_layout` with the mesh's own bases, lengths, directions and palm), so
that the bones turn the skin about its own knuckles.

**The grip frame** is where a hand holds: in front of the palm, +x out of the palm (toward what it
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


#: The named poses: each finger's three angles (rad), thumb first; the thumb's first is how far it
#: is opposed, the rest flex. ``open`` is a relaxed hand, the fingers slightly curled and the thumb
#: alongside; ``power`` wraps a handle, the thumb over the fingers; ``pinch`` meets thumb and index;
#: ``flat`` presses with the palm; ``point`` leaves the index straight to press with, the others
#: curled under the thumb.
_C = math.radians
POSES = {
    "open": [
        [_C(35), _C(10), _C(5)],
        [_C(8), _C(12), _C(8)],
        [_C(10), _C(14), _C(9)],
        [_C(12), _C(16), _C(10)],
        [_C(14), _C(18), _C(12)],
    ],
    "power": [[_C(65), _C(15), _C(10)]] + [[_C(75), _C(95), _C(55)]] * 4,
    "pinch": [
        [_C(70), _C(20), _C(5)],
        [_C(40), _C(50), _C(25)],
        [_C(45), _C(60), _C(35)],
        [_C(50), _C(65), _C(40)],
        [_C(55), _C(70), _C(45)],
    ],
    "flat": [[_C(10), _C(0), _C(0)]] + [[_C(0), _C(0), _C(0)]] * 4,
    "point": [
        [_C(75), _C(15), _C(10)],
        [_C(0), _C(0), _C(0)],
        [_C(80), _C(100), _C(60)],
        [_C(85), _C(100), _C(60)],
        [_C(85), _C(100), _C(60)],
    ],
}
#: How fast a hand changes pose: rad/s of angle per bone, at most.
CURL_RATE = 6.0
#: When each finger starts toward a pose (s), thumb first in the list: a grip closes from the index
#: to the little finger and the thumb comes over last; the others go together.
STAGGER = {
    "power": [0.15, 0.0, 0.04, 0.08, 0.12],
    "point": [0.10, 0.0, 0.0, 0.03, 0.06],
    "pinch": [0.0, 0.0, 0.03, 0.06, 0.09],
}


@dataclass(frozen=True)
class HandLayout:
    """Where one hand's fingers are, in the wrist's frame: each finger's base, its bones' lengths,
    its rest direction and each bone's axis; the palm's normal; and the grip frame (position,
    rotation) in the wrist's frame."""

    side: str
    bases: np.ndarray  # (5, 3)
    lengths: np.ndarray  # (5, 3)
    directions: np.ndarray  # (5, 3) unit, a finger's rest direction
    axes: np.ndarray  # (5, 3, 3) unit, the axis each bone turns about (in its parent bone's frame)
    grip_pos: np.ndarray
    grip_quat: np.ndarray
    palm: np.ndarray = field(default_factory=lambda: np.zeros(3))  # unit, out of the palm


def _unit(v) -> np.ndarray:
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def make_layout(side: str, bases, lengths, directions, palm, grip_pos) -> HandLayout:
    """A hand from its geometry in the wrist's frame: the fingers' ``bases`` (5, 3), bone
    ``lengths`` (5, 3) and rest ``directions`` (5, 3), thumb first; the ``palm``'s normal; and the
    grip frame's origin ``grip_pos``. The axes follow: each finger bone flexes toward the palm
    (about its direction × the palm's normal); the thumb's first bone opposes it, about the axis
    halfway between its line and the palm's normal, and its other two flex. The grip frame's +x is the palm's normal, its +z the fingers' mean direction."""
    bases = np.asarray(bases, float)
    lengths = np.asarray(lengths, float)
    directions = np.array([_unit(d) for d in directions])
    n = _unit(palm)
    axes = np.empty((5, BONES, 3))
    for f in range(5):
        axes[f, :] = _unit(np.cross(directions[f], n))
    along = _unit(np.mean(directions[1:], axis=0))
    # The thumb's base joint opposes it about the axis halfway between its own line and the palm's
    # normal: one turn that swings it in front of the palm and turns its pad toward the fingers.
    # (Its sense mirrors with the hand, as a flexion's does: the left hand's turn is the right's in
    # a mirror.)
    mirror = -np.sign(np.dot(np.cross(directions[0], n), along)) or 1.0
    axes[0, 0] = mirror * _unit(directions[0] + n)
    z = _unit(along - np.dot(along, n) * n)
    x = n
    y = np.cross(z, x)
    grip_quat = Q.from_matrix(np.column_stack([x, y, z]))
    return HandLayout(
        side, bases, lengths, directions, axes, np.asarray(grip_pos, float), grip_quat, n
    )


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
            [
                [0.030, palm * 0.020, -0.035],
                [0.028, 0.0, -0.095],
                [0.008, 0.0, -0.100],
                [-0.012, 0.0, -0.095],
                [-0.030, 0.0, -0.085],
            ],
        ),
        float,
    ) * (1.0 if "bases" in block else scale)
    lengths = np.array(
        block.get(
            "lengths",
            [
                [0.035, 0.030, 0.025],
                [0.042, 0.026, 0.020],
                [0.046, 0.028, 0.021],
                [0.043, 0.026, 0.020],
                [0.034, 0.020, 0.018],
            ],
        ),
        float,
    ) * (1.0 if "lengths" in block else scale)
    # the thumb beside the palm, spread from the fingers in the palm's plane
    directions = np.array([[0.6, 0.0, -0.8]] + [[0.0, 0.0, -1.0]] * 4)
    grip = np.array(block.get("grip", [0.0, palm * 0.035, -0.09]), float) * (
        1.0 if "grip" in block else scale
    )
    return make_layout(side, bases, lengths, directions, [0.0, palm, 0.0], grip)


def _axis_angles(axes: np.ndarray, angles: np.ndarray) -> np.ndarray:
    half = angles[:, None] / 2.0
    return np.hstack([np.cos(half), axes * np.sin(half)])


def _qmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a.T
    bw, bx, by, bz = b.T
    return np.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        axis=1,
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
        q = _qmul(q, _axis_angles(lay.axes[:, k], curls[:, k]))
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
        return _matrix(np.asarray(wrist_quat, float)) @ self._tips[finger] + np.asarray(
            wrist_pos, float
        )

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


def _smoothstep(u):
    return u * u * (3.0 - 2.0 * u)


@dataclass
class HandState:
    """One hand's pose: the angles now, the pose it is going to, and, while a grip closes, which
    fingers have met what they hold and stopped.

    A pose change eases each finger from where it was to the pose's angles (smoothstep), all of its
    bones arriving together, over the time its largest change takes at :data:`CURL_RATE` at most;
    each finger starts after its :data:`STAGGER` delay. A change of pose part-way starts from
    where the fingers are, so a hand never jumps."""

    curls: np.ndarray = field(default_factory=lambda: np.array(POSES["open"], float))
    pose: str = "open"
    stopped: np.ndarray = field(default_factory=lambda: np.zeros(5, bool))

    def __post_init__(self):
        self._goal = np.array(POSES[self.pose], float)
        self._from = self.curls.copy()
        self._t = 0.0
        self._delay = np.zeros(5)
        self._span = np.zeros(5)

    def set_pose(self, name: str) -> None:
        if name not in POSES:
            raise ValueError(f"no hand pose {name!r}: the poses are {sorted(POSES)}")
        self.pose = name
        self.stopped[:] = False
        self._goal = np.array(POSES[name], float)
        self._from = self.curls.copy()
        self._t = 0.0
        self._delay = np.array(STAGGER.get(name, [0.0] * 5), float)
        # smoothstep's peak rate is 1.5 times its mean: the span keeps the peak at CURL_RATE
        self._span = 1.5 * np.max(np.abs(self._goal - self._from), axis=1) / CURL_RATE

    def advance(self, dt: float) -> bool:
        """Move the angles on by ``dt`` toward the pose, stopped fingers held; whether every finger
        has arrived or stopped."""
        self._t += max(dt, 0.0)
        live = ~self.stopped
        u = np.ones(5)
        moving = self._span > 0.0
        u[moving] = np.clip((self._t - self._delay[moving]) / self._span[moving], 0.0, 1.0)
        s = _smoothstep(u)[:, None]
        new = self._from + s * (self._goal - self._from)
        self.curls = np.where(live[:, None], new, self.curls)
        return bool(np.all(self.stopped | (u >= 1.0)))

    def close_on(self, touching) -> None:
        """Stop each finger whose tip ``touching(finger_index)`` reports at what it holds."""
        for i in range(5):
            if not self.stopped[i] and touching(i):
                self.stopped[i] = True
