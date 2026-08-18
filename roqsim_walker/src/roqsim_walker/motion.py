"""Motion clips for the mocap-driven human + a procedural bootstrap.

Ported from our earlier in-house nav prototype's ``mujoco_nav.motion``.

A :class:`Clip` is the compact runtime format the CARLA/BVH converters write and the walker player
samples: per-joint **local** rotations over time, with the root's XY + heading stripped out (those
come from the nav stack). Root vertical bob is kept. The joint order matches
:data:`roqsim_walker.humanoid.JOINT_NAMES`.

:func:`procedural_walk` / :func:`procedural_idle` synthesise small clips in the same format so a
walker animates out-of-the-box when a ``.npz`` clip is missing.
"""

from __future__ import annotations

import math

import numpy as np

from roqsim_walker.humanoid import JOINT_NAMES

_J = len(JOINT_NAMES)
_IDX = {n: i for i, n in enumerate(JOINT_NAMES)}


def _pitch(angle: float) -> np.ndarray:
    """Quaternion (w,x,y,z) for a rotation about +Y (leg/arm swing axis)."""
    h = angle / 2.0
    return np.array([math.cos(h), 0.0, math.sin(h), 0.0])


def _roll(angle: float) -> np.ndarray:
    """Quaternion about +X (small lateral sway)."""
    h = angle / 2.0
    return np.array([math.cos(h), math.sin(h), 0.0, 0.0])


class Clip:
    """Per-joint local rotations over time + root vertical bob."""

    def __init__(self, joint_rot, root_z, fps=30.0, stride_len=0.75, loop=True, joint_names=None):
        self.joint_rot = np.asarray(joint_rot, dtype=float)  # (T, J, 4)
        self.root_z = np.asarray(root_z, dtype=float)  # (T,)
        self.fps = float(fps)
        self.stride_len = float(stride_len)
        self.loop = bool(loop)
        self.joint_names = list(joint_names) if joint_names is not None else list(JOINT_NAMES)
        self.num_frames = self.joint_rot.shape[0]

    @property
    def duration(self) -> float:
        return self.num_frames / self.fps

    @classmethod
    def load(cls, path) -> Clip:
        d = np.load(path, allow_pickle=False)
        names = [str(n) for n in d["joint_names"]] if "joint_names" in d else None
        clip = cls(
            d["joint_rot"],
            d["root_z"],
            fps=float(d["fps"]),
            stride_len=float(d["stride_len"]),
            loop=bool(d["loop"]),
            joint_names=names,
        )
        return clip._reordered()

    def save(self, path) -> None:
        np.savez(
            path,
            joint_rot=self.joint_rot,
            root_z=self.root_z,
            fps=self.fps,
            stride_len=self.stride_len,
            loop=self.loop,
            joint_names=np.array(self.joint_names),
        )

    def _reordered(self) -> Clip:
        """Permute clip joints into humanoid JOINT_NAMES order (missing -> identity)."""
        if self.joint_names == list(JOINT_NAMES):
            return self
        rot = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (self.num_frames, _J, 1))
        src = {n: i for i, n in enumerate(self.joint_names)}
        for j, name in enumerate(JOINT_NAMES):
            if name in src:
                rot[:, j, :] = self.joint_rot[:, src[name], :]
        self.joint_rot, self.joint_names = rot, list(JOINT_NAMES)
        return self

    def sample_array(self, phase: float):
        """``(joint_rot (J,4), root_z)`` at fractional ``phase`` in [0, 1)."""
        t = (phase % 1.0) * self.num_frames
        i0 = int(math.floor(t)) % self.num_frames
        i1 = (i0 + 1) % self.num_frames if self.loop else min(i0 + 1, self.num_frames - 1)
        frac = t - math.floor(t)
        q = _nlerp(self.joint_rot[i0], self.joint_rot[i1], frac)  # (J, 4)
        z = (1 - frac) * self.root_z[i0] + frac * self.root_z[i1]
        return q, float(z)

    def sample(self, phase: float):
        """``(joint_rot dict, root_z)`` at fractional ``phase`` in [0, 1)."""
        q, z = self.sample_array(phase)
        return {name: q[_IDX[name]] for name in JOINT_NAMES}, z


def _nlerp(q0, q1, t):
    """Per-row normalised lerp of quaternions (q0, q1: (J, 4))."""
    q1 = np.where((np.sum(q0 * q1, axis=1, keepdims=True) < 0), -q1, q1)
    q = (1.0 - t) * q0 + t * q1
    return q / np.linalg.norm(q, axis=1, keepdims=True)


def blend_quats(qa, qb, w):
    """Blend two pose arrays ((J,4)) by weight ``w`` (0 -> qa, 1 -> qb)."""
    return _nlerp(qa, qb, float(w))


def smoothstep(x, lo, hi) -> float:
    """0 below ``lo``, 1 above ``hi``, smooth Hermite ease between."""
    if hi <= lo:
        return 1.0 if x >= hi else 0.0
    t = min(1.0, max(0.0, (x - lo) / (hi - lo)))
    return t * t * (3.0 - 2.0 * t)


# -- procedural bootstrap clips (same format; replaced by real mocap) --------------------------
def procedural_walk(frames=30, fps=30.0, stride_len=0.75) -> Clip:
    """A simple forward alternating gait. Sign conventions in our X-forward, Z-up frame: flexing a
    downward limb *forward* is a NEGATIVE pitch about +Y; a knee flexes the shin *backward*
    (POSITIVE pitch) and only during that leg's forward swing. Left/right are half a cycle apart;
    arms oppose the legs."""
    rot = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (frames, _J, 1))
    root_z = np.zeros(frames)
    a_hip, a_knee, a_arm = 0.5, 1.0, 0.4
    for f in range(frames):
        ph = f / frames  # one full gait cycle (two steps)
        tau = 2.0 * math.pi * ph
        left = math.cos(tau)  # +1 = left leg forward, -1 = back
        right = math.cos(tau + math.pi)
        _set(rot, f, "hip_l", _pitch(-a_hip * left))  # thigh forward (flex)
        _set(rot, f, "hip_r", _pitch(-a_hip * right))
        # knee flexes mid-swing: left swings forward over ph in (0.5, 1.0).
        _set(rot, f, "knee_l", _pitch(a_knee * max(0.0, math.sin(2 * math.pi * (ph - 0.5)))))
        _set(rot, f, "knee_r", _pitch(a_knee * max(0.0, math.sin(tau))))
        _set(rot, f, "shoulder_l", _pitch(a_arm * left))  # arm opposes same-side leg
        _set(rot, f, "shoulder_r", _pitch(a_arm * right))
        _set(rot, f, "elbow_l", _pitch(-0.25))
        _set(rot, f, "elbow_r", _pitch(-0.25))
        root_z[f] = -0.03 * abs(math.sin(tau))  # dip on each foot-plant
    return Clip(rot, root_z, fps=fps, stride_len=stride_len, loop=True)


def procedural_idle(frames=60, fps=30.0) -> Clip:
    """A gentle standing idle: tiny sway + breathing bob."""
    rot = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (frames, _J, 1))
    root_z = np.zeros(frames)
    for f in range(frames):
        th = 2.0 * math.pi * f / frames
        _set(rot, f, "pelvis", _roll(0.02 * math.sin(th)))
        _set(rot, f, "elbow_l", _pitch(-0.15))
        _set(rot, f, "elbow_r", _pitch(-0.15))
        root_z[f] = 0.01 * math.sin(th)
    return Clip(rot, root_z, fps=fps, stride_len=1.0, loop=True)


def _set(rot, f, name, quat) -> None:
    rot[f, _IDX[name], :] = quat
