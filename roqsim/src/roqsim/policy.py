"""A policy's observation layout and control parameters, as data beside its checkpoint.

Why this exists: ``g1.yaml`` (flat) and ``oli/walk_param.yaml`` (nested under ``HumanoidRobotCfg``) are
two mutually incompatible schemas read by two hand-written loaders, and each observation vector is
assembled by hand in its plugin. A third policy would mean a third loader and a third hand-assembled
observation -- and an observation that disagrees with the checkpoint it feeds fails *silently*: the robot
does not error, it twitches. So a policy declares its own layout here, in one place, and the builder
below is the only thing that assembles it.

**In core, and family-agnostic on purpose.** Every policy-driven robot here needs this -- the humanoids in
``roqsim_humanoid`` and Spot in ``roqsim_quadruped`` alike -- so scoping it to one family would force the other
to depend sideways on it. It costs core nothing to own: this module describes checkpoints, it never *loads*
them, so it imports only ``numpy`` and ``yaml``, both already core dependencies. Checkpoint loading stays
in the family plugins, which is where ``torch``/``onnxruntime`` belong and where core must not follow.

Deliberately small, and deliberately not a framework: no registry, no plugin group, no base class to
inherit. A dataclass, a term table, and one builder (see architecture.rst §9, on a generic error-model
framework that was deleted for adding indirection without demonstrated reuse -- the guard against this
growing into one).

A spec is a YAML file next to the checkpoint::

    name: g1_stand
    checkpoint: stand.pt          # relative to the spec file
    joints:
      actuated: [left_hip_pitch_joint, ...]   # policy outputs, in policy order
      observed: [left_shoulder_pitch_joint, ...]  # in the observation, NOT commanded
    control:
      decimation: 10
      action_scale: 0.25          # scalar, or one value per actuated joint
      default_angles: [...]
      kp: [...]
      kd: [...]
    observation:                  # ordered; the list IS the layout
      - {term: base_ang_vel, scale: 0.25}
      - {term: projected_gravity}
      - {term: command, scale: [2.0, 2.0, 0.25]}
      - {term: actuated_pos_rel_default, scale: 1.0}
      - {term: actuated_vel, scale: 0.05}
      - {term: prev_action}
      - {term: observed_pos}
      - {term: observed_vel, scale: 0.05}
    envelope:                     # optional; what the policy was TRAINED for
      payload_kg: [0.0, 1.0]
      payload_frame: left_grasp
      note: table-height grasping only; the legs never crouch

``envelope`` is recorded rather than enforced here: a policy trained on a bounded envelope does not fail
outside it, it just balances worse, so the ranges have to be *written down somewhere machine-readable*
for a caller to check against. :func:`PolicySpec.check_envelope` is that check; whether to warn or raise
is the caller's call, since only the caller knows what the world contains.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml


def gravity_orientation(quat) -> np.ndarray:
    """Projected-gravity direction in the base frame from a ``(w, x, y, z)`` quaternion.

    Verbatim from unitree_rl_gym's ``deploy_mujoco.get_gravity_orientation`` -- the form the pretrained
    G1 policy was trained against. It is algebraically ``R^T @ [0, 0, -1]`` for a unit quaternion, so it
    is the standard projected gravity and not a Unitree-specific convention.
    """
    qw, qx, qy, qz = quat
    return np.array(
        [
            2 * (-qz * qx + qw * qy),
            -2 * (qz * qy + qw * qx),
            1 - 2 * (qw * qw + qz * qz),
        ],
        dtype=np.float32,
    )


@dataclass(frozen=True)
class Envelope:
    """What a policy was trained for. Recorded so a caller can check, not silently assumed."""

    payload_kg: tuple[float, float] | None = None
    payload_frame: str = ""
    note: str = ""

    def check_payload(self, mass_kg: float) -> str:
        """Return a human-readable complaint if *mass_kg* is outside the trained range, else ""."""
        if self.payload_kg is None:
            return ""
        lo, hi = self.payload_kg
        if not lo <= mass_kg <= hi:
            return (
                f"payload {mass_kg} kg is outside the policy's trained range [{lo}, {hi}] kg. The "
                f"policy will not error -- it will simply balance worse. {self.note}".strip()
            )
        return ""


@dataclass(frozen=True)
class PolicySpec:
    name: str
    checkpoint: Path
    actuated: tuple[str, ...]
    observed: tuple[str, ...]
    decimation: int
    action_scale: np.ndarray
    default_angles: np.ndarray
    kp: np.ndarray
    kd: np.ndarray
    terms: tuple[dict, ...]
    envelope: Envelope = field(default_factory=Envelope)

    @property
    def num_obs(self) -> int:
        """Observation width implied by the term list -- the number the checkpoint must expect."""
        return sum(_term_width(t, self) for t in self.terms)

    @classmethod
    def from_yaml(cls, path: str | Path) -> PolicySpec:
        path = Path(path)
        raw = yaml.safe_load(path.read_text()) or {}
        joints = raw.get("joints") or {}
        control = raw.get("control") or {}
        actuated = tuple(joints.get("actuated") or ())
        if not actuated:
            raise ValueError(f"{path}: joints.actuated is required and must be non-empty")
        terms = tuple(dict(t) for t in (raw.get("observation") or ()))
        if not terms:
            raise ValueError(f"{path}: observation must list at least one term")
        unknown = sorted({t.get("term") for t in terms} - set(_TERMS))
        if unknown:
            raise ValueError(
                f"{path}: unknown observation term(s) {unknown}; known terms: {sorted(_TERMS)}"
            )

        n = len(actuated)
        env_raw = raw.get("envelope") or {}
        payload = env_raw.get("payload_kg")
        return cls(
            name=str(raw.get("name") or path.stem),
            checkpoint=(path.parent / str(raw["checkpoint"])).resolve(),
            actuated=actuated,
            observed=tuple(joints.get("observed") or ()),
            decimation=int(control.get("decimation", 10)),
            action_scale=_as_vec(control.get("action_scale", 1.0), n, "action_scale", path),
            default_angles=_as_vec(control.get("default_angles", 0.0), n, "default_angles", path),
            kp=_as_vec(control.get("kp", 0.0), n, "kp", path),
            kd=_as_vec(control.get("kd", 0.0), n, "kd", path),
            terms=terms,
            envelope=Envelope(
                payload_kg=(float(payload[0]), float(payload[1])) if payload else None,
                payload_frame=str(env_raw.get("payload_frame", "")),
                note=str(env_raw.get("note", "")),
            ),
        )

    def build_observation(
        self, state: ObservationState, out: np.ndarray | None = None
    ) -> np.ndarray:
        """Assemble the observation in declared term order.

        Writes into *out* when given (the caller preallocates once, as the plugins do, so there is no
        per-tick allocation on the physics thread).
        """
        if out is None:
            out = np.zeros(self.num_obs, dtype=np.float32)
        i = 0
        for term in self.terms:
            values = _TERMS[term["term"]](state, self)
            scale = term.get("scale")
            if scale is not None:
                values = values * np.asarray(scale, dtype=np.float32)
            width = len(values)
            out[i : i + width] = values
            i += width
        return out


@dataclass
class ObservationState:
    """Everything the terms can read, gathered once per policy tick by the caller.

    A plain data carrier on purpose: the terms stay pure functions of it, so the builder can be tested
    against recorded numbers without a simulator.
    """

    base_ang_vel: np.ndarray  # body-frame angular velocity (3)
    base_quat: np.ndarray  # (w, x, y, z)
    command: np.ndarray  # (3) body-frame [vx, vy, yaw_rate]
    actuated_pos: np.ndarray
    actuated_vel: np.ndarray
    prev_action: np.ndarray
    # Optional, because which of these a policy needs is exactly what differs between policies: Spot's
    # Isaac policy observes base linear velocity and the Unitree humanoid ones do not; only a walking
    # policy has a gait phase; only a policy watching an arm it does not command has observed joints.
    base_lin_vel: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    observed_pos: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    observed_vel: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    phase: float = 0.0  # gait phase in [0, 1)


def _as_vec(value, n: int, what: str, path) -> np.ndarray:
    """A scalar broadcast to *n*, or a length-*n* list. Per-joint arrays are needed (Oli uses them)."""
    arr = (
        np.full(n, float(value), dtype=np.float32)
        if np.isscalar(value)
        else np.asarray(value, dtype=np.float32)
    )
    if arr.shape != (n,):
        raise ValueError(f"{path}: control.{what} has {arr.shape[0]} values, expected {n}")
    return arr


def _term_width(term: dict, spec: PolicySpec) -> int:
    return _WIDTHS[term["term"]](spec)


# Observation terms. Each is a pure function of the gathered state; the width function lets `num_obs`
# be known before any simulation exists, which is what makes a spec checkable at load.
_TERMS = {
    # Body-frame linear velocity. Quadruped policies trained in Isaac Lab observe it (Spot's does, as
    # its first three dims); the Unitree humanoid policies do not. Both shapes are just term lists.
    "base_lin_vel": lambda s, spec: np.asarray(s.base_lin_vel, dtype=np.float32),
    "base_ang_vel": lambda s, spec: np.asarray(s.base_ang_vel, dtype=np.float32),
    "projected_gravity": lambda s, spec: gravity_orientation(s.base_quat),
    "command": lambda s, spec: np.asarray(s.command, dtype=np.float32),
    "actuated_pos_rel_default": lambda s, spec: (
        np.asarray(s.actuated_pos, dtype=np.float32) - spec.default_angles
    ),
    "actuated_vel": lambda s, spec: np.asarray(s.actuated_vel, dtype=np.float32),
    "prev_action": lambda s, spec: np.asarray(s.prev_action, dtype=np.float32),
    "observed_pos": lambda s, spec: np.asarray(s.observed_pos, dtype=np.float32),
    "observed_vel": lambda s, spec: np.asarray(s.observed_vel, dtype=np.float32),
    # sin/cos of the gait phase. A walking policy needs it; a stand policy must NOT declare it -- the
    # phase is what keeps that policy stepping (measured: 27 foot-lifts / 10 s at zero command).
    "gait_phase": lambda s, spec: np.array(
        [math.sin(2 * math.pi * s.phase), math.cos(2 * math.pi * s.phase)], dtype=np.float32
    ),
}

_WIDTHS = {
    "base_lin_vel": lambda spec: 3,
    "base_ang_vel": lambda spec: 3,
    "projected_gravity": lambda spec: 3,
    "command": lambda spec: 3,
    "actuated_pos_rel_default": lambda spec: len(spec.actuated),
    "actuated_vel": lambda spec: len(spec.actuated),
    "prev_action": lambda spec: len(spec.actuated),
    "observed_pos": lambda spec: len(spec.observed),
    "observed_vel": lambda spec: len(spec.observed),
    "gait_phase": lambda spec: 2,
}
