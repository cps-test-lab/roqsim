"""Core's policy spec reaches the quadruped too -- proven against Spot, not asserted.

`roqsim.policy` claims to be family-agnostic. The only way to mean that is to show it describes a policy
from a *different* family than the one it was written for: Spot's pretrained Isaac policy, whose
observation is 48-dim, starts with base linear velocity (the humanoid policies have no such term), and
computes projected gravity through Isaac Lab's `quat_rotate_inverse` rather than Unitree's
`get_gravity_orientation`.

`spot_locomotion` is deliberately NOT migrated onto the spec. It works and it is covered; this test gets
the proof that the format is general without touching a working control path.
"""

from __future__ import annotations

import numpy as np
import pytest
import yaml

from roqsim.policy import ObservationState, PolicySpec, gravity_orientation
from roqsim_quadruped.plugins.spot_locomotion import _GRAVITY_W, _quat_rotate_inverse
from roqsim_quadruped.policy import POLICY_DIR

DEPLOY = yaml.safe_load((POLICY_DIR / "spot.yaml").read_text())
JOINTS = list(DEPLOY["joint_order"])


def _spot_spec(tmp_path) -> PolicySpec:
    (tmp_path / "spot_policy.pt").write_bytes(b"")  # only the path is resolved, never loaded
    path = tmp_path / "spot.spec.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "name": "spot_walk",
                "checkpoint": "spot_policy.pt",
                "joints": {"actuated": JOINTS},
                "control": {
                    "decimation": DEPLOY["control_decimation"],
                    "action_scale": DEPLOY["action_scale"],
                    "default_angles": DEPLOY["default_angles"],
                },
                "observation": [
                    # Spot's order, from the Isaac flat env: linear velocity FIRST.
                    {"term": "base_lin_vel", "scale": DEPLOY["lin_vel_scale"]},
                    {"term": "base_ang_vel", "scale": DEPLOY["ang_vel_scale"]},
                    {"term": "projected_gravity"},
                    {"term": "command", "scale": DEPLOY["cmd_scale"]},
                    {"term": "actuated_pos_rel_default", "scale": DEPLOY["dof_pos_scale"]},
                    {"term": "actuated_vel", "scale": DEPLOY["dof_vel_scale"]},
                    {"term": "prev_action"},
                ],
            }
        )
    )
    return PolicySpec.from_yaml(path)


def test_projected_gravity_is_the_same_function_in_both_families(tmp_path):
    """Unitree's `get_gravity_orientation` and Isaac Lab's `quat_rotate_inverse(q, -z)` agree exactly.

    This is why ONE `projected_gravity` term serves both a humanoid and a quadruped: the two upstreams
    write it differently but compute the identical quantity, R(q)^T @ [0, 0, -1].
    """
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(200):
        quat = rng.normal(size=4)
        quat /= np.linalg.norm(quat)
        worst = max(
            worst,
            float(np.abs(_quat_rotate_inverse(quat, _GRAVITY_W) - gravity_orientation(quat)).max()),
        )
    assert worst < 1e-6, f"the two upstream forms disagree by {worst:g}"


def test_spec_width_matches_spots_policy(tmp_path):
    spec = _spot_spec(tmp_path)
    n = len(JOINTS)
    assert spec.num_obs == 3 + 3 + 3 + 3 + 3 * n == 48


def test_spec_reproduces_spots_own_observation(tmp_path):
    """Element-wise against what `spot_locomotion.pre_step` assembles by hand."""
    spec = _spot_spec(tmp_path)
    rng = np.random.default_rng(1)
    n = len(JOINTS)

    for _ in range(20):
        quat = rng.normal(size=4).astype(np.float32)
        quat /= np.linalg.norm(quat)
        lin_world = rng.normal(size=3).astype(np.float32)
        state = ObservationState(
            base_lin_vel=_quat_rotate_inverse(quat, lin_world),
            base_ang_vel=rng.normal(size=3).astype(np.float32),
            base_quat=quat,
            command=rng.normal(size=3).astype(np.float32),
            actuated_pos=rng.normal(size=n).astype(np.float32),
            actuated_vel=rng.normal(size=n).astype(np.float32),
            prev_action=rng.normal(size=n).astype(np.float32),
        )

        # Hand-assembled exactly as spot_locomotion does it.
        expected = np.zeros(48, dtype=np.float32)
        expected[0:3] = _quat_rotate_inverse(quat, lin_world) * DEPLOY["lin_vel_scale"]
        expected[3:6] = state.base_ang_vel * DEPLOY["ang_vel_scale"]
        expected[6:9] = _quat_rotate_inverse(quat, _GRAVITY_W)
        expected[9:12] = state.command * np.array(DEPLOY["cmd_scale"], dtype=np.float32)
        expected[12 : 12 + n] = (
            state.actuated_pos - np.array(DEPLOY["default_angles"], dtype=np.float32)
        ) * DEPLOY["dof_pos_scale"]
        expected[12 + n : 12 + 2 * n] = state.actuated_vel * DEPLOY["dof_vel_scale"]
        expected[12 + 2 * n : 12 + 3 * n] = state.prev_action

        assert np.allclose(spec.build_observation(state), expected, atol=1e-6)


def test_spot_declares_no_gait_phase(tmp_path):
    """Unlike the G1 walk policy: the term list differs per policy, which is the point of the format."""
    assert "gait_phase" not in {t["term"] for t in _spot_spec(tmp_path).terms}


def test_per_joint_action_scale_from_spots_own_config(tmp_path):
    """Spot's action_scale is a scalar today, but the schema must accept either shape."""
    spec = _spot_spec(tmp_path)
    assert spec.action_scale.shape == (len(JOINTS),)
    assert spec.action_scale[0] == pytest.approx(DEPLOY["action_scale"])
