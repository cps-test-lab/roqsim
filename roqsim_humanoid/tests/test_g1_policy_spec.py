"""The spec format is checked against the policy that already runs, not just the one being added.

A spec designed around one policy is that policy's shape wearing a general name. So the load-bearing test
here rebuilds the **existing G1 walk observation** -- the 47-dim vector `motion.pt` was trained on -- from
a spec, and compares it element-wise with the vector `g1_locomotion` assembles by hand. If those agree,
the format describes a policy that demonstrably works rather than one invented to fit the format.

The walk plugin is deliberately NOT migrated onto the spec: it works, it is covered, and swapping its
observation builder would risk a silent regression for no gain today. This test gets the proof without
the risk.
"""

from __future__ import annotations

import numpy as np
import pytest
import yaml

from roqsim_humanoid.plugins.g1_locomotion import LEG_JOINTS, _gravity_orientation
from roqsim_humanoid.policy import DEFAULT_CONFIG
from roqsim.policy import ObservationState, PolicySpec, gravity_orientation

# The walk policy's own deploy config is the source of truth for the numbers; the spec must not restate
# them by hand, or the test would only prove the two hand-copies agree.
DEPLOY = yaml.safe_load(DEFAULT_CONFIG.read_text())


def _walk_spec(tmp_path) -> PolicySpec:
    """A spec describing `motion.pt`: 47 dims, gait phase, no observed-but-unactuated joints."""
    (tmp_path / "motion.pt").write_bytes(b"")  # only the path is resolved, never loaded here
    spec_file = tmp_path / "g1_walk.spec.yaml"
    spec_file.write_text(
        yaml.safe_dump(
            {
                "name": "g1_walk",
                "checkpoint": "motion.pt",
                "joints": {"actuated": list(LEG_JOINTS)},
                "control": {
                    "decimation": DEPLOY["control_decimation"],
                    "action_scale": DEPLOY["action_scale"],
                    "default_angles": DEPLOY["default_angles"],
                    "kp": DEPLOY["kps"],
                    "kd": DEPLOY["kds"],
                },
                "observation": [
                    {"term": "base_ang_vel", "scale": DEPLOY["ang_vel_scale"]},
                    {"term": "projected_gravity"},
                    {"term": "command", "scale": DEPLOY["cmd_scale"]},
                    {"term": "actuated_pos_rel_default", "scale": DEPLOY["dof_pos_scale"]},
                    {"term": "actuated_vel", "scale": DEPLOY["dof_vel_scale"]},
                    {"term": "prev_action"},
                    {"term": "gait_phase"},
                ],
            }
        )
    )
    return PolicySpec.from_yaml(spec_file)


def test_spec_matches_the_checkpoints_declared_width(tmp_path):
    """47 is what `motion.pt` expects; a spec that disagrees would twitch the robot, not error."""
    assert _walk_spec(tmp_path).num_obs == DEPLOY["num_obs"] == 47


def test_spec_reproduces_the_running_walk_observation(tmp_path):
    """The acid test: same numbers as `g1_locomotion` assembles by hand, element for element."""
    spec = _walk_spec(tmp_path)
    rng = np.random.default_rng(0)
    n = len(LEG_JOINTS)

    for _ in range(20):
        quat = rng.normal(size=4).astype(np.float32)
        quat /= np.linalg.norm(quat)
        state = ObservationState(
            base_ang_vel=rng.normal(size=3).astype(np.float32),
            base_quat=quat,
            command=rng.normal(size=3).astype(np.float32),
            actuated_pos=rng.normal(size=n).astype(np.float32),
            actuated_vel=rng.normal(size=n).astype(np.float32),
            prev_action=rng.normal(size=n).astype(np.float32),
            phase=float(rng.uniform()),
        )

        # Hand-assembled exactly as g1_locomotion.pre_step does it.
        expected = np.zeros(47, dtype=np.float32)
        expected[:3] = state.base_ang_vel * DEPLOY["ang_vel_scale"]
        expected[3:6] = _gravity_orientation(state.base_quat)
        expected[6:9] = state.command * np.array(DEPLOY["cmd_scale"], dtype=np.float32)
        expected[9 : 9 + n] = (
            state.actuated_pos - np.array(DEPLOY["default_angles"], dtype=np.float32)
        ) * DEPLOY["dof_pos_scale"]
        expected[9 + n : 9 + 2 * n] = state.actuated_vel * DEPLOY["dof_vel_scale"]
        expected[9 + 2 * n : 9 + 3 * n] = state.prev_action
        expected[9 + 3 * n : 9 + 3 * n + 2] = (
            np.sin(2 * np.pi * state.phase),
            np.cos(2 * np.pi * state.phase),
        )

        assert np.allclose(spec.build_observation(state), expected, atol=1e-6)


def test_gravity_term_is_the_plugins_own_function(tmp_path):
    """Guards against the two copies drifting apart -- the term must stay bit-identical."""
    rng = np.random.default_rng(1)
    for _ in range(10):
        quat = rng.normal(size=4).astype(np.float32)
        quat /= np.linalg.norm(quat)
        assert np.allclose(gravity_orientation(quat), _gravity_orientation(quat), atol=1e-7)


def test_the_shipped_stand_spec_is_valid_and_73_dim():
    """The spec ships before its checkpoint does, so it has to stand on its own until training lands."""
    from roqsim_humanoid.plugins.g1_locomotion import LEG_JOINTS
    from roqsim_humanoid.policy import find_spec

    spec = PolicySpec.from_yaml(find_spec("g1_stand"))
    assert spec.num_obs == 73
    assert tuple(spec.actuated) == LEG_JOINTS  # the arms are MoveIt's, not the policy's
    assert len(spec.observed) == 14
    # The omission that matters: a gait phase is what keeps the walk policy stepping at zero command.
    assert "gait_phase" not in {t["term"] for t in spec.terms}
    # And the trained envelope travels with it, machine-readable.
    assert spec.envelope.payload_kg == (0.0, 1.0)
    assert spec.envelope.payload_frame == "left_grasp"
    assert "outside the policy's trained range" in spec.envelope.check_payload(2.0)


def test_missing_checkpoint_fails_loudly_with_how_to_get_it(tmp_path):
    """Until training runs there is no stand.pt, and that must be an actionable error, not a crash."""
    from roqsim.config import load_config_from_dict
    from roqsim.engine import Engine

    cfg = load_config_from_dict(
        {
            "sim": {"timestep": 0.002},
            "plugins": [
                {"spawn_robot": {"model": "unitree_g1_dex1", "name": "robot", "pos": [0, 0]}},
                {"g1_locomotion": {"robot": "robot", "policy": "g1_stand"}},
            ],
        },
        base_dir=tmp_path,
    )
    with pytest.raises(RuntimeError, match="no checkpoint at"):
        Engine(cfg).setup()
