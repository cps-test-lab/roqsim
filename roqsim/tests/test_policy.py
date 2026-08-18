"""The policy-spec format: schema behaviour, widths, and the envelope record.

Lives in core because the format does, and deliberately imports nothing from a robot-family package --
a core test that reached into `roqsim_humanoid` would invert the dependency it is meant to keep clean. The
proof that the format can describe a policy which actually runs lives with that policy, in
`roqsim_humanoid/tests/test_policy_spec.py`.
"""

from __future__ import annotations

import pytest
import yaml

from roqsim.policy import PolicySpec

LEGS = [f"leg_{i}" for i in range(12)]


def _write(tmp_path, name, body):
    (tmp_path / "x.pt").write_bytes(b"")
    path = tmp_path / name
    path.write_text(yaml.safe_dump(body))
    return path


def test_observed_joints_widen_the_observation(tmp_path):
    """The stand policy's shape: legs actuated, arms observed but not commanded."""
    (tmp_path / "stand.pt").write_bytes(b"")
    spec_file = tmp_path / "stand.spec.yaml"
    spec_file.write_text(
        yaml.safe_dump(
            {
                "checkpoint": "stand.pt",
                "joints": {
                    "actuated": LEGS,
                    "observed": [f"arm_{i}" for i in range(14)],
                },
                "control": {"default_angles": [0.0] * 12},
                "observation": [
                    {"term": "base_ang_vel"},
                    {"term": "projected_gravity"},
                    {"term": "command"},
                    {"term": "actuated_pos_rel_default"},
                    {"term": "actuated_vel"},
                    {"term": "prev_action"},
                    {"term": "observed_pos"},
                    {"term": "observed_vel"},
                ],
            }
        )
    )
    spec = PolicySpec.from_yaml(spec_file)
    assert spec.num_obs == 3 + 3 + 3 + 12 + 12 + 12 + 14 + 14 == 73
    # No gait_phase term: the cycling phase is what keeps the walk policy stepping.
    assert "gait_phase" not in {t["term"] for t in spec.terms}


def test_unknown_term_is_refused(tmp_path):
    (tmp_path / "x.pt").write_bytes(b"")
    spec_file = tmp_path / "x.spec.yaml"
    spec_file.write_text(
        yaml.safe_dump(
            {
                "checkpoint": "x.pt",
                "joints": {"actuated": ["a"]},
                "observation": [{"term": "base_ang_vel"}, {"term": "moon_phase"}],
            }
        )
    )
    with pytest.raises(ValueError, match="moon_phase"):
        PolicySpec.from_yaml(spec_file)


def test_per_joint_scale_arrays_are_supported(tmp_path):
    """Oli uses per-joint action_scale arrays, so the schema has to accept them, not just scalars."""
    (tmp_path / "x.pt").write_bytes(b"")
    spec_file = tmp_path / "x.spec.yaml"
    spec_file.write_text(
        yaml.safe_dump(
            {
                "checkpoint": "x.pt",
                "joints": {"actuated": ["a", "b", "c"]},
                "control": {"action_scale": [0.1, 0.2, 0.3]},
                "observation": [{"term": "prev_action"}],
            }
        )
    )
    assert list(PolicySpec.from_yaml(spec_file).action_scale) == pytest.approx([0.1, 0.2, 0.3])

    # And a wrong length is refused rather than broadcast into silence -- a per-joint array that is one
    # short is exactly the kind of edit that would otherwise mis-scale every joint after it.
    short = tmp_path / "short.spec.yaml"
    short.write_text(
        yaml.safe_dump(
            {
                "checkpoint": "x.pt",
                "joints": {"actuated": ["a", "b", "c"]},
                "control": {"action_scale": [0.1, 0.2]},
                "observation": [{"term": "prev_action"}],
            }
        )
    )
    with pytest.raises(ValueError, match="action_scale"):
        PolicySpec.from_yaml(short)


def test_envelope_flags_an_out_of_range_payload(tmp_path):
    """The point of recording the envelope: outside it the policy degrades silently, it does not error."""
    (tmp_path / "x.pt").write_bytes(b"")
    spec_file = tmp_path / "x.spec.yaml"
    spec_file.write_text(
        yaml.safe_dump(
            {
                "checkpoint": "x.pt",
                "joints": {"actuated": ["a"]},
                "observation": [{"term": "prev_action"}],
                "envelope": {"payload_kg": [0.0, 1.0], "payload_frame": "left_grasp"},
            }
        )
    )
    envelope = PolicySpec.from_yaml(spec_file).envelope
    assert envelope.check_payload(0.5) == ""
    assert "outside the policy's trained range" in envelope.check_payload(5.0)
