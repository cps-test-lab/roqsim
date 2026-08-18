"""The exporter's numerical check actually catches a bad export.

The check is the whole reason the exporter exists as a script rather than three lines inline: a subtly
wrong weight copy produces a policy that looks badly trained, and days get spent tuning rewards to fix a
transposed matrix. So the guard itself is tested -- including that it FAILS when it should.

Lives here rather than under external/train/ because that directory is not a package and has no venv in
CI; the exporter's verification half needs only torch and numpy, both of which the sim venv has.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "external" / "train"))
from export_policy import build_torch_mlp, export, verify_export  # noqa: E402

NUM_OBS, NUM_ACT = 73, 12


def _layers(rng, num_obs=NUM_OBS):
    """(out, in) weight matrices, as nn.Linear expects."""
    return [
        (rng.normal(size=(64, num_obs)) * 0.1, rng.normal(size=64) * 0.1),
        (rng.normal(size=(64, 64)) * 0.1, rng.normal(size=64) * 0.1),
        (rng.normal(size=(NUM_ACT, 64)) * 0.1, rng.normal(size=NUM_ACT) * 0.1),
    ]


def test_faithful_export_passes_and_round_trips(tmp_path):
    rng = np.random.default_rng(0)
    layers = _layers(rng)
    reference = build_torch_mlp(layers)
    reference.eval()

    out = tmp_path / "stand.pt"
    worst = export(
        layers,
        out,
        NUM_OBS,
        reference=lambda obs: reference(torch.from_numpy(obs)).detach().numpy(),
    )
    assert worst < 1e-6
    assert out.exists()

    # And the saved file is loadable exactly as the plugin loads it.
    loaded = torch.jit.load(str(out))
    obs = torch.from_numpy(rng.normal(size=(1, NUM_OBS)).astype(np.float32))
    with torch.no_grad():
        assert loaded(obs).shape == (1, NUM_ACT)


def test_transposed_weights_are_caught(tmp_path):
    """The mistake this guard exists for: a JAX kernel is (in, out), nn.Linear wants (out, in)."""
    rng = np.random.default_rng(1)
    square = [
        (rng.normal(size=(64, 64)) * 0.1, rng.normal(size=64) * 0.1),
        (rng.normal(size=(64, 64)) * 0.1, rng.normal(size=64) * 0.1),
    ]
    reference = build_torch_mlp(square)
    reference.eval()
    # Same shapes, transposed contents -- so it builds and runs, and is silently wrong.
    wrong = torch.jit.script(build_torch_mlp([(w.T, b) for w, b in square]).eval())

    with pytest.raises(RuntimeError, match="export mismatch"):
        verify_export(
            lambda obs: reference(torch.from_numpy(obs)).detach().numpy(), wrong, num_obs=64
        )


def test_export_refuses_to_write_when_verification_fails(tmp_path):
    """A checkpoint that does not reproduce the trained policy must not reach the package."""
    rng = np.random.default_rng(2)
    layers = _layers(rng)
    out = tmp_path / "bad.pt"
    with pytest.raises(RuntimeError, match="export mismatch"):
        export(layers, out, NUM_OBS, reference=lambda obs: np.zeros((1, NUM_ACT), dtype=np.float32))
    assert not out.exists()


def test_exported_width_matches_the_shipped_stand_spec():
    """The exporter and the spec must agree on 73, or the runtime feeds the wrong-sized vector."""
    from roqsim.policy import PolicySpec
    from roqsim_humanoid.policy import find_spec

    assert PolicySpec.from_yaml(find_spec("g1_stand")).num_obs == NUM_OBS
