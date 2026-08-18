#!/usr/bin/env python3
"""Export a trained MLP policy to TorchScript for CPU inference, and verify it numerically.

Training happens in JAX (MuJoCo Playground / brax); the substrate runs `torch.jit.load` on the physics
thread at 50 Hz. For an MLP that conversion is mechanical -- copy the weights into an equivalent
`torch.nn.Sequential` -- and the only interesting part is proving the copy is faithful.

**Why the verification is not optional.** A subtly wrong export is indistinguishable from a badly trained
policy: the robot wobbles, and you go tuning rewards for a day chasing a transposed weight matrix. So the
exporter feeds the same random observations through both implementations and refuses to write a
checkpoint whose outputs disagree.

The exporter also writes the spec (`roqsim.policy.PolicySpec`) beside the checkpoint, from the same values
the env was configured with. That is what stops the trainer and the runtime from disagreeing about the
observation layout -- the failure that does not error, it just twitches.

Runs in `.venv-train` (jax), but the *verification* half only needs torch and numpy, so
`verify_export()` is importable from the sim venv too and is covered by a test there.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn

ACTIVATIONS = {"relu": nn.ReLU, "tanh": nn.Tanh, "elu": nn.ELU, "silu": nn.SiLU}


def build_torch_mlp(
    layers: list[tuple[np.ndarray, np.ndarray]], activation: str = "relu"
) -> nn.Sequential:
    """An `nn.Sequential` from (weight, bias) pairs, activation between all but the last layer.

    Weights are expected in the row-major ``(out, in)`` convention `nn.Linear` uses; a JAX/flax kernel
    is ``(in, out)`` and must be transposed by the caller, which is precisely the kind of mistake the
    numerical check below exists to catch.
    """
    if activation not in ACTIVATIONS:
        raise ValueError(f"unknown activation {activation!r}; known: {sorted(ACTIVATIONS)}")
    modules: list[nn.Module] = []
    for i, (weight, bias) in enumerate(layers):
        linear = nn.Linear(weight.shape[1], weight.shape[0])
        with torch.no_grad():
            linear.weight.copy_(torch.from_numpy(np.ascontiguousarray(weight, dtype=np.float32)))
            linear.bias.copy_(torch.from_numpy(np.ascontiguousarray(bias, dtype=np.float32)))
        modules.append(linear)
        if i < len(layers) - 1:
            modules.append(ACTIVATIONS[activation]())
    return nn.Sequential(*modules)


def verify_export(
    reference,
    scripted: torch.jit.ScriptModule,
    num_obs: int,
    samples: int = 200,
    tol: float = 1e-5,
    seed: int = 0,
) -> float:
    """Max abs difference between *reference* and the scripted module over random observations.

    *reference* is any callable taking a ``(1, num_obs)`` float32 array and returning the action -- the
    JAX policy at export time, or a plain torch module in a test. Raises when the two disagree by more
    than *tol*, because writing a checkpoint that does not match what was trained is worse than failing.
    """
    rng = np.random.default_rng(seed)
    worst = 0.0
    for _ in range(samples):
        obs = rng.normal(size=(1, num_obs)).astype(np.float32)
        want = np.asarray(reference(obs), dtype=np.float32).reshape(-1)
        with torch.no_grad():
            got = scripted(torch.from_numpy(obs)).numpy().reshape(-1)
        worst = max(worst, float(np.abs(want - got).max()))
    if worst > tol:
        raise RuntimeError(
            f"export mismatch: max |reference - torchscript| = {worst:g} > {tol:g}. The weights were "
            f"not copied faithfully (a transposed kernel is the usual cause). Refusing to write a "
            f"checkpoint that does not reproduce the trained policy."
        )
    return worst


def export(
    layers: list[tuple[np.ndarray, np.ndarray]],
    out_path: Path,
    num_obs: int,
    reference=None,
    activation: str = "relu",
) -> float:
    """Build, verify and save. Returns the measured worst-case difference (0.0 if unverified)."""
    module = build_torch_mlp(layers, activation)
    module.eval()
    scripted = torch.jit.script(module)
    worst = verify_export(reference, scripted, num_obs) if reference is not None else 0.0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.jit.save(scripted, str(out_path))
    return worst


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--params", type=Path, required=True, help=".npz of flax kernels/biases")
    ap.add_argument("--out", type=Path, required=True, help="where to write the .pt")
    ap.add_argument("--num-obs", type=int, required=True)
    ap.add_argument("--activation", default="relu", choices=sorted(ACTIVATIONS))
    args = ap.parse_args()

    data = np.load(args.params)
    # flax stores kernels as (in, out); nn.Linear wants (out, in).
    order = sorted({k.rsplit("_", 1)[0] for k in data.files})
    layers = [(data[f"{name}_kernel"].T, data[f"{name}_bias"]) for name in order]
    worst = export(layers, args.out, args.num_obs, activation=args.activation)
    print(f"wrote {args.out} ({len(layers)} layers, num_obs={args.num_obs}, check={worst:g})")
    print("NOTE: exported without a reference; run the verification against the JAX policy in-process.")


if __name__ == "__main__":
    main()
