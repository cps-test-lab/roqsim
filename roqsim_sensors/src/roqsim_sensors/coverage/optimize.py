"""Deterministic placement search -- a non-LLM baseline over the same coverage evaluator.

:func:`greedy_baseline` is submodular max-coverage: precompute each candidate mount's visible-point set
once, then repeatedly add the candidate that satisfies the most still-unsatisfied points (a point is
satisfied once ``k`` sensors see it), until the target fraction or the sensor budget is reached. This is
the standard greedy set-cover heuristic and a good warm-start; it is not optimal.

:func:`generate_candidates` lays out a grid of down-looking mounts inside the room, so ``greedy`` is
usable without hand-authoring candidate poses.
"""

from __future__ import annotations

import numpy as np

from . import sampling
from .adapters import build_fov
from .catalog import placed_from_proposal
from .engine import CoverageResult, coverage

# Down-looking mount orientations by sensor type (see adapters for the frame conventions).
_DOWN_RPY = {
    "oakd_camera": [0.0, np.pi / 2, 0.0],  # optical axis -> straight down
    "realsense_d435": [0.0, np.pi / 2, 0.0],
    "realsense_d415": [0.0, np.pi / 2, 0.0],
    "livox_mid360": [float(np.pi), 0.0, 0.0],  # invert the dome so the +52..-7 band faces down
    "lidar": [0.0, 0.0, 0.0],  # planar, horizontal
}


def generate_candidates(model, data, *, types, spacing=3.0, z=3.0) -> list[dict]:
    """A grid of down-looking candidate mounts at height ``z``, restricted to enclosed room positions."""
    lo, hi = sampling.world_bounds(model, data)
    xs = np.arange(lo[0] + spacing / 2, hi[0], spacing)
    ys = np.arange(lo[1] + spacing / 2, hi[1], spacing)
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    grid = np.column_stack([gx.ravel(), gy.ravel(), np.full(gx.size, float(z))])
    mask = sampling._classify_points(
        model,
        data,
        grid,
        max_dist=float(np.linalg.norm(hi - lo)) + 1.0,
        geomgroup=sampling._geomgroup_mask(),
    )
    candidates = []
    for pos in grid[mask]:
        for t in types:
            t = t.strip()
            candidates.append(
                {
                    "type": t,
                    "pos": [float(v) for v in pos],
                    "rpy": _DOWN_RPY.get(t, [0.0, 0.0, 0.0]),
                }
            )
    return candidates


def greedy_baseline(
    model,
    data,
    candidates: list[dict],
    points: np.ndarray,
    *,
    labels=None,
    label_names=None,
    target_frac: float = 0.95,
    max_sensors: int = 10,
    k: int = 1,
) -> tuple[list[dict], CoverageResult]:
    """Greedily select candidate mounts to reach ``target_frac`` coverage at level ``k``."""
    if not candidates:
        raise ValueError("greedy_baseline needs at least one candidate")
    # Precompute each candidate's visible mask (single-sensor coverage) once.
    masks = []
    for i, prop in enumerate(candidates):
        fov = build_fov(model, data, placed_from_proposal(prop, index=i))
        masks.append(coverage(model, data, [fov], points).by_sensor[:, 0])

    n = len(points)
    counts = np.zeros(n, dtype=int)
    chosen: list[int] = []
    while len(chosen) < max_sensors:
        satisfied = counts >= k
        if satisfied.mean() >= target_frac:
            break
        best, best_gain = -1, 0
        for i, mask in enumerate(masks):
            if i in chosen:
                continue
            gain = int(np.sum(mask & ~satisfied))
            if gain > best_gain:
                best, best_gain = i, gain
        if best < 0 or best_gain == 0:
            break  # no candidate adds anything -- report the shortfall rather than looping
        chosen.append(best)
        counts += masks[best].astype(int)

    chosen_props = [candidates[i] for i in chosen]
    fovs = [
        build_fov(model, data, placed_from_proposal(p, index=i)) for i, p in enumerate(chosen_props)
    ]
    result = coverage(model, data, fovs, points, labels=labels, label_names=label_names)
    return chosen_props, result
