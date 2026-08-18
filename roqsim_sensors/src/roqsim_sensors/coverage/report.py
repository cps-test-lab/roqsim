"""The agent-digestible coverage report.

:func:`build_report` reduces a :class:`~.engine.CoverageResult` to a compact JSON-serialisable dict an
LLM (or a human) can reason over: the achieved coverage against the target, per-object coverage,
per-sensor contribution (to prune redundant sensors), and the uncovered regions (to decide where to
add one). It intentionally omits the raw per-point arrays -- those are for the visualiser, not the
planner.
"""

from __future__ import annotations

import numpy as np

from .engine import CoverageResult
from .sampling import cluster_gaps


def build_report(
    result: CoverageResult,
    *,
    world: str = "",
    target: dict | None = None,
    placements: list[dict] | None = None,
    gap_resolution: float = 0.25,
    max_gaps: int = 10,
    per_region: list[dict] | None = None,
) -> dict:
    """Build the coverage report dict. ``target`` is e.g. ``{"metric": "fraction_covered", "k": 1,
    "value": 0.95}``; ``placements`` is the list of placement dicts that produced ``result``.

    ``per_region`` (from :func:`~.regions.per_region_coverage`) is an optional per-area coverage
    breakdown -- the "how much of THIS room is covered" answer -- attached verbatim under ``per_region``.
    """
    counts = result.counts
    n = int(len(counts))
    achieved = {
        "fraction_covered_k1": result.coverage_fraction(1),
        "fraction_covered_k2": result.coverage_fraction(2),
        "fraction_covered_k3": result.coverage_fraction(3),
        "mean_count": float(np.mean(counts)) if n else 0.0,
        "max_count": int(counts.max()) if n else 0,
        "n_points": n,
    }

    per_object = []
    if result.labels is not None and result.label_names:
        for idx, name in enumerate(result.label_names):
            sel = result.labels == idx
            m = int(sel.sum())
            if m == 0:
                continue
            obj_counts = counts[sel]
            per_object.append(
                {
                    "name": name,
                    "n_points": m,
                    "fraction_covered": float(np.mean(obj_counts >= 1)),
                    "mean_count": float(np.mean(obj_counts)),
                }
            )
        per_object.sort(key=lambda o: o["fraction_covered"])

    seen = result.per_sensor_seen()
    # Unique contribution: points this sensor is the *only* one to see.
    per_sensor = []
    for s, fov in enumerate(result.fovs):
        col = result.by_sensor[:, s]
        unique = int(np.sum(col & (counts == 1)))
        per_sensor.append(
            {
                "id": s,
                "type": fov.sensor_type,
                "label": fov.label,
                "total_points": int(seen[s]),
                "unique_points": unique,
            }
        )

    target = target or {}
    k = int(target.get("k", 1))
    metric_key = f"fraction_covered_k{k}"
    achieved_value = achieved.get(metric_key, achieved["fraction_covered_k1"])
    report = {
        "world": world,
        "target": target,
        "achieved": achieved,
        "target_met": bool(target and achieved_value >= float(target.get("value", 1.0))),
        "placements": placements or [],
        "per_object": per_object,
        "per_region": per_region or [],
        "per_sensor_contribution": per_sensor,
        "uncovered_regions": [],
    }

    # Uncovered regions: cluster the points that fail the target's k (default k=1).
    uncovered = result.points[counts < k] if n else result.points
    gaps = cluster_gaps(uncovered, resolution=gap_resolution)[:max_gaps]
    report["uncovered_regions"] = [
        {
            "centroid": g.centroid,
            "n_points": g.n_points,
            "bbox_min": g.bbox_min,
            "bbox_max": g.bbox_max,
        }
        for g in gaps
    ]
    return report
