"""Named 2D regions (rooms / zones) for per-area coverage breakdown and target restriction.

A :class:`Region` is a named polygon in the world xy-plane with an optional z-band: a sample point is
inside iff its ``(x, y)`` is in the polygon and its ``z`` lies in ``[z_min, z_max]``. Regions are the
lens the coverage evaluator looks through -- the sampler and the scorer still work on the *whole*
world; regions answer "how much of THIS room is covered?" (a per-region report) and "optimise for
THESE rooms only" (restrict the sample to their union).

Deliberately world-agnostic. Nothing here knows about any particular building: regions come from an
explicit JSON file (polygons or axis-aligned bboxes) or are reconstructed from a scene-builder sketch
(:func:`regions_from_sketch`), and the same :class:`Region` drives both the report and the greedy
objective. Point-in-polygon is a self-contained numpy even-odd test, so this module needs no plotting
or GIS dependency (it works without the ``coverage`` extra).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Region:
    """A named polygon in the world xy-plane, with an optional inclusive z-band."""

    name: str
    polygon: np.ndarray  # (N, 2) vertices in world metres
    z_min: float = -math.inf
    z_max: float = math.inf

    def __post_init__(self) -> None:
        self.polygon = np.asarray(self.polygon, dtype=np.float64).reshape(-1, 2)
        if len(self.polygon) < 3:
            raise ValueError(
                f"region {self.name!r} needs >= 3 polygon vertices, got {len(self.polygon)}"
            )

    def contains(self, points: np.ndarray) -> np.ndarray:
        """(P,) bool mask: which ``points`` (P, 3 world) fall inside this region."""
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        inside = _points_in_polygon(self.polygon, pts[:, :2])
        if math.isfinite(self.z_min) or math.isfinite(self.z_max):
            inside &= (pts[:, 2] >= self.z_min) & (pts[:, 2] <= self.z_max)
        return inside


def _points_in_polygon(polygon: np.ndarray, pts_xy: np.ndarray) -> np.ndarray:
    """Vectorised even-odd (ray-casting) point-in-polygon test. Boundary inclusion is unspecified."""
    x = pts_xy[:, 0]
    y = pts_xy[:, 1]
    inside = np.zeros(len(pts_xy), dtype=bool)
    n = len(polygon)
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        # Edge (j -> i) straddles the horizontal ray at y, and the crossing is to the right of x.
        straddles = (yi > y) != (yj > y)
        x_cross = (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi
        inside ^= straddles & (x < x_cross)
        j = i
    return inside


def _bbox_polygon(bbox) -> np.ndarray:
    """Axis-aligned ``[xmin, ymin, xmax, ymax]`` -> a 4-vertex polygon (CCW)."""
    xmin, ymin, xmax, ymax = (float(v) for v in bbox)
    return np.array([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]])


def _region_from_spec(spec: dict) -> Region:
    name = str(spec.get("name", spec.get("id", "region")))
    if "polygon" in spec:
        poly = np.asarray(spec["polygon"], dtype=np.float64).reshape(-1, 2)
    elif "bbox" in spec:
        poly = _bbox_polygon(spec["bbox"])
    else:
        raise ValueError(f"region {name!r} needs a 'polygon' or a 'bbox'")
    z_min = float(spec.get("z_min", -math.inf))
    z_max = float(spec.get("z_max", math.inf))
    return Region(name=name, polygon=poly, z_min=z_min, z_max=z_max)


def load_regions(spec) -> list[Region]:
    """Load regions from ``spec``: a path to JSON, an already-parsed dict/list, or a sketch.

    Accepted JSON shapes (all world-agnostic):

    * ``{"regions": [{"name", "polygon"|"bbox", "z_min"?, "z_max"?}, ...]}`` or a bare list of those.
    * A scene **floorplan** (has ``"rooms"`` and ``"lines"``) -> :func:`regions_from_sketch`, so
      you can point ``--regions`` straight at a scene's ``floorplan.json``.
    """
    if isinstance(spec, (str, Path)):
        obj = json.loads(Path(spec).read_text())
    else:
        obj = spec
    if isinstance(obj, dict) and "rooms" in obj and "lines" in obj:
        return regions_from_sketch(obj)
    items = obj["regions"] if isinstance(obj, dict) else obj
    if not isinstance(items, list):
        raise ValueError("regions spec must be a list, {'regions': [...]}, or a scene sketch")
    return [_region_from_spec(it) for it in items]


def regions_from_sketch(sketch, *, bridge: float = 1.6) -> list[Region]:
    """Reconstruct one :class:`Region` per named room from a scene-builder sketch.

    The sketch stores rooms as unordered wall-segment id lists (``line_ids`` into ``lines`` of
    ``{x0_m, y0_m, x1_m, y1_m}``), so this chains the segments end-to-end into a closed loop, bridging
    gaps up to ``bridge`` metres (door openings leave the wall in two pieces). A room whose walls will
    not chain into >= 3 vertices is skipped rather than yielding a bogus polygon.
    """
    if isinstance(sketch, (str, Path)):
        sketch = json.loads(Path(sketch).read_text())
    lines = {line["id"]: line for line in sketch.get("lines", [])}
    out: list[Region] = []
    for room in sketch.get("rooms", []):
        segs = [
            (
                np.array([line["x0_m"], line["y0_m"]], dtype=np.float64),
                np.array([line["x1_m"], line["y1_m"]], dtype=np.float64),
            )
            for lid in room.get("line_ids", [])
            if (line := lines.get(lid)) is not None
        ]
        if len(segs) < 3:
            continue
        used = [False] * len(segs)
        chain = [segs[0][0], segs[0][1]]
        used[0] = True
        for _ in range(len(segs) - 1):
            cur = chain[-1]
            best_i, best_end, best_d = -1, None, bridge
            for i, (a, b) in enumerate(segs):
                if used[i]:
                    continue
                for p, q in ((a, b), (b, a)):
                    d = float(np.linalg.norm(cur - p))
                    if d < best_d:
                        best_i, best_end, best_d = i, q, d
            if best_i < 0:
                break
            used[best_i] = True
            chain.append(best_end)
        if len(chain) >= 3:
            out.append(
                Region(name=str(room.get("name", room.get("id", "room"))), polygon=np.array(chain))
            )
    return out


def select(regions: list[Region], names) -> list[Region]:
    """Keep the regions whose name is in ``names`` (list or comma-separated string), preserving order.

    Fails loudly on a name that matches nothing -- a typo'd room name silently covering the whole world
    would be worse than an error."""
    if isinstance(names, str):
        names = [n.strip() for n in names.split(",") if n.strip()]
    wanted = list(names)
    by_name = {r.name: r for r in regions}
    missing = [n for n in wanted if n not in by_name]
    if missing:
        raise ValueError(f"region name(s) {missing} not found; available: {sorted(by_name)}")
    return [by_name[n] for n in wanted]


def union_mask(points: np.ndarray, regions: list[Region]) -> np.ndarray:
    """(P,) bool: points inside ANY region (the target footprint for ``--restrict``)."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    mask = np.zeros(len(pts), dtype=bool)
    for r in regions:
        mask |= r.contains(pts)
    return mask


def footprint_mask(points: np.ndarray, regions: list[Region]) -> np.ndarray:
    """(P,) bool: points whose ``(x, y)`` is in ANY region polygon, **ignoring the z-band**.

    For restricting candidate *mounts* (at ceiling height) to the target rooms' floor footprint -- a
    region's z-band constrains sample points, not where a sensor may hang above them."""
    arr = np.asarray(points, dtype=np.float64)
    arr = arr.reshape(len(arr), -1)[:, :2]
    mask = np.zeros(len(arr), dtype=bool)
    for r in regions:
        mask |= _points_in_polygon(r.polygon, arr)
    return mask


def per_region_coverage(
    points: np.ndarray, counts: np.ndarray, regions: list[Region]
) -> list[dict]:
    """Per-region coverage summary from evaluated ``points`` + their sensor ``counts``.

    Each region is scored independently over the points inside it (regions may overlap; a shared point
    counts for every region it lies in). ``fraction_covered_k1/k2`` are the fraction of the region's
    points seen by >= 1 / >= 2 sensors -- the "how much of THIS room is covered" answer.
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    counts = np.asarray(counts).reshape(-1)
    out = []
    for r in regions:
        sel = r.contains(pts)
        m = int(sel.sum())
        c = counts[sel]
        out.append(
            {
                "name": r.name,
                "n_points": m,
                "fraction_covered_k1": float(np.mean(c >= 1)) if m else 0.0,
                "fraction_covered_k2": float(np.mean(c >= 2)) if m else 0.0,
                "mean_count": float(np.mean(c)) if m else 0.0,
            }
        )
    return out
