# Copyright (C) 2026 Frederik Pasch
#
# SPDX-License-Identifier: Apache-2.0

"""What a bake depends on, recorded part by part, so a stale bake says what changed.

The parts:

* ``geoms`` -- per baked geom (by its key: its name, else ``<body>#<index>``): its world triangles
  and albedo UVs as a short numeric summary (count, area, area-weighted centroid and second moments,
  bounding box, UV moments) and its material exactly (colour, emission, a hash of the texture's
  pixels);
* ``occluders`` -- what moves, at its reset pose, summarised the same way;
* ``lights`` -- the resolved light sources (:mod:`.lights`), fixtures and emitters included;
* ``settings`` -- the expanded preset and the format version.

Geometry is compared as numbers within a tolerance (:data:`ABS_TOL`, :data:`REL_TOL`), not hashed: a
hash of rounded coordinates flips for a coordinate that sits at a rounding boundary, so float noise
(another machine, another order of the same sums) would read as an edit.

A baked geom the world no longer has is not staleness: a world drawn with its ceiling removed
(``ceiling``, ``roqsim render --no-ceiling``) keeps its bake for everything else, and the record lists
what was missing (``partial``).
"""

from __future__ import annotations

import json

import numpy as np

from .export import Export, _areas
from .lights import rounded
from .settings import FORMAT_VERSION, BakeSettings

#: Two summaries agree when every number is within ``ABS_TOL + REL_TOL * |value|``: 0.1 mm on a
#: coordinate, and 1 µm of float noise passes.
ABS_TOL = 1e-4
REL_TOL = 1e-4


def summary(tri: np.ndarray, uv: np.ndarray | None = None) -> list[float]:
    """A triangle set's numeric summary (see the module docstring)."""
    tri = np.asarray(tri, float).reshape(-1, 3, 3)
    if len(tri) == 0:
        return [0.0]
    a = _areas(tri)
    total = float(a.sum())
    w = a / total if total > 0 else np.full(len(a), 1.0 / len(a))
    c = tri.mean(1)
    centroid = (c * w[:, None]).sum(0)
    corners = tri.reshape(-1, 3)
    wc = np.repeat(w, 3) / 3.0
    second = [
        float((corners[:, i] * corners[:, j] * wc).sum())
        for i, j in ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))
    ]
    out = [
        float(len(tri)),
        total,
        *centroid.tolist(),
        *corners.min(0).tolist(),
        *corners.max(0).tolist(),
        *second,
    ]
    if uv is not None:
        u = np.asarray(uv, float).reshape(-1, 2)
        out += [
            *u.mean(0).tolist(),
            float((u[:, 0] ** 2).mean()),
            float((u[:, 1] ** 2).mean()),
            float((u[:, 0] * u[:, 1]).mean()),
        ]
    return out


def material_record(mat: dict) -> dict:
    return {
        "rgba": rounded(mat["rgba"]),
        "emission": rounded(mat["emission"]),
        "texture": mat["texture_hash"],
    }


def compute(export: Export, light: dict, settings: BakeSettings) -> dict:
    """The fingerprint of *export* baked with *light* (:func:`roqsim.bake.lights.resolve`) and *settings*."""
    geoms = {}
    for ent in export.entities.values():
        for gi in np.unique(ent.geom):
            rec = export.geoms[int(gi)]
            sel = ent.geom == gi
            part = geoms.setdefault(
                rec.key,
                {"entities": {}, "material": material_record(export.materials[rec.material])},
            )
            part["entities"][ent.name] = summary(ent.pos[sel], ent.uv1[sel])
    return {
        "geoms": geoms,
        "occluders": summary(export.occluders),
        "lights": rounded(_light_record(light)),
        "settings": {"format": FORMAT_VERSION, **settings.as_record()},
    }


def _light_record(light: dict) -> dict:
    return {
        "rule": light["rule"],
        "lights": light["lights"],
        "sky": light["sky"],
        "emitters": [
            {k: e[k] for k in ("geom", "strength_w_m2", "colour")} for e in light["emitters"]
        ],
    }


def _close(a, b) -> bool:
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_close(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_close(x, y) for x, y in zip(a, b, strict=True))
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        return abs(float(a) - float(b)) <= ABS_TOL + REL_TOL * abs(float(b))
    return a == b


def compare(stored: dict, current: dict) -> dict:
    """How a bake made with *stored* fits a world whose fingerprint is *current*.

    ``{"status": "fresh" | "stale" | "partial", "reasons": [...], "changed": [...], "added": [...],
    "missing": [...]}``: ``stale`` when anything it was baked from differs (a baked geom changed or a
    new one is there, what moves stands elsewhere, the lights or the settings differ); ``partial`` when
    only baked geoms are gone.
    """
    reasons, changed, added, missing = [], [], [], []
    sg, cg = stored.get("geoms", {}), current.get("geoms", {})
    for key, part in cg.items():
        if key not in sg:
            added.append(key)
        elif not _close(part, sg[key]):
            changed.append(key)
    missing = sorted(k for k in sg if k not in cg)
    if changed:
        reasons.append(f"{len(changed)} baked geom(s) changed: {_few(changed)}")
    if added:
        reasons.append(f"{len(added)} static geom(s) the bake does not have: {_few(added)}")
    if not _close(current.get("occluders"), stored.get("occluders")):
        reasons.append("what moves stands elsewhere at reset (its shadow is in the bake)")
    if not _close(current.get("lights"), stored.get("lights")):
        reasons.append("the lights differ")
    if current.get("settings") != stored.get("settings"):
        diff = sorted(
            k
            for k in set(current.get("settings", {})) | set(stored.get("settings", {}))
            if current.get("settings", {}).get(k) != stored.get("settings", {}).get(k)
        )
        reasons.append(f"the bake's settings differ ({', '.join(diff)})")
    status = "stale" if reasons else ("partial" if missing else "fresh")
    return {
        "status": status,
        "reasons": reasons,
        "changed": changed,
        "added": added,
        "missing": missing,
    }


def _few(keys: list[str], n: int = 4) -> str:
    return ", ".join(keys[:n]) + (f" and {len(keys) - n} more" if len(keys) > n else "")


def digest(fp: dict) -> str:
    """A short name for a fingerprint, for a record to cite (not what staleness is decided by)."""
    import hashlib

    return hashlib.sha256(json.dumps(rounded(fp, 3), sort_keys=True).encode()).hexdigest()[:12]
