#!/usr/bin/env python3
"""Convert the Robotiq Hand-E visual meshes from Robotiq's own ``robotiq_description``.

Source: https://github.com/robotiq/ros at :data:`COMMIT`, ``grippers/robotiq_description`` (BSD-3-Clause,
Copyright Robotiq; see ``ROBOTIQ_HANDE_LICENSE`` beside the model). Its three Collada files --
``meshes/visual/hand_e/hand-e.dae`` (the gripper body), ``left_finger.dae`` and ``right_finger.dae``
-- are in metres, in the frames of the links ``robotiq_hand_e_macro.urdf.xacro`` hangs them on, with
identity visual origins. Vertices are taken verbatim: ``hand-e.dae`` is tagged ``Y_UP`` but is already
Z-up, and honouring the tag would lay the body on its side (see ``dae2obj.py``).

Each mesh is written as one binary STL. The meshes carry at most one useful colour each, and the
MJCF takes its materials from the xacro's ``<color>`` instead. They are light enough (10k triangles
for the body, 3k and 6k for the fingers) to need no decimation.

Robotiq ships no coupling mesh; ``robotiq_hande.xml`` draws the GRP-CPL-062 coupling as a disc.
Only the visual meshes come from here: every collision geom of the model is a primitive, so nothing
here affects the physics.

Usage (needs ``pip install pycollada``)::

    python3 external/convert/convert_robotiq_hande_meshes.py
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sources import resolve_source  # noqa: E402

URL = "https://github.com/robotiq/ros"
COMMIT = "c39cd970a3f950c8be206d0e4340c7b69e018d7a"
PACKAGE = "grippers/robotiq_description"

OUT = (
    Path(__file__).resolve().parents[2]
    / "roqsim_manipulation_assets/src/roqsim_manipulation_assets/models/robotiq_hande/meshes"
)

#: Source mesh under meshes/visual/hand_e/ -> the STL written beside the model.
MESHES = {
    "hand-e.dae": "hand_e.stl",
    "left_finger.dae": "hande_left_finger.stl",
    "right_finger.dae": "hande_right_finger.stl",
}


def _load(path: Path) -> np.ndarray:
    """Every triangle of the scene, (n, 3, 3) in metres, in the scene's own frame."""
    import collada

    c = collada.Collada(str(path), ignore=[collada.common.DaeError])
    tris = [
        np.asarray(prim.vertex)[np.asarray(prim.vertex_index).reshape(-1, 3)]
        for geom in c.scene.objects("geometry")
        for prim in geom.primitives()
        if prim.vertex is not None and prim.vertex_index is not None and len(prim.vertex_index)
    ]
    if not tris:
        raise RuntimeError(f"no triangles in {path}")
    return np.concatenate(tris)


def _write_stl(path: Path, tri: np.ndarray) -> None:
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    with open(path, "wb") as f:
        f.write(b"robotiq_hande".ljust(80, b" "))
        f.write(struct.pack("<I", len(tri)))
        for n, t in zip(normals, tri, strict=True):
            f.write(struct.pack("<12fH", *n, *t.reshape(-1), 0))


def main() -> int:
    src = resolve_source(
        "robotiq_ros", URL, COMMIT, sparse=PACKAGE, subdir=f"{PACKAGE}/meshes/visual/hand_e"
    )
    OUT.mkdir(parents=True, exist_ok=True)
    for source, target in MESHES.items():
        tri = _load(src / source)
        _write_stl(OUT / target, tri)
        lo, hi = tri.reshape(-1, 3).min(0), tri.reshape(-1, 3).max(0)
        print(f"{target}: {len(tri)} triangles, bounds {lo.round(4)} .. {hi.round(4)} m")
    return 0


if __name__ == "__main__":
    sys.exit(main())
