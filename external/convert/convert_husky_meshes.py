"""Convert the Clearpath Husky A200 (husky_description) Collada meshes to MuJoCo OBJ.

Usage::

    python external/convert/convert_husky_meshes.py \
        --src /path/to/husky_description/meshes \
        --dst roqsim_mobile/src/roqsim_mobile/models/husky_a200/meshes

Why a direct copy instead of Blender? The sibling turtlebot4 meshes were produced with Blender, and
it remains the preferred tool for decimation/normal work. These Husky sources, however, are Collada
with ``<up_axis>Z_UP`` whose raw geometry is *already* in the exact URDF/MuJoCo link frame (the frame
assimp/RViz load it in). Routing them through Blender's Collada importer in this environment added an
axis transform that was hard to reconcile (and the local distro Blender ships without Collada support
at all), whereas these are visual-only geoms for which MuJoCo recomputes normals and all collision is
authored as primitives (base box + wheel cylinders) — so no decimation/smoothing is needed. A direct
geometry copy is therefore both simplest and highest-fidelity: exact source vertices and triangles,
zero reinterpretation, no external dependency. If a source ever needs real mesh work, do it in Blender.

Verified in the MuJoCo geom (rendered) frame — note MuJoCo re-frames ``mesh_vert`` to a canonical
inertial frame internally, so orientation must be checked via ``mesh_quat``/``mesh_pos``, not raw
``mesh_vert``:
    base  ext (x,y,z) = (0.809, 0.571, 0.228) m   -> length / width / height, sits on z>=0
    wheel ext (x,y,z) = (0.356, 0.114, 0.355) m   -> axle along Y, radius 0.178

Provenance: Clearpath Robotics `husky_description`, meshes base_link/wheel/top_plate/bumper. BSD-3.
"""

from __future__ import annotations

import argparse
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

# source .dae  ->  output .obj basename (husky_ prefix avoids clashing with turtlebot meshes)
MESHES = {
    "base_link.dae": "husky_base.obj",
    "wheel.dae": "husky_wheel.obj",
    "top_plate.dae": "husky_top_plate.obj",
    "bumper.dae": "husky_bumper.obj",
}


def _strip_ns(tag: str) -> str:
    return tag.split("}", 1)[-1]


def _find(el, name):
    return [c for c in el.iter() if _strip_ns(c.tag) == name]


def parse_collada(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (vertices Nx3, triangles Mx3 int) in the file's native coordinate frame.

    Accumulates every <geometry>/<mesh>; within a mesh, reads the POSITION source and every
    <triangles>/<polylist> group, honouring the per-input offset stride so index de-interleaving is
    correct. Polylists are assumed triangulated (vcount all 3), which holds for these Husky meshes.
    """
    root = ET.parse(path).getroot()
    all_v: list[np.ndarray] = []
    all_f: list[np.ndarray] = []
    base = 0
    for mesh in _find(root, "mesh"):
        # id -> float values, for every <source>
        sources: dict[str, np.ndarray] = {}
        for src in _find(mesh, "source"):
            fa = _find(src, "float_array")
            if fa:
                sources["#" + src.get("id")] = np.fromstring(fa[0].text, sep=" ")
        # <vertices> maps a vertices-id to a POSITION source
        vert_src: dict[str, str] = {}
        for verts in _find(mesh, "vertices"):
            for inp in _find(verts, "input"):
                if inp.get("semantic") == "POSITION":
                    vert_src["#" + verts.get("id")] = inp.get("source")
        # positions for this mesh: resolve the vertices element's POSITION source
        pos = None
        for psrc in vert_src.values():
            pos = sources[psrc].reshape(-1, 3)
        if pos is None:
            continue
        all_v.append(pos)
        for prim in _find(mesh, "triangles") + _find(mesh, "polylist"):
            inputs = _find(prim, "input")
            stride = max(int(i.get("offset", 0)) for i in inputs) + 1
            voff = next(int(i.get("offset", 0)) for i in inputs if i.get("semantic") == "VERTEX")
            p = _find(prim, "p")[0]
            idx = np.fromstring(re.sub(r"\s+", " ", p.text.strip()), sep=" ", dtype=np.int64)
            vidx = idx.reshape(-1, stride)[:, voff]
            all_f.append(vidx.reshape(-1, 3) + base)
        base += len(pos)
    if not all_v:
        raise ValueError(f"no mesh geometry parsed from {path}")
    return np.vstack(all_v), np.vstack(all_f)


def write_obj(path: Path, v: np.ndarray, f: np.ndarray) -> None:
    lines = [f"# converted from Collada by convert_husky_meshes.py ({len(v)} verts, {len(f)} tris)"]
    lines += [f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in v]
    lines += [f"f {a + 1} {b + 1} {c + 1}" for a, b, c in f]  # OBJ is 1-indexed
    path.write_text("\n".join(lines) + "\n")


def convert(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for dae, obj in MESHES.items():
        src_path = src / dae
        if not src_path.exists():
            raise FileNotFoundError(f"source mesh missing: {src_path}")
        v, f = parse_collada(src_path)
        out = dst / obj
        write_obj(out, v, f)
        ext = (v.max(0) - v.min(0)).round(3)
        print(f"[husky] {dae} -> {out}  ({len(v)}v {len(f)}f, ext={ext})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--dst", required=True, type=Path)
    args = ap.parse_args()
    convert(args.src.expanduser(), args.dst.expanduser())


if __name__ == "__main__":
    main()
