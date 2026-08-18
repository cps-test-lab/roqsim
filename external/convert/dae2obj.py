"""Exact DAE->OBJ conversion via pycollada (MuJoCo cannot load Collada; stock Linux Blender lacks
OpenCOLLADA).

Reads each Collada file, loads vertices **verbatim** (the source authors them in the URDF link-local
frame with identity visual origins; the ``<up_axis>`` tag is vestigial -- honouring it explodes arms).
Normals are dropped so MuJoCo derives them from winding (a Blender obj_export axis remap reflected the
meshes and inverted the winding). No remeshing -- geometry is preserved exactly.

MuJoCo also ignores OBJ/MTL materials, so a mesh's per-material colours are lost unless split out. Each
pycollada *bound primitive* is exactly one material, so this writes:
  - ``<stem>.obj``            the combined mesh (all materials), used by the MJCF compile / kinematics
  - ``<stem>__m<k>.obj``      one sub-mesh per material (only when the mesh has >1 material)
  - ``materials.json``        {"<stem>": [["<sub-stem>", [r,g,b]], ...]} -- diffuse colour per sub-mesh
build_g2_mjcf.py reads materials.json to emit MJCF <material>s and one visual sub-geom per material.

    python external/convert/dae2obj.py <src_meshes_dir> <out_meshes_dir>
"""

import json
import sys
from pathlib import Path

import collada
import numpy as np


def _write_obj(path, V, F):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for p in V:
            f.write(f"v {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
        for t in F:
            f.write(f"f {t[0] + 1} {t[1] + 1} {t[2] + 1}\n")  # OBJ is 1-indexed


def _diffuse(mat):
    """(r,g,b) from a bound Material's effect diffuse, or None for a texture / missing."""
    try:
        d = mat.effect.diffuse
    except AttributeError:
        return None
    if isinstance(d, (tuple, list, np.ndarray)) and len(d) >= 3:
        return [round(float(x), 4) for x in d[:3]]
    return None  # a Map (texture) -- none of this robot's meshes hit this


def convert(dae_path, out_dir, rel_stem):
    """Write combined + per-material sub-OBJs; return [(sub_stem, [r,g,b]), ...]."""
    c = collada.Collada(str(dae_path), ignore=[collada.common.DaeError])
    if c.scene is None:
        raise RuntimeError(f"no scene in {dae_path}")
    prims = []  # (V, F, color)
    for geom in c.scene.objects("geometry"):
        for prim in geom.primitives():
            v, idx = prim.vertex, prim.vertex_index
            if v is None or idx is None or len(idx) == 0:
                continue
            prims.append((np.asarray(v), np.asarray(idx).reshape(-1, 3), _diffuse(prim.material)))
    if not prims:
        raise RuntimeError(f"no triangles in {dae_path}")

    # combined (all materials) -- referenced by the URDF/compile
    off, allV, allF = 0, [], []
    for v, f, _ in prims:
        allV.append(v)
        allF.append(f + off)
        off += len(v)
    _write_obj(out_dir / (rel_stem + ".obj"), np.vstack(allV), np.vstack(allF))

    # per-material sub-meshes (merge primitives that share a colour)
    by_color = {}
    for v, f, col in prims:
        key = tuple(col) if col else (0.5, 0.5, 0.5)
        by_color.setdefault(key, []).append((v, f))
    out = []
    multi = len(by_color) > 1
    for k, (color, parts) in enumerate(by_color.items()):
        off, vs, fs = 0, [], []
        for v, f in parts:
            vs.append(v)
            fs.append(f + off)
            off += len(v)
        sub_stem = f"{rel_stem}__m{k}" if multi else rel_stem
        if multi:
            _write_obj(out_dir / (sub_stem + ".obj"), np.vstack(vs), np.vstack(fs))
        out.append([sub_stem, list(color)])
    return out


def main():
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    materials = {}
    daes = sorted(src.rglob("*.dae"))
    for dae in daes:
        rel_stem = str(dae.relative_to(src).with_suffix(""))
        subs = convert(dae, out, rel_stem)
        materials[rel_stem] = subs
        print(f"  {rel_stem}: {len(subs)} material(s)")
    (out / "materials.json").write_text(json.dumps(materials, indent=0))
    print(f"TOTAL: {len(daes)} meshes, materials.json written")


if __name__ == "__main__":
    main()
