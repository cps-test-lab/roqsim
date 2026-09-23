"""Convert the Clearpath Husky A200 meshes to MuJoCo OBJ.

Usage::

    # the PACS top plate and the horizontal PACS bracket, from pinned clearpath_common checkouts
    python external/convert/convert_husky_meshes.py \
        --dst roqsim_mobile/src/roqsim_mobile/models/husky_a200/meshes

    # also the chassis, wheel, top chassis and bumper, from a husky_description checkout
    python external/convert/convert_husky_meshes.py \
        --src /path/to/husky_description/meshes \
        --dst roqsim_mobile/src/roqsim_mobile/models/husky_a200/meshes

Two sources:

* ``husky_description`` (ROS 1, BSD-3) Collada meshes: ``base_link``, ``wheel``, ``top_plate`` and
  ``bumper``. Its ``top_plate.dae`` holds the geometry Clearpath's ROS 2 description ships as
  ``clearpath_platform_description/meshes/a200/top_chassis.stl`` (x +-0.406, y +-0.2069,
  z -0.0021 .. 0.2243, equal to the millimetre), the chassis cover whose top is ``default_mount``
  (a200.urdf.xacro:156-159, z 0.224) -- so it is written as ``husky_top_chassis.obj``.
* ``clearpath_common`` (BSD-3) STL meshes, fetched at a pinned commit by ``sources.resolve_source``:
  ``attachments/pacs_top_plate.stl`` at ``CLEARPATH_COMMON_COMMIT`` (the pin
  build_scanner_devices.py uses), and ``clearpath_mounts_description/meshes/pacs/
  bracket_horizontal.stl``, which that commit does not carry, at ``CLEARPATH_COMMON_MOUNTS_COMMIT``
  on ``jazzy``.

Why a direct copy instead of Blender? These sources are Collada with ``<up_axis>Z_UP`` and metre STL
whose raw geometry is *already* in the exact URDF/MuJoCo link frame. They are visual-only geoms for
which MuJoCo recomputes normals, and all collision is authored as primitives, so no decimation or
smoothing is needed. A direct geometry copy is therefore both simplest and highest-fidelity: exact
source vertices and triangles, zero reinterpretation.

Verified in the MuJoCo geom (rendered) frame -- note MuJoCo re-frames ``mesh_vert`` to a canonical
inertial frame internally, so orientation must be checked via ``mesh_quat``/``mesh_pos``, not raw
``mesh_vert``:
    base  ext (x,y,z) = (0.809, 0.571, 0.228) m   -> length / width / height, sits on z>=0
    wheel ext (x,y,z) = (0.356, 0.114, 0.355) m   -> axle along Y; the tyre mesh is r 0.178, larger
                                                     than the 0.1651 m rolling radius Clearpath's
                                                     description collides with (its own outdoor.dae
                                                     is the same size)

Each STL's extents are checked against the vendor collision box before it is written, so a wrong file
or a unit change fails the conversion.
"""

from __future__ import annotations

import argparse
import re
import struct
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sources import resolve_source  # noqa: E402

CLEARPATH_COMMON_URL = "https://github.com/clearpathrobotics/clearpath_common.git"
#: `jazzy` -- the pin build_scanner_devices.py, build_ridgeback_mjcf.py and build_warthog_mjcf.py use.
CLEARPATH_COMMON_COMMIT = "b0f6d920422ad302372a1c65e31d61648da884ed"
#: `jazzy`, the first pin that carries clearpath_mounts_description/meshes/pacs/.
CLEARPATH_COMMON_MOUNTS_COMMIT = "33e4b311dd9a2dcaa8e8d262ae75fab2c53e560b"

# source .dae  ->  output .obj basename (husky_ prefix avoids clashing with turtlebot meshes)
MESHES = {
    "base_link.dae": "husky_base.obj",
    "wheel.dae": "husky_wheel.obj",
    "top_plate.dae": "husky_top_chassis.obj",
    "bumper.dae": "husky_bumper.obj",
}

#: output .obj -> (checkout name, commit, sparse path, file in the checkout, expected extent (m)).
#: The extents are the vendor collision boxes: top_plate.urdf.xacro:117-131 (0.67 x 0.59; the mesh is
#: 6.4 mm thick against the box's 6.35) and pacs/bracket.urdf.xacro:36-42 (0.09 x 0.09 x 0.010125).
STL_MESHES = {
    "husky_pacs_top_plate.obj": (
        "clearpath_common",
        CLEARPATH_COMMON_COMMIT,
        None,
        "clearpath_platform_description/meshes/a200/attachments/pacs_top_plate.stl",
        (0.67, 0.59, 0.0064),
    ),
    "husky_bracket_horizontal.obj": (
        "clearpath_common_mounts",
        CLEARPATH_COMMON_MOUNTS_COMMIT,
        "clearpath_mounts_description",
        "clearpath_mounts_description/meshes/pacs/bracket_horizontal.stl",
        (0.09, 0.09, 0.0101),
    ),
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


def parse_stl(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (vertices Nx3, triangles Mx3 int) of a binary or ASCII STL, shared corners merged."""
    raw = path.read_bytes()
    count = struct.unpack("<I", raw[80:84])[0] if len(raw) >= 84 else -1
    if len(raw) == 84 + 50 * count:
        record = np.dtype([("normal", "<3f4"), ("corners", "<9f4"), ("attr", "<u2")])
        corners = np.frombuffer(raw, dtype=record, count=count, offset=84)["corners"]
    else:
        found = re.findall(rb"vertex\s+(\S+)\s+(\S+)\s+(\S+)", raw)
        if not found or len(found) % 3:
            raise ValueError(f"{path} is neither a binary nor an ASCII STL")
        corners = np.array(found, dtype=np.float64)
    corners = np.asarray(corners, dtype=np.float64).reshape(-1, 3)
    verts, index = np.unique(np.round(corners, 7), axis=0, return_inverse=True)
    return verts, index.reshape(-1, 3)


def write_obj(path: Path, v: np.ndarray, f: np.ndarray, source: str) -> None:
    lines = [
        f"# converted from {source} by convert_husky_meshes.py ({len(v)} verts, {len(f)} tris)"
    ]
    lines += [f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in v]
    lines += [f"f {a + 1} {b + 1} {c + 1}" for a, b, c in f]  # OBJ is 1-indexed
    path.write_text("\n".join(lines) + "\n")


def convert_collada(src: Path, dst: Path) -> None:
    for dae, obj in MESHES.items():
        src_path = src / dae
        if not src_path.exists():
            raise FileNotFoundError(f"source mesh missing: {src_path}")
        v, f = parse_collada(src_path)
        out = dst / obj
        write_obj(out, v, f, "Collada")
        ext = (v.max(0) - v.min(0)).round(3)
        print(f"[husky] {dae} -> {out}  ({len(v)}v {len(f)}f, ext={ext})")


def convert_clearpath(dst: Path) -> None:
    for obj, (name, commit, sparse, file, expected) in STL_MESHES.items():
        checkout = resolve_source(name, CLEARPATH_COMMON_URL, commit, sparse=sparse)
        src_path = checkout / file
        if not src_path.is_file():
            raise FileNotFoundError(f"source mesh missing: {src_path}")
        v, f = parse_stl(src_path)
        ext = v.max(0) - v.min(0)
        if not np.allclose(ext, expected, atol=5e-4):
            raise ValueError(f"{file} @ {commit[:7]} spans {ext.round(4)}, expected {expected}")
        out = dst / obj
        write_obj(out, v, f, f"clearpath_common @ {commit[:7]} {file}")
        print(f"[husky] {file} @ {commit[:7]} -> {out}  ({len(v)}v {len(f)}f, ext={ext.round(4)})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--src", type=Path, help="husky_description/meshes; omit to convert only the STLs"
    )
    ap.add_argument("--dst", required=True, type=Path)
    args = ap.parse_args()
    dst = args.dst.expanduser()
    dst.mkdir(parents=True, exist_ok=True)
    if args.src is not None:
        convert_collada(args.src.expanduser(), dst)
    convert_clearpath(dst)


if __name__ == "__main__":
    main()
