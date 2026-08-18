"""Convert the ROBOTIS TurtleBot3 Waffle STL meshes to decimated MuJoCo OBJ.

Usage::

    python external/convert/convert_turtlebot3_meshes.py \
        --src /path/to/robotis_mujoco_menagerie/robotis_tb3/assets \
        --dst roqsim_mobile/src/roqsim_mobile/models/turtlebot3_waffle/meshes

    # or let it clone the pinned source itself into a temp dir:
    python external/convert/convert_turtlebot3_meshes.py --clone \
        --dst roqsim_mobile/src/roqsim_mobile/models/turtlebot3_waffle/meshes

Provenance: ROBOTIS `robotis_mujoco_menagerie`, directory `robotis_tb3`, pinned at commit
``d8344c0dbe7a00208d0301111523dde65efc174a`` (Apache-2.0; see ``turtlebot3_waffle_LICENSE`` beside the
model). The source ships MJCFs for the Burger and the **Waffle Pi** plus, in ``assets/``, the plain
**Waffle** base mesh — which is the platform `hussain_maze_ros2_2024` actually names, so this port uses
``waffle_base.stl`` rather than the Waffle Pi's.

Why decimate: ``waffle_base.stl`` is 325,838 triangles (the Waffle Pi's is 157,576). The vendor MJCF
uses that mesh for BOTH visual and collision geometry. In roqsim the meshes are visual-only — all
collision is authored as primitives (chassis box, wheel cylinders, caster spheres) — so the budget here
is about model size and render cost, not physics.

**Budgets are per-part and were chosen by LOOKING at the result, not by ratio.** The tires and the
scanner housing are smooth solids and survive aggressive collapse; the chassis is a lattice and does
not (see the comment on its entry below). Rendering the converted mesh is the only check that catches
this — extents, vertex counts and masses all pass on a shredded mesh.

Scale: the source STLs are in millimetres (the vendor MJCF applies ``scale="0.001 0.001 0.001"``).
``--scale 0.001`` bakes metres into the OBJ instead, so the MJCF needs no scale attribute and the mesh
files are self-describing.

Verified extents after conversion (metres, raw OBJ bounds):
    waffle_base  (x, y, z) = (0.271, 0.279, 0.124)  chassis plate; datasheet 0.281 x 0.306 x 0.141
                                                     includes bumper/camera/lidar protrusions
    left_tire / right_tire = (0.066, 0.018, 0.066)  radius 0.033, width 0.0182, axle along Y
    lds                    = (0.094, 0.069, 0.039)  LDS-01 scanner housing
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

SOURCE_REPO = "https://github.com/ROBOTIS-GIT/robotis_mujoco_menagerie.git"
SOURCE_COMMIT = "d8344c0dbe7a00208d0301111523dde65efc174a"
SOURCE_SUBDIR = "robotis_tb3/assets"

# source .stl -> (output .obj, triangle budget). Budgets are per-part: the chassis carries the
# silhouette and gets the largest share; the tires are near-cylinders that collapse very well.
MESHES = {
    # 100k, not less: the Waffle plate is a LATTICE (the "waffle" grid of holes and
    # standoffs), and collapse decimation shreds it. At a 12k budget (which the collapse
    # ratio actually delivered as 29k) the plate rendered as jagged spikes with holes torn
    # through it while every numeric check — extents, vertex count, mass — still passed.
    # At 100k it is visually indistinguishable from the 326k source, at a third the size.
    # Cost of carrying it: essentially nothing, because the lidar plugin excludes
    # base_link from its ray casts and all collision is primitives.
    "waffle_base.stl": ("waffle_base.obj", 100000),
    "left_tire.stl": ("left_tire.obj", 2000),
    "right_tire.stl": ("right_tire.obj", 2000),
    "lds.stl": ("lds.obj", 3000),
}

# reduce_mesh.py lives in roqsim_assets/tools/ (the prop pipeline's step 2); reused here rather than
# re-implementing decimation. external/ is a sibling of the packages, hence parents[2].
REDUCE = Path(__file__).resolve().parents[2] / "roqsim_assets/tools/reduce_mesh.py"


def _clone(dst: Path) -> Path:
    """Shallow-clone the pinned source and return its assets directory."""
    subprocess.run(["git", "init", "-q", str(dst)], check=True)
    subprocess.run(["git", "-C", str(dst), "remote", "add", "origin", SOURCE_REPO], check=True)
    subprocess.run(
        ["git", "-C", str(dst), "fetch", "-q", "--depth", "1", "origin", SOURCE_COMMIT], check=True
    )
    subprocess.run(["git", "-C", str(dst), "checkout", "-q", "FETCH_HEAD"], check=True)
    return dst / SOURCE_SUBDIR


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--src", type=Path, help=f"{SOURCE_SUBDIR} of a checkout of {SOURCE_REPO}")
    ap.add_argument("--clone", action="store_true", help="clone the pinned source into a temp dir")
    ap.add_argument("--dst", type=Path, required=True, help="output mesh directory")
    ap.add_argument(
        "--blender", default="blender", help="Blender binary (default: 'blender' on PATH)"
    )
    args = ap.parse_args()

    if not REDUCE.is_file():
        raise SystemExit(f"missing mesh reducer: {REDUCE}")
    if bool(args.src) == bool(args.clone):
        raise SystemExit("pass exactly one of --src or --clone")

    tmp = None
    if args.clone:
        tmp = tempfile.TemporaryDirectory()
        src = _clone(Path(tmp.name) / "robotis_mujoco_menagerie")
    else:
        src = args.src
    if not src.is_dir():
        raise SystemExit(f"source mesh directory not found: {src}")

    args.dst.mkdir(parents=True, exist_ok=True)
    for stl, (obj, budget) in MESHES.items():
        s = src / stl
        if not s.is_file():
            raise SystemExit(f"missing source mesh: {s}")
        out = args.dst / obj
        cmd = [
            sys.executable,
            str(REDUCE),
            str(s),
            str(out),
            "--target-faces",
            str(budget),
            "--scale",
            "0.001",
            "--blender",
            args.blender,
        ]
        print("+", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)
        if not out.is_file():
            raise SystemExit(f"reducer produced no output: {out}")

    # The MTLs the reducer emits are noise here: MuJoCo ignores OBJ materials, and every geom gets an
    # MJCF <material> instead (the source meshes carry flat vendor colours, no textures).
    for mtl in args.dst.glob("*.mtl"):
        mtl.unlink()

    print(f"\nwrote {len(MESHES)} meshes to {args.dst}")
    print(f"source: {SOURCE_REPO} @ {SOURCE_COMMIT} ({SOURCE_SUBDIR})")
    if tmp:
        tmp.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
