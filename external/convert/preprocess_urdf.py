"""Preprocess the expanded G2 URDF into a MuJoCo-compilable URDF + flat meshdir.

- flattens mesh refs to unique names (visual .obj from dae2obj's output, collision .STL from the raw tree)
- welds the 4 swerve steer joints (…_joint1) to fixed  (diff-drive-as-caster approximation)
- injects a <mujoco><compiler> block so MuJoCo reads the meshes and keeps the tree intact
- flattens the per-material sub-meshes too, and writes materials_flat.json (mesh -> [(submesh, rgb)])
  for build_g2_mjcf.py to recover colours

Run from a work dir (or pass it as argv[1]) containing:
    g2_crs_omnipicker.urdf   (expanded xacro)
    g2col/                   (dae2obj output: combined + per-material .obj + materials.json)
    g2meshes/                (original source meshes incl. .../convex/*.STL)
It writes g2_mj.urdf + a flat mjmesh/ + materials_flat.json there; MuJoCo then compiles g2_mj.urdf to
g2_base.xml and build_g2_mjcf.py turns that into the model.
"""

import json
import re
import shutil
import sys
from pathlib import Path

SCR = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path.cwd()
URDF = SCR / "g2_crs_omnipicker.urdf"
VIS = SCR / "g2col"  # dae2obj output: combined <stem>.obj + per-material <stem>__m<k>.obj + materials.json
RAW = SCR / "g2meshes"  # original tree (has …/convex/*.STL collision hulls)
MESHDIR = SCR / "mjmesh"  # flat output meshdir (unique names)
OUT = SCR / "g2_mj.urdf"

PREFIX = "package://genie_sim_robot_model/robots/genie/g2/meshes/"
STEER = [
    "idx111_chassis_lwheel_front_joint1",
    "idx131_chassis_rwheel_front_joint1",
    "idx141_chassis_rwheel_rear_joint1",
    "idx121_chassis_lwheel_rear_joint1",
]


def flat_name(rel: str) -> str:
    """G2/base_link.dae -> g2_base_link ; G2/base_link__m0 -> g2_base_link_m0 ; etc."""
    return re.sub(r"[^A-Za-z0-9]+", "_", rel.rsplit(".", 1)[0] if "." in rel else rel).strip("_").lower()


def main():
    if MESHDIR.exists():
        shutil.rmtree(MESHDIR)
    MESHDIR.mkdir(parents=True)
    txt = URDF.read_text()

    # rewrite every URDF mesh filename to a flat unique file in MESHDIR, copying the right source file
    # (visual .dae -> dae2obj's combined .obj; collision .STL -> raw). The combined visual mesh is what
    # the compile uses; build_g2_mjcf.py later splits multi-material ones into the per-material subs.
    def repl(m):
        rel = m.group(1)
        stem = flat_name(rel)
        if rel.lower().endswith(".dae"):
            src, dst = VIS / (rel[:-4] + ".obj"), MESHDIR / (stem + ".obj")
        else:
            src, dst = RAW / rel, MESHDIR / (stem + Path(rel).suffix)
        if not dst.exists():
            if not src.exists():
                raise FileNotFoundError(src)
            shutil.copy(src, dst)
        return f'filename="{dst.name}"'

    txt = re.sub(r'filename="' + re.escape(PREFIX) + r'([^"]*)"', repl, txt)

    # weld steer joints
    for j in STEER:
        txt = re.sub(
            r'(<joint name="' + re.escape(j) + r'") type="revolute"', r'\1 type="fixed"', txt
        )

    # inject mujoco compiler block right after <robot ...>
    inject = (
        '\n  <mujoco><compiler meshdir="mjmesh" balanceinertia="true" '
        'discardvisual="false" fusestatic="false" strippath="false" '
        'autolimits="true"/></mujoco>\n'
    )
    txt = re.sub(r"(<robot\b[^>]*>)", r"\1" + inject, txt, count=1)
    OUT.write_text(txt)

    # flatten the per-material sub-meshes + emit materials_flat.json keyed by flat mesh name
    materials = json.loads((VIS / "materials.json").read_text())
    flat_materials = {}
    for stem, subs in materials.items():
        flat_materials[flat_name(stem)] = [[flat_name(sub), rgb] for sub, rgb in subs]
        for sub, _ in subs:
            dst = MESHDIR / (flat_name(sub) + ".obj")
            if not dst.exists():
                shutil.copy(VIS / (sub + ".obj"), dst)
    (SCR / "materials_flat.json").write_text(json.dumps(flat_materials, indent=0))

    n = len(list(MESHDIR.iterdir()))
    print(f"wrote {OUT} + materials_flat.json  ({n} unique meshes in {MESHDIR})")


if __name__ == "__main__":
    main()
