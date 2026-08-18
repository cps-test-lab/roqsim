"""View a single mesh file in isolation — minimal repro for meshes that render
offscreen but vanish in the interactive viewer (e.g. the Oli forearm/thigh shells).

Live viewer (the path that drops meshes on this machine):

    .venv/bin/python external/convert/view_mesh.py \
        roqsim_humanoid/src/roqsim_humanoid/models/meshes/oli/left_hip_yaw_link.STL

Offscreen reference render (known-good path) for the same mesh:

    .venv/bin/python external/convert/view_mesh.py --offscreen /tmp/mesh.png <mesh.STL>

If the PNG shows the mesh but the viewer window does not, the mesh is fine and
the on-screen GL context is the culprit.
"""

import argparse
import pathlib
import sys

import mujoco


def build_model(mesh_path: pathlib.Path) -> mujoco.MjModel:
    xml = f"""
<mujoco model="single_mesh">
  <visual>
    <global offwidth="1200" offheight="900"/>
  </visual>
  <asset>
    <mesh name="m" file="{mesh_path.resolve()}"/>
  </asset>
  <worldbody>
    <light pos="0.5 -0.5 1.5" dir="-0.3 0.3 -1"/>
    <light pos="-0.5 0.5 1.5" dir="0.3 -0.3 -1"/>
    <geom type="mesh" mesh="m" rgba="0.75 0.75 0.75 1"/>
  </worldbody>
</mujoco>
"""
    return mujoco.MjModel.from_xml_string(xml)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mesh", type=pathlib.Path, help="mesh file (STL/OBJ/MSH)")
    ap.add_argument(
        "--offscreen",
        metavar="OUT_PNG",
        type=pathlib.Path,
        help="render to PNG via the offscreen path instead of opening the viewer",
    )
    args = ap.parse_args()
    if not args.mesh.is_file():
        sys.exit(f"error: no such mesh file: {args.mesh}")

    model = build_model(args.mesh)
    # Loaded-geometry evidence independent of any window: if these are non-zero,
    # the mesh file itself is sound.
    print(f"mesh loaded: {model.mesh_vertnum[0]} vertices, {model.mesh_facenum[0]} faces")

    if args.offscreen:
        import os

        os.environ.setdefault("MUJOCO_GL", "egl")
        import numpy as np
        from PIL import Image

        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        renderer = mujoco.Renderer(model, 900, 1200)
        cam = mujoco.MjvCamera()
        # frame the mesh from its bounding box
        verts = model.mesh_vert[: model.mesh_vertnum[0]]
        center = verts.mean(axis=0)
        extent = float(np.linalg.norm(verts.max(axis=0) - verts.min(axis=0)))
        cam.lookat[:] = center
        cam.distance = 2.0 * extent
        cam.azimuth, cam.elevation = 135, -20
        renderer.update_scene(data, cam)
        Image.fromarray(renderer.render()).save(args.offscreen)
        print(f"offscreen render written to {args.offscreen}")
    else:
        from mujoco import viewer

        viewer.launch(model)


if __name__ == "__main__":
    main()
