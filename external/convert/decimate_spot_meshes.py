"""Decimate the Spot visual meshes for viewer/offscreen-render performance.

Run headless with Blender (tested with 4.2):

    blender --background --python external/convert/decimate_spot_meshes.py -- <src_assets_dir> <out_meshes_dir> [cap]

For every ``*.obj`` in ``src_assets_dir`` it applies a Collapse Decimate modifier down to a triangle
``cap`` (default 2000) and writes the result to ``out_meshes_dir``; meshes already under the cap are
written through unchanged. MuJoCo builds the convex hull of a mesh for collision, so reducing the
*visual* triangle count does not change contact behaviour -- the locomotion policy walks identically.

This mirrors the decimation used for the Unitree G1 in ``roqsim_humanoid``. The source assets come
from MuJoCo Menagerie's ``boston_dynamics_spot/assets`` (see ``THIRD_PARTY.md``).
"""

import sys
from pathlib import Path

import bpy


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def tri_count(obj):
    mesh = obj.data
    return sum(len(p.vertices) - 2 for p in mesh.polygons)


def process(src_dir, out_dir, cap):
    src_dir, out_dir = Path(src_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    total_before = total_after = 0
    for obj_path in sorted(src_dir.glob("*.obj")):
        clear_scene()
        bpy.ops.wm.obj_import(filepath=str(obj_path))
        objs = [o for o in bpy.context.scene.objects if o.type == "MESH"]
        before = sum(tri_count(o) for o in objs)
        for o in objs:
            if tri_count(o) > cap:
                bpy.context.view_layer.objects.active = o
                mod = o.modifiers.new(name="decimate", type="DECIMATE")
                mod.decimate_type = "COLLAPSE"
                mod.ratio = min(1.0, cap / max(1, tri_count(o)))
                bpy.ops.object.modifier_apply(modifier=mod.name)
        after = sum(tri_count(o) for o in objs)
        bpy.ops.object.select_all(action="SELECT")
        out_path = out_dir / obj_path.name
        bpy.ops.wm.obj_export(
            filepath=str(out_path),
            export_selected_objects=True,
            export_materials=False,
            export_normals=True,
            export_uv=False,
        )
        total_before += before
        total_after += after
        print(f"  {obj_path.name}: {before} -> {after} tris")
    pct = 100.0 * total_after / max(1, total_before)
    print(f"TOTAL: {total_before} -> {total_after} tris ({pct:.1f}%)")


def main():
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    if len(argv) < 2:
        print("usage: blender --background --python decimate_spot_meshes.py -- <src> <out> [cap]")
        sys.exit(1)
    src, out = argv[0], argv[1]
    cap = int(argv[2]) if len(argv) > 2 else 2000
    process(src, out, cap)


if __name__ == "__main__":
    main()
