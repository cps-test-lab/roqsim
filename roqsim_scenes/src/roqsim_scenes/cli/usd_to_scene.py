"""Convert an IsaacSim/USD world into a roqsim static scene (per-object OBJ + a JSON manifest).

Run with a bundled Blender (4.x, ships a USD importer)::

    blender --background --roqsim scenes usd-to-scene -- \
        <input.usd> src/roqsim_scenes/scenes/<name> <name> [--unit-scale 0.01]

For each USD mesh prim it writes ``<name>/meshes/<obj>.obj`` (world-space, in **metres**, MuJoCo
Z-up) and records the prim in ``<name>/scene.json`` with its diffuse ``rgba`` and a ``collide`` flag.
Stage 2, ``scene_to_mjcf.py``, reads that manifest and bakes one MuJoCo geom per prim into a plain
MJCF world -- rendered from the true triangles, collided by each prim's own convex hull.

Why per-object (not one merged mesh): MuJoCo collides a mesh by its **convex hull**. One merged
building mesh hulls into a solid block that fills the interior (why ``roqsim_mobile``'s floorplan
disables mesh collision); keeping the USD's separate prims means each wall/desk hulls into a sane
solid on its own, so the scene is collidable for free.

IsaacSim authors in centimetres by default, so the default ``--unit-scale 0.01`` converts to metres;
Blender's USD importer already resolves the stage up-axis to Blender's Z-up, and we export OBJ with
``up=Z, forward=Y`` so the coordinates land in MuJoCo's frame unchanged.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

import bpy
import mathutils

# Default gray for a prim whose material carries no usable diffuse colour.
_DEFAULT_RGBA = [0.7, 0.7, 0.72, 1.0]


def _argv() -> list[str]:
    """Args after the ``--`` separator Blender uses to hand script args through."""
    return sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []


def _safe(name: str, used: set[str]) -> str:
    """A filesystem-safe, unique slug for a prim name (OBJ filename + geom name)."""
    slug = re.sub(r"[^0-9A-Za-z_.-]", "_", name).strip("_") or "mesh"
    base, i = slug, 1
    while slug in used:
        slug = f"{base}_{i}"
        i += 1
    used.add(slug)
    return slug


def _rgba(obj: bpy.types.Object) -> list[float]:
    """Diffuse RGBA of a prim's first material (Principled BSDF base colour, else viewport colour)."""
    for slot in obj.material_slots:
        mat = slot.material
        if not mat:
            continue
        if mat.use_nodes:
            bsdf = next((n for n in mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED"), None)
            if bsdf is not None:
                c = bsdf.inputs["Base Color"].default_value
                a = bsdf.inputs["Alpha"].default_value
                return [round(c[0], 4), round(c[1], 4), round(c[2], 4), round(float(a), 4)]
        c = mat.diffuse_color
        return [round(c[0], 4), round(c[1], 4), round(c[2], 4), round(c[3], 4)]
    return list(_DEFAULT_RGBA)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("input_usd")
    ap.add_argument("out_dir")
    ap.add_argument("scene_name")
    ap.add_argument("--unit-scale", type=float, default=0.01, help="USD units -> metres (cm=0.01)")
    args = ap.parse_args(_argv())

    meshes_dir = os.path.join(args.out_dir, "meshes")
    os.makedirs(meshes_dir, exist_ok=True)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.wm.usd_import(filepath=os.path.abspath(args.input_usd), import_materials=True)

    scale = mathutils.Matrix.Scale(args.unit_scale, 4)
    objs = [o for o in bpy.data.objects if o.type == "MESH" and o.data.polygons]

    used: set[str] = set()
    entries: list[dict] = []
    lo = [float("inf")] * 3
    hi = [float("-inf")] * 3

    for obj in objs:
        # Bake the cm->m scale into the world transform so the exported OBJ is metres at the origin.
        obj.matrix_world = scale @ obj.matrix_world

        for c in obj.bound_box:
            w = obj.matrix_world @ mathutils.Vector(c)
            for i in range(3):
                lo[i] = min(lo[i], w[i])
                hi[i] = max(hi[i], w[i])

        slug = _safe(obj.name, used)
        rel = os.path.join("meshes", f"{slug}.obj")
        bpy.ops.object.select_all(action="DESELECT")
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        bpy.ops.wm.obj_export(
            filepath=os.path.join(args.out_dir, rel),
            export_selected_objects=True,
            apply_modifiers=True,
            export_materials=False,
            export_triangulated_mesh=True,
            forward_axis="Y",
            up_axis="Z",
        )
        entries.append({"name": slug, "mesh": rel, "rgba": _rgba(obj), "collide": True})

    manifest = {
        "name": args.scene_name,
        "source": os.path.basename(args.input_usd),
        "unit_scale": args.unit_scale,
        "bounds_min": [round(v, 4) for v in lo],
        "bounds_max": [round(v, 4) for v in hi],
        "objects": entries,
    }
    with open(os.path.join(args.out_dir, "scene.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)

    print(
        f"SCENE_OK objects={len(entries)} bounds_min={manifest['bounds_min']} "
        f"bounds_max={manifest['bounds_max']}"
    )


if __name__ == "__main__":
    main()
