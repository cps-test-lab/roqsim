"""Blender-side processing for the Livox Mid-360 meshes (run via ``blender -b -P``).

Reads one Open CASCADE OBJ (produced by cascadio): the housing assembly, and writes two MuJoCo-ready
OBJs -- grey housing (``mid360_body``) and laser dome window (``mid360_dome``). Both are decimated,
scaled mm->m, have their base dropped to z=0 with the housing centred on x=y=0, and are exported axes
1:1 (Z up).

The dome is the housing CAD's material ``mat_2``. The sensor's FOV is no longer a baked mesh --
``spawn_sensor`` synthesises it as an angular sector from the datasheet angles in
``mid360.manifest.yaml`` -- so this script produces only the two housing meshes. Invoked by
``livox_mid360_convert.py``; not run directly.

Args after ``--``:  <housing_obj> <body_out> <dome_out>
"""

import sys

import bpy
from mathutils import Vector

BODY_TARGET = 6500  # target triangle counts after decimation
DOME_TARGET = 900

argv = sys.argv[sys.argv.index("--") + 1 :]
if len(argv) != 3:
    raise SystemExit("usage: blender -b -P livox_mid360_blender.py -- "
                     "<housing_obj> <body_out> <dome_out>")
housing_obj, body_out, dome_out = argv


def reset():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def import_join(path):
    bpy.ops.wm.obj_import(filepath=path)
    meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    for o in meshes:
        o.select_set(True)
    bpy.context.view_layer.objects.active = meshes[0]
    if len(meshes) > 1:
        bpy.ops.object.join()
    return bpy.context.view_layer.objects.active


def tris(o):
    return sum(len(p.vertices) - 2 for p in o.data.polygons)


def decimate(o, target):
    n = tris(o)
    if n > target:
        m = o.modifiers.new("dec", "DECIMATE")
        m.ratio = target / n
        bpy.context.view_layer.objects.active = o
        bpy.ops.object.modifier_apply(modifier=m.name)


def material_slot(o, token):
    return next(
        (i for i, s in enumerate(o.material_slots) if s.material and token in s.material.name),
        None,
    )


def world_bbox_min_z(objs):
    return min((o.matrix_world @ Vector(c)).z for o in objs for c in o.bound_box)


def export(o, path):
    for x in bpy.context.scene.objects:
        x.select_set(False)
    o.select_set(True)
    bpy.context.view_layer.objects.active = o
    bpy.ops.wm.obj_export(
        filepath=path,
        export_selected_objects=True,
        forward_axis="Y",
        up_axis="Z",
        export_materials=False,
        export_normals=True,
        export_uv=False,
    )
    print(f"wrote {path}")


# --- housing assembly: split the dome (mat_2) from the body; compute the base-drop here -----------
reset()
obj = import_join(housing_obj)
obj.name = "mid360"
dome_slot = material_slot(obj, "mat_2")
if dome_slot is None:
    raise SystemExit("dome material 'mat_2' not found in the housing OBJ")
bpy.ops.object.mode_set(mode="OBJECT")
for p in obj.data.polygons:
    p.select = p.material_index == dome_slot
bpy.ops.object.mode_set(mode="EDIT")
bpy.ops.mesh.separate(type="SELECTED")
bpy.ops.object.mode_set(mode="OBJECT")

parts = [o for o in bpy.context.scene.objects if o.type == "MESH"]
dome = min(parts, key=lambda o: len(o.data.polygons))
body = max(parts, key=lambda o: len(o.data.polygons))
dome.name, body.name = "mid360_dome", "mid360_body"

# Bake the importer's Y-up->Z-up rotation, then scale mm->m.
for o in parts:
    o.select_set(True)
bpy.context.view_layer.objects.active = body
bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)
for o in parts:
    o.scale = (0.001, 0.001, 0.001)
bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

base_shift = -world_bbox_min_z(parts)  # drop the base to z=0
for o in parts:
    o.location.z += base_shift
bpy.ops.object.transform_apply(location=True, rotation=False, scale=True)

decimate(body, BODY_TARGET)
decimate(dome, DOME_TARGET)
export(body, body_out)
export(dome, dome_out)
print(f"BASE_SHIFT_m {base_shift:.7f}")
print("done")
