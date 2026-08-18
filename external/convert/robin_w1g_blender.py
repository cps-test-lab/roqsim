"""Blender-side processing for the Seyond Robin W1G meshes (run via ``blender -b -P``).

Reads the Open CASCADE OBJ (produced by cascadio from Seyond's Robin W1G STEP) and writes one
MuJoCo-ready OBJ: the dark sensor housing (``robin_w1g_body``). The body is reframed from the CAD axes
into roqsim's mount convention, scaled mm->m, centred on x=y=0 with its base dropped to z=0,
decimated, and exported axes 1:1 (Z up). The sensor's FOV is no longer a baked mesh --
``spawn_sensor`` synthesises it as a bounded 120 deg x 70 deg angular sector from the datasheet angles
in ``robin_w1g.manifest.yaml`` -- so this script produces only the housing mesh.

CAD -> sim reframe. The Creo export has +y up (device height 85 mm), +z pointing to the rear
(connector/mounting plate), and the optical window on the -z front. roqsim's standalone lidar
mount convention is +x = boresight (forward), +z = up, base on z=0 -- the same frame the
``livox_mid360`` caster and the ``robin_w1g`` site are authored against. The mapping that takes the
window to +x and height to +z is:

    sim_x = -cad_z   sim_y = -cad_x   sim_z = +cad_y      (a proper rotation, det = +1)

The device is a wedge: a near-vertical optical window on the front (+x after reframing), a top face
sloping down toward the rear, the connector and product-label/datum face on the rear/side. Only the
housing is kept: the CAD's giant ~29 m context solid (Open CASCADE material ``mat_10``, and anything
whose bbox is larger than the device) is dropped. Invoked by ``robin_w1g_convert.py``; not run
directly.

Args after ``--``:  <robin_obj> <body_out>
"""

import sys

import bpy
from mathutils import Matrix, Vector

BODY_TARGET = 8000  # target triangle count after decimation
DEVICE_MAX_M = 0.25  # drop any CAD part larger than this (the ~29 m context solid, mat_10)

# CAD (Creo) -> sim reframe: sim_x=-cad_z, sim_y=-cad_x, sim_z=+cad_y. See the module docstring.
CAD_TO_SIM = Matrix(
    (
        (0.0, 0.0, -1.0, 0.0),
        (-1.0, 0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )
)

argv = sys.argv[sys.argv.index("--") + 1 :]
if len(argv) != 2:
    raise SystemExit("usage: blender -b -P robin_w1g_blender.py -- <robin_obj> <body_out>")
robin_obj, body_out = argv


def reset():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def tris(o):
    return sum(len(p.vertices) - 2 for p in o.data.polygons)


def decimate(o, target):
    n = tris(o)
    if n > target:
        m = o.modifiers.new("dec", "DECIMATE")
        m.ratio = target / n
        bpy.context.view_layer.objects.active = o
        bpy.ops.object.modifier_apply(modifier=m.name)


def world_bbox(objs):
    cs = [o.matrix_world @ Vector(c) for o in objs for c in o.bound_box]
    lo = Vector((min(c.x for c in cs), min(c.y for c in cs), min(c.z for c in cs)))
    hi = Vector((max(c.x for c in cs), max(c.y for c in cs), max(c.z for c in cs)))
    return lo, hi


def export(o, path):
    for x in bpy.context.scene.objects:
        x.select_set(False)
    o.select_set(True)
    bpy.context.view_layer.objects.active = o
    # Identity export (matches the identity import below): the mesh is already authored in the sim
    # frame (Z up), so no axis remap -- the coords are written exactly as reframed.
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


# --- housing: import, split by material, drop the context solid, join, reframe, scale, ground -------
reset()
# Identity import (forward=Y, up=Z): load the Open CASCADE OBJ coords unchanged so the CAD_TO_SIM
# reframe below acts in the raw CAD frame. The default importer would apply its own Y->Z rotation and
# double-transform the mesh.
bpy.ops.wm.obj_import(filepath=robin_obj, forward_axis="Y", up_axis="Z")
meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
if not meshes:
    raise SystemExit("robin_w1g: no meshes imported from the OBJ")

# Open CASCADE writes the whole assembly as one object (per-material `usemtl` groups, no per-object
# split), so split it into one object per CAD material first -- the ~29 m context/aiming solid is its
# own material (`mat_10`) and must be separable from the ~0.1 m device.
for o in meshes:
    o.select_set(True)
bpy.context.view_layer.objects.active = meshes[0]
if len(meshes) > 1:
    bpy.ops.object.join()
whole = bpy.context.view_layer.objects.active
bpy.ops.object.mode_set(mode="EDIT")
bpy.ops.mesh.separate(type="MATERIAL")
bpy.ops.object.mode_set(mode="OBJECT")

# Keep only device-sized parts; a part larger than DEVICE_MAX_M in any axis is the CAD context solid.
device = []
for o in [x for x in bpy.context.scene.objects if x.type == "MESH"]:
    dim = max(o.dimensions) * 0.001  # importer units are mm
    if dim > DEVICE_MAX_M:
        print(f"drop oversized part {o.name!r} ({dim:.2f} m -- the CAD context solid)")
        bpy.data.objects.remove(o, do_unlink=True)
    else:
        device.append(o)
if not device:
    raise SystemExit(
        "robin_w1g: every imported part exceeded the device size -- check DEVICE_MAX_M"
    )

for o in device:
    o.select_set(True)
bpy.context.view_layer.objects.active = device[0]
if len(device) > 1:
    bpy.ops.object.join()
body = bpy.context.view_layer.objects.active
body.name = "robin_w1g_body"

# Reframe CAD->sim (identity import above, so this acts in the raw CAD frame), then scale mm->m.
body.data.transform(CAD_TO_SIM)
body.scale = (0.001, 0.001, 0.001)
bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

lo, hi = world_bbox([body])
centre = Vector(((lo.x + hi.x) / 2, (lo.y + hi.y) / 2, lo.z))  # x/y centre, z base to 0
body.location -= centre
bpy.ops.object.transform_apply(location=True, rotation=False, scale=True)

decimate(body, BODY_TARGET)
export(body, body_out)

lo, hi = world_bbox([body])
print(f"BODY_BBOX_m x[{lo.x:.4f},{hi.x:.4f}] y[{lo.y:.4f},{hi.y:.4f}] z[{lo.z:.4f},{hi.z:.4f}]")
print("done")
