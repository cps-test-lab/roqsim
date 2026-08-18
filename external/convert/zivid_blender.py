"""Blender-side processing for the Zivid 3 XL250 meshes (run via ``blender -b -P``).

Reads Zivid's official binary STL (SolidWorks export, mm, Z up) and writes two MuJoCo-ready OBJs:
the dark camera housing (``zivid_body``) and a square FOV coverage frustum (``zivid_fov``). The body
is decimated, scaled mm->m, centred on x=y=0 with its base dropped to z=0, and exported axes 1:1
(Z up). The FOV mesh is authored here (not from CAD -- Zivid ships no FOV solid): a 39 deg square
pyramid from the optical origin out to the 2.5 m focus distance, matching the datasheet's
1750 x 1750 mm field of view at focus.

The STL is a single dark part (no CAD materials to split, unlike the Livox pipeline), so there is no
cascadio/OBJ material step -- Blender imports the STL directly. Optical geometry (baseline, FOV,
focus) comes from the datasheet, not the mesh; only the housing shape and the optical z-height come
from the CAD. Invoked by ``zivid_convert.py``; not run directly.

Args after ``--``:  <zivid_stl> <body_out> <fov_out>
"""

import math
import sys

import bpy
from mathutils import Vector

BODY_TARGET = 8000        # target triangle count after decimation
BASELINE_M = 0.25         # datasheet: distance between the two optical modules
FOV_DEG = 39.0            # datasheet: horizontal == vertical (square) field of view
FOCUS_M = 2.5             # datasheet: focus distance; FOV frustum is drawn out to here
OPTICAL_Y_M = 0.030       # optical origin depth: at the front glass, just inside the bezel
FOV_FWD = Vector((0.0, 1.0, 0.0))  # optical axis in the normalised body frame (+Y, the optics face)

argv = sys.argv[sys.argv.index("--") + 1 :]
if len(argv) != 3:
    raise SystemExit("usage: blender -b -P zivid_blender.py -- <zivid_stl> <body_out> <fov_out>")
zivid_stl, body_out, fov_out = argv


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


# --- housing: scale mm->m, centre on x=y=0, drop base to z=0, decimate --------------------------
reset()
bpy.ops.wm.stl_import(filepath=zivid_stl)
body = next(o for o in bpy.context.scene.objects if o.type == "MESH")
body.name = "zivid_body"

# Some Blender builds (e.g. 4.0's stl_import) hand back a multi-user mesh datablock, which makes
# transform_apply below abort ("Cannot apply to a multi user"). Force single-user so the transforms
# bake regardless of importer/version.
bpy.ops.object.select_all(action="DESELECT")
body.select_set(True)
bpy.context.view_layer.objects.active = body
bpy.ops.object.make_single_user(object=True, obdata=True)

body.scale = (0.001, 0.001, 0.001)  # SolidWorks STL is in mm; MuJoCo works in m
bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

lo, hi = world_bbox([body])
centre = Vector(((lo.x + hi.x) / 2, (lo.y + hi.y) / 2, lo.z))  # x/y centre, z base
body.location -= centre
bpy.ops.object.transform_apply(location=True, rotation=False, scale=True)

decimate(body, BODY_TARGET)
export(body, body_out)

lo, hi = world_bbox([body])
optical_z = lo.z + 0.011  # optical strip sits ~11 mm above the base (measured from the CAD)
print(f"BODY_BBOX_m x[{lo.x:.4f},{hi.x:.4f}] y[{lo.y:.4f},{hi.y:.4f}] z[{lo.z:.4f},{hi.z:.4f}]")
print(f"OPTICAL_ORIGIN_m x=+{BASELINE_M / 2:.4f} y={OPTICAL_Y_M:.4f} z={optical_z:.4f}")

# --- FOV frustum: 39 deg square pyramid, apex at the optical origin, out to the focus distance ----
# Authored (Zivid ships no FOV solid). Apex at the origin pointing +Y; the mid360-style ``_fov``
# geom in zivid.xml is placed at the optical origin, so the cone opens from the imaging lens.
reset()
half = math.tan(math.radians(FOV_DEG) / 2.0) * FOCUS_M  # half-width of the square at focus
apex = Vector((0.0, 0.0, 0.0))
# Square far plane at +Y = FOCUS_M, spanning +/-half in the two axes orthogonal to the optical axis.
far = FOCUS_M
verts = [
    apex,
    Vector((-half, far, -half)),
    Vector((half, far, -half)),
    Vector((half, far, half)),
    Vector((-half, far, half)),
]
faces = [
    (0, 1, 2), (0, 2, 3), (0, 3, 4), (0, 4, 1),  # four side triangles
    (1, 4, 3), (1, 3, 2),                          # far cap
]
mesh = bpy.data.meshes.new("zivid_fov")
mesh.from_pydata([tuple(v) for v in verts], [], faces)
mesh.update()
fov = bpy.data.objects.new("zivid_fov", mesh)
bpy.context.scene.collection.objects.link(fov)
export(fov, fov_out)
print("done")
