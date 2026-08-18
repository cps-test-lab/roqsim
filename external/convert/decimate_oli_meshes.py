"""OPTIONAL: decimate the Oli visual meshes in place, for offscreen-camera render performance.

Not part of the default build -- the Oli ships full-res.
Run this only if you want to shave camera-render cost and accept slightly softer visuals.

The Oli meshes are CAD parts: flat panels wrapped into thin-walled shells (forearm/thigh/shin covers)
around solid joint housings. A ratio-based **Collapse** decimate (what the G1 uses) destroys those
thin shells -- it merges the shell's near-coincident front/back walls and the forearms, thighs and
shin-to-foot covers vanish. So this uses **Planar (Dissolve)** decimation: it merges only triangles
within ``angle`` degrees of coplanar, collapsing flat panels while leaving thin walls/edges intact.
(Dissolve reduces these curved parts only modestly -- ~a few %; that is the accepted cost of not
breaking the shells.)

Collision for the Oli is PRIMITIVE geoms (see build_oli.py), not the meshes, so this is purely a
rendering optimisation with zero effect on dynamics.

Run:  blender --background --python external/convert/decimate_oli_meshes.py -- [angle_deg]
Default 5 deg. Re-run after build_oli.py (which copies fresh full-res meshes).
"""

import sys
from math import radians
from pathlib import Path

import bpy

# this script lives in roqsim/external/convert/; the meshes live in the humanoid package (sibling of external/)
MESHES = Path(__file__).resolve().parents[2] / "roqsim_humanoid/src/roqsim_humanoid/models/meshes/oli"
argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
ANGLE = float(argv[0]) if argv else 5.0

# Blender 4.0's new wm.stl_export is off by default; the legacy io_mesh_stl exporter is reliable.
try:
    bpy.ops.preferences.addon_enable(module="io_mesh_stl")
except Exception:
    pass


def clear():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


total_before = total_after = 0
for stl in sorted(MESHES.glob("*.STL")):
    clear()
    bpy.ops.wm.stl_import(filepath=str(stl))
    obj = bpy.context.selected_objects[0]
    bpy.context.view_layer.objects.active = obj
    n0 = len(obj.data.polygons)
    total_before += n0
    mod = obj.modifiers.new("dec", "DECIMATE")
    mod.decimate_type = "DISSOLVE"  # planar: merge coplanar faces, preserve thin walls/edges
    mod.angle_limit = radians(ANGLE)
    # face_count reflects the post-modifier tri count once evaluated; report from the eval'd mesh.
    dg = bpy.context.evaluated_depsgraph_get()
    n1 = len(obj.evaluated_get(dg).data.polygons)
    total_after += n1
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    # The exporter bakes the modifier (use_mesh_modifiers), sidestepping the multi-user apply guard.
    bpy.ops.export_mesh.stl(filepath=str(stl), use_selection=True, use_mesh_modifiers=True)
    print(f"  {stl.name}: {n0} -> {n1} tris")

pct = 100 * total_after / max(total_before, 1)
print(f"TOTAL tris {total_before} -> {total_after} ({pct:.0f}%), planar angle {ANGLE} deg")
