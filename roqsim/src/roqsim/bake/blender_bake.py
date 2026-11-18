# Copyright (C) 2026 Frederik Pasch
#
# SPDX-License-Identifier: Apache-2.0

"""The bake's Blender side: run *inside* Blender by ``roqsim bake``, never imported by roqsim.

    blender -b --factory-startup --python blender_bake.py -- --work DIR --out DIR --entities a,b
        [--device CPU|CUDA|OPTIX|HIP|METAL] [--threads N] [--uv-from DIR]

Reads the work directory ``roqsim bake`` wrote (:func:`roqsim.bake.run.write_work`): ``manifest.json``
(entities, materials, light sources, settings), ``entities/<e>.npz`` (world triangles, normals, albedo
UVs with v up, material and geom per triangle), ``emitters.npz``, ``occluders.npz`` and the albedo
textures. Builds the scene once -- every entity a mesh with its normals and albedo, the occluders grey,
the emitters emissive (a face that looks at a surface within 2 cm emits nothing: the back of a ceiling
panel), the lights -- and, per entity of ``--entities``:

1. lays out the lightmap UVs (unless ``--uv-from`` has them): Smart UV Project without a margin, a
   shell face nothing in the world sees (a wall's outside) shrunk to a twentieth, Pack Islands by the
   islands' outline with four texels between them, then every island under six texels across or
   three wide scaled up (at most 8 x) and packed again, five rounds at most. Corners of one geom at
   one place are welded to 10 µm first, so a prop's parts are islands, not its triangles;
2. bakes, at the entity's size: Cycles' DIFFUSE pass with DIRECT and INDIRECT and without COLOR (the
   light alone); the DIRECT pass alone (no bounces: cheap), from which the indirect light is the
   difference; AO (1 m); NORMAL (world space). A two-sided entity (a prop) is baked again from the
   other side of every face. Every light pass is denoised by Open Image Denoise (Blender's own),
   guided by the normal pass, unless the manifest says not to;
3. writes ``OUT/<e>.<pass>.npy`` (float16, rows bottom first), ``OUT/<e>.uv2.npy`` (per triangle
   corner, in the export's order) and ``OUT/times.<first entity>.json``.

A GPU type that is not there exits 3 naming ``--device cpu``; OIDN not found exits 4 naming
``denoise: false``.
"""

import ctypes
import glob
import json
import math
import os
import sys
import time
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

argv = sys.argv[sys.argv.index("--") + 1 :]
opts = dict(zip(argv[0::2], argv[1::2], strict=True))
WORK, OUT = Path(opts["--work"]), Path(opts["--out"])
ENTITIES = [e for e in opts["--entities"].split(",") if e]
DEVICE = opts.get("--device", "CPU").upper()
THREADS = int(opts.get("--threads", 0))
UV_FROM = Path(opts["--uv-from"]) if opts.get("--uv-from") else None
OUT.mkdir(parents=True, exist_ok=True)
man = json.loads((WORK / "manifest.json").read_text())
SET = man["settings"]
timing = {"device": DEVICE, "entities": {}}
t_start = time.time()

#: The least an island spans, and the least it is wide, in texels, once packed; how often small islands
#: are scaled up and the layout packed again, and the most an island grows in all rounds together.
MIN_ISLAND, MIN_WIDTH, ISLAND_ROUNDS, MAX_GROWTH = 6.0, 3.0, 5, 8.0
#: Smart UV Project's angle: a prop's islands are its separate parts more than its curvature.
ANGLE_DEG = 66.0
#: The distance ambient occlusion looks, in metres.
AO_DISTANCE = 1.0
#: A face of an emitter whose normal meets a surface this near emits nothing.
EMITTER_BACKING = 0.02


def srgb_to_linear(c):
    c = np.asarray(c, float)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


# -- scene ------------------------------------------------------------------------------------------
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.render.engine = "CYCLES"
prefs = bpy.context.preferences.addons["cycles"].preferences
if DEVICE == "CPU":
    # Cycles' compute device types are the GPU back-ends only; the CPU is the scene's device.
    prefs.compute_device_type = "NONE"
    scene.cycles.device = "CPU"
else:
    try:
        prefs.compute_device_type = DEVICE
    except TypeError:
        print(
            f"roqsim-bake: this Blender knows no {DEVICE} device type; --device cpu bakes on the CPU"
        )
        sys.exit(3)
    prefs.get_devices()
    for dev in prefs.devices:
        dev.use = dev.type == DEVICE
    # Cycles renders on the CPU without a word when no device of the type is enabled.
    if not any(dev.use for dev in prefs.devices):
        print(f"roqsim-bake: no {DEVICE} device found; --device cpu bakes on the CPU")
        sys.exit(3)
    scene.cycles.device = "GPU"
if THREADS > 0:
    scene.render.threads_mode = "FIXED"
    scene.render.threads = THREADS
scene.cycles.samples = int(SET["samples"])
scene.cycles.use_denoising = False
scene.cycles.max_bounces = int(SET["bounces"]) + 2
scene.cycles.diffuse_bounces = int(SET["bounces"])
scene.cycles.glossy_bounces = 0
scene.cycles.transmission_bounces = 0
scene.cycles.caustics_reflective = False
scene.cycles.caustics_refractive = False
scene.view_settings.view_transform = "Standard"
world = bpy.data.worlds.new("sky")
world.use_nodes = True
world.node_tree.nodes["Background"].inputs[1].default_value = float(man["light"].get("sky", 0.0))
world.light_settings.distance = AO_DISTANCE
scene.world = world


def make_material(i, spec):
    mat = bpy.data.materials.new(f"mat{i}")
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfDiffuse")
    bsdf.inputs["Roughness"].default_value = 0.0
    nt.links.new(bsdf.outputs[0], out.inputs[0])
    rgb = list(srgb_to_linear(spec["rgba"][:3]))
    if spec.get("texture_file"):
        img = bpy.data.images.load(str(WORK / spec["texture_file"]))
        img.colorspace_settings.name = "sRGB"
        tex = nt.nodes.new("ShaderNodeTexImage")
        tex.image = img
        tex.extension = "REPEAT"
        uvn = nt.nodes.new("ShaderNodeUVMap")
        uvn.uv_map = "albedo"
        nt.links.new(uvn.outputs[0], tex.inputs[0])
        mix = nt.nodes.new("ShaderNodeMix")
        mix.data_type = "RGBA"
        mix.blend_type = "MULTIPLY"
        mix.inputs["Factor"].default_value = 1.0
        nt.links.new(tex.outputs[0], mix.inputs[6])
        mix.inputs[7].default_value = rgb + [1.0]
        nt.links.new(mix.outputs[2], bsdf.inputs["Color"])
    else:
        bsdf.inputs["Color"].default_value = rgb + [1.0]
    return mat


def emission_material(name, colour, strength):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    em = nt.nodes.new("ShaderNodeEmission")
    em.inputs["Color"].default_value = list(colour) + [1.0]
    em.inputs["Strength"].default_value = float(strength)
    nt.links.new(em.outputs[0], out.inputs[0])
    return mat


materials = [make_material(i, s) for i, s in enumerate(man["materials"])]


def mesh_object(name, pos, nrm=None, uv1=None, mat=None, part=None, mats=None):
    """One object from per-corner arrays; corners of one part at one place are welded, normals and
    UVs stay per corner."""
    mats = materials if mats is None else mats
    n = len(pos)
    me = bpy.data.meshes.new(name)
    corners = pos.reshape(-1, 3).astype(np.float64)
    tag = np.repeat(np.asarray(part if part is not None else np.zeros(n), float), 3)[:, None]
    # To 10 µm: a prop turned in the room carries float32 noise past a finer weld.
    _, inv = np.unique(np.concatenate([tag, np.round(corners, 5)], 1), axis=0, return_inverse=True)
    inv = inv.reshape(n, 3)
    bad = (inv[:, 0] == inv[:, 1]) | (inv[:, 1] == inv[:, 2]) | (inv[:, 0] == inv[:, 2])
    nv = int(inv.max()) + 1 if n else 0
    extra = np.arange(nv, nv + 3 * int(bad.sum())).reshape(-1, 3)
    inv[bad] = extra
    index = inv.ravel()
    verts = np.zeros((nv + 3 * int(bad.sum()), 3))
    verts[index] = corners
    me.vertices.add(len(verts))
    me.vertices.foreach_set("co", verts.ravel())
    me.loops.add(len(corners))
    me.loops.foreach_set("vertex_index", index)
    me.polygons.add(n)
    me.polygons.foreach_set("loop_start", np.arange(0, 3 * n, 3))
    me.update()
    if uv1 is not None:
        uv = me.uv_layers.new(name="albedo")
        uv.data.foreach_set("uv", uv1.reshape(-1, 2).ravel())
    if mat is not None:
        used = np.unique(mat)
        for m in used:
            me.materials.append(mats[int(m)])
        remap = {int(m): k for k, m in enumerate(used)}
        me.polygons.foreach_set("material_index", np.array([remap[int(m)] for m in mat]))
    if nrm is not None:
        if len(me.polygons) != n:
            sys.exit(f"roqsim-bake: {name}: {len(me.polygons)} faces of {n}")
        me.normals_split_custom_set(nrm.reshape(-1, 3).tolist())
    ob = bpy.data.objects.new(name, me)
    scene.collection.objects.link(ob)
    return ob


objects = {}
for ename in man["entities"]:
    z = np.load(WORK / "entities" / f"{ename}.npz")
    objects[ename] = mesh_object(ename, z["pos"], z["nrm"], z["uv1"], z["mat"], z["geom"])
occ = np.load(WORK / "occluders.npz")["pos"]
if len(occ):
    grey = make_material(len(materials), {"rgba": [0.6, 0.6, 0.6, 1]})
    mesh_object("occluders", occ.reshape(-1, 3, 3), mat=np.zeros(len(occ) // 3, int), mats=[grey])

for i, L in enumerate(man["light"]["lights"]):
    t = L["type"]
    kind = {"area": "AREA", "point": "POINT", "spot": "SPOT", "sun": "SUN"}[t]
    ld = bpy.data.lights.new(f"light{i}", kind)
    ld.color = L.get("colour", [1, 1, 1])
    if t == "sun":
        ld.energy = float(L["strength"])
    else:
        ld.energy = float(L["power_w"])
    if t == "area":
        ld.shape = "RECTANGLE"
        ld.size, ld.size_y = L["size"]
    if t == "spot":
        ld.spot_size = math.radians(float(L["spot_deg"]))
    lo = bpy.data.objects.new(f"light{i}", ld)
    if "pos" in L:
        lo.location = L["pos"]
    if "dir" in L:
        # A light points down its -z: turn -z onto the light's direction.
        lo.rotation_mode = "QUATERNION"
        lo.rotation_quaternion = Vector((0, 0, -1)).rotation_difference(
            Vector(L["dir"]).normalized()
        )
    elif t == "area":
        lo.rotation_euler = (0.0, 0.0, math.radians(L.get("yaw_deg", 0.0)))
    scene.collection.objects.link(lo)


# Every surface a ray can meet, for the tests below: the entities, the occluders.
def bvh_of(obs):
    allv, allf = [], []
    for ob in obs:
        me = ob.data
        off = len(allv)
        allv += [v.co.copy() for v in me.vertices]
        allf += [tuple(i + off for i in p.vertices) for p in me.polygons]
    return BVHTree.FromPolygons(allv, allf) if allf else None


if man["emitters"]:
    z = np.load(WORK / "emitters.npz")
    src = z["source"]
    keys = sorted(set(int(s) for s in src))
    emats = [
        emission_material(
            f"emitter{k}", man["emitters"][k]["colour"], man["emitters"][k]["strength_w_m2"]
        )
        for k in keys
    ]
    dark = make_material(len(materials) + 1, {"rgba": [0.05, 0.05, 0.05, 1]})
    surfaces = bvh_of([o for o in scene.objects if o.type == "MESH"])
    pos = z["pos"].astype(np.float64)
    fn = np.cross(pos[:, 1] - pos[:, 0], pos[:, 2] - pos[:, 0])
    fn /= np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-12)
    centre = pos.mean(1)
    backed = np.zeros(len(pos), bool)
    if surfaces is not None:
        for i in range(len(pos)):
            hit = surfaces.ray_cast(
                Vector(centre[i] + fn[i] * 1e-4), Vector(fn[i]), EMITTER_BACKING
            )
            backed[i] = hit[0] is not None
    mat_index = np.array([keys.index(int(s)) for s in src])
    mat_index[backed] = len(keys)
    mesh_object("emitters", z["pos"], z["nrm"], mat=mat_index, part=src, mats=emats + [dark])
timing["scene_s"] = round(time.time() - t_start, 2)

BVH = bvh_of([o for o in scene.objects if o.type == "MESH"])


def unseen_faces(ob):
    """The faces a ray along whose normal meets nothing: a wall's outside, the ceiling's top."""
    out = []
    for p in ob.data.polygons:
        hit = BVH.ray_cast(p.center + p.normal * 1e-3, p.normal, 50.0)
        if hit[0] is None:
            out.append(p.index)
    return out


# -- OIDN, Blender's own --------------------------------------------------------------------------------
_OIDN = None


def oidn():
    global _OIDN
    if _OIDN is not None:
        return _OIDN
    here = Path(bpy.app.binary_path).resolve().parent
    names = []
    for base in (here, here / "lib", here.parent / "lib", here.parent / "Resources" / "lib"):
        names += glob.glob(str(base / "*OpenImageDenoise*"))
    names = [n for n in names if "core" not in n and "device" not in n]
    if not names:
        print(
            f"roqsim-bake: Open Image Denoise not found beside {bpy.app.binary_path}; `denoise: false` bakes without it"
        )
        sys.exit(4)
    lib = ctypes.CDLL(sorted(names, key=len)[0])
    lib.oidnNewDevice.restype = ctypes.c_void_p
    lib.oidnNewFilter.restype = ctypes.c_void_p
    lib.oidnNewFilter.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    lib.oidnCommitDevice.argtypes = [ctypes.c_void_p]
    lib.oidnSetSharedFilterImage.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
    ]
    lib.oidnSetFilterBool.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_bool]
    lib.oidnCommitFilter.argtypes = [ctypes.c_void_p]
    lib.oidnExecuteFilter.argtypes = [ctypes.c_void_p]
    lib.oidnReleaseFilter.argtypes = [ctypes.c_void_p]
    lib.oidnGetDeviceError.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_char_p)]
    dev = lib.oidnNewDevice(1)  # the CPU
    lib.oidnCommitDevice(dev)
    _OIDN = (lib, dev)
    return _OIDN


def denoise(color, normal):
    """OIDN's RT filter over a light map (hdr), guided by the normal pass and a white albedo."""
    lib, dev = oidn()
    h, w, _ = color.shape
    color = np.ascontiguousarray(color, np.float32)
    nrm = np.ascontiguousarray(normal * 2.0 - 1.0, np.float32)
    alb = np.ones_like(color)
    res = np.empty_like(color)
    f = lib.oidnNewFilter(dev, b"RT")
    for name, a in ((b"color", color), (b"normal", nrm), (b"albedo", alb), (b"output", res)):
        lib.oidnSetSharedFilterImage(f, name, a.ctypes.data, 3, w, h, 0, 0, 0)
    lib.oidnSetFilterBool(f, b"hdr", True)
    lib.oidnCommitFilter(f)
    lib.oidnExecuteFilter(f)
    msg = ctypes.c_char_p()
    if lib.oidnGetDeviceError(dev, ctypes.byref(msg)):
        sys.exit(f"roqsim-bake: oidn: {msg.value}")
    lib.oidnReleaseFilter(f)
    return res


# -- per entity: lightmap UVs, passes -------------------------------------------------------------------
def uv_array(layer):
    a = np.empty(len(layer.data) * 2, np.float64)
    layer.data.foreach_get("uv", a)
    return a.reshape(-1, 3, 2)


def check_uv(name, layer, step):
    a = uv_array(layer)
    if not (
        np.isfinite(a).all() and a.min() > -1e-4 and a.max() < 1 + 1e-4 and a.max() - a.min() > 0.05
    ):
        sys.exit(f"roqsim-bake: {name}: the lightmap UVs after {step} span {a.min()}..{a.max()}")


def pack(margin):
    """Pack Islands by the islands' outline, *margin* (a fraction of the square) between them.

    Twice with the margin added as it is (``ADD``), not once as a fraction of the result
    (``FRACTION``): ``FRACTION`` searches for the scale and took 14 s for a layout ``ADD`` packs in 2.5;
    the first packing brings the islands to their final scale, so the second one's margin is the
    final margin, within the little the margins themselves shrink it."""
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    for _ in range(2):
        bpy.ops.uv.select_all(action="SELECT")
        bpy.ops.uv.pack_islands(
            udim_source="CLOSEST_UDIM",
            rotate=True,
            scale=True,
            merge_overlap=False,
            margin_method="ADD",
            margin=margin,
            shape_method="CONCAVE",
        )
    bpy.ops.object.mode_set(mode="OBJECT")


def uv_islands(vid, uv):
    """Per triangle, its island: triangles joined where they share a vertex at one UV."""
    key, parent = {}, list(range(len(vid)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    q = np.round(uv * 2**20).astype(np.int64)
    for t in range(len(vid)):
        for k in range(3):
            kk = (int(vid[t, k]), int(q[t, k, 0]), int(q[t, k, 1]))
            o = key.setdefault(kk, t)
            if o != t:
                ra, rb = find(o), find(t)
                if ra != rb:
                    parent[ra] = rb
    roots = np.array([find(t) for t in range(len(vid))])
    _, lab = np.unique(roots, return_inverse=True)
    return lab.ravel()


def small_islands(lab, uv, size, skip):
    area = (
        np.abs(
            (uv[:, 1, 0] - uv[:, 0, 0]) * (uv[:, 2, 1] - uv[:, 0, 1])
            - (uv[:, 2, 0] - uv[:, 0, 0]) * (uv[:, 1, 1] - uv[:, 0, 1])
        )
        / 2
        * size
        * size
    )
    out = {}
    for isl in np.unique(lab):
        t = lab == isl
        if skip[t].all():
            continue
        p = uv[t].reshape(-1, 2)
        span = max(float((p.max(0) - p.min(0)).max()) * size, 1e-3)
        width = float(area[t].sum()) / span
        need = max(MIN_ISLAND / span, MIN_WIDTH / max(width, 1e-3))
        if need > 1.0:
            out[int(isl)] = need * 1.5
    return out


def layout(name, ob, size, shrink_unseen):
    me = ob.data
    uv2 = me.uv_layers.new(name="lightmap")
    me.uv_layers.active = uv2
    bpy.ops.object.select_all(action="DESELECT")
    ob.select_set(True)
    bpy.context.view_layer.objects.active = ob
    info = {}
    if UV_FROM is not None and (UV_FROM / f"{name}.uv2.npy").exists():
        uv2.data.foreach_set("uv", np.load(UV_FROM / f"{name}.uv2.npy").astype(np.float32).ravel())
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.object.mode_set(mode="OBJECT")
        info["uv_reused"] = True
        return info
    unseen = set(unseen_faces(ob)) if shrink_unseen else set()
    margin = 4.0 / size
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    # No margin here: Smart UV Project spends it per island; Pack Islands keeps it.
    bpy.ops.uv.smart_project(
        angle_limit=math.radians(ANGLE_DEG),
        island_margin=0.0,
        area_weight=0.0,
        correct_aspect=True,
        scale_to_bounds=False,
    )
    bpy.ops.object.mode_set(mode="OBJECT")
    uv2 = me.uv_layers["lightmap"]  # leaving edit mode reallocates the layers
    check_uv(name, uv2, "smart project")
    if unseen and len(unseen) < len(me.polygons):
        data = uv2.data
        for p in me.polygons:
            if p.index in unseen:
                idx = list(p.loop_indices)
                c = sum((data[i].uv for i in idx), Vector((0, 0))) / len(idx)
                for i in idx:
                    data[i].uv = c + (data[i].uv - c) * 0.05
    pack(margin)
    vid = np.empty(len(me.loops), np.int64)
    me.loops.foreach_get("vertex_index", vid)
    vid = vid.reshape(-1, 3)
    skip = np.zeros(len(me.polygons), bool)
    skip[list(unseen)] = True
    skip |= np.array([p.area < 1e-6 for p in me.polygons])
    grown, enlarged = {}, 0
    for _ in range(ISLAND_ROUNDS):
        uv2 = me.uv_layers["lightmap"]
        a = uv_array(uv2)
        lab = uv_islands(vid, a)
        small = small_islands(lab, a, size, skip)
        small = {i: min(f, MAX_GROWTH / grown.get(i, 1.0)) for i, f in small.items()}
        small = {i: f for i, f in small.items() if f > 1.01}
        if not small:
            break
        for isl, factor in small.items():
            t = lab == isl
            c = a[t].reshape(-1, 2).mean(0)
            a[t] = c + (a[t] - c) * factor
            grown[isl] = grown.get(isl, 1.0) * factor
        enlarged += len(small)
        uv2.data.foreach_set("uv", a.ravel())
        pack(margin)
    uv2 = me.uv_layers["lightmap"]
    a = uv_array(uv2)
    lo, hi = float(a.min()), float(a.max())
    if lo < 0 or hi > 1:
        # Pack Islands can spill past the square when the islands are many; scaled back into it.
        a = (a - min(lo, 0.0)) / (max(hi, 1.0) - min(lo, 0.0))
        uv2.data.foreach_set("uv", a.astype(np.float32).ravel())
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.object.mode_set(mode="OBJECT")
    check_uv(name, me.uv_layers["lightmap"], "packing")
    info.update(
        {
            "islands": int(uv_islands(vid, uv_array(me.uv_layers["lightmap"])).max() + 1),
            "islands_enlarged": enlarged,
            "unseen_faces": len(unseen),
        }
    )
    return info


def target(ob, img):
    for mat in ob.data.materials:
        nt = mat.node_tree
        node = nt.nodes.get("lightmap_target") or nt.nodes.new("ShaderNodeTexImage")
        node.name = "lightmap_target"
        node.image = img
        uvn = nt.nodes.get("lightmap_uv") or nt.nodes.new("ShaderNodeUVMap")
        uvn.name = "lightmap_uv"
        uvn.uv_map = "lightmap"
        if not node.inputs[0].links:
            nt.links.new(uvn.outputs[0], node.inputs[0])
        nt.nodes.active = node
        node.select = True


def bake_pass(ob, size, kind, **kw):
    img = bpy.data.images.new(f"{ob.name}_{kind}", size, size, float_buffer=True, alpha=False)
    img.colorspace_settings.name = "Non-Color"
    target(ob, img)
    bpy.ops.object.bake(
        margin=16,
        margin_type="EXTEND",
        use_clear=True,
        target="IMAGE_TEXTURES",
        uv_layer="lightmap",
        **kw,
    )
    px = np.empty(size * size * 4, np.float32)
    img.pixels.foreach_get(px)
    bpy.data.images.remove(img)
    return px.reshape(size, size, 4)[..., :3].copy()  # rows bottom first, as Blender keeps them


def flip_normals(ob):
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.flip_normals()
    bpy.ops.object.mode_set(mode="OBJECT")


def passes(name, ob, size, suffix=""):
    t = {}
    out = {}
    t0 = time.time()
    out["normal"] = bake_pass(ob, size, "normal", type="NORMAL", normal_space="OBJECT")
    t["normal" + suffix] = round(time.time() - t0, 2)
    t0 = time.time()
    out["light"] = bake_pass(ob, size, "light", type="DIFFUSE", pass_filter={"DIRECT", "INDIRECT"})
    t["light" + suffix] = round(time.time() - t0, 2)
    t0 = time.time()
    bounces = scene.cycles.diffuse_bounces
    scene.cycles.diffuse_bounces = 0
    out["direct"] = bake_pass(ob, size, "direct", type="DIFFUSE", pass_filter={"DIRECT"})
    scene.cycles.diffuse_bounces = bounces
    t["direct" + suffix] = round(time.time() - t0, 2)
    t0 = time.time()
    out["ao"] = bake_pass(ob, size, "ao", type="AO")
    t["ao" + suffix] = round(time.time() - t0, 2)
    if SET["denoise"]:
        t0 = time.time()
        for k in ("light", "direct", "ao"):
            out[k] = denoise(out[k], out["normal"])
        t["denoise" + suffix] = round(time.time() - t0, 2)
    for k in ("light", "direct", "ao", "normal"):
        np.save(OUT / f"{name}.{k}{suffix}.npy", out[k].astype(np.float16))
    return t


for ename in ENTITIES:
    spec = man["entities"][ename]
    ob, size = objects[ename], int(spec["size"])
    t0 = time.time()
    info = layout(ename, ob, size, spec["class"] == "shell")
    t_uv = round(time.time() - t0, 2)
    uv = uv_array(ob.data.uv_layers["lightmap"]).astype(np.float32)
    np.save(OUT / f"{ename}.uv2.npy", uv)
    t = passes(ename, ob, size)
    if spec["two_sided"]:
        flip_normals(ob)
        t.update(passes(ename, ob, size, "_back"))
        flip_normals(ob)
    timing["entities"][ename] = {
        "size": size,
        "uv_s": t_uv,
        "bake_s": t,
        "faces": len(ob.data.polygons),
        **info,
    }
    print(f"roqsim-bake: baked {ename}: {size}px, uv {t_uv}s, passes {t}", flush=True)

timing["total_s"] = round(time.time() - t_start, 2)
timing["blender"] = bpy.app.version_string
(OUT / f"times.{ENTITIES[0] if ENTITIES else 'none'}.json").write_text(json.dumps(timing, indent=1))
os._exit(0)
