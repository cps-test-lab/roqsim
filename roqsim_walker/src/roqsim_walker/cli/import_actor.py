#!/usr/bin/env python3
"""Convert a rigged COLLADA character (a Gazebo/Open-RMF <actor> skin) into a walker blueprint.

Writes a ``people/<Name>/`` folder -- OBJ + MTL + textures, ``<name>.walker.json`` and the authored
``<name>.weights.npz`` -- from a ``.dae`` that carries a skin controller. Only the *skin* is taken:
locomotion comes from this package's clips (``models/anims/``), so the source's ``<animation>`` tracks
are ignored. This is the counterpart to the CARLA character export pipeline.

Note what that means for licensing: the clips are CARLA retargets under CC-BY 4.0, so an imported
actor is a CC0 or CC-BY skin animated by CC-BY clips. Whatever the source's own terms are, the
resulting walker still carries CARLA attribution -- see the package ``THIRD_PARTY.md``.

    roqsim walker import-actor FemaleVisitorWalk.dae --name FemaleVisitorWalk --anim-set female \
        --credits-source https://fuel.gazebosim.org/1.0/Luca/models/FemaleVisitorWalk \
        --credits-licence "CC-BY 4.0" --credits-author "Wan Yi Seow"

Like the prop tools in ``roqsim_assets/tools``, this **only produces files** -- adding them to git
and honouring the source licence is up to you (CC0 / CC-BY / CC-BY-SA only; see THIRD_PARTY.md).

**Rig requirements.** The source skeleton is mapped onto roqsim_walker's fixed 17-joint topology
via ``JOINT_MAP``; a rig whose joints cannot be mapped is rejected rather than guessed at. The
Open-RMF actor family (FemaleVisitorWalk, MaleVisitorWalk, the hospital set) shares one rig and maps
cleanly. Note the mapping is *semantic*, not positional: bone **lengths** are read from the source
bind pose, but the emitted offsets are written in this package's canonical **arms-down** rest
convention (see ``skeleton_from_bind``) because the locomotion clips are authored against it.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil

import numpy as np

from roqsim_walker.blueprint import models_dir
from roqsim_walker.humanoid import CHILDREN, JOINT_NAMES

#: Source joint -> our joint, for *bone geometry* (one source joint per ours).
JOINT_MAP = {
    "pelvis": "Spine1",
    "spine": "Spine3",
    "head": "Head",
    "shoulder_l": "UpperArm_L",
    "elbow_l": "LowerArm_L",
    "wrist_l": "Hand_L",
    "shoulder_r": "UpperArm_R",
    "elbow_r": "LowerArm_R",
    "wrist_r": "Hand_R",
    "hip_l": "UpperLeg_L",
    "knee_l": "LowerLeg_L",
    "ankle_l": "Foot_L",
    "toe_l": "Toe_L",
    "hip_r": "UpperLeg_R",
    "knee_r": "LowerLeg_R",
    "ankle_r": "Foot_R",
    "toe_r": "Toe_R",
}

#: Source joint -> our joint, for *skin weights* (many source joints per ours). The source rig is
#: finer than ours: its extra spine link and its fingers/thumbs have no counterpart, so their weight
#: folds into the nearest joint we do model. Anything not listed falls back to JOINT_MAP's inverse.
WEIGHT_FOLD = {
    "Spine2": "spine",
    "Neck": "head",
    **{f"Fingers{i}_{s}": f"wrist_{s.lower()}" for i in (1, 2, 3) for s in ("L", "R")},
    **{f"Thumb{i}_{s}": f"wrist_{s.lower()}" for i in (1, 2, 3) for s in ("L", "R")},
}

#: Joints whose offset keeps the source's full xyz (a real attachment point or an angled foot).
#: Everything else is a limb bone, collapsed to a straight-down (0, 0, -length) rest offset.
_KEEP_XYZ = {"spine", "head", "shoulder_l", "shoulder_r", "hip_l", "hip_r", "toe_l", "toe_r"}

_PARENT = {c: p for p, kids in CHILDREN.items() for c in kids}


def load_skin(dae_path: str):
    """``(joint_bind_world, weights_by_source_joint, mesh_parts, materials)`` from a rigged DAE."""
    # pycollada is the `import` extra, not a runtime dep: reading a DAE is authoring, not simulating.
    # Named here rather than left as a bare ModuleNotFoundError -- `roqsim walker --help` advertises this
    # tool to everyone, so the first thing most callers hit is the missing extra.
    try:
        import collada
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "import-actor needs pycollada: pip install 'roqsim_walker[import]'"
        ) from exc

    doc = collada.Collada(dae_path)
    if not doc.controllers:
        raise SystemExit(
            f"{dae_path}: no <library_controllers> skin -- this mesh is not rigged. A static mesh "
            f"can still be used via skin.py's auto-rig, but needs a hand-authored skeleton."
        )
    if (doc.assetInfo.upaxis or "").upper() != "Z_UP":
        raise SystemExit(f"{dae_path}: up_axis is {doc.assetInfo.upaxis}, expected Z_UP")
    unit = float(doc.assetInfo.unitmeter or 1.0)
    if abs(unit - 1.0) > 1e-6:
        raise SystemExit(f"{dae_path}: unitmeter={unit}, expected metres")

    ctrl = doc.controllers[0]
    if not np.allclose(np.array(ctrl.bind_shape_matrix), np.eye(4)):
        raise SystemExit(f"{dae_path}: non-identity bind_shape_matrix is not supported")

    names = [str(x) for x in ctrl.weight_joints.data.flatten()]
    bind = {}
    for n in names:  # world bind pose = inverse of the inverse-bind matrix
        if n not in bind:
            bind[n] = np.linalg.inv(np.array(ctrl.joint_matrices[n]).reshape(4, 4))
    wsrc = ctrl.weights.data.flatten()

    # per-vertex [(joint_idx, weight_idx), ...] -> dense (nvert, n_source_joint)
    nvert = len(ctrl.index)
    W = np.zeros((nvert, len(names)))
    for vi, pairs in enumerate(ctrl.index):
        for ji, wi in np.asarray(pairs).reshape(-1, 2):
            W[vi, int(ji)] += float(wsrc[int(wi)])

    parts = list(ctrl.geometry.primitives)  # the skinned geometry's per-material groups
    for prim in parts:
        if not getattr(prim, "texcoordset", None):
            raise SystemExit(f"{dae_path}: material group {prim.material!r} has no UVs")
    materials = {}
    for mat in doc.materials:
        tex = None
        dif = mat.effect.diffuse
        if hasattr(dif, "sampler") and dif.sampler is not None:
            tex = os.path.basename(str(dif.sampler.surface.image.path))
        materials[mat.id] = {"texture": tex, "path": _image_path(dae_path, dif)}
    return doc, ctrl, names, bind, W, parts, materials


def _image_path(dae_path: str, diffuse) -> str | None:
    """Absolute path of a diffuse texture, resolved relative to the DAE (paths may escape meshes/)."""
    if not hasattr(diffuse, "sampler") or diffuse.sampler is None:
        return None
    raw = str(diffuse.sampler.surface.image.path)
    if raw.startswith("file://"):
        raw = raw[7:]
    cand = raw if os.path.isabs(raw) else os.path.join(os.path.dirname(dae_path), raw)
    return os.path.normpath(cand) if os.path.isfile(cand) else None


def skeleton_from_bind(bind: dict) -> tuple[dict, float]:
    """``(offsets, root_height)`` in this package's canonical rest convention.

    The source is T-posed (arms out along +-Y) but the locomotion clips are authored against an
    **arms-down** rest, and ``skin.add_skin(tpose=True)`` is what rotates the bind arms back out to
    meet the mesh. So limb bones are emitted as ``(0, 0, -length)`` -- length from the source,
    direction canonical -- while real attachment points (shoulder/hip) and the angled foot keep
    their measured xyz. Baking the T-pose into the offsets here would make every clip splay the arms.
    """
    pos = {ours: bind[src][:3, 3] for ours, src in JOINT_MAP.items()}
    offsets = {}
    for ours in JOINT_NAMES:
        if ours == "pelvis":
            continue
        delta = pos[ours] - pos[_PARENT[ours]]
        offsets[ours] = (
            [round(float(v), 4) for v in delta]
            if ours in _KEEP_XYZ
            else [0.0, 0.0, -round(float(np.linalg.norm(delta)), 4)]
        )
    return offsets, round(float(pos["pelvis"][2]), 4)


def fold_weights(W: np.ndarray, names: list[str]) -> np.ndarray:
    """(nvert, n_source) -> (nvert, 17), summing each source joint onto the joint we model."""
    inv = {src: ours for ours, src in JOINT_MAP.items()}
    col = {n: i for i, n in enumerate(JOINT_NAMES)}
    out = np.zeros((W.shape[0], len(JOINT_NAMES)))
    unmapped = set()
    for i, src in enumerate(names):
        ours = inv.get(src) or WEIGHT_FOLD.get(src)
        if ours is None:
            unmapped.add(src)
            continue
        out[:, col[ours]] += W[:, i]
    if unmapped:
        raise SystemExit(
            f"unmapped source joints: {sorted(unmapped)}. Extend JOINT_MAP/WEIGHT_FOLD -- refusing "
            f"to silently drop their skin weight."
        )
    return out / (out.sum(axis=1, keepdims=True) + 1e-9)


def measure_collision(verts: np.ndarray, w: np.ndarray, bind_ours: dict) -> dict:
    """joint -> capsule radius (m): the spread of the vertices this joint dominates, measured off
    its bone axis. Mirrors what a CARLA Phys asset gives, but derived from geometry.

    Measured against ``bind_ours``, the bones the skin is bound to at runtime in the mesh's own
    (T-)pose (:func:`measure` takes them from ``skin._bind_bones``), which are what each capsule
    rides. Not the arms-down rest pose, which would place the arm bones vertically while the mesh's
    arms stick out sideways; and not the source rig's own bones, which the walker's straight limbs
    do not keep.
    """
    dom = np.argmax(w, axis=1)
    out = {}
    for ji, name in enumerate(JOINT_NAMES):
        if name in ("toe_l", "toe_r"):  # leaf feet use humanoid._TERMINALS, not a radius
            continue
        vs = verts[dom == ji]
        if len(vs) < 4:
            continue
        a = bind_ours[name]
        kids = CHILDREN[name]
        b = bind_ours[kids[0]] if kids else a + np.array([0.0, 0.0, 0.06])
        ab = b - a
        t = np.clip((vs - a) @ ab / (ab @ ab + 1e-9), 0.0, 1.0)
        d = np.linalg.norm(vs - (a + t[:, None] * ab), axis=1)
        out[name] = round(float(np.percentile(d, 75)), 4)
    return out


def _lowest(points: np.ndarray, rear: bool, what: str) -> np.ndarray:
    """The lowest of ``points``; of several as low (to 0.02 mm), the rearmost or the foremost."""
    if not len(points):
        raise SystemExit(f"no vertex follows {what}; cannot measure its sole")
    low = points[points[:, 2] <= points[:, 2].min() + 2e-5]
    return low[np.argmin(low[:, 0])] if rear else low[np.argmax(low[:, 0])]


def measure_sole(verts: np.ndarray, w: np.ndarray, bind_ours: dict) -> tuple[dict, list]:
    """``(sole, foot_tip)``: where each shoe actually touches the floor.

    ``sole[s]['heel']`` is the lowest point of the vertices the **ankle** carries, as an offset from
    the ankle; ``sole[s]['toe']`` the lowest of those the **toe** carries, from the toe -- which is
    how ``animation._lowest_sole`` reads them (``joint + rotate(q, offset)``), each riding its
    joint. Taking the lowest point (ties to the rearmost and foremost) puts the standing walker's
    lowest point exactly on the floor whatever the sole's shape; a point merely low on the shoe
    (the rearmost of its lowest centimetre) can stand above it and let the shoe sink.

    ``bind_ours`` are the bones the skin is bound to at runtime (:func:`measure` takes them from
    ``skin._bind_bones``); at rest their frames are world-aligned, so a plain world-space delta is
    what the grounding wants.
    """
    dom = np.argmax(w, axis=1)
    col = {n: i for i, n in enumerate(JOINT_NAMES)}
    sole, tips = {}, []
    for s in ("l", "r"):
        heel_pt = _lowest(verts[dom == col[f"ankle_{s}"]], rear=True, what=f"ankle_{s}")
        toe_pt = _lowest(verts[dom == col[f"toe_{s}"]], rear=False, what=f"toe_{s}")
        sole[s] = {
            "heel": [round(float(v), 4) for v in (heel_pt - bind_ours[f"ankle_{s}"])],
            "toe": [round(float(v), 4) for v in (toe_pt - bind_ours[f"toe_{s}"])],
        }
        tips.append(toe_pt - bind_ours[f"toe_{s}"])
    return sole, [round(float(v), 4) for v in np.mean(tips, axis=0)]


def measure_hull(verts: np.ndarray, w: np.ndarray) -> dict:
    """``side -> {"joints", "at", "w"}``: the points of each shoe its foot grounding is taken on.

    The lowest point of a body turned any way lies on its convex hull, so each shoe is sampled by
    the hull's corners -- one hull per joint the shoe's vertices follow (ankle, toe), since the toe
    bends against the foot. Each point keeps its bind position in the walker's frame (``at``) and
    its skin weights over that leg's joints (``w``, columns ``joints``), so the runtime skins it
    exactly as the mesh (``animation._lowest_sole``) and the shoe can turn no point of its skin
    under the floor unseen -- as the toe cap at toe-off, which a heel and a toe point miss.
    """
    from scipy.spatial import ConvexHull

    dom = np.argmax(w, axis=1)
    col = {n: i for i, n in enumerate(JOINT_NAMES)}
    out = {}
    for s in ("l", "r"):
        leg = [f"hip_{s}", f"knee_{s}", f"ankle_{s}", f"toe_{s}"]
        idx = []
        for j in (f"ankle_{s}", f"toe_{s}"):
            group = np.flatnonzero(dom == col[j])
            if len(group) < 4:
                raise SystemExit(f"the {j} shoe has {len(group)} vertices; cannot take its hull")
            idx.extend(group[ConvexHull(verts[group]).vertices])
        idx = np.unique(idx)
        wl = w[idx][:, [col[j] for j in leg]]
        off_leg = 1.0 - wl.sum(axis=1)
        if off_leg.max() > 1e-3:
            raise SystemExit(
                f"a {s} shoe vertex is {off_leg.max():.3f} skinned off its leg; refusing to drop it"
            )
        used = wl.max(axis=0) > 1e-3
        wl = wl[:, used] / wl[:, used].sum(axis=1, keepdims=True)
        out[s] = {
            "joints": [j for j, u in zip(leg, used, strict=True) if u],
            "at": np.round(verts[idx], 4).tolist(),
            "w": np.round(wl, 4).tolist(),
        }
    return out


def measure(verts: np.ndarray, w: np.ndarray, skeleton: dict, tpose: bool, flip: bool) -> dict:
    """What a blueprint's ``walker.json`` says of its mesh's shape, measured on the mesh as MuJoCo
    poses it: ``{"height", "foot_tip", "collision", "sole"}``, the sole with each shoe's ``hull``.

    ``verts`` (soles on z = 0) and their skin weights ``w`` are turned as the skin turns them
    (``skin._aligner``) and measured against the bones the skin is bound to at runtime -- ``skin._bind_bones``
    of ``skeleton`` (a ``walker.json`` skeleton dict) and ``tpose`` -- which are what the foot
    grounding adds a sole offset to and what a capsule rides. The source rig's own bones are not
    those: the walker's limbs are straight where the rig's bend, which puts the runtime ankle
    centimetres from the rig's. Anything that derives a blueprint from an import measures here too.
    """
    from roqsim_walker import skin
    from roqsim_walker.humanoid import to_skeleton

    verts = skin._aligner(verts, flip=flip)(np.asarray(verts, float))
    poses = skin._bind_bones(tpose=tpose, skeleton=to_skeleton(skeleton))
    bind = {name: pos for name, (pos, _) in poses.items()}
    collision = measure_collision(verts, w, bind)
    sole, foot_tip = measure_sole(verts, w, bind)
    for side, hull in measure_hull(verts, w).items():
        sole[side]["hull"] = hull
    return {
        "height": round(float(verts[:, 2].max()), 4),
        "foot_tip": foot_tip,
        "collision": collision,
        "sole": sole,
    }


def write_obj(path: str, parts, verts: np.ndarray, mtl_name: str) -> None:
    """Multi-material OBJ: one ``usemtl`` group per source material, sharing one vertex pool, so
    ``skin._load_parts`` (trimesh) splits it back into the groups walker.json keys by.

    The source groups index one shared vertex pool and one shared UV source, so both are written
    once and the indices carry over unchanged. Polylists are triangulated (the source mixes tris,
    quads and n-gons; MuJoCo skins want triangles).
    """
    tsets = [(p.material, p.triangleset()) for p in parts]
    uvs = np.asarray(tsets[0][1].texcoordset[0])
    for _, ts in tsets[1:]:
        if not np.array_equal(np.asarray(ts.texcoordset[0]), uvs):
            raise SystemExit("material groups do not share one UV source; unsupported")
    with open(path, "w") as f:
        f.write(f"# generated by `roqsim walker import-actor`\nmtllib {mtl_name}\n")
        for v in verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for uv in uvs:
            f.write(f"vt {uv[0]:.6f} {uv[1]:.6f}\n")
        for material, ts in tsets:
            f.write(f"usemtl {material}\n")
            vi, ti = np.asarray(ts.vertex_index), np.asarray(ts.texcoord_indexset[0])
            for tv, tt in zip(vi, ti, strict=True):
                f.write(
                    f"f {tv[0] + 1}/{tt[0] + 1} {tv[1] + 1}/{tt[1] + 1} {tv[2] + 1}/{tt[2] + 1}\n"
                )


def write_mtl(path: str, materials: dict) -> None:
    with open(path, "w") as f:
        f.write(
            "# Material table for the imported actor. roqsim_walker resolves textures from the\n"
            "# *.walker.json, NOT from this file -- but trimesh needs the .mtl present to split the\n"
            "# OBJ into per-material groups and key them by name (skin.add_skin matches on that).\n"
        )
        for name in materials:
            f.write(
                f"\nnewmtl {name}\nKa 1.000000 1.000000 1.000000\nKd 0.800000 0.800000 0.800000\nd 1.000000\nillum 2\n"
            )


def credits_text(name: str, source: str, licence: str, author: str) -> str:
    """The ``CREDITS.txt`` an imported blueprint carries: its own terms, and the clips' CARLA ones."""
    return (
        f"{name}\n\nSource:  {source}\n"
        f"Licence: {licence}\n"
        f"Author:  {author}\n\n"
        "Every file here is regenerated from that source by the import below;\n"
        "none is edited by hand. Its walker.json is measured on the mesh against\n"
        "the bones the walker's skin is posed on at runtime, and grounds each\n"
        "shoe on its hull.\n\n"
        "Imported with `roqsim walker import-actor`: the skin (mesh, textures,\n"
        "skeleton, skin weights) is the source's. Locomotion comes from this\n"
        "package's models/anims/, which are CARLA retargets under CC-BY 4.0 --\n"
        "so this walker carries CARLA attribution too; see THIRD_PARTY.md.\n"
    )


def main(argv: list | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("dae", help="rigged source .dae")
    ap.add_argument("--name", required=True, help="blueprint folder name, e.g. FemaleVisitorWalk")
    ap.add_argument(
        "--out", default=None, help="people/ dir (default: this package's models/people)"
    )
    ap.add_argument(
        "--anim-set",
        default=None,
        choices=["adult", "female", "kid"],
        help="locomotion set; blueprint._infer_gender cannot read these names, so set it",
    )
    ap.add_argument(
        "--flip",
        action="store_true",
        help="mesh faces -X (CARLA); default assumes +X like the Open-RMF actors",
    )
    ap.add_argument(
        "--exclude-material",
        action="append",
        default=[],
        metavar="NAME",
        help="drop this material group (a held prop like a handbag); repeatable",
    )
    # Required: a blueprint is committed with the CREDITS.txt written here or not at all, and one
    # written without its source, licence or author is an asset with no recorded licence.
    ap.add_argument("--credits-source", required=True, help="where the actor came from (URL)")
    ap.add_argument("--credits-licence", required=True, help='its licence, e.g. "CC-BY 4.0"')
    ap.add_argument("--credits-author", required=True, help="its author, as the source names them")
    args = ap.parse_args(argv)

    # Default to the same people/ the loader reads, so an import is visible to `resolve_walker`
    # without a path argument. Asking blueprint rather than recomputing it keeps one definition.
    out_root = args.out or os.path.join(models_dir(), "people")
    wdir = os.path.join(out_root, args.name)
    os.makedirs(wdir, exist_ok=True)
    stem = args.name[0].lower() + args.name[1:]

    doc, ctrl, names, bind, W, parts, materials = load_skin(args.dae)
    offsets, root_height = skeleton_from_bind(bind)
    w17 = fold_weights(W, names)

    # Drop excluded material groups (e.g. a held handbag) from both the mesh and the material table.
    drop = set(args.exclude_material)
    if drop:
        unknown = drop - {p.material for p in parts}
        if unknown:
            raise SystemExit(
                f"--exclude-material {sorted(unknown)}: no such group in {list(materials)}"
            )
        parts = [p for p in parts if p.material not in drop]
        materials = {m: i for m, i in materials.items() if m not in drop}

    # Mesh in the frame skin.add_skin will see: soles dropped to z=0 (its _aligner does the same).
    verts = np.asarray(ctrl.geometry.primitives[0].vertex, dtype=float).copy()
    zshift = verts[:, 2].min()
    verts[:, 2] -= zshift

    # Only the vertices the KEPT groups reference feed the geometry measurements -- so a dropped bag
    # skinned to the hand does not inflate the wrist's collision radius.
    kept = np.unique(
        np.concatenate([np.asarray(p.triangleset().vertex_index).ravel() for p in parts])
    )
    shape = measure(
        verts[kept],
        w17[kept],
        {"root_height": root_height, "offsets": offsets},
        tpose=True,
        flip=args.flip,
    )
    height, collision, sole = shape["height"], shape["collision"], shape["sole"]

    write_obj(os.path.join(wdir, f"{stem}.obj"), parts, verts, f"{stem}.mtl")
    write_mtl(os.path.join(wdir, f"{stem}.mtl"), materials)
    for info in materials.values():
        if info["path"]:
            shutil.copy2(info["path"], os.path.join(wdir, os.path.basename(info["path"])))
    np.savez_compressed(
        os.path.join(wdir, f"{stem}.weights.npz"),
        positions=verts,
        weights=w17,
        joints=np.array(JOINT_NAMES),
    )

    meta = {
        "obj": f"{stem}.obj",
        "tpose": True,
        "flip": bool(args.flip),
        "materials": {m: {"texture": i["texture"], "normal": None} for m, i in materials.items()},
        "skeleton": {
            "root_height": root_height,
            "height": height,
            "foot_tip": shape["foot_tip"],
            "offsets": offsets,
        },
        "collision": collision,
        "sole": sole,
    }
    if args.anim_set:
        meta["anim_set"] = args.anim_set
    with open(os.path.join(wdir, f"{stem}.walker.json"), "w") as f:
        json.dump(meta, f, indent=1)

    with open(os.path.join(wdir, "CREDITS.txt"), "w") as f:
        f.write(
            credits_text(args.name, args.credits_source, args.credits_licence, args.credits_author)
        )

    print(f"wrote {wdir}")
    print(f"  height {height:.3f} m | root_height {root_height:.3f} m | {len(verts)} verts")
    print(f"  materials: {', '.join(materials)}")
    print(f"  collision radii: {len(collision)} joints | anim_set: {args.anim_set or 'inferred'}")


if __name__ == "__main__":
    main()
