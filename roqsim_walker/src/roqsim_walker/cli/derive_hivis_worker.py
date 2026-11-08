# Copyright (C) 2026 Frederik Pasch
# SPDX-License-Identifier: Apache-2.0
"""Derive the WorkerHiVisWalk blueprint from an import of the Open-RMF OpScrubsWalk actor.

    roqsim walker import-actor OpScrubsWalk.dae --name OpScrubsWalk --out /tmp/people \\
        --credits-source https://fuel.gazebosim.org/1.0/Luca/models/OpScrubsWalk \\
        --credits-licence "CC-BY 4.0" --credits-author "Wan Yi Seow (Open Robotics); Fuel owner: Luca"
    roqsim walker derive-hivis-worker /tmp/people/OpScrubsWalk \\
        roqsim_walker/src/roqsim_walker/models/people/WorkerHiVisWalk

No redistributable rigged character with a high-visibility vest was found, so this dresses one: the
actor in scrubs keeps its skeleton, its skin weights, its body and its long trousers, and

* loses its surgical mask and its name badge (their faces are removed, and the hole the mask
  covered is closed with a fan of faces, skinned as the face around it);
* is subdivided once (Loop's scheme, texture coordinates and skin weights carried along), so its
  silhouette is drawn round up close rather than as the source's facets;
* is repainted on its own texture, regions found by the 3-D position each texel lands on: the scrubs
  top a dark work shirt with a high-visibility yellow vest over the torso and two reflective bands
  round it, the scrub trousers dark work trousers, the clogs dark safety shoes, the scrub cap hair,
  and the mouth the mask hid skin with lips.

The vest is a texture region on the same skeleton, not a mesh of its own: it moves with the torso
exactly and can never part from it. Nothing of another source is pasted in; every colour is painted
here, each region keeping the source's shading (its texel brightness) under the new colour.

The source layout this relies on (its UV islands, by their centres) is checked: a different source
fails, naming the island it cannot place, rather than being painted wrongly.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import trimesh
from PIL import Image
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

NAME = "WorkerHiVisWalk"
STEM = "workerHiVisWalk"
MATERIAL = "WorkerHiVis-material"
TEXTURE = "WorkerHiVis.png"

#: The source's UV islands by their centres (u, v; v up) and what each is.
ISLANDS = {
    (0.050, 0.573): "badge",
    (0.095, 0.977): "badge",
    (0.014, 0.791): "badge",
    (0.017, 0.984): "badge",
    (0.303, 0.197): "shirt",
    (0.531, 0.388): "shirt",
    (0.148, 0.193): "shirt",
    (0.935, 0.211): "shirt",  # the V-neck's inner layer
    (0.911, 0.472): "shirt",  # the collar
    (0.937, 0.660): "shoe",  # the clogs' heel straps
    (0.975, 0.658): "shoe",
    (0.089, 0.429): "shoe",
    (0.237, 0.429): "shoe",
    (0.807, 0.611): "shoe",  # soles
    (0.772, 0.612): "shoe",
    (0.924, 0.919): "skin",  # hands
    (0.788, 0.521): "skin",
    (0.636, 0.927): "skin",  # wrists
    (0.634, 0.869): "skin",
    (0.864, 0.396): "skin",  # neck
    (0.306, 0.813): "face",
    (0.477, 0.537): "eye",
    (0.155, 0.526): "eye",
    (0.403, 0.061): "mask",
    (0.789, 0.851): "trousers",
    (0.786, 0.646): "trousers",
    (0.688, 0.109): "hair",  # the scrub cap
    (0.995, 0.212): "ties",  # the cap's and the mask's ties
    (0.994, 0.241): "ties",
}
REMOVED = {"badge", "mask", "ties"}

#: sRGB colours the regions are painted in, each multiplied by the source's relative shading.
SHIRT = (52, 58, 70)  # dark grey-blue work shirt
TROUSERS = (46, 50, 58)  # charcoal work trousers
SHOE = (34, 30, 28)  # black-brown leather
HAIR = (44, 33, 26)  # dark brown
VEST = (222, 245, 40)  # high-visibility yellow
BAND = (196, 198, 200)  # retro-reflective silver
LIPS = (0.78, 0.55, 0.52)  # the skin's colour, scaled

#: The vest, in the mesh's bind (T-) pose, metres: the torso between its hem and the shoulders, out
#: to where the sleeves begin, with a V at the neck in front and two bands round it.
VEST_Z = (0.84, 1.37)
VEST_HALF_WIDTH = 0.165
EDGE = 0.006  # the width (m) over which the vest's edge fades into the shirt
NECK_V = (1.20, 1.37, 0.075)  # z where the V starts, z at the top, half its width at the top
BANDS_Z = ((0.925, 0.968), (1.025, 1.068))
#: The face below this (m) is painted as skin: the source shades the mask onto it.
LOWER_FACE_Z = 1.447
MOUTH = (0.0, 1.427, 0.021, 0.0045)  # y, z of the lips' centre, half width, half height


def load(src_dir: str):
    """The import: its OBJ's vertices, faces, texture coordinates, its texture, json and weights."""
    meta_path = next(p for p in os.listdir(src_dir) if p.endswith(".walker.json"))
    meta = json.load(open(os.path.join(src_dir, meta_path)))
    mesh = trimesh.load(os.path.join(src_dir, meta["obj"]), process=False)
    (material,) = meta["materials"].values()
    texture = Image.open(os.path.join(src_dir, material["texture"])).convert("RGB")
    weights = np.load(os.path.join(src_dir, os.path.splitext(meta["obj"])[0] + ".weights.npz"))
    v, f = np.asarray(mesh.vertices, float), np.asarray(mesh.faces)
    uv = np.asarray(mesh.visual.uv, float)
    return v, f, uv, texture, meta, weights


def islands(v, f, uv) -> np.ndarray:
    """Each face's region (:data:`ISLANDS`), from the UV island it lies in."""
    key = np.unique(np.round(np.hstack([v, uv]), 6), axis=0, return_inverse=True)[1].ravel()
    corners = key[f]
    rows = np.repeat(np.arange(len(f)), 3)
    incidence = coo_matrix((np.ones(len(rows)), (rows, corners.ravel()))).tocsr()
    _, label = connected_components(incidence @ incidence.T, directed=False)
    region = np.empty(len(f), dtype=object)
    centres = np.array(list(ISLANDS))
    for island in np.unique(label):
        faces = label == island
        centre = uv[f[faces]].reshape(-1, 2).mean(axis=0)
        d = np.linalg.norm(centres - centre, axis=1)
        if d.min() > 0.01:
            raise SystemExit(
                f"UV island at {np.round(centre, 3)} is not the source's: {d.min():.3f}"
            )
        region[faces] = ISLANDS[tuple(centres[d.argmin()])]
    return region


def close_mouth(P, F, Fuv, region, weights):
    """Close the hole the mask covered: the boundary loop in front of the face, as a fan of faces
    round a vertex at its centre, raised to the face's curve, skinned and textured as its loop."""
    edges = np.sort(np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1)
    unique, count = np.unique(edges, axis=0, return_counts=True)
    boundary = unique[count == 1]
    nxt: dict[int, list[int]] = {}
    for a, b in boundary:
        nxt.setdefault(a, []).append(b)
        nxt.setdefault(b, []).append(a)
    seen, loops = set(), []
    for start in nxt:
        if start in seen:
            continue
        loop, prev, cur = [start], None, start
        seen.add(start)
        while True:
            step = [n for n in nxt[cur] if n != prev and (n not in seen or n == start)]
            if not step or step[0] == start:
                break
            prev, cur = cur, step[0]
            loop.append(cur)
            seen.add(cur)
        loops.append(loop)
    face_faces = region == "face"
    face_verts = set(np.unique(F[face_faces]))
    mouth = [lp for lp in loops if len(lp) > 8 and set(lp) <= face_verts]
    mouth = [lp for lp in mouth if P[lp].mean(axis=0)[0] > 0.02 and 1.38 < P[lp].mean(axis=0)[2]]
    if len(mouth) != 1:
        raise SystemExit(f"expected one hole in front of the face, found {len(mouth)}")
    (loop,) = mouth
    loop = np.array(loop)
    # each loop vertex's texture coordinate, as the face's faces give it
    uv_of = {}
    for fi in np.flatnonzero(face_faces):
        for k in range(3):
            uv_of.setdefault(F[fi, k], Fuv[fi, k])
    centre = P[loop].mean(axis=0)
    centre[0] += 0.35 * (P[loop][:, 0].max() - centre[0])  # bulge forward as the face does
    c = len(P)
    P = np.vstack([P, centre])
    weights = np.vstack([weights, weights[loop].mean(axis=0)])
    c_uv = np.mean([uv_of[i] for i in loop], axis=0)
    new_f, new_uv = [], []
    for a, b in zip(loop, np.roll(loop, -1), strict=True):
        new_f.append([a, b, c])
        new_uv.append([uv_of[a], uv_of[b], c_uv])
    new_f, new_uv = np.array(new_f), np.array(new_uv)
    # face outward (+x), as the face around it does
    n = np.cross(P[new_f[:, 1]] - P[new_f[:, 0]], P[new_f[:, 2]] - P[new_f[:, 0]])
    if n[:, 0].sum() < 0:
        new_f, new_uv = new_f[:, ::-1], new_uv[:, ::-1]
    F = np.vstack([F, new_f])
    Fuv = np.vstack([Fuv, new_uv])
    region = np.concatenate([region, np.full(len(new_f), "mouth", dtype=object)])
    return P, F, Fuv, region, weights


def loop_subdivide(P, F, Fuv, region, weights):
    """One step of Loop subdivision of the welded mesh; texture coordinates per face corner and
    skin weights linearly along, so a UV seam stays where it was."""
    edges = np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    opposite = np.concatenate([F[:, 2], F[:, 0], F[:, 1]])
    key = np.sort(edges, axis=1)
    unique, inverse, count = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    inverse = inverse.ravel()
    interior = count == 2
    opp_sum = np.zeros((len(unique), 3))
    np.add.at(opp_sum, inverse, P[opposite])
    a, b = P[unique[:, 0]], P[unique[:, 1]]
    mid = np.where(interior[:, None], 0.375 * (a + b) + 0.125 * opp_sum, 0.5 * (a + b))
    # even vertices
    n_vert = len(P)
    nbr_sum = np.zeros_like(P)
    valence = np.zeros(n_vert)
    for i, j in ((0, 1), (1, 0)):
        np.add.at(nbr_sum, unique[:, i], P[unique[:, j]])
        np.add.at(valence, unique[:, i], 1)
    bnd = unique[~interior]
    bnd_sum = np.zeros_like(P)
    bnd_count = np.zeros(n_vert)
    for i, j in ((0, 1), (1, 0)):
        np.add.at(bnd_sum, bnd[:, i], P[bnd[:, j]])
        np.add.at(bnd_count, bnd[:, i], 1)
    beta = np.where(valence > 3, 3.0 / (8.0 * np.maximum(valence, 1)), 3.0 / 16.0)
    even = (1 - valence * beta)[:, None] * P + beta[:, None] * nbr_sum
    even = np.where((bnd_count == 2)[:, None], 0.75 * P + 0.125 * bnd_sum, even)
    even = np.where(((bnd_count > 0) & (bnd_count != 2))[:, None], P, even)
    P2 = np.vstack([even, mid])
    W2 = np.vstack([weights, 0.5 * (weights[unique[:, 0]] + weights[unique[:, 1]])])
    m = n_vert + inverse.reshape(3, -1).T  # (faces, 3): midpoints of edges 01, 12, 20
    a, b, c = F[:, 0], F[:, 1], F[:, 2]
    ab, bc, ca = m[:, 0], m[:, 1], m[:, 2]
    F2 = np.concatenate(
        [np.stack(t, axis=1) for t in ((a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca))]
    )
    ua, ub, uc = Fuv[:, 0], Fuv[:, 1], Fuv[:, 2]
    uab, ubc, uca = 0.5 * (ua + ub), 0.5 * (ub + uc), 0.5 * (uc + ua)
    Fuv2 = np.concatenate(
        [
            np.stack(t, axis=1)
            for t in ((ua, uab, uca), (uab, ub, ubc), (uca, ubc, uc), (uab, ubc, uca))
        ]
    )
    return P2, F2, Fuv2, np.tile(region, 4), W2


def rasterize(P, F, Fuv, region, size):
    """Per texel: the region of the face it lies in ("" for none), and its bind-pose position."""
    W, H = size
    label = np.full((H, W), "", dtype=object)
    pos = np.zeros((H, W, 3))
    for fi in range(len(F)):
        t = Fuv[fi] * [W, H]
        t[:, 1] = H - t[:, 1]
        x0, y0 = np.floor(t.min(axis=0)).astype(int)
        x1, y1 = np.ceil(t.max(axis=0)).astype(int)
        x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, W - 1), min(y1, H - 1)
        if x1 < x0 or y1 < y0:
            continue
        xs, ys = np.meshgrid(np.arange(x0, x1 + 1) + 0.5, np.arange(y0, y1 + 1) + 0.5)
        m = np.array(
            [[t[0, 0] - t[2, 0], t[1, 0] - t[2, 0]], [t[0, 1] - t[2, 1], t[1, 1] - t[2, 1]]]
        )
        if abs(np.linalg.det(m)) < 1e-12:
            continue
        rel = np.stack([xs - t[2, 0], ys - t[2, 1]], axis=-1) @ np.linalg.inv(m).T
        bary = np.concatenate([rel, 1 - rel.sum(axis=-1, keepdims=True)], axis=-1)
        inside = (bary >= -1e-3).all(axis=-1)
        yy, xx = np.nonzero(inside)
        label[y0 + yy, x0 + xx] = region[fi]
        pos[y0 + yy, x0 + xx] = bary[inside] @ P[F[fi]]
    return label, pos


def paint(texture: Image.Image, label, pos) -> Image.Image:
    """The texture repainted region by region (see the module docstring)."""
    src = np.asarray(texture, float) / 255.0
    lum = src @ [0.2126, 0.7152, 0.0722]
    out = src.copy()

    def shade(mask, colour, strength=1.0):
        rel = lum / max(np.median(lum[mask]), 1e-3)
        rel = 1.0 + strength * (np.clip(rel, 0.4, 1.6) - 1.0)
        out[mask] = np.clip(np.array(colour) / 255.0 * rel[mask][:, None], 0, 1)

    y, z, x = pos[..., 1], pos[..., 2], pos[..., 0]
    shirt = label == "shirt"
    shade(shirt, SHIRT)
    # the vest's cover of the shirt, 0..1: its edges a few millimetres soft, so the armhole and the
    # V do not step along the texels
    v_start, v_top, v_half = NECK_V
    v_width = np.clip((z - v_start) / (v_top - v_start), 0, None) * v_half

    def inside(d):
        return np.clip(d / EDGE + 0.5, 0, 1)

    cover = inside(VEST_HALF_WIDTH - np.abs(y)) * inside(z - VEST_Z[0]) * inside(VEST_Z[1] - z)
    in_v = (x > 0) & (z > v_start)
    cover = np.where(in_v, cover * inside(np.abs(y) - v_width), cover) * shirt
    shirt_colour = out.copy()
    vest = cover > 0
    shade(vest, VEST, strength=0.6)
    for lo, hi in BANDS_Z:
        shade(vest & (z > lo) & (z < hi), BAND, strength=0.15)
    out[vest] = cover[vest][:, None] * out[vest] + (1 - cover[vest][:, None]) * shirt_colour[vest]
    shade(label == "trousers", TROUSERS)
    shade(label == "shoe", SHOE, strength=0.5)
    shade(label == "hair", HAIR, strength=1.2)
    # the mouth the mask hid: the face's own skin around it, then lips
    face = label == "face"
    mouth = label == "mouth"
    dark = face & (lum < 0.12)  # the unpainted area under the mask, inside the face's island
    skin_ref = face & (z > LOWER_FACE_Z + 0.01) & (z < LOWER_FACE_Z + 0.05) & ~dark
    skin = np.median(src[skin_ref], axis=0)
    fill = mouth | dark
    out[fill] = skin
    # below the nose the source paints the mask's shadow and edge onto the skin: skin there,
    # blended into the face's own texture over the last centimetre
    lower = face & (z < LOWER_FACE_Z + 0.01)
    blend = np.clip((LOWER_FACE_Z + 0.01 - z) / 0.01, 0, 1)[lower][:, None]
    out[lower] = blend * skin + (1 - blend) * src[lower]
    my, mz, mw, mh = MOUTH
    lips = fill & (((y - my) / mw) ** 2 + ((z - mz) / mh) ** 2 < 1.0)
    out[lips] = skin * LIPS
    painted = shirt | (label == "trousers") | (label == "shoe") | (label == "hair") | fill
    # carry each painted region's border texels outwards, so a mip level or a filtered sample at
    # an island's edge reads the new colour, not the old one beside it
    region_id = np.zeros(label.shape, int)
    for k, name in enumerate(("shirt", "trousers", "shoe", "hair"), start=1):
        region_id[label == name] = k
    region_id[fill] = 5
    _, (iy, ix) = ndimage.distance_transform_edt(label == "", return_indices=True)
    near = region_id[iy, ix]
    grow = (label == "") & (near > 0) & (ndimage.distance_transform_edt(label == "") < 6)
    out[grow] = out[iy[grow], ix[grow]]
    del painted
    return Image.fromarray((out * 255 + 0.5).astype(np.uint8))


def write(dst: str, P, F, Fuv, W, texture: Image.Image, meta: dict) -> None:
    os.makedirs(dst, exist_ok=True)
    corner_key = np.hstack([F.reshape(-1, 1), np.round(Fuv.reshape(-1, 2), 7)])
    uniq, inverse = np.unique(corner_key, axis=0, return_inverse=True)
    pos_index = uniq[:, 0].astype(int)
    uv = uniq[:, 1:]
    faces = inverse.ravel().reshape(-1, 3)
    with open(os.path.join(dst, f"{STEM}.obj"), "w") as o:
        o.write(f"# derived by roqsim walker derive-hivis-worker\nmtllib {STEM}.mtl\n")
        for p in P:
            o.write(f"v {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
        for t in uv:
            o.write(f"vt {t[0]:.6f} {t[1]:.6f}\n")
        o.write(f"usemtl {MATERIAL}\n")
        for fc in faces:
            o.write("f " + " ".join(f"{pos_index[i] + 1}/{i + 1}" for i in fc) + "\n")
    with open(os.path.join(dst, f"{STEM}.mtl"), "w") as o:
        o.write(
            "# Material table for the walker. roqsim_walker resolves textures from the\n"
            "# *.walker.json, NOT from this file -- but trimesh needs the .mtl present to key the\n"
            "# OBJ's material group by name (skin.add_skin matches on that).\n"
            f"newmtl {MATERIAL}\nKd 1 1 1\nmap_Kd {TEXTURE}\n"
        )
    texture.save(os.path.join(dst, TEXTURE), optimize=True)
    from roqsim_walker.humanoid import JOINT_NAMES

    np.savez_compressed(
        os.path.join(dst, f"{STEM}.weights.npz"),
        positions=P,
        weights=W,
        joints=np.array(JOINT_NAMES),
    )
    meta = dict(meta)
    meta["obj"] = f"{STEM}.obj"
    meta["materials"] = {MATERIAL: {"texture": TEXTURE, "normal": None}}
    meta["skeleton"] = dict(meta["skeleton"], height=round(float(P[:, 2].max()), 4))
    with open(os.path.join(dst, f"{STEM}.walker.json"), "w") as o:
        json.dump(meta, o, indent=1)


def credits_text(source: str) -> str:
    """The blueprint's CREDITS.txt: the source's attribution as import-actor recorded it, and what
    was changed, which CC-BY asks a derivative to say."""
    lines = source.splitlines()
    keep = [ln for ln in lines if ln.startswith(("Source:", "Licence:", "Author:"))]
    if len(keep) != 3:
        raise SystemExit("the import's CREDITS.txt has no Source/Licence/Author lines")
    return (
        f"{NAME}\n\n"
        + "\n".join(keep)
        + "\n\nA derivative of that actor, under the same licence: imported with `roqsim walker\n"
        "import-actor` (the skin -- mesh, texture, skeleton, skin weights -- is the source's), then\n"
        "changed by `roqsim walker derive-hivis-worker`: the surgical mask, its ties and the name\n"
        "badge removed and the mouth closed, the mesh subdivided once, and the texture repainted\n"
        "(work shirt with a high-visibility vest, work trousers, safety shoes, hair, mouth).\n"
        "Locomotion comes from this package's models/anims/, which are CARLA retargets under\n"
        "CC-BY 4.0 -- so this walker carries CARLA attribution too; see THIRD_PARTY.md.\n"
    )


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("src", help="the import-actor output folder of OpScrubsWalk")
    ap.add_argument("dst", help=f"the {NAME} blueprint folder to write")
    args = ap.parse_args(argv)
    v, f, uv, texture, meta, weights = load(args.src)
    region = islands(v, f, uv)
    keep = ~np.isin(region, list(REMOVED))
    f, region = f[keep], region[keep]
    # weld to one vertex per position; texture coordinates per face corner
    P, pos_of = np.unique(np.round(v, 6), axis=0, return_inverse=True)
    pos_of = pos_of.ravel()
    F = pos_of[f]
    Fuv = uv[f]
    lookup = {
        tuple(np.round(p, 6)): w
        for p, w in zip(weights["positions"], weights["weights"], strict=True)
    }
    W = np.array([lookup[tuple(p)] for p in P])
    used = np.unique(F)
    remap = np.full(len(P), -1)
    remap[used] = np.arange(len(used))
    P, W, F = P[used], W[used], remap[F]
    P, F, Fuv, region, W = close_mouth(P, F, Fuv, region, W)
    P, F, Fuv, region, W = loop_subdivide(P, F, Fuv, region, W)
    P[:, 2] -= P[:, 2].min()  # soles on z = 0, as the importer leaves them
    label, pos = rasterize(P, F, Fuv, region, texture.size)
    write(args.dst, P, F, Fuv, W, paint(texture, label, pos), meta)
    with open(os.path.join(args.src, "CREDITS.txt")) as f:
        source = f.read()
    with open(os.path.join(args.dst, "CREDITS.txt"), "w") as f:
        f.write(credits_text(source))
    print(f"wrote {args.dst}: {len(P)} positions, {len(F)} faces")


if __name__ == "__main__":
    main()
