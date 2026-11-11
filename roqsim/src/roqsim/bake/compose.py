# Copyright (C) 2026 Frederik Pasch
#
# SPDX-License-Identifier: Apache-2.0

"""MuJoCo's classic renderer's part of a bake: what it draws, composed in memory when the world loads.

The classic renderer applies one texture per material, and lights it again. So per baked entity the
light map (:mod:`.store`) is multiplied by the model's own surface colour (its material's colour times
its texture, read through the geom's own UVs) and divided by the run-time rig's light for each texel's
normal (:func:`roqsim.bake.settings.rig_light`): MuJoCo multiplies it back and draws the bake. Where an
entity's texture carries more detail than its light map, the composed texture is finer (up to
:data:`MAX_UPSCALE` and :data:`MAX_TEXTURE`), the light resampled to it.

What the spec gets, per baked geom:

* a visual mesh of the same surface carrying the lightmap UVs and its entity's composed texture, in
  geom group :data:`roqsim.presence.WINDOW_GEOM_GROUP` (drawn, passed by every ray, colliding with
  nothing), named ``<geom>__baked`` and recorded as drawing it (:data:`DRAWN_FOR_TEXT`,
  :func:`baked_sources`), so what labels a geom by its name labels its drawn surface too;
* the geom itself moved to group 3: it still collides and every ray still meets it, and it is no
  longer drawn by default, so sensing and contacts are those of the unbaked world;
* the light sources glow (their material's emission 1); the world's lights are replaced by the rig;
* every baked geom and light source casts no shadow (``castshadow`` off): its shadows are in the
  bake already, and the rig's key throws only what moves.
"""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from .. import presence
from .analysis import Analysis
from .settings import RIG, fill_direction, rig_light
from .store import BakeError, EntityMaps, linear_to_srgb, read_entity, resize, srgb_to_linear

#: Entities composed at once.
COMPOSE_THREADS = 8
#: The most a composed texture is finer than its light map, and the largest it may be.
MAX_UPSCALE = 8
MAX_TEXTURE = 4096
#: Texture detail beyond this (texels per metre) is not carried into a composed texture.
MAX_DETAIL_PER_M = 512.0
#: How far (texels) a composed texture's islands are extended into the space between them, so filtering
#: and mip levels read the island's own colour.
PAD_TEXELS = 8


class ComposeError(Exception):
    """A bake that does not fit the world it is drawn into."""


#: The model text element naming, for each visual mesh a bake is drawn by, the geom it draws (as
#: ``{visual name: [body name, index among the body's geoms]}``, JSON): what labels a geom by its name
#: (a segmentation camera's ``geoms:`` class) reads to label its drawn surface the same way.
DRAWN_FOR_TEXT = "roqsim:baked_drawn_for"


def baked_sources(model: mujoco.MjModel) -> dict[int, int]:
    """The geom each visual mesh of a baked world draws, ``{visual geom id: source geom id}``; empty
    for a world drawn unbaked."""
    tid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TEXT, DRAWN_FOR_TEXT)
    if tid < 0:
        return {}
    adr, size = int(model.text_adr[tid]), int(model.text_size[tid])
    raw = bytes(model.text_data[adr : adr + size]).split(b"\0", 1)[0]
    out = {}
    for visual, (body, index) in json.loads(raw.decode()).items():
        vid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, visual)
        bid = 0 if body == "world" else mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
        if vid < 0 or bid < 0 or index >= int(model.body_geomnum[bid]):
            raise ValueError(
                f"{DRAWN_FOR_TEXT}: {visual!r} names {body!r}'s geom {index}, not in the model"
            )
        out[vid] = int(model.body_geomadr[bid]) + int(index)
    return out


def _spec_geom(spec: mujoco.MjSpec, rec) -> mujoco.MjsGeom:
    if rec.name:
        g = spec.geom(rec.name)
        if g is None:
            raise ComposeError(f"no geom {rec.name!r} in the world")
        return g
    body = spec.worldbody if rec.body == "world" else spec.body(rec.body)
    if body is None:
        raise ComposeError(f"no body {rec.body!r} in the world")
    geoms = list(body.geoms)
    if rec.index_in_body >= len(geoms):
        raise ComposeError(
            f"body {rec.body!r} has {len(geoms)} geoms, the bake names its {rec.index_in_body + 1}th"
        )
    return geoms[rec.index_in_body]


def rasterize(uv: np.ndarray, size: int, pad: int = PAD_TEXELS) -> np.ndarray:
    """Per texel of a ``size x size`` map (rows bottom first), the triangle covering it, -1 where none;
    then extended *pad* texels into the space between islands."""
    img = Image.new("I", (size, size), -1)
    draw = ImageDraw.Draw(img)
    pts = (uv * size).astype(np.float64)
    for i, t in enumerate(pts):
        draw.polygon([(t[0, 0], t[0, 1]), (t[1, 0], t[1, 1]), (t[2, 0], t[2, 1])], fill=i)
    ids = np.asarray(img, np.int32).copy()  # image row y is v = y / size: bottom first
    for _ in range(pad):
        empty = ids < 0
        if not empty.any():
            break
        for shifted in (
            np.roll(ids, 1, 0),
            np.roll(ids, -1, 0),
            np.roll(ids, 1, 1),
            np.roll(ids, -1, 1),
        ):
            take = empty & (shifted >= 0)
            ids[take] = shifted[take]
            empty &= ~take
    return ids


def barycentric(uv_tri: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Barycentric weights of points *p* ``(k, 2)`` in triangles ``(k, 3, 2)``, clamped into them."""
    a, b, c = uv_tri[:, 0], uv_tri[:, 1], uv_tri[:, 2]
    v0, v1, v2 = b - a, c - a, p - a
    d00 = (v0 * v0).sum(1)
    d01 = (v0 * v1).sum(1)
    d11 = (v1 * v1).sum(1)
    d20 = (v2 * v0).sum(1)
    d21 = (v2 * v1).sum(1)
    den = d00 * d11 - d01 * d01
    ok = np.abs(den) > 1e-20
    den = np.where(ok, den, 1.0)
    w1 = np.where(ok, (d11 * d20 - d01 * d21) / den, 1 / 3)
    w2 = np.where(ok, (d00 * d21 - d01 * d20) / den, 1 / 3)
    w = np.clip(np.stack([1 - w1 - w2, w1, w2], 1), 0, None)
    return w / np.maximum(w.sum(1, keepdims=True), 1e-12)


#: sRGB-encoded byte to linear, and linear (on a fine grid over [0, 1]) to an sRGB-encoded byte.
_TO_LINEAR = None
_TO_SRGB_STEPS = 16384
_TO_SRGB = None


def _luts():
    global _TO_LINEAR, _TO_SRGB
    if _TO_LINEAR is None:
        _TO_LINEAR = srgb_to_linear(np.arange(256, dtype=np.float32) / 255.0).astype(np.float32)
        x = np.arange(_TO_SRGB_STEPS + 1, dtype=np.float32) / _TO_SRGB_STEPS
        _TO_SRGB = linear_to_srgb(x).astype(np.float32)
    return _TO_LINEAR, _TO_SRGB


def to_srgb(x: np.ndarray) -> np.ndarray:
    """Linear values in [0, 1] (clipped) to sRGB-encoded ones, through a table."""
    lut = _luts()[1]
    i = np.clip(x, 0.0, 1.0)
    i *= _TO_SRGB_STEPS
    return lut[(i + 0.5).astype(np.int32)]


#: Encoded byte values (bilinearly blended, so fractional) to linear, on a grid of this many steps.
_BYTE_STEPS = 4096
_BYTE_TO_LINEAR = None


def sample(tex: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """*tex* ``(h, w, 3)`` uint8, sRGB-encoded (MuJoCo's rows: row 0 is v = 0), at *uv*: bilinear in
    the encoded values, as the renderer filters it, repeating; returned linear, float32."""
    global _BYTE_TO_LINEAR
    if _BYTE_TO_LINEAR is None:
        _BYTE_TO_LINEAR = srgb_to_linear(np.arange(_BYTE_STEPS + 1, dtype=np.float32) / _BYTE_STEPS)
    h, w = tex.shape[:2]
    flat = tex.reshape(-1, 3)
    x = uv[:, 0] * np.float32(w) - np.float32(0.5)
    y = uv[:, 1] * np.float32(h) - np.float32(0.5)
    x0f, y0f = np.floor(x), np.floor(y)
    fx, fy = (x - x0f)[:, None], (y - y0f)[:, None]
    x0 = np.mod(x0f.astype(np.int32), w)
    y0 = np.mod(y0f.astype(np.int32), h)
    x1 = x0 + 1
    x1[x1 == w] = 0
    y1 = y0 + 1
    y1[y1 == h] = 0
    r0, r1 = y0 * w, y1 * w
    top = np.take(flat, r0 + x0, axis=0) * (1 - fx) + np.take(flat, r0 + x1, axis=0) * fx
    bottom = np.take(flat, r1 + x0, axis=0) * (1 - fx) + np.take(flat, r1 + x1, axis=0) * fx
    v = top * (1 - fy) + bottom * fy
    v *= _BYTE_STEPS / 255.0
    v += 0.5
    return _BYTE_TO_LINEAR[v.astype(np.int32)]


def affine_uv(uv2: np.ndarray, uv1: np.ndarray) -> np.ndarray:
    """Per triangle, the affine map from its lightmap UVs to its albedo UVs: ``(n, 2, 3)``, so that
    ``uv1 = M[:, :, :2] @ uv2 + M[:, :, 2]``. A degenerate triangle maps everything to its first
    corner."""
    a, b, c = uv2[:, 0], uv2[:, 1], uv2[:, 2]
    e = np.stack([b - a, c - a], 2)  # (n, 2, 2): columns are the edges
    det = e[:, 0, 0] * e[:, 1, 1] - e[:, 0, 1] * e[:, 1, 0]
    ok = np.abs(det) > 1e-14
    inv = np.zeros_like(e)
    inv[ok, 0, 0] = e[ok, 1, 1] / det[ok]
    inv[ok, 0, 1] = -e[ok, 0, 1] / det[ok]
    inv[ok, 1, 0] = -e[ok, 1, 0] / det[ok]
    inv[ok, 1, 1] = e[ok, 0, 0] / det[ok]
    f = np.stack([uv1[:, 1] - uv1[:, 0], uv1[:, 2] - uv1[:, 0]], 2)  # (n, 2, 2)
    lin = f @ inv
    off = uv1[:, 0] - np.einsum("nij,nj->ni", lin, a)
    return np.concatenate([lin, off[..., None]], 2)


def detail_upscale(pos, uv1, mat, materials, textures, light_size: int) -> int:
    """How much finer than its light map an entity's composed texture is drawn: enough for its
    textures' own detail (at most :data:`MAX_DETAIL_PER_M`), 1 for an untextured one."""
    textured = np.array([materials[int(k)]["texture"] >= 0 for k in mat])
    if not textured.any():
        return 1
    tri, uvt = pos[textured], uv1[textured]
    world = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    e1, e2 = uvt[:, 1] - uvt[:, 0], uvt[:, 2] - uvt[:, 0]
    uv_area = 0.5 * np.abs(e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0])
    texels = np.array(
        [
            textures[materials[int(k)]["texture"]].shape[0]
            * textures[materials[int(k)]["texture"]].shape[1]
            for k in mat[textured]
        ],
        float,
    )
    ok = world > 1e-9
    if not ok.any():
        return 1
    density = np.sqrt(uv_area[ok] * texels[ok] / world[ok])
    order = np.argsort(density)
    cum = np.cumsum(world[ok][order])
    detail = float(min(density[order][np.searchsorted(cum, cum[-1] / 2)], MAX_DETAIL_PER_M))
    all_area = float(
        0.5 * np.linalg.norm(np.cross(pos[:, 1] - pos[:, 0], pos[:, 2] - pos[:, 0]), axis=1).sum()
    )
    light_density = light_size / max(np.sqrt(all_area * 1.4), 1e-6)
    k = int(np.ceil(detail / max(light_density, 1e-6)))
    k = max(1, min(MAX_UPSCALE, k, MAX_TEXTURE // max(light_size, 1)))
    return k


def _material_tables(materials):
    rgb = np.array(
        [srgb_to_linear(np.asarray(m["rgba"][:3], np.float32)) for m in materials], np.float32
    )
    textured = np.array([m["texture"] >= 0 for m in materials])
    return rgb, textured


def compose_entity(
    maps: EntityMaps, pos, nrm, uv1, mat, materials, textures, exposure: float
) -> tuple[np.ndarray, float, dict]:
    """The composed texture of one entity (``(C, C, 3)`` uint8, rows bottom first), its material's
    emission and what it cost. *pos*, *nrm*, *uv1*, *mat* are aligned to ``maps.uv``'s triangles.

    What varies slowly is computed at the light map's resolution: the light times the exposure and
    the rig's light for each texel's normal (whose 99.5th percentile sets the material's emission, so
    a surface facing down does not clip). At the composed resolution, finer where the entity is
    textured, each texel takes the two by bilinear interpolation and its own surface colour: its
    material's, times its texture read through the geom's own UVs where it has one."""
    t0 = time.perf_counter()
    S = maps.light.shape[0]
    rgb_table, textured_table = _material_tables(materials)
    # -- at the light map's resolution: the light and the rig's light for the normal.
    ids = rasterize(maps.uv, S)
    yy, xx = np.nonzero(ids >= 0)
    tri = ids[yy, xx]
    p = np.stack([(xx + 0.5) / S, (yy + 0.5) / S], 1)
    w = barycentric(maps.uv[tri].astype(np.float64), p).astype(np.float32)
    n = (nrm[tri].astype(np.float32) * w[..., None]).sum(1)
    n = np.where(maps.flip[tri][:, None], -n, n)
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
    lit = rig_light(n)
    emission = float(max(0.0, 1.0 - np.percentile(lit, 99.5))) if len(lit) else 0.0
    rig_map = np.ones((S, S), np.float32)
    rig_map[yy, xx] = rig_light(n, emission)
    q = (exposure * maps.light).astype(np.float32)
    # -- at the composed resolution.
    k = detail_upscale(pos, uv1, mat, materials, textures, S)
    C = S * k
    if k == 1:
        idc = ids
        q_c, rig_c = q, rig_map
    else:
        idc = rasterize(maps.uv, C, pad=0)
        empty = idc < 0
        # The space between islands takes the padded ids of the light map's resolution.
        coarse = np.repeat(np.repeat(ids, k, 0), k, 1)
        idc[empty] = coarse[empty]
        del coarse
        q_c, rig_c = resize(q, C), resize(rig_map, C)
    covered = idc >= 0
    albedo = np.zeros((C, C, 3), np.float32)
    tri_c = np.where(covered, idc, 0)
    mats = mat[tri_c]
    albedo[covered] = rgb_table[mats[covered]]
    tex_sel = covered & textured_table[mats]
    if tex_sel.any():
        flat_idx = np.flatnonzero(tex_sel)
        ty, tx = np.divmod(flat_idx, C)
        tt = idc.ravel()[flat_idx]
        # Each triangle's albedo UVs are an affine function of its lightmap UVs.
        M = affine_uv(maps.uv.astype(np.float64), uv1.astype(np.float64)).astype(np.float32)
        Mt = M[tt]
        u2x = (tx.astype(np.float32) + 0.5) / C
        u2y = (ty.astype(np.float32) + 0.5) / C
        uv = np.stack(
            [
                Mt[:, 0, 0] * u2x + Mt[:, 0, 1] * u2y + Mt[:, 0, 2],
                Mt[:, 1, 0] * u2x + Mt[:, 1, 1] * u2y + Mt[:, 1, 2],
            ],
            1,
        )
        del Mt, u2x, u2y, ty, tx
        tm = mat[tt]
        values = np.empty((len(flat_idx), 3), np.float32)
        for mi in np.unique(tm):
            sel = tm == mi
            values[sel] = sample(textures[materials[int(mi)]["texture"]], uv[sel])
        flat_albedo = albedo.reshape(-1, 3)
        flat_albedo[flat_idx] *= values
    albedo *= q_c if q_c.ndim == 3 else q_c[..., None]
    display = to_srgb(albedo)
    del albedo
    display /= np.maximum(rig_c, 1e-3)[..., None]
    clipped = float((display.max(-1)[covered] > 1.0).mean()) if covered.any() else 0.0
    np.clip(display, 0.0, 1.0, out=display)
    display *= 255.0
    display += 0.5
    out = display.astype(np.uint8)
    out[~covered] = 0
    return (
        out,
        emission,
        {
            "size": C,
            "upscale": k,
            "clipped_share": round(clipped, 4),
            "seconds": round(time.perf_counter() - t0, 3),
        },
    )


def _mesh_arrays(pos, nrm, uv, flip):
    """Per-corner arrays of a mesh in its body's frame, the flipped faces turned round."""
    pos, nrm, uv = pos.copy(), nrm.copy(), uv.copy()
    pos[flip] = pos[flip][:, ::-1]
    nrm[flip] = -nrm[flip][:, ::-1]
    uv[flip] = uv[flip][:, ::-1]
    if len(pos) < 2:  # MuJoCo wants four vertices at least
        pos, nrm, uv = (np.concatenate([a, a]) for a in (pos, nrm, uv))
    return pos, nrm, uv


def compose(
    spec: mujoco.MjSpec,
    analysis: Analysis,
    d: mujoco.MjData,
    root: Path,
    meta: dict,
    *,
    moving_shadows: bool = True,
    prefix: str = "baked",
) -> dict:
    """Draw the bake at *root* (its ``bake.json`` *meta*) into *spec*. *analysis* is the world as it
    loads (:func:`roqsim.bake.analysis.analyse_spec`), *d* its data at the pose analysed. Returns what
    was composed and what it cost."""
    t0 = time.perf_counter()
    export = analysis.export
    by_key = {g.key: g for g in export.geoms}
    position = {g.key: i for i, g in enumerate(export.geoms)}
    stored = meta["geoms"]
    exposure = float(meta["exposure"])
    stats = {"entities": {}, "geoms": 0, "missing": []}
    drawn_for: dict[str, list] = {}
    replaced: dict[str, mujoco.MjsGeom] = {}
    # What each entity draws, matched to the world as it loads.
    jobs = []
    for ename, entry in meta["entities"].items():
        try:
            maps = read_entity(root, ename, entry)
        except (BakeError, OSError) as err:
            raise ComposeError(f"{ename}'s maps cannot be read: {err}") from None
        cur = export.entities.get(ename)
        n = len(maps.uv)
        idx = np.full(n, -1, np.int64)
        present_geoms = []
        for gi in np.unique(maps.geom):
            key = stored[int(gi)]["key"]
            rec = by_key.get(key)
            mine = np.nonzero(maps.geom == gi)[0]
            if rec is None or cur is None or rec.kind != "baked":
                stats["missing"].append(key)
                continue
            cur_tris = np.nonzero(cur.geom == position[key])[0]
            if len(cur_tris) != len(mine):
                raise ComposeError(
                    f"{key}: the bake has {len(mine)} triangles in {ename}, the world "
                    f"{len(cur_tris)}; the world changed since the bake"
                )
            idx[mine] = cur_tris
            present_geoms.append((int(gi), rec, mine))
        if not present_geoms:
            continue
        keep = idx >= 0
        sub = EntityMaps(
            maps.light, maps.indirect, maps.ao, maps.uv[keep], maps.geom[keep], maps.flip[keep]
        )
        jobs.append((ename, cur, sub, idx[keep], keep, present_geoms))
    # Every geom it changes, found before the spec is changed: a bake that does not fit fails here,
    # with the world as it was, never half composed.
    for *_, present_geoms in jobs:
        for _gi, rec, _mine in present_geoms:
            if rec.key not in replaced:
                replaced[rec.key] = _spec_geom(spec, rec)
    emitters = []
    for i, src in enumerate(meta.get("light", {}).get("emitters", [])):
        rec = by_key.get(src["geom"])
        if rec is not None and rec.kind == "emitter":
            emitters.append((i, src, rec, _spec_geom(spec, rec)))

    # Composed side by side: the work is numpy's, which runs outside the interpreter's lock.
    def one(job):
        ename, cur, sub, ci, _, _ = job
        return compose_entity(
            sub,
            cur.pos[ci],
            cur.nrm[ci],
            cur.uv1[ci],
            cur.mat[ci],
            export.materials,
            export.textures,
            exposure,
        )

    order = sorted(range(len(jobs)), key=lambda i: -jobs[i][2].light.shape[0])
    with ThreadPoolExecutor(max_workers=max(1, min(COMPOSE_THREADS, os.cpu_count() or 1))) as pool:
        results = dict(zip(order, pool.map(one, [jobs[i] for i in order]), strict=True))
    for i, (ename, cur, sub, ci, keep, present_geoms) in enumerate(jobs):
        tex, emission, info = results[i]
        C = tex.shape[0]
        tname = f"{prefix}_{ename}"
        spec.add_texture(
            name=tname,
            type=mujoco.mjtTexture.mjTEXTURE_2D,
            width=C,
            height=C,
            nchannel=3,
            colorspace=mujoco.mjtColorSpace.mjCOLORSPACE_LINEAR,
        ).data = tex.tobytes()
        material = spec.add_material(
            name=tname,
            rgba=[1, 1, 1, 1],
            emission=emission,
            specular=0.0,
            shininess=0.0,
            reflectance=0.0,
        )
        material.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = tname
        stats["entities"][ename] = {**info, "emission": round(emission, 4)}
        # Per geom: its visual mesh in its body's frame.
        new_index = {int(j): i for i, j in enumerate(np.nonzero(keep)[0])}
        for _gi, rec, mine in present_geoms:
            rows = np.array([new_index[int(j)] for j in mine])
            body_id = int(analysis.model.geom_bodyid[rec.id])
            R, p = d.xmat[body_id].reshape(3, 3), d.xpos[body_id]
            pos = (cur.pos[ci[rows]] - p) @ R
            nrm = cur.nrm[ci[rows]] @ R
            pos, nrm, uv = _mesh_arrays(pos, nrm, sub.uv[rows].astype(np.float64), sub.flip[rows])
            base = rec.name or f"{rec.body}_{rec.index_in_body}"
            gname = f"{base}__baked" if len(rec.entities) <= 1 else f"{base}__baked_{ename}"
            mesh = spec.add_mesh(name=f"{prefix}_{gname}")
            mesh.inertia = mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL
            k = len(pos) * 3
            mesh.uservert = pos.reshape(-1).tolist()
            mesh.usernormal = nrm.reshape(-1).tolist()
            mesh.usertexcoord = uv.reshape(-1).tolist()
            mesh.userface = list(range(k))
            mesh.userfacenormal = list(range(k))
            mesh.userfacetexcoord = list(range(k))
            body = spec.worldbody if rec.body == "world" else spec.body(rec.body)
            visual = body.add_geom(
                name=gname,
                type=mujoco.mjtGeom.mjGEOM_MESH,
                meshname=mesh.name,
                material=tname,
                contype=0,
                conaffinity=0,
                group=presence.WINDOW_GEOM_GROUP,
                density=0.0,
                pos=[0, 0, 0],
                quat=[1, 0, 0, 0],
            )
            visual.castshadow = False
            drawn_for[gname] = [rec.body, rec.index_in_body]
            stats["geoms"] += 1
    for g in replaced.values():
        g.group = 3  # still collides, still met by every ray; drawn as its bake
    # The light sources glow, in their own colour or the one the world named.
    for i, src, rec, g in emitters:
        rgba = src.get("display_rgba") or export.materials[rec.material]["rgba"]
        m = spec.add_material(
            name=f"{prefix}_emitter_{i}",
            rgba=list(rgba),
            emission=1.0,
            specular=0.0,
            shininess=0.0,
            reflectance=0.0,
        )
        g.material = m.name
        g.castshadow = False
    _rig(spec, moving_shadows, prefix)
    for text in list(spec.texts):
        if text.name == DRAWN_FOR_TEXT:
            spec.delete(text)
    spec.add_text(name=DRAWN_FOR_TEXT, data=json.dumps(drawn_for, sort_keys=True))
    stats["seconds"] = round(time.perf_counter() - t0, 3)
    stats["textures_bytes_gpu"] = int(
        sum(v["size"] ** 2 * 3 * 4 / 3 for v in stats["entities"].values())
    )
    return stats


def _welded_to_world(body, spec: mujoco.MjSpec) -> bool:
    """Whether *body* (of *spec*) is welded to the world: no joint and no mocap body on its way there."""
    while body is not None and body.name != spec.worldbody.name:
        if len(body.joints) or body.mocap:
            return False
        body = body.parent
    return True


def _rig(spec: mujoco.MjSpec, moving_shadows: bool, prefix: str) -> None:
    """Replace the world's lights by :data:`roqsim.bake.settings.RIG`. A light on what moves (a
    robot's headlamp) is not the world's and stays."""
    for light in list(spec.lights):
        if _welded_to_world(light.parent, spec):
            spec.delete(light)
    hl = spec.visual.headlight
    hl.active = 0
    hl.ambient = [0, 0, 0]
    hl.diffuse = [0, 0, 0]
    hl.specular = [0, 0, 0]
    spec.visual.quality.shadowsize = int(RIG["shadowsize"])
    k = RIG["key"]
    key = spec.worldbody.add_light(
        name=f"{prefix}_key",
        type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
        pos=[0, 0, 3.0],
        dir=list(k["dir"]),
        castshadow=bool(moving_shadows),
    )
    key.ambient = [k["ambient"]] * 3
    key.diffuse = [k["diffuse"]] * 3
    key.specular = [0, 0, 0]
    for i, f in enumerate(RIG["fills"]):
        fill = spec.worldbody.add_light(
            name=f"{prefix}_fill{i}",
            type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
            pos=[0, 0, 3.0],
            dir=fill_direction(f).tolist(),
            castshadow=False,
        )
        fill.ambient = [f["ambient"]] * 3
        fill.diffuse = [f["diffuse"]] * 3
        fill.specular = [0, 0, 0]
