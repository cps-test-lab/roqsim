# Copyright (C) 2026 Frederik Pasch
#
# SPDX-License-Identifier: Apache-2.0

"""The bake on disk, renderer-neutral: the light alone, where it falls, and what it was baked from.

A bake is a directory (``<world>.bake/``) committed with the world:

``bake.json``
    ``format``; ``fingerprint`` (:mod:`.fingerprint`) and its ``digest``; ``settings``; ``light`` (the
    resolved sources and the rule they came from, :mod:`.lights`); ``exposure`` (what the stored light
    is multiplied by to give the value a white surface is drawn at, linear); ``geoms`` (every baked geom:
    ``key``, ``name``, ``body``, ``index_in_body``, ``collides``, ``geom_group``); ``emitters``;
    ``entities`` (per light map: its ``class``, ``size``, files and scales); ``times`` (seconds per
    step); ``device``, ``blender``, ``bytes``.

``<entity>.light.png``
    The direct plus indirect light arriving at the surface, **without the surface colour** (Cycles'
    DIFFUSE pass, DIRECT and INDIRECT): its luminance as an 8-bit greyscale PNG, sRGB-encoded,
    ``light = srgb_to_linear(png / 255) * light_scale``.

``<entity>.chroma.png``
    The light's colour at an eighth of the resolution (bounced colour varies slowly): RGB,
    ``chroma = png / 255 * 3``, so ``light_rgb = light * chroma`` (chroma's luminance is 1).

``<entity>.indirect.png``
    At a quarter of the resolution (it varies slowly, and the difference of two passes is the noisiest
    map): the indirect light's luminance (greyscale channel, sRGB-encoded, ``* indirect_scale``) and the ambient occlusion (alpha, linear, 1 = open). What a physically based
    renderer, which does direct light and shadows itself, takes as its emission and occlusion inputs.

``<entity>.uv.npz``
    ``uv``: the lightmap UV of every triangle corner, ``(n, 3, 2)`` uint16 over [0, 1], v up (v = 0 is
    the image's bottom row; glTF's v points down: ``1 - v``); ``geom``: per triangle, its geom (index
    into ``geoms``); ``flip``: per triangle, whether the map holds the side *opposite* to its winding (a
    two-sided prop's face lit from behind), so the renderer draws it turned round.

The triangles are those :mod:`.export` makes of each geom, in its order; a renderer re-derives them
from the world it loads (the fingerprint says whether they still are the same).
"""

from __future__ import annotations

import io
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

BAKE_JSON = "bake.json"
UV_SCALE = 65535.0
#: The chroma map's and the indirect map's resolution, as a divisor of the light map's.
CHROMA_DIVISOR = 8
INDIRECT_DIVISOR = 4
LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)


def linear_to_srgb(x):
    x = np.clip(np.asarray(x, np.float32), 0.0, None)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - 0.055)


def srgb_to_linear(x):
    x = np.asarray(x, np.float32)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


class BakeError(Exception):
    """A bake directory that cannot be read as one."""


@dataclass
class EntityMaps:
    """One entity's maps as arrays, rows bottom first (row ``r`` is ``v = (r + 0.5) / size``)."""

    light: np.ndarray  # (S, S, 3) float32, linear, before the exposure
    indirect: np.ndarray  # (S/4, S/4) float32
    ao: np.ndarray  # (S/4, S/4) float32
    uv: np.ndarray  # (n, 3, 2) float32
    geom: np.ndarray  # (n,) int32
    flip: np.ndarray  # (n,) bool


def _png_bytes(arr: np.ndarray, mode: str) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr[::-1], mode).save(buf, "PNG", optimize=True)
    return buf.getvalue()


def resize(a: np.ndarray, size: int) -> np.ndarray:
    """*a* ``(h, w, ...)`` resampled to ``(size, size, ...)``, bilinear, texel centres aligned."""
    h, w = a.shape[:2]
    if (h, w) == (size, size):
        return a

    def axis(n_in):
        x = (np.arange(size) + 0.5) * n_in / size - 0.5
        x0 = np.clip(np.floor(x).astype(int), 0, n_in - 1)
        x1 = np.clip(x0 + 1, 0, n_in - 1)
        return x0, x1, np.clip(x - x0, 0.0, 1.0).astype(np.float32)

    y0, y1, fy = axis(h)
    x0, x1, fx = axis(w)
    extra = (None,) * (a.ndim - 2)
    fx = fx[(None, slice(None), *extra)]
    fy = fy[(slice(None), None, *extra)]
    top = a[y0][:, x0] * (1 - fx) + a[y0][:, x1] * fx
    bottom = a[y1][:, x0] * (1 - fx) + a[y1][:, x1] * fx
    return top * (1 - fy) + bottom * fy


def _down(a: np.ndarray, f: int) -> np.ndarray:
    """*a* ``(S, S, ...)`` averaged over ``f x f`` blocks (S a multiple of f, else padded by edge)."""
    s = a.shape[0]
    n = -(-s // f) * f
    if n != s:
        pad = [(0, n - s), (0, n - s)] + [(0, 0)] * (a.ndim - 2)
        a = np.pad(a, pad, mode="edge")
    return a.reshape(n // f, f, n // f, f, *a.shape[2:]).mean((1, 3))


def encode_entity(name: str, maps: EntityMaps) -> tuple[dict[str, bytes], dict]:
    """The files of one entity and its ``bake.json`` entry (the scales)."""
    y = maps.light @ LUMA
    scale = float(np.percentile(y, 99.9)) or 1.0
    light_png = np.round(linear_to_srgb(np.clip(y / scale, 0, 1)) * 255).astype(np.uint8)
    blocks = _down(maps.light, CHROMA_DIVISOR)
    by = blocks @ LUMA
    chroma = np.where(by[..., None] > 1e-6, blocks / np.maximum(by[..., None], 1e-6), 1.0)
    chroma_png = np.round(np.clip(chroma / 3.0, 0, 1) * 255).astype(np.uint8)
    iscale = float(np.percentile(maps.indirect, 99.9)) or 1.0
    ind = np.round(linear_to_srgb(np.clip(maps.indirect / iscale, 0, 1)) * 255).astype(np.uint8)
    ao = np.round(np.clip(maps.ao, 0, 1) * 255).astype(np.uint8)
    uvq = np.round(np.clip(maps.uv, 0, 1) * UV_SCALE).astype(np.uint16)
    npz = io.BytesIO()
    np.savez_compressed(
        npz, uv=uvq, geom=maps.geom.astype(np.int32), flip=maps.flip.astype(np.uint8)
    )
    files = {
        f"{name}.light.png": _png_bytes(light_png, "L"),
        f"{name}.chroma.png": _png_bytes(chroma_png, "RGB"),
        f"{name}.indirect.png": _png_bytes(np.stack([ind, ao], -1), "LA"),
        f"{name}.uv.npz": npz.getvalue(),
    }
    entry = {
        "size": int(maps.light.shape[0]),
        "light_scale": scale,
        "indirect_scale": iscale,
        "triangles": int(len(maps.uv)),
        "files": sorted(files),
    }
    return files, entry


def write(root: Path, meta: dict, files: dict[str, bytes]) -> int:
    """Write a bake into *root* (replacing what was there); returns its size in bytes."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    keep = set(files) | {BAKE_JSON}
    for old in root.iterdir():
        if (
            old.is_file()
            and old.name not in keep
            and (old.suffix in (".png", ".npz") or old.name == BAKE_JSON)
        ):
            old.unlink()
    total = 0
    for name, data in files.items():
        (root / name).write_bytes(data)
        total += len(data)
    meta = {**meta, "bytes": total}
    text = json.dumps(meta, indent=1, sort_keys=True) + "\n"
    (root / BAKE_JSON).write_text(text)
    return total + len(text.encode())


def size_on_disk(root: Path) -> int:
    root = Path(root)
    return sum(p.stat().st_size for p in root.iterdir() if p.is_file())


def read_meta(root: Path) -> dict | None:
    """The bake's ``bake.json``, or ``None`` when there is no bake at *root*."""
    path = Path(root) / BAKE_JSON
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as err:
        raise BakeError(f"{path}: not a bake ({err})") from None


def _png(path: Path) -> np.ndarray:
    if not path.is_file():
        raise BakeError(f"{path}: missing from the bake")
    return np.asarray(Image.open(path))[::-1]


def read_entity(root: Path, name: str, entry: dict) -> EntityMaps:
    """One entity's maps, decoded."""
    root = Path(root)
    y = srgb_to_linear(_png(root / f"{name}.light.png").astype(np.float32) / 255) * float(
        entry["light_scale"]
    )
    s = y.shape[0]
    chroma = _png(root / f"{name}.chroma.png").astype(np.float32) / 255 * 3.0
    up = resize(chroma, s)
    ind = _png(root / f"{name}.indirect.png").astype(np.float32) / 255
    with np.load(root / f"{name}.uv.npz") as z:
        uv = z["uv"].astype(np.float32) / UV_SCALE
        geom, flip = z["geom"].astype(np.int32), z["flip"].astype(bool)
    return EntityMaps(
        light=y[..., None] * up,
        indirect=srgb_to_linear(ind[..., 0]) * float(entry["indirect_scale"]),
        ao=ind[..., 1],
        uv=uv,
        geom=geom,
        flip=flip,
    )


def bake_meta(analysis, entries: dict, exposure: float, extra: dict) -> dict:
    """``bake.json`` of a bake of *analysis* (:class:`roqsim.bake.analysis.Analysis`): its entities' *entries*
    (:func:`encode_entity`), the *exposure* and *extra* keys (times, device, Blender's version)."""
    from . import fingerprint as fp
    from .settings import FORMAT_VERSION

    export = analysis.export
    geoms = [
        {
            "key": g.key,
            "name": g.name,
            "body": g.body,
            "index_in_body": g.index_in_body,
            "collides": g.collides,
            "geom_group": g.geom_group,
            "entities": g.entities,
        }
        for g in export.geoms
    ]
    return {
        "format": FORMAT_VERSION,
        "fingerprint": analysis.fingerprint,
        "digest": fp.digest(analysis.fingerprint),
        "settings": analysis.settings.as_record(),
        "light": analysis.light,
        "exposure": exposure,
        "geoms": geoms,
        "entities": entries,
        "notes": export.notes,
        **extra,
    }
