# Copyright (C) 2026 Frederik Pasch
#
# SPDX-License-Identifier: Apache-2.0

"""Compile a raw mesh into a minimal, lit, grounded MJCF scene -- the one place that knows how.

Not a tool: a library used by ``roqsim render`` (to show a mesh that is not a finalized model yet) and by
``roqsim assets render-thumbnails`` (for walker blueprints and prop OBJs). It is the single owner of
that scene, so the two cannot disagree about what a previewed mesh looks like.

It lives in the core rather than beside the prop pipeline that motivated it, because ``roqsim render``
names ``.obj``/``.stl`` in its own ``--help``: a documented core capability cannot be satisfied by
reaching into an optional sibling, or the promise holds only when that sibling happens to be installed.
Nothing here is asset-library-specific either -- a checker floor, a light, and an MTL lookup are
OBJ-format concerns, and ``roqsim_assets`` depends on ``roqsim`` like every other sibling, never the reverse.

Two details here are not obvious and are the reason this is shared rather than re-derived:

* **Shell inertia.** A decimated import can have flipped faces, giving a near-zero computed volume that
  fails MuJoCo's default volume-based inertia. These scenes are static previews, so surface-based
  inertia is both sufficient and robust.
* **The baseColor texture.** MuJoCo never reads textures from an OBJ/MTL, so the sibling ``.mtl``'s
  ``map_Kd`` PNG is wired into an in-memory material here -- which is what ``roqsim assets finalize-mujoco`` later
  bakes into ``<name>.xml``. Without it a textured prop previews as flat grey and looks wrong for a
  reason that has nothing to do with the mesh.

Geometry *facts* about a mesh (bounds, scale, origin, materials) are **not** here: read them from the
raw OBJ with ``roqsim assets inspect-prop``, which is the deterministic ground truth. MuJoCo recentres a
mesh on its centre of mass at compile time, so anything measured off a compiled model reports a nicely
centred box even for a prop metres off origin.
"""

from __future__ import annotations

import os

import mujoco

#: The mesh geom's name in the generated scene, so a caller can look it up (``model.mesh("prop")``).
MESH_NAME = "prop"


def color_png(mesh_path: str) -> str | None:
    """Absolute path to the mesh's baseColor **PNG** via a sibling ``.mtl``'s ``map_Kd``, or ``None``.

    Textures are converted to PNG up front on import (``finalize_mujoco.pngify``), so this just reads
    what is on disk -- nothing is transcoded. Two kinds of staleness are tolerated on purpose, and both
    matter in practice: the ``.mtl`` may name the file by a bare or stale *path* (so it is resolved by
    basename anywhere under the mesh's folder, where Sketchfab drops ``textures/``), and it may still
    name the pre-conversion *extension* (so the match is on the stem -- an MTL saying ``wood.jpg`` finds
    the converted ``wood.png``).
    """
    directory = os.path.dirname(os.path.abspath(mesh_path))
    if not os.path.isdir(directory):
        return None
    wanted: str | None = None
    for mtl in sorted(f for f in os.listdir(directory) if f.lower().endswith(".mtl")):
        try:
            with open(os.path.join(directory, mtl), encoding="utf-8", errors="replace") as handle:
                for line in handle:
                    tokens = line.split()
                    if tokens and tokens[0] == "map_Kd":
                        wanted = os.path.splitext(os.path.basename(tokens[-1]))[0].lower()
                        break
        except OSError:
            continue
        if wanted:
            break
    if not wanted:
        return None
    for root, _dirs, files in os.walk(directory):
        for name in files:
            if os.path.splitext(name)[0].lower() == wanted and name.lower().endswith(".png"):
                return os.path.join(root, name)
    return None


def build_mesh_scene(mesh_path: str) -> mujoco.MjModel:
    """Compile ``mesh_path`` into a scene with a checker floor, a light and the mesh's own texture."""
    spec = mujoco.MjSpec()
    spec.worldbody.add_light(pos=[2, -2, 3], dir=[-1, 1, -2], diffuse=[0.8, 0.8, 0.8])
    spec.add_texture(
        name="grid",
        type=mujoco.mjtTexture.mjTEXTURE_2D,
        builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
        width=256,
        height=256,
        rgb1=[0.3, 0.3, 0.3],
        rgb2=[0.4, 0.4, 0.4],
    )
    mat = spec.add_material(name="grid")
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "grid"
    mat.texrepeat = [8, 8]
    mat.texuniform = True
    floor = spec.worldbody.add_geom()
    floor.type = mujoco.mjtGeom.mjGEOM_PLANE
    floor.size = [5, 5, 0.1]
    floor.material = "grid"

    mesh = spec.add_mesh()
    mesh.name = MESH_NAME
    mesh.file = os.path.abspath(mesh_path)
    mesh.inertia = mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL
    geom = spec.worldbody.add_geom()
    geom.type = mujoco.mjtGeom.mjGEOM_MESH
    geom.meshname = MESH_NAME

    png = color_png(mesh_path)
    if png:
        spec.add_texture(name="prop_color", type=mujoco.mjtTexture.mjTEXTURE_2D, file=png)
        prop_mat = spec.add_material(name="prop_color")
        prop_mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "prop_color"
        geom.material = "prop_color"
    else:
        geom.rgba = [0.8, 0.8, 0.82, 1.0]
    return spec.compile()
