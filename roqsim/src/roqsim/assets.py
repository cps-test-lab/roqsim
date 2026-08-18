"""De-duplicate identical file-backed assets in an ``MjSpec`` before compilation.

A world that attaches many copies of the same model (e.g. ``spawn_model`` placing 240 storage
trays) ends up with one mesh, one texture, and one material *per copy* in the merged spec:
``spec.attach`` deep-copies the child's assets under a unique prefix, with no sharing. MuJoCo then
parses every OBJ, decodes every PNG, and computes a convex hull for every copy — 240× the work and
240× the RAM for byte-identical geometry, and the viewer uploads each duplicate to the GPU.

This pass runs between the build loop and ``spec.compile()``. It merges duplicates by content and
retargets references onto the survivor, then deletes the redundant elements. Deduping is by
*identity of the source file*, which is sound only because :func:`roqsim.models.apply_assets`
has already rewritten every ref to an absolute path — same path within one process means same bytes.

Why two levels (meshes and materials) rather than meshes and textures directly: a geom/site/mesh
references a mesh and a material by a plain **name string** (``geom.meshname`` / ``.material``),
which rewrites cleanly. A material references its textures through a fixed role vector, and that
binding is **immutable once the material came in via** ``spec.attach`` (assigning the slot reads
back changed but the compiler still resolves the original name). So textures are not repointed
directly; instead identical materials are merged (via the name-string references that *do* rewrite),
after which each duplicate texture is left unreferenced and can simply be deleted.

Deliberately conservative:

* Only file-backed meshes and 2D textures are candidates. Procedural/builtin meshes and textures (a
  checker floor, a gradient skybox) carry no file, are few and cheap, and their generative
  parameters are not worth key-matching — they are left as-is, and a material that references one is
  never merged away.
* Cube maps and skyboxes (texture type != 2D) are never deleted: they are bound by the renderer, not
  through a material slot, so they would not be seen as referenced.
"""

from __future__ import annotations

from pathlib import Path

import mujoco


def _mesh_key(mesh):
    """Content key for a file-backed mesh: source file plus everything that alters the compiled
    geometry (scale, reference frame, hull/inertia options)."""
    return (
        str(Path(mesh.file).resolve()),
        tuple(mesh.scale),
        tuple(mesh.refpos),
        tuple(mesh.refquat),
        int(mesh.smoothnormal),
        int(mesh.maxhullvert),
        int(mesh.inertia),
        bool(mesh.needsdf),
    )


def _texture_content(tex):
    """Content identity of a texture, used when keying the materials that reference it.

    File-backed 2D textures collapse by file + decode attributes. Anything else (builtin/procedural,
    cube maps, skyboxes) is keyed by its unique name, so materials referencing distinct such textures
    never merge — conservative, and those textures are never deleted anyway.
    """
    if tex.file and tex.type == mujoco.mjtTexture.mjTEXTURE_2D:
        return (
            "file",
            str(Path(tex.file).resolve()),
            int(tex.colorspace),
            int(tex.nchannel),
            bool(tex.hflip),
            bool(tex.vflip),
            str(tex.content_type),
        )
    return ("named", tex.name)


def _material_key(mat, tex_content):
    """Content key for a material: its visual parameters plus the *content* of each texture role it
    references (resolved through ``tex_content`` so two materials pointing at byte-identical textures
    under different attach prefixes compare equal)."""
    roles = tuple(
        (i, tex_content.get(name, ("missing", name))) for i, name in enumerate(mat.textures) if name
    )
    return (
        tuple(mat.rgba),
        float(mat.specular),
        float(mat.shininess),
        float(mat.reflectance),
        float(mat.metallic),
        float(mat.roughness),
        tuple(mat.emission) if hasattr(mat.emission, "__len__") else float(mat.emission),
        tuple(mat.texrepeat),
        int(mat.texuniform),
        roles,
    )


def _dedup(elements, key_fn):
    """Group ``elements`` by ``key_fn``; return (rename map old_name->survivor, list of drops)."""
    survivors: dict = {}
    rename: dict[str, str] = {}
    drops = []
    for el in elements:
        key = key_fn(el)
        keep = survivors.get(key)
        if keep is None:
            survivors[key] = el
        else:
            rename[el.name] = keep.name
            drops.append(el)
    return rename, drops


def _retarget(elements, attr, rename):
    for el in elements:
        new = rename.get(getattr(el, attr))
        if new is not None:
            setattr(el, attr, new)


def deduplicate_assets(spec: mujoco.MjSpec) -> dict:
    """Merge byte-identical file-backed meshes and materials in ``spec`` in place, dropping the
    textures that then fall unreferenced.

    Returns counts ``{"meshes_removed", "materials_removed", "textures_removed"}`` for the load
    report / logging. Safe on a spec with no duplicates (returns zeros). Must run before
    ``spec.compile()``: it mutates the asset tables and reference strings that compilation reads.
    """
    # Elements that reference a mesh/material by name string (all rewrite cleanly, unlike a
    # material's texture-role vector). Skins carry a material too; sites live under bodies.
    geoms = list(spec.geoms)
    sites = list(spec.sites)
    meshes = list(spec.meshes)
    skins = list(getattr(spec, "skins", []))
    mat_referrers = (geoms, sites, meshes, skins)

    # 1) meshes: dedup identical geometry, retarget geom.meshname.
    mesh_rename, mesh_drops = _dedup([m for m in meshes if m.file], _mesh_key)
    _retarget(geoms, "meshname", mesh_rename)
    for mesh in mesh_drops:
        spec.delete(mesh)

    # 2) materials: dedup by params + referenced-texture content, retarget every .material ref.
    tex_content = {t.name: _texture_content(t) for t in spec.textures}
    mat_rename, mat_drops = _dedup(list(spec.materials), lambda m: _material_key(m, tex_content))
    for referrers in mat_referrers:
        _retarget(referrers, "material", mat_rename)
    for mat in mat_drops:
        spec.delete(mat)

    # 3) textures: whatever no surviving material references now (file-backed 2D only) is dead.
    referenced = {name for mat in spec.materials for name in mat.textures if name}
    tex_drops = [
        t
        for t in spec.textures
        if t.file and t.type == mujoco.mjtTexture.mjTEXTURE_2D and t.name not in referenced
    ]
    for tex in tex_drops:
        spec.delete(tex)

    return {
        "meshes_removed": len(mesh_drops),
        "materials_removed": len(mat_drops),
        "textures_removed": len(tex_drops),
    }
