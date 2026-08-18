"""One appearance vocabulary for a surface: an appearance dict -> a MuJoCo texture + material.

A world says what a floor or a wall should look like in the same words wherever it is built:

    floor:
      rgb1: [0.85, 0.85, 0.85]   # builtin-checker colour A (0..1 RGB); ignored when `texture` is set
      rgb2: [0.78, 0.78, 0.79]   # builtin-checker colour B
      texture: roqsim_assets:Concrete030   # a PNG replaces the checker (RGB/albedo channel only)
      reflectance: 0.2           # explicit > the texture's manifest > the caller's default
      physical_size: 1.8         # metres one tile spans; scalar or [x, y]
      rgba: [2.2, 2.2, 2.2, 1]   # optional multiplicative tint; RGB > 1 brightens (MuJoCo does not clamp)

Lives in the core because it has two consumers on different branches of the package graph, and belongs
to neither: :mod:`roqsim_mobile.plugins.floorplan` builds a room's floor and walls at run time, and
``roqsim scenes scene-to-mjcf`` bakes a scene's ground plane. Same reasoning as
:mod:`roqsim.floorplan_collision`. The payoff is that a floor looks the same whether it was baked or
built by the plugin, and that ``floor:`` means one thing across the substrate.

Kept out of :mod:`roqsim.textures`, which resolves texture *files* and deliberately imports no MuJoCo.
"""

from __future__ import annotations

import mujoco

from .textures import resolve_texture, texture_manifest

#: Metres one texture tile spans when neither the world nor the texture's manifest says.
DEFAULT_PHYSICAL_SIZE = 1.0


def surface_material(
    spec: mujoco.MjSpec,
    tex_name: str,
    mat_name: str,
    raw: dict,
    defaults: dict,
    default_physical_size: float = DEFAULT_PHYSICAL_SIZE,
) -> str:
    """Create texture *tex_name* + material *mat_name* from the appearance dict *raw*; return *mat_name*.

    A builtin checker (``rgb1``/``rgb2``) unless an image ``texture`` is set. Texture scale is
    real-world: ``texuniform`` makes ``texrepeat`` repetitions-per-metre, so one tile spans
    ``physical_size`` metres consistently on a plane and on a mesh whose OBJ UVs are already in metres.
    """
    cfg = {**defaults, **raw}
    tex = spec.add_texture()
    tex.name = tex_name
    tex.type = mujoco.mjtTexture.mjTEXTURE_2D
    texture_path = None
    if cfg.get("texture"):
        # An image texture replaces the builtin checker (only the RGB/albedo channel is wired).
        # Leave width/height at 0 so the loader takes them from the image file.
        texture_path = str(resolve_texture(cfg["texture"]))
        tex.builtin = mujoco.mjtBuiltin.mjBUILTIN_NONE
        tex.file = texture_path
    else:
        tex.builtin = mujoco.mjtBuiltin.mjBUILTIN_CHECKER
        tex.width = tex.height = 512
        tex.rgb1 = list(cfg["rgb1"])
        tex.rgb2 = list(cfg["rgb2"])

    mat = spec.add_material()
    mat.name = mat_name
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = tex_name
    size_x, size_y = physical_size(raw, texture_path, default_physical_size)
    mat.texuniform = True
    mat.texrepeat = [1.0 / size_x, 1.0 / size_y]
    mat.reflectance = reflectance(raw, texture_path, defaults["reflectance"])
    if "rgba" in raw:
        # Multiplicative tint on the texture/checker (like Poly Haven's base colour); RGB > 1
        # brightens (MuJoCo does not clamp material rgba to 1).
        mat.rgba = [float(c) for c in raw["rgba"]]
    return mat_name


def physical_size(
    raw: dict, texture_path: str | None, default: float = DEFAULT_PHYSICAL_SIZE
) -> tuple[float, float]:
    """(size_x, size_y) metres one texture tile spans: explicit config > texture manifest > *default*.

    A scalar means a square tile. Drives ``texrepeat`` so a texture displays at its real-world size.
    """
    val = raw.get("physical_size")
    if val is None and texture_path:
        val = texture_manifest(texture_path).get("physical_size")
    if val is None:
        val = default
    if isinstance(val, (list, tuple)):
        return float(val[0]), float(val[1])
    return float(val), float(val)


def reflectance(raw: dict, texture_path: str | None, default: float) -> float:
    """Material reflectance, in precedence order: explicit config > texture manifest > *default*."""
    if "reflectance" in raw:
        return float(raw["reflectance"])
    if texture_path:
        manifest_val = texture_manifest(texture_path).get("reflectance")
        if manifest_val is not None:
            return float(manifest_val)
    return float(default)
