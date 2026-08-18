"""Scene plugin: load a floorplan **mesh** as the world -- a ground plane (+light) fitted to the mesh,
the mesh itself as visual + lidar walls, and exact convex wall colliders from its json-ld.

Provides a ground plane named ``floor`` (the TurtleBot caster contact pair references that name) grown
to the mesh's XY footprint, plus a ceiling light. The floorplan mesh (e.g. an ``.stl`` from
Floorplan-DSL / scenery_builder; ported from our earlier in-house nav prototype's ``environment.py``) *is* the walls.

The mesh itself is **visual + lidar only** (``contype``/``conaffinity`` = 0): MuJoCo collides a mesh by
its *convex hull*, which for a building outline is a solid block filling the interior, so the robot
would spawn inside it and jam. The lidar raycaster (``mj_multiRay``) tests the real triangles and
ignores contype/conaffinity, so it still sees the true walls (doorways included) -- which is what a
costmap-based navigation stack needs.

Physics walls come from *exact* convex colliders, one per wall segment / column, read from the
floorplan's json-ld source next to the mesh (``<env>/json-ld/``, see
:mod:`roqsim.floorplan_collision`). They are invisible and hidden from the renderer, but
solid to physics, so the robot physically cannot drive through walls (doorways stay open). The json-ld
is **required**: a mesh without it fails validation.

This plugin requires a ``mesh``. A scene that only needs a bare floor + light should omit the plugin
and use the engine's default world (``sim.world``; unset -> ``empty_room``).

Config::

    floorplan:
      mesh: <path>         # floorplan mesh (.stl); absolute, or relative to the process cwd (REQUIRED)
      mesh_scale: 1.0      # float or [x, y, z]
      mesh_pos: [0, 0, 0]  # placement offset of the mesh in the world frame
      floor:               # ground-plane appearance + physics (all keys optional; default = light gray)
        rgb1: [0.85, 0.85, 0.85]        # builtin-checker colour A (0..1 RGB)
        rgb2: [0.78, 0.78, 0.79]        # builtin-checker colour B
        reflectance: 0.2                # 0..1; if omitted, a texture's manifest value (else 0.2) is used
        texture: null                   # PNG image; overrides rgb1/rgb2 when set. A package-qualified
                                        #   name ('roqsim_assets:Concrete030') or a PNG path
                                        #   (absolute / cwd-relative). MuJoCo loads PNG only.
        rgba: null                      # optional multiplicative tint on the texture/checker (like
                                        #   Poly Haven's base colour). RGB >1 brightens (not clamped
                                        #   to 1); e.g. [2.2, 2.2, 2.2, 1] = much brighter.
        physical_size: 1.8              # metres one tile spans (real-world scale); scalar or [x, y].
                                        #   If omitted, a texture's manifest value (else 1.0) is used.
        friction: [2.0, 0.005, 0.0001]  # geom contact friction [sliding, torsional, rolling]
      wall:                # floorplan-mesh appearance (same keys as 'floor' minus friction; default = gray)
        rgb1: [0.8, 0.8, 0.82]          # solid colour when rgb1 == rgb2 (the default)
        rgb2: [0.8, 0.8, 0.82]
        reflectance: 0.0
        texture: null                   # PNG image (see 'floor.texture'), applied to the wall mesh.
        rgba: null                      # optional tint (see 'floor.rgba')
        physical_size: 2.4              # metres one tile spans. Honoured for both a UV-less .stl (via
                                        #   texuniform) and a UV'd .obj (its UVs are scaled to match).
      light:               # a single overhead light at the floorplan centre + a global ambient
        height: 2.5                     # metres above the floor for the light
        diffuse: [0.35, 0.35, 0.35]     # light colour/intensity (flat across the cone)
        cutoff: 90.0                    # spot half-angle (deg); 90 = hemisphere, no visible cone edge
        fill: [0.3, 0.3, 0.3]           # uniform global ambient (not a light); [0, 0, 0] disables it

Textures are resolved via :func:`roqsim.textures.resolve_texture`: a package-qualified
``<package>:<name>`` (e.g. ``roqsim_assets:Concrete030``, from the shared :mod:`roqsim_assets`)
or a PNG path -- no cross-package name search. A texture folder may carry a ``manifest.yaml`` (next to
the PNG) with surface properties -- ``reflectance`` and ``physical_size`` -- used when the world does
not set the matching ``<floor|wall>`` key explicitly. When no manifest exists, the defaults are used.
"""

from __future__ import annotations

import logging
import os

import mujoco
import numpy as np

from roqsim.context import SimContext
from roqsim.plugin import Plugin
from roqsim.surfaces import physical_size, surface_material
from roqsim.textures import TextureError, UVScaler, resolve_texture

logger = logging.getLogger("roqsim_mobile.floorplan")

# Ground-plane half-extent used only when the mesh bounds can't be read (trimesh missing/unreadable).
_FALLBACK_HALF_EXTENT = 5.0

# Real-world size (m) one texture tile spans when a surface sets no 'physical_size' and its texture's
# manifest carries none. Applies to the builtin checker too (a 1 m tile => 0.5 m checker cells).
_DEFAULT_PHYSICAL_SIZE = 1.0

# Default ground-plane appearance/physics: a light-gray checker (common indoor look) with the
# original contact friction, so worlds that don't set 'floor' keep the same wheel behaviour.
_FLOOR_DEFAULTS = {
    "rgb1": [0.85, 0.85, 0.85],
    "rgb2": [0.78, 0.78, 0.79],
    "reflectance": 0.2,
    "texture": None,
    "friction": [2.0, 0.005, 0.0001],
}

# Default floorplan-mesh (wall) appearance. rgb1 == rgb2 => a solid colour (the mesh carries no UVs, so
# a 'texture' is auto-projected planar-XY by MuJoCo). Matches the previous solid wall colour, so worlds
# that don't set 'wall' look unchanged. Same keys as 'floor' minus friction (the mesh has no contacts).
_WALL_DEFAULTS = {
    "rgb1": [0.8, 0.8, 0.82],
    "rgb2": [0.8, 0.8, 0.82],
    "reflectance": 0.0,
    "texture": None,
}

# Lighting: a single overhead light at the floorplan centre + a uniform global ambient so wall-shadowed
# corners aren't black.
#
# MuJoCo's default spotlight (exponent=10, cutoff=45deg) concentrates the beam into a bright hotspot --
# the "spotty" look. exponent=0 makes intensity uniform across the cone, and 'cutoff' then just sets how
# wide a footprint the light covers (radius ~= height * tan(cutoff)).
_LIGHT_DEFAULTS = {
    "height": 2.5,  # metres above the floor for the light
    "diffuse": [0.35, 0.35, 0.35],  # light colour/intensity
    "cutoff": 90.0,  # spot half-angle (deg); 90 = hemisphere, i.e. no visible cone edge
    "fill": [0.3, 0.3, 0.3],  # uniform global ambient; [0, 0, 0] disables it
}


def _footprint(colliders: list) -> tuple[float, float, float, float]:
    """(center_x, center_y, half_x, half_y) of the floorplan's XY extent from its wall colliders.

    The json-ld colliders are world-frame, so this centres and sizes the ground plane on the actual
    floorplan (scenery_builder does not place it at the origin -- an origin-centred plane sized to
    ``max(|min|, |max|)`` would be ~4x too big and cover the floorplan with only one quadrant). Falls
    back to a default square at the origin when there are no colliders (validate_config requires them).
    """
    if colliders:
        pts = np.concatenate([np.asarray(v, dtype=float).reshape(-1, 3) for v in colliders])
        lo = pts[:, :2].min(axis=0)
        hi = pts[:, :2].max(axis=0)
        half = (hi - lo) / 2.0
        if half[0] > 0 and half[1] > 0:
            center = (lo + hi) / 2.0
            return float(center[0]), float(center[1]), float(half[0]), float(half[1])
    return 0.0, 0.0, _FALLBACK_HALF_EXTENT, _FALLBACK_HALF_EXTENT


class FloorplanPlugin(Plugin):
    # Builds its own ground plane + light fitted to the mesh, so it overrides the engine's default
    # sim.world (see roqsim.world). This is the mobile-robot scene; fixed cells use sim.world.
    provides_world = True

    def validate_config(self, config: dict) -> list[str]:
        errors = []
        mesh = config.get("mesh")
        if not mesh:
            errors.append(
                "'mesh' is required; omit the floorplan plugin to use the default empty_room world"
            )
        elif not os.path.exists(mesh):
            errors.append(f"'mesh' file does not exist: {mesh}")
        else:
            from roqsim.floorplan_collision import wall_colliders

            if not wall_colliders(mesh):
                errors.append(
                    f"'mesh' has no json-ld wall colliders next to it (expected <env>/json-ld/): {mesh}"
                )
            scale = config.get("mesh_scale", 1.0)
            values = scale if isinstance(scale, (list, tuple)) else [scale]
            if len(values) not in (1, 3) or any(float(v) <= 0 for v in values):
                errors.append(
                    "'mesh_scale' must be a positive float or a list of 3 positive floats"
                )
        errors.extend(self._validate_floor(config.get("floor") or {}))
        errors.extend(self._validate_appearance(config.get("wall") or {}, "wall"))
        errors.extend(self._validate_light(config.get("light") or {}))
        return errors

    def _validate_light(self, light: dict) -> list[str]:
        errors = []
        if "height" in light:
            try:
                if float(light["height"]) <= 0:
                    errors.append("'light.height' must be > 0")
            except (TypeError, ValueError):
                errors.append("'light.height' must be a number")
        if "cutoff" in light:
            try:
                if not (0.0 < float(light["cutoff"]) <= 90.0):
                    errors.append("'light.cutoff' must be a number in (0, 90] degrees")
            except (TypeError, ValueError):
                errors.append("'light.cutoff' must be a number")
        for key in ("diffuse", "fill"):
            if key in light:
                v = light[key]
                if (
                    not isinstance(v, (list, tuple))
                    or len(v) != 3
                    or any(not (0.0 <= float(c) <= 1.0) for c in v)
                ):
                    errors.append(f"'light.{key}' must be a list of 3 floats in [0, 1]")
        return errors

    def sources(self) -> list:
        """The mesh, and the json-ld wall colliders beside it.

        Both, because they are one artifact split across two directories: the colliders are
        found at ``<env>/json-ld/`` *relative to the mesh*, so a caller that staged only the
        mesh would get a floorplan whose walls are visual-only -- which this plugin refuses to
        build. A caller enumerating dependencies has no way to know that rule; this does.
        """
        from roqsim.floorplan_collision import _json_ld_dir

        mesh = self.config.get("mesh")
        if not mesh:
            return []
        found = [os.path.abspath(mesh)]
        jdir = _json_ld_dir(mesh)
        if jdir:
            found.extend(
                os.path.join(jdir, name)
                for name in sorted(os.listdir(jdir))
                if name.endswith(".json")
            )
        return found

    def _validate_floor(self, floor: dict) -> list[str]:
        errors = self._validate_appearance(floor, "floor")
        if "friction" in floor:
            fric = floor["friction"]
            if (
                not isinstance(fric, (list, tuple))
                or len(fric) != 3
                or any(float(v) < 0 for v in fric)
            ):
                errors.append("'floor.friction' must be a list of 3 non-negative floats")
        return errors

    def _validate_appearance(self, block: dict, prefix: str) -> list[str]:
        """Validate a surface appearance block (rgb1/rgb2/rgba/reflectance/texture); floor + wall."""
        errors = []
        for key in ("rgb1", "rgb2"):
            if key in block:
                rgb = block[key]
                if (
                    not isinstance(rgb, (list, tuple))
                    or len(rgb) != 3
                    or any(not (0.0 <= float(c) <= 1.0) for c in rgb)
                ):
                    errors.append(f"'{prefix}.{key}' must be a list of 3 floats in [0, 1]")
        if "rgba" in block:
            # A multiplicative tint on the material (texture or checker). >1 brightens (MuJoCo does not
            # clamp it to 1); alpha in [0, 1]. So only the RGB scale is unbounded above.
            rgba = block["rgba"]
            if (
                not isinstance(rgba, (list, tuple))
                or len(rgba) != 4
                or any(float(c) < 0 for c in rgba)
                or not (0.0 <= float(rgba[3]) <= 1.0)
            ):
                errors.append(
                    f"'{prefix}.rgba' must be a list of 4 non-negative floats (alpha in [0, 1]); "
                    f"RGB >1 brightens the texture"
                )
        if "reflectance" in block and not (0.0 <= float(block["reflectance"]) <= 1.0):
            errors.append(f"'{prefix}.reflectance' must be a float in [0, 1]")
        if "physical_size" in block:
            ps = block["physical_size"]
            vals = ps if isinstance(ps, (list, tuple)) else [ps]
            if len(vals) not in (1, 2) or any(float(v) <= 0 for v in vals):
                errors.append(
                    f"'{prefix}.physical_size' must be a positive number or a list of 2 positive numbers"
                )
        texture = block.get("texture")
        if texture:
            try:
                resolve_texture(texture)
            except TextureError as exc:
                errors.append(f"'{prefix}.texture' {exc}")
        return errors

    def build(self, spec: mujoco.MjSpec, ctx: SimContext) -> None:
        from roqsim.floorplan_collision import wall_colliders

        mesh = self.config.get("mesh")  # required; enforced by validate_config
        floor_raw = self.config.get("floor") or {}
        floor_cfg = {**_FLOOR_DEFAULTS, **floor_raw}
        # World-frame wall colliders from the json-ld (required, validated): also give the exact XY
        # footprint so the ground plane is centred + sized on the floorplan, which scenery_builder does
        # NOT place at the origin (an origin-centred plane would be ~4x too big and off to one corner).
        colliders = wall_colliders(mesh)
        cx, cy, half_x, half_y = _footprint(colliders)

        surface_material(spec, "grid", "floor_mat", floor_raw, _FLOOR_DEFAULTS)

        floor = spec.worldbody.add_geom()
        floor.name = "floor"
        floor.type = mujoco.mjtGeom.mjGEOM_PLANE
        floor.pos = [cx, cy, 0.0]
        floor.size = [half_x, half_y, 0.05]
        floor.material = "floor_mat"
        floor.friction = [float(v) for v in floor_cfg["friction"]]

        self._add_lights(spec, (cx, cy))

        wall_raw = self.config.get("wall") or {}
        wall_mat = surface_material(spec, "wall_grid", "wall_mat", wall_raw, _WALL_DEFAULTS)
        self._add_mesh(spec, mesh, wall_mat)
        self._add_colliders(spec, colliders)

    def _add_lights(self, spec, floor_center) -> None:
        """A single overhead light at the floorplan centre + a uniform global ambient.

        The light is a spotlight with ``exponent = 0`` so its intensity is flat across the cone
        (MuJoCo's default exponent=10 is what makes lights look "spotty"); ``cutoff`` sets the covered
        radius. ``fill`` is applied as the scene's global ambient rather than another positional light,
        so it lifts wall-shadowed corners evenly instead of adding a second hotspot.
        """
        cfg = {**_LIGHT_DEFAULTS, **(self.config.get("light") or {})}
        fill = [float(v) for v in cfg["fill"]]
        if any(fill):
            spec.visual.headlight.ambient = fill

        cx, cy = floor_center
        light = spec.worldbody.add_light()
        light.pos = [cx, cy, float(cfg["height"])]
        light.dir = [0, 0, -1]
        light.diffuse = [float(v) for v in cfg["diffuse"]]
        light.exponent = 0.0  # flat across the cone -> no hotspot
        light.cutoff = float(cfg["cutoff"])

    def _add_mesh(self, spec: mujoco.MjSpec, mesh_path: str, material: str) -> None:
        """Add the floorplan mesh as a visual + lidar-only geom (no physics contacts).

        See the module docstring: a mesh collides by its convex hull, which would fill the building's
        interior, so contacts are disabled and the lidar (which raycasts the real triangles) provides
        the obstacles. ``material`` carries the wall appearance (see :func:`roqsim.surfaces.surface_material`).
        """
        scale = self.config.get("mesh_scale", 1.0)
        scale = [float(s) for s in (scale if isinstance(scale, (list, tuple)) else [scale] * 3)]

        mesh_file = os.path.abspath(mesh_path)
        # If the wall carries a texture and the mesh has baked UVs, MuJoCo ignores texrepeat on it, so
        # 'wall.physical_size' is applied by scaling the mesh UVs (no-op for a UV-less .stl -- there the
        # material's texuniform+texrepeat set the scale). See roqsim.textures.UVScaler.
        wall_raw = self.config.get("wall") or {}
        if wall_raw.get("texture"):
            texture_path = str(resolve_texture(wall_raw["texture"]))
            size_x, _ = physical_size(wall_raw, texture_path, _DEFAULT_PHYSICAL_SIZE)
            if size_x > 0:
                self._uv_scaler = UVScaler(prefix="roqsim_mobile_uv_")
                mesh_file = self._uv_scaler.scaled(mesh_file, 1.0 / size_x, "floorplan")

        m = spec.add_mesh()
        m.name = "floorplan"
        m.file = mesh_file
        m.scale = scale

        g = spec.worldbody.add_geom()
        g.name = "floorplan"
        g.type = mujoco.mjtGeom.mjGEOM_MESH
        g.meshname = "floorplan"
        g.pos = [float(v) for v in self.config.get("mesh_pos", [0.0, 0.0, 0.0])]
        g.material = material
        g.contype = 0
        g.conaffinity = 0

    def _add_colliders(self, spec: mujoco.MjSpec, colliders: list) -> None:
        """Inject one collidable geom per exact convex wall/column (world-frame vertex sets).

        Hidden behind the visual mesh (transparent, render group 3) but solid to physics. MuJoCo
        convex-hulls each ``uservert`` set at compile time, which is exact here because every part is
        already convex. ``colliders`` comes from the mesh's json-ld, validated to exist.
        """
        for i, verts in enumerate(colliders):
            name = f"floorplan_col_{i}"
            m = spec.add_mesh()
            m.name = name
            m.uservert = np.asarray(verts, dtype=float).flatten()

            g = spec.worldbody.add_geom()
            g.name = name
            g.type = mujoco.mjtGeom.mjGEOM_MESH
            g.meshname = name
            g.rgba = [0.0, 0.0, 0.0, 0.0]  # invisible; default contype/conaffinity = collidable
            g.group = 3  # hidden by the renderer
