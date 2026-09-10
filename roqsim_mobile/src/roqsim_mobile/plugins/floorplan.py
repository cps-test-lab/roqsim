"""Scene plugin: a floorplan as the world -- ground plane, light and walls, from a mesh or from
wall segments.

**Two sources, one plugin, because they are one thing.** A floorplan is a floorplan whether it
arrives as geometry or as a layout, and a world author picks by what they HAVE:

``mesh:``
    the building already exists as geometry -- imported from CAD, or produced by
    Floorplan-DSL / scenery_builder. The mesh is the walls; its json-ld gives exact colliders.
``lines:`` / ``floorplan:``
    the building is a layout: wall segments and door openings, built here as boxes with no mesh
    anywhere. Reach for this when the walls are the experiment's VARIABLE -- a corridor width is
    then an ordinary config value that a sweep varies and the run's provenance records, rather
    than a file baked ahead of time that nothing downstream can tell apart from another file.

Provides a ground plane named ``floor`` (the TurtleBot caster contact pair references that name)
grown to the walls' XY footprint, plus a ceiling light. Because it builds those itself it fills the
same slot as a world definition: ``sim.world`` alongside it is refused, not overridden (see
:mod:`roqsim.world`).

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

**From segments**, each wall becomes one box: visible AND collidable, because a box is already
convex, so unlike the mesh source there is nothing to hide behind -- what is drawn is what is
collided with, and the lidar sees the same wall the renderer does. A door is a hole with a beam
above it rather than a full-height gap, so a room stays enclosed over head height. The arithmetic
is :mod:`roqsim.floorplan_geometry`, shared with the mesh baker and the plan-view renderer, so a
preview, a baked world and this plugin cut the same openings.

Exactly one source, and naming neither is refused. A scene that only needs a bare floor + light
should omit this plugin and use a world definition instead (``sim.world``; unset -> ``empty_room``).

Config::

    floorplan:
      # --- one of these two sources ---
      mesh: <path>         # floorplan mesh (.stl); absolute, or relative to the process cwd
      # ...or the layout instead, in the floorplan JSON's own vocabulary:
      floorplan: rooms.json  # what `roqsim scenes dxf-to-floorplan` and the sketch window write
      lines:                 # ...or the segments inline
        - {id: 0, x0_m: 0.0, y0_m: 0.0, x1_m: 6.0, y1_m: 0.0}
      doors: [{line_id: 0, t: 0.5, width_m: 0.9}]   # t is 0..1 along that wall
      height: 2.5          # segments only: ceiling height (m)
      thickness: 0.12      # segments only: wall thickness (m)
      opening_height: 2.0  # segments only: door height; the wall above one becomes a lintel
      # --- the rest applies to both ---
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


def _boxes_as_points(boxes: list) -> list:
    """Wall boxes as world-frame corner points, so one footprint rule serves both sources.

    ``_footprint`` reduces vertex sets to an XY extent; giving it the boxes' own corners means the
    ground plane is sized the same way whether the walls came from a mesh or from segments.
    """
    import math

    out = []
    for (cx, cy, _cz), (hx, hy, _hz), yaw in boxes:
        ca, sa = math.cos(yaw), math.sin(yaw)
        corners = []
        for sx in (-hx, hx):
            for sy in (-hy, hy):
                corners.append([cx + sx * ca - sy * sa, cy + sx * sa + sy * ca, 0.0])
        out.append(corners)
    return out


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

    #: Defaults for the segment source. A wall is a real wall, not a line: it has a thickness, and
    #: a door is an opening in it with a beam above rather than a full-height gap.
    SEGMENT_DEFAULTS = {"height": 2.5, "thickness": 0.12, "opening_height": 2.0}

    def _validate_segments(self, config: dict) -> list[str]:
        errors = []
        if config.get("lines") is not None and config.get("floorplan") is not None:
            errors.append(
                "name one of 'lines' (inline) or 'floorplan' (a JSON file), not both: two layouts "
                "with no rule for which wins"
            )
        if config.get("lines") is not None and not config.get("lines"):
            # Told apart from naming no source at all: an author who wrote `lines: []` wrote
            # something, and the message has to be about what they wrote.
            errors.append("'lines' is empty: a floorplan with no walls builds nothing to drive in")
        for key in ("lines", "doors"):
            if config.get(key) is not None and not isinstance(config[key], list):
                errors.append(f"'{key}' must be a list")
        for key in self.SEGMENT_DEFAULTS:
            if key in config and float(config[key]) <= 0:
                errors.append(f"'{key}' must be > 0")
        height = float(config.get("height", self.SEGMENT_DEFAULTS["height"]))
        opening = float(config.get("opening_height", self.SEGMENT_DEFAULTS["opening_height"]))
        if opening > height:
            errors.append(
                "'opening_height' is taller than 'height': a door cannot be higher than the wall "
                "it is cut into"
            )
        return errors

    def validate_config(self, config: dict) -> list[str]:
        errors = []
        mesh = config.get("mesh")
        segments = config.get("lines") is not None or config.get("floorplan") is not None
        if bool(mesh) == bool(segments):
            errors.append(
                "name exactly one source: 'mesh' (a floorplan mesh with its json-ld colliders), or "
                "'lines'/'floorplan' (wall segments, built as boxes with no mesh at all). Omit the "
                "plugin entirely to use the default empty_room world."
            )
        if segments:
            errors.extend(self._validate_segments(config))
        elif not mesh:
            pass
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

    def _wall_boxes(self) -> list:
        """``(centre, half_size, yaw)`` per wall box, from the segments this entry states.

        The arithmetic is :mod:`roqsim.floorplan_geometry`, which the mesh baker and the plan-view
        renderer already share -- so a preview, a baked world and this plugin cut the same openings.
        A door is a hole with a beam over it, not a gap, which is what keeps a room enclosed above
        head height.
        """
        import json
        import math
        from pathlib import Path

        from roqsim.floorplan_geometry import wall_pieces

        source = self.config.get("floorplan")
        if source:
            path = Path(source)
            if not path.is_absolute():
                path = Path(self.base_dir or ".") / path
            if not path.is_file():
                raise RuntimeError(
                    f"floorplan[{self.label}]: {str(path)!r} does not exist. It is resolved "
                    f"relative to the world file; `roqsim scenes dxf-to-floorplan` and the "
                    f"scene-builder's sketch window both write this shape."
                )
            doc = json.loads(path.read_text(encoding="utf-8"))
            lines, doors = list(doc.get("lines") or []), list(doc.get("doors") or [])
        else:
            lines = list(self.config.get("lines") or [])
            doors = list(self.config.get("doors") or [])

        cfg = {
            **self.SEGMENT_DEFAULTS,
            **{k: self.config[k] for k in self.SEGMENT_DEFAULTS if k in self.config},
        }
        thickness = float(cfg["thickness"])
        boxes = []
        for (x0, y0), (x1, y1), z0, z1 in wall_pieces(
            lines, doors, float(cfg["height"]), float(cfg["opening_height"])
        ):
            length = math.hypot(x1 - x0, y1 - y0)
            if length <= 0.0:
                continue
            boxes.append(
                (
                    ((x0 + x1) / 2.0, (y0 + y1) / 2.0, (z0 + z1) / 2.0),
                    (length / 2.0, thickness / 2.0, (z1 - z0) / 2.0),
                    math.atan2(y1 - y0, x1 - x0),
                )
            )
        return boxes

    def _add_wall_boxes(self, spec: mujoco.MjSpec, boxes: list, material: str) -> None:
        """One box geom per wall piece: visible AND collidable, with no mesh anywhere.

        Unlike the mesh source there is nothing to hide behind: a box is already convex, so the
        thing that is drawn is the thing that is collided with and a lidar sees the same wall the
        renderer does.
        """
        import math

        for i, ((cx, cy, cz), half, yaw) in enumerate(boxes):
            g = spec.worldbody.add_geom()
            g.name = f"floorplan_wall_{i}"
            g.type = mujoco.mjtGeom.mjGEOM_BOX
            g.size = list(half)
            g.pos = [cx, cy, cz]
            g.quat = [math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)]
            g.material = material

    def build(self, spec: mujoco.MjSpec, ctx: SimContext) -> None:
        from roqsim.floorplan_collision import wall_colliders

        mesh = self.config.get("mesh")  # one of the two sources; enforced by validate_config
        floor_raw = self.config.get("floor") or {}
        floor_cfg = {**_FLOOR_DEFAULTS, **floor_raw}
        # World-frame wall colliders from the json-ld (required, validated): also give the exact XY
        # footprint so the ground plane is centred + sized on the floorplan, which scenery_builder does
        # NOT place at the origin (an origin-centred plane would be ~4x too big and off to one corner).
        boxes = [] if mesh else self._wall_boxes()
        if not mesh and not boxes:
            raise RuntimeError(
                f"floorplan[{self.label}]: the segments produced no walls. A floorplan with "
                f"nothing in it builds a world that quietly measures nothing."
            )
        colliders = wall_colliders(mesh) if mesh else _boxes_as_points(boxes)
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
        if mesh:
            self._add_mesh(spec, mesh, wall_mat)
            self._add_colliders(spec, colliders)
        else:
            self._add_wall_boxes(spec, boxes, wall_mat)

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
