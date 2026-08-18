"""Scene plugin: open a world's roof by deleting its ceiling geometry at build time.

A generic, world-agnostic ``with_ceiling`` switch. When disabled it removes every geom whose geometry
lies **entirely above** a height cut (``above_z``) -- the roof, ceiling fans, hung fixtures and
clerestory windows -- while walls, pillars and floor-standing clutter (which span *down* into the
room) stay.

Deletion, and not hiding, because hiding cannot separate the two things that have to come apart.
Contact ignores appearance entirely. And ``mj_ray``/``mj_multiRay`` skip a geom exactly when its
*resolved* alpha is 0 -- independently of any ``geomgroup`` mask -- so making a roof transparent to
open the view also removes it from every lidar, which is not "opening the world" but deleting the roof
by a slower route. Deleting it outright is the honest version, and it leaves every other geom's physics
and sensing untouched.

Height is the whole definition -- no per-object name list -- so the same plugin opens any baked scene.

Config::

    ceiling:
      enabled: true      # the with_ceiling switch. true = keep the ceiling (no-op, the default, so
                         # adding this plugin never surprise-deletes geometry). false = remove it.
      above_z: 2.5       # a geom is "ceiling" iff its whole world-space AABB is above this height (m)

Set ``enabled: false`` in the world YAML (or via ``--set`` / an ``extends`` override / a campaign
factor) to open the roof. Removal happens in ``build`` (pre-compile); the engine's ``dedup_assets``
pass then drops the textures the removed geoms leave unreferenced.
"""

from __future__ import annotations

import itertools
import logging

import mujoco
import numpy as np

from ..context import SimContext
from ..plugin import Plugin

_log = logging.getLogger(__name__)


class CeilingPlugin(Plugin):
    parallel_safe = True  # build-only; no per-step work

    def __init__(self, config=None, *, name=None):
        super().__init__(config, name=name)
        # Default enabled=true: a bare `- ceiling: {}` keeps the ceiling, so dropping this plugin into
        # a world never deletes geometry unless the world explicitly asks (enabled: false).
        self.enabled = bool(self.config.get("enabled", True))
        self.above_z = float(self.config.get("above_z", 2.5))

    def validate_config(self, config: dict) -> list[str]:
        errors = []
        if "enabled" in config and not isinstance(config["enabled"], bool):
            errors.append("'enabled' must be a boolean")
        if "above_z" in config:
            try:
                z = float(config["above_z"])
                if not np.isfinite(z):
                    errors.append("'above_z' must be finite")
            except (TypeError, ValueError):
                errors.append("'above_z' must be a number")
        return errors

    def build(self, spec: mujoco.MjSpec, ctx: SimContext) -> None:
        if self.enabled:
            return  # keep the ceiling: no-op

        # Measure world-space geom heights on a throwaway compile (the only reliable source of mesh
        # bounds before the real compile). MjSpec is built for edit-then-recompile, so compiling to
        # measure, deleting, and letting the engine recompile is safe.
        model = spec.compile()
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)

        corners = np.array(list(itertools.product((-1.0, 1.0), repeat=3)))  # 8 unit-cube corners
        ceiling_names: set[str] = set()
        for g in range(model.ngeom):
            center = model.geom_aabb[g][:3]
            half = model.geom_aabb[g][3:]
            # AABB corners in the geom frame -> world z (handles the inertia-frame recentre + rotation
            # MuJoCo gives mesh geoms). world = xpos + R @ local, so world_z = xpos_z + R[2,:] . local;
            # R = geom_xmat (row-major), so R[2,:] is row 2 -> geom_xmat.reshape(3,3)[2].
            local = center + corners * half
            world_z = data.geom_xpos[g][2] + local @ data.geom_xmat[g].reshape(3, 3)[2]
            if world_z.min() > self.above_z:
                nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g)
                if nm:  # unnamed geoms can't be addressed on the spec; scene geoms are all named
                    ceiling_names.add(nm)

        if not ceiling_names:
            _log.info("ceiling: nothing above z=%.2f m to remove", self.above_z)
            return

        # Delete the ceiling geoms, remembering their meshes so we can drop the ones nothing else uses.
        orphan_meshes: set[str] = set()
        for geom in list(spec.geoms):
            if geom.name in ceiling_names:
                if geom.meshname:
                    orphan_meshes.add(geom.meshname)
                spec.delete(geom)
        still_used = {geom.meshname for geom in spec.geoms if geom.meshname}
        meshes_removed = 0
        for mesh in list(spec.meshes):
            if mesh.name in orphan_meshes and mesh.name not in still_used:
                spec.delete(mesh)
                meshes_removed += 1

        _log.info(
            "ceiling: removed %d geoms above z=%.2f m (+%d meshes); world is open",
            len(ceiling_names),
            self.above_z,
            meshes_removed,
        )
