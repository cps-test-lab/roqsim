# Copyright (C) 2026 Frederik Pasch
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""Baked lighting: realistic, photoreal light for any world -- global illumination, soft shadows and
ambient occlusion path-traced once into a light map, drawn by MuJoCo's classic renderer.

The classic renderer lights a world with a few lamps and hard shadows: rooms come out flat, walls
evenly bright, nothing darkens under a table, and the default lamp's shadows look wrong. This plugin
draws a world's static geometry with its light baked by Blender's Cycles (direct and bounced light,
contact shadows, ambient occlusion), and lights what moves -- a robot, a door's leaf -- live, throwing
its shadow on the baked surfaces. A world turns it on with one entry:

    components:
      - baked_lighting: {}

**Baking is done by hand, once**: ``roqsim bake <world>`` (a few minutes on a CPU for a furnished
room, ``--device optix|cuda|hip|metal`` for a GPU). The bake is kept beside the world's MJCF
(``<world>.bake/``, a few MB) and committed with it, so every checkout loads it as it is and nothing
is baked again. Loading never bakes.

**A missing or stale bake does not stop the world**: it loads unbaked, drawn as without this plugin,
with a warning that says which (missing, or stale and what changed) and gives the ``roqsim bake``
command. The warning is a *world condition* (:mod:`roqsim.conditions`), so it is recorded with the run
(a recording's provenance, ``roqsim render``'s JSON record, ``roqsim_conditions.json`` in the run's
output directory) and a result made unbaked can be told from a baked one. ``require: true`` stops a
world whose bake is missing or stale, saying the same.

What is baked is the static, drawn geometry (welded to the world); what moves is lit live; glass is
drawn as it is; the light sources are the world's emitting fixtures, else its MJCF lights, else a
documented default (:mod:`roqsim.bake.lights`). It acts in ``build_late``, after every other plugin has
built, so it sees the world however it was made. Costs: on a GPU the colour camera runs at half to two
thirds of the unbaked world's frame rate (mostly the key light's shadow pass); in software GL set
``moving_shadows: false`` and the baked world draws at the unbaked world's rate; textures take more GPU memory. ``docs/baked_lighting.rst`` has it all.

Its keys, with their defaults (``roqsim plugins describe baked_lighting`` publishes them from the
schema)::

    baked_lighting:
      path: ""              # the bake's directory; default <mjcf stem>.bake beside the world's MJCF
      require: false        # true: a missing or stale bake stops the world instead of a warning
      quality: default      # default (sized for a CPU) | high (full quality, for a GPU; > 5 MB)
      texels_per_m: {}      # light-map resolution per class, over the preset: {shell, floor, props}
      samples: 0            # Cycles samples per texel, over the preset (0: the preset's)
      denoise: true         # Open Image Denoise over the passes
      lights: auto          # auto (fixtures, else MJCF lights, else the default) | a list of lights
      emitters: []          # lamps no material declares: [{match: glob, strength_w_m2: 20.0}]
      exclude: []           # static geoms (glob on the name) left unbaked, lit live
      occluders: null       # the moving geoms (globs) whose shadow is baked; when not set, what
                            #   hangs on a joint to the world does and what floats free does not
      moving_shadows: true  # the key light throws the shadow of what moves (off for software GL)
      max_mb: 5.0           # the most the bake may take on disk; roqsim bake fails above it
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import mujoco

from ..conditions import raise_condition
from ..context import SimContext
from ..plugin import Plugin, PluginError
from ..schema import Field

log = logging.getLogger(__name__)

#: The blackboard key ``roqsim bake`` sets: the world is loaded to be baked, not drawn.
BAKING = "baked_lighting/baking"


class BakedLightingPlugin(Plugin):
    """Draw a world's static geometry with its baked light (see the module docstring)."""

    CONFIG_SCHEMA = {
        "path": Field(
            str, default="", doc="the bake's directory; default <mjcf stem>.bake beside the MJCF"
        ),
        "require": Field(bool, default=False, doc="a missing or stale bake stops the world"),
        "quality": Field(str, default="default", choices=("default", "high"), doc="the preset"),
        "texels_per_m": Field(
            dict, default=None, doc="light-map texels per metre: {shell, floor, props}"
        ),
        "samples": Field(
            int, default=0, minimum=0, doc="Cycles samples per texel (0: the preset's)"
        ),
        "denoise": Field(bool, default=True, doc="Open Image Denoise over the passes"),
        "lights": Field((str, list), default="auto", doc="auto, or a list of lights"),
        "emitters": Field(
            list, default=None, doc="lamps no material declares: [{match, strength_w_m2}]"
        ),
        "exclude": Field(list, default=None, doc="static geoms (glob) left unbaked"),
        "occluders": Field(list, default=None, doc="moving geoms (glob) whose shadow is baked"),
        "moving_shadows": Field(bool, default=True, doc="the key light throws what moves' shadow"),
        "max_mb": Field(
            float, default=5.0, minimum=0.0, unit="MB", doc="the bake's size cap on disk"
        ),
    }

    def validate_config(self, config: dict) -> list[str]:
        from ..bake.lights import check_lights
        from ..bake.settings import CLASSES

        errors = []
        lights = config.get("lights", "auto")
        if isinstance(lights, str) and lights != "auto":
            errors.append(f"lights: 'auto' or a list of lights, not {lights!r}")
        elif isinstance(lights, list):
            errors += check_lights(lights)
        for key in config.get("texels_per_m") or {}:
            if key not in CLASSES:
                errors.append(f"texels_per_m.{key}: one of {CLASSES}")
        for i, e in enumerate(config.get("emitters") or []):
            if not isinstance(e, dict) or "match" not in e:
                errors.append(f"emitters[{i}]: a mapping with 'match' (a glob on the geom name)")
        return errors

    # -- what a bake is made with -----------------------------------------------------------------

    def bake_settings(self):
        from ..bake.settings import settings_for

        c = self.config
        return settings_for(
            c.get("quality", "default"),
            texels_per_m=c.get("texels_per_m") or None,
            samples=c.get("samples") or None,
            denoise=c.get("denoise", True),
        )

    def bake_dir(self, ctx: SimContext) -> Path:
        """Where this world's bake is kept."""
        given = self.config.get("path") or ""
        if given:
            p = Path(given).expanduser()
            return (p if p.is_absolute() else Path(self.base_dir) / p).resolve()
        mjcf = getattr(ctx, "world_path", None)
        if mjcf is None:
            raise PluginError(
                f"{self.address}: the world names no MJCF file (sim.world), so there is no default "
                "place for its bake: set `path:` (a directory beside the world YAML, say "
                "`path: <world>.bake`)"
            )
        mjcf = Path(mjcf)
        return mjcf.with_name(mjcf.stem + ".bake")

    def bake_command(self, ctx: SimContext) -> str:
        source = getattr(ctx, "world_source", None)
        return f"roqsim bake {source}" if source else "roqsim bake <world.yaml>"

    # -- the build --------------------------------------------------------------------------------

    def build_late(self, spec: mujoco.MjSpec, ctx: SimContext) -> None:
        from ..bake import fingerprint as fp
        from ..bake.analysis import analyse_spec
        from ..bake.store import BakeError, read_meta

        t0 = time.perf_counter()
        c = self.config
        settings = self.bake_settings()
        try:
            self.analysis, _ = analyse_spec(
                spec,
                settings,
                lights=c.get("lights", "auto"),
                emitters=c.get("emitters") or [],
                exclude=tuple(c.get("exclude") or ()),
                occluders=tuple(c["occluders"]) if c.get("occluders") is not None else None,
            )
        except ValueError as err:
            raise PluginError(f"{self.address}: {err}") from None
        self.analysis_seconds = time.perf_counter() - t0
        self.root = self.bake_dir(ctx)
        if ctx.blackboard.get(BAKING, False):
            return  # roqsim bake: the world as it is, to be baked
        command = self.bake_command(ctx)
        try:
            meta = read_meta(self.root)
        except BakeError as err:
            meta = None
            problem = f"the bake at {self.root} cannot be read ({err})"
        else:
            problem = f"no bake at {self.root}"
        if meta is None:
            self._weaker(
                ctx,
                "bake_missing",
                f"{problem}: the world is drawn unbaked",
                command,
                {"path": str(self.root)},
            )
            return
        verdict = fp.compare(meta.get("fingerprint", {}), self.analysis.fingerprint)
        if verdict["status"] == "stale":
            self._weaker(
                ctx,
                "bake_stale",
                f"the bake at {self.root} is stale ({'; '.join(verdict['reasons'])}): the world "
                "is drawn unbaked",
                command,
                {
                    "path": str(self.root),
                    "reasons": verdict["reasons"],
                    "changed": verdict["changed"][:50],
                    "added": verdict["added"][:50],
                },
            )
            return
        self._compose(spec, ctx, meta, verdict, t0)

    def _weaker(self, ctx, code: str, message: str, command: str, details: dict) -> None:
        if self.config.get("require", False):
            raise PluginError(
                f"{self.address}: {message.split(': the world')[0]}; `require: true` "
                f"stops the world -- bake it: {command}"
            )
        raise_condition(
            ctx, source=self.address, code=code, message=message, fix=command, details=details
        )

    def _compose(self, spec, ctx, meta: dict, verdict: dict, t0: float) -> None:
        from ..bake import fingerprint as fp
        from ..bake.compose import ComposeError, compose

        m = self.analysis.model
        d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)
        try:
            stats = compose(
                spec,
                self.analysis,
                d,
                self.root,
                meta,
                moving_shadows=bool(self.config.get("moving_shadows", True)),
            )
        except ComposeError as err:
            self._weaker(
                ctx,
                "bake_stale",
                f"the bake at {self.root} does not fit the world ({err}): "
                "the world is drawn unbaked",
                self.bake_command(ctx),
                {"path": str(self.root)},
            )
            return
        self.compose_stats = stats
        if verdict["status"] == "partial":
            raise_condition(
                ctx,
                source=self.address,
                code="bake_partial",
                level="info",
                message=f"{len(verdict['missing'])} baked geom(s) are not in this world (drawn baked: the rest)",
                details={"missing": verdict["missing"][:50]},
            )
        raise_condition(
            ctx,
            source=self.address,
            code="bake_used",
            level="info",
            message=f"drawn with the bake at {self.root}",
            details={
                "path": str(self.root),
                "digest": meta.get("digest") or fp.digest(meta.get("fingerprint", {})),
                "quality": self.config.get("quality", "default"),
                "baked_at": meta.get("baked_at"),
                "load_seconds": round(time.perf_counter() - t0, 3),
                "compose_seconds": stats["seconds"],
                "geoms": stats["geoms"],
            },
        )
