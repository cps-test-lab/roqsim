# Copyright (C) 2026 Frederik Pasch
#
# SPDX-License-Identifier: Apache-2.0

"""One call from a compiled world to what a bake needs and is judged by: the export, the resolved
light and the fingerprint. The plugin calls it when the world loads; ``roqsim bake`` uses the same
result, taken from the same plugin, so what is baked is what is checked."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco

from . import fingerprint as fp
from . import lights as bake_lights
from .export import Export, export_world
from .settings import BakeSettings


@dataclass
class Analysis:
    export: Export
    light: dict
    fingerprint: dict
    settings: BakeSettings
    #: The compiled model it was taken from (a copy, for a plugin that analyses its spec).
    model: mujoco.MjModel | None = None


def analyse(
    m: mujoco.MjModel,
    d: mujoco.MjData,
    settings: BakeSettings,
    *,
    lights="auto",
    emitters: list[dict] | None = None,
    exclude: tuple[str, ...] = (),
    occluders: tuple[str, ...] | None = None,
) -> Analysis:
    """Export *m* at the pose in *d* (call ``mj_forward`` first) and fingerprint it."""
    geom_to_source, sources = bake_lights.emitter_geoms(m, list(emitters or []))
    export = export_world(
        m, d, settings, emitter_geoms=geom_to_source, exclude=tuple(exclude), occluders=occluders
    )
    light = bake_lights.resolve(m, d, lights, sources, export.bounds)
    return Analysis(export, light, fp.compute(export, light, settings), settings, model=m)


def analyse_spec(
    spec: mujoco.MjSpec, settings: BakeSettings, **kwargs
) -> tuple[Analysis, mujoco.MjModel]:
    """:func:`analyse` of a copy of *spec* compiled at its reset pose (``qpos0``). The spec itself
    is not compiled: compiling it would freeze its name lookup for the plugins after this one."""
    m = spec.copy().compile()
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    return analyse(m, d, settings, **kwargs), m
