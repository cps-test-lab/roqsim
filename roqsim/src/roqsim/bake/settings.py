# Copyright (C) 2026 Frederik Pasch
#
# SPDX-License-Identifier: Apache-2.0

"""The bake's settings, its presets, and the run-time light rig a baked world is drawn under.

**The presets.** ``default`` is sized for a machine with only a CPU: a furnished room of about 50 m²
with 30 props bakes in about four minutes on a 20-core desktop CPU, at a mean error of about two 8-bit
levels against the full-quality bake. What it
gives up: shadow edges on walls and floor about four times softer, small parts of props less exact,
bounced colour coarse. ``high`` is the full-quality bake (the resolution and samples that matched
a real room's photos), meant for a GPU, and larger than the default's size cap.

Resolution is in **light-map texels per metre**, per class of surface: the room's shell (walls,
ceiling), its floor (where the contact shadows are) and the props (small parts, many islands). Light
varies slowly across a surface except at shadow edges, so a light map holds it at a fraction of the
textures' resolution; the surface colour is not in the bake at all.

**The rig.** MuJoCo's classic renderer sums a material's emission and every light's ambient and
diffuse terms, clamps the sum at 1 and only then multiplies the texture. So a baked texel is stored
divided by that sum for its own normal (:func:`rig_light`), and the world is drawn under a fixed rig
whose sum is a function of the normal alone: no headlight; a key light straight down that throws the
shadow of what moves; three shadowless fills 20 degrees below the horizontal, 120 degrees apart, the
first carrying the ambient term (MuJoCo's shadow darkens a light's ambient term too, so it sits on a
light that throws none). What moves is lit by the same rig, so it is lit the way the baked surfaces
were divided: brighter on top, darker underneath.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, replace

import numpy as np

#: The bake's format. A bake of another format is stale (its ``settings`` part differs).
FORMAT_VERSION = 1

#: Where a surface's class decides its resolution.
CLASSES = ("shell", "floor", "props")


@dataclass(frozen=True)
class BakeSettings:
    """Everything that changes a bake's result (and is therefore in its fingerprint)."""

    #: Light-map texels per metre per class.
    texels_per_m: dict = field(default_factory=lambda: {"shell": 50.0, "floor": 108.0, "props": 400.0})
    #: The most a prop's light map may be across, in texels (a large flat prop needs fewer per metre).
    prop_cap: int = 512
    #: The most any light map may be across; a shell larger than this is split.
    map_cap: int = 4096
    #: Cycles' samples per texel.
    samples: int = 64
    #: Cycles' diffuse bounces (the indirect light).
    bounces: int = 6
    #: Open Image Denoise over each pass, guided by the normal pass.
    denoise: bool = True

    def as_record(self) -> dict:
        return asdict(self)


PRESETS = {
    "default": BakeSettings(),
    "high": BakeSettings(
        texels_per_m={"shell": 200.0, "floor": 430.0, "props": 800.0},
        prop_cap=1024,
        samples=512,
    ),
}


def settings_for(quality: str, texels_per_m: dict | None = None, samples=None, denoise=None) -> BakeSettings:
    """The preset *quality* with any of the overrides a world sets."""
    if quality not in PRESETS:
        raise ValueError(f"quality {quality!r}: one of {sorted(PRESETS)}")
    s = PRESETS[quality]
    if texels_per_m:
        unknown = set(texels_per_m) - set(CLASSES)
        if unknown:
            raise ValueError(f"texels_per_m: unknown classes {sorted(unknown)}; one of {CLASSES}")
        s = replace(s, texels_per_m={**s.texels_per_m, **{k: float(v) for k, v in texels_per_m.items()}})
    if samples is not None:
        s = replace(s, samples=int(samples))
    if denoise is not None:
        s = replace(s, denoise=bool(denoise))
    return s


# -- the run-time rig --------------------------------------------------------------------------------

#: The rig the classic renderer draws a baked world under (see the module docstring). Measured to do
#: what it should against the bake's own values: 0.52 of 255 levels mean difference over 16.6 M pixels
#: of a furnished room.
RIG = {
    "key": {"dir": (0.0, 0.0, -1.0), "diffuse": 0.2, "ambient": 0.0},
    "fills": (
        {"azimuth_deg": 30.0, "elevation_deg": -20.0, "diffuse": 0.25, "ambient": 0.55},
        {"azimuth_deg": 150.0, "elevation_deg": -20.0, "diffuse": 0.25, "ambient": 0.0},
        {"azimuth_deg": 270.0, "elevation_deg": -20.0, "diffuse": 0.25, "ambient": 0.0},
    ),
    "shadowsize": 4096,
}


def fill_direction(fill: dict) -> np.ndarray:
    """A fill's direction of travel (MuJoCo's ``dir``)."""
    a, e = math.radians(fill["azimuth_deg"]), math.radians(fill["elevation_deg"])
    return np.array([math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)])


def rig_light(normals: np.ndarray, emission: float | np.ndarray = 0.0) -> np.ndarray:
    """MuJoCo's lighting sum under :data:`RIG` for unit *normals* ``(..., 3)`` and a material's
    *emission*, clamped at 1: the factor the classic renderer multiplies a texel by."""
    n = np.asarray(normals, float)
    key = np.asarray(RIG["key"]["dir"], float)
    s = emission + RIG["key"]["ambient"] + RIG["key"]["diffuse"] * np.clip(n @ -key, 0, None)
    for fill in RIG["fills"]:
        s = s + fill["diffuse"] * np.clip(n @ -fill_direction(fill), 0, None) + fill["ambient"]
    return np.clip(s, 0.0, 1.0)


#: The exposure rule: the bake's light is scaled so that the 95th percentile of the shell's lit
#: vertical surfaces is this (linear). A white wall then reads 0.75 as a display value there, the level
#: that matched a real room's photos.
SHELL_P95 = 0.55
