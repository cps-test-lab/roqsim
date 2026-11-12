# Copyright (C) 2026 Frederik Pasch
#
# SPDX-License-Identifier: Apache-2.0

"""Where a bake's light comes from: what the world declares, by a fixed order of rules.

With ``lights: auto`` (the default) the first rule that finds anything gives the lights, and the bake
records which (``rule``):

1. **emitting fixtures** -- static drawn geoms whose material's emission is at least
   :data:`FIXTURE_EMISSION` (a ceiling panel, a lamp's diffuser drawn glowing). Each is an emissive
   mesh in the bake at :data:`FIXTURE_W_M2` and glows at run time. The world's MJCF ``<light>``\\ s are
   then taken for the fixtures' run-time stand-ins (a world without a bake needs *some* lamp where
   its panels are) and not baked a second time;
2. **the MJCF lights** -- each ``<light>``: a point light, a spot (its ``cutoff``; 90 degrees or more is
   a point), a directional one as a sun; its power in proportion to its ``diffuse``;
3. **the default**, for a world that declares neither: one area light the size of the static
   geometry's footprint, :data:`DEFAULT_BELOW_TOP` under its highest point, facing down, and a dim
   uniform sky (:data:`DEFAULT_SKY`), as an overcast skylight would light a room without a lamp.

``emitters`` adds light sources no material declares, by geom name (fnmatch): the world's own lamps
drawn without emission (a photographer's LED panel). An ``emitters`` entry also sets the strength of a
fixture it matches. ``lights`` as a list replaces rules 2 and 3 with the lights it states (fixtures
and emitters still apply).

Only the ratio between light sources matters: one exposure per world (:data:`roqsim.bake.settings.
SHELL_P95`) sets the brightness.
"""

from __future__ import annotations

import fnmatch
import math

import mujoco
import numpy as np

from .export import geom_key, is_static, resolved_rgba

#: A material at least this emissive is a light source (a ceiling panel drawn glowing is 1; a ceiling's
#: emission standing in for bounced light, 0.35, is not).
FIXTURE_EMISSION = 0.5
#: A fixture's radiant exitance in the bake (W/m²): a 60 W panel of 1.2 x 0.6 m.
FIXTURE_W_M2 = 83.0
#: A point or spot light of diffuse 1, in watts.
POINT_W_PER_DIFFUSE = 300.0
#: A directional light of diffuse 1, as a sun's strength (W/m²).
SUN_PER_DIFFUSE = 3.0
#: The default light's distance under the top of the static geometry, in metres.
DEFAULT_BELOW_TOP = 0.1
#: The default light's radiant exitance (W/m²) and the sky's strength.
DEFAULT_W_M2 = 40.0
DEFAULT_SKY = 0.05

LIGHT_TYPES = ("area", "point", "spot", "sun")


def _srgb_to_linear(c):
    c = np.asarray(c, float)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def emitter_geoms(m: mujoco.MjModel, emitters: list[dict]) -> tuple[dict[int, int], list[dict]]:
    """The geoms that are light sources: ``{geom id: index into sources}`` and the sources.

    A source is ``{"geom", "strength_w_m2", "colour", "display_rgba", "from"}``: ``geom`` its key (its
    name, else ``<body>#<index among its body's geoms>``, as :mod:`.export` names it), ``from``
    ``"fixture"`` (its material glows) or ``"emitters"`` (the world named it).
    """
    out: dict[int, int] = {}
    sources: list[dict] = []
    for g in range(m.ngeom):
        if int(m.geom_group[g]) >= 3 or not is_static(m, g):
            continue
        rgba = resolved_rgba(m, g)
        if rgba[3] <= 0:
            continue
        name = m.geom(g).name
        entry = next((e for e in emitters if fnmatch.fnmatch(name, e["match"])), None)
        mat = int(m.geom_matid[g])
        fixture = mat >= 0 and float(m.mat_emission[mat]) >= FIXTURE_EMISSION
        if entry is None and not fixture:
            continue
        entry = entry or {}
        colour = entry.get("colour") or list(_srgb_to_linear(rgba[:3]))
        out[g] = len(sources)
        sources.append(
            {
                "geom": geom_key(m, g),
                "strength_w_m2": float(entry.get("strength_w_m2", FIXTURE_W_M2)),
                "colour": [float(c) for c in colour],
                "display_rgba": entry.get("display_rgba"),
                "from": "emitters" if entry else "fixture",
            }
        )
    unmatched = [
        e["match"]
        for e in emitters
        if not any(fnmatch.fnmatch(s["geom"], e["match"]) for s in sources)
    ]
    if unmatched:
        raise ValueError(f"emitters that match no drawn static geom: {unmatched}")
    return out, sources


def model_lights(m: mujoco.MjModel, d: mujoco.MjData) -> list[dict]:
    """The MJCF ``<light>``\\ s of the world as bake lights (rule 2): those welded to the world. A
    light on what moves (a robot's headlamp) is lit live, as what carries it is."""
    out = []
    T = mujoco.mjtLightType
    for i in range(m.nlight):
        if not bool(m.light_active[i]):
            continue
        body = int(m.light_bodyid[i])
        if int(m.body_weldid[body]) != 0 or int(m.body_mocapid[body]) >= 0:
            continue
        diffuse = float(np.mean(m.light_diffuse[i]))
        if diffuse <= 0:
            continue
        colour = (np.asarray(m.light_diffuse[i], float) / max(diffuse, 1e-9)).clip(0, None)
        colour = (colour / max(colour.max(), 1e-9)).tolist()
        pos, direction = d.light_xpos[i].tolist(), d.light_xdir[i].tolist()
        name = m.light(i).name or f"light{i}"
        ltype = int(m.light_type[i])
        if ltype == int(T.mjLIGHT_DIRECTIONAL):
            out.append(
                {
                    "type": "sun",
                    "name": name,
                    "dir": direction,
                    "strength": SUN_PER_DIFFUSE * diffuse,
                    "colour": colour,
                }
            )
            continue
        cutoff = float(m.light_cutoff[i])
        light = {
            "name": name,
            "pos": pos,
            "power_w": POINT_W_PER_DIFFUSE * diffuse,
            "colour": colour,
        }
        if ltype == int(T.mjLIGHT_SPOT) and cutoff < 90.0:
            out.append({"type": "spot", "dir": direction, "spot_deg": 2.0 * cutoff, **light})
        else:
            out.append({"type": "point", **light})
    return out


def default_lights(bounds) -> tuple[list[dict], float]:
    """Rule 3: the area light over the static footprint, and the sky's strength."""
    lo, hi = np.asarray(bounds[0], float), np.asarray(bounds[1], float)
    size = np.maximum(hi[:2] - lo[:2], 0.5)
    centre = (lo[:2] + hi[:2]) / 2
    z = (
        float(hi[2] - DEFAULT_BELOW_TOP)
        if hi[2] - lo[2] > 2 * DEFAULT_BELOW_TOP
        else float(lo[2] + 3.0)
    )
    light = {
        "type": "area",
        "name": "default",
        "pos": [float(centre[0]), float(centre[1]), z],
        "size": [float(size[0]), float(size[1])],
        "yaw_deg": 0.0,
        "power_w": DEFAULT_W_M2 * float(size[0] * size[1]),
        "colour": [1.0, 1.0, 1.0],
    }
    return [light], DEFAULT_SKY


def check_lights(lights: list) -> list[str]:
    """Errors of a world's explicit ``lights`` list."""
    errors = []
    for i, light in enumerate(lights):
        if not isinstance(light, dict):
            errors.append(f"lights[{i}]: a mapping")
            continue
        t = light.get("type")
        if t not in LIGHT_TYPES:
            errors.append(f"lights[{i}].type: one of {LIGHT_TYPES}, not {t!r}")
            continue
        need = {
            "area": ("pos", "size", "power_w"),
            "point": ("pos", "power_w"),
            "spot": ("pos", "dir", "spot_deg", "power_w"),
            "sun": ("dir", "strength"),
        }[t]
        missing = [k for k in need if k not in light]
        if missing:
            errors.append(f"lights[{i}] ({t}): missing {missing}")
    return errors


def resolve(m: mujoco.MjModel, d: mujoco.MjData, lights, sources: list[dict], bounds) -> dict:
    """The bake's light: ``{"rule", "lights", "sky", "emitters", "stand_ins"}``."""
    fixtures = [s for s in sources if s["from"] == "fixture"]
    declared = model_lights(m, d)
    if isinstance(lights, list):
        return {
            "rule": "explicit",
            "lights": [dict(x) for x in lights],
            "sky": 0.0,
            "emitters": sources,
            "stand_ins": [x["name"] for x in declared],
        }
    if fixtures:
        return {
            "rule": "fixtures",
            "lights": [],
            "sky": 0.0,
            "emitters": sources,
            "stand_ins": [x["name"] for x in declared],
        }
    if declared:
        return {
            "rule": "mjcf_lights",
            "lights": declared,
            "sky": 0.0,
            "emitters": sources,
            "stand_ins": [],
        }
    default, sky = default_lights(bounds)
    return {"rule": "default", "lights": default, "sky": sky, "emitters": sources, "stand_ins": []}


def rounded(value, digits: int = 4):
    """*value* with every float rounded, so a fingerprint does not see float noise."""
    if isinstance(value, float):
        return round(value, digits) if math.isfinite(value) else value
    if isinstance(value, dict):
        return {k: rounded(v, digits) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [rounded(v, digits) for v in value]
    return value
