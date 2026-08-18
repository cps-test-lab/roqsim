"""The sensor catalog: which sensor types may be placed, their default FOV, cost, and mount limits.

A catalog entry pairs a sensor *type* (which selects the FOV adapter) with a default FOV config and a
:class:`MountConstraint` describing the realistic *constrained-mount* placement space -- where in the
room this sensor may go (wall/ceiling) and at what height. The placement search proposes poses inside
these constraints; the evaluator turns each into a :class:`~.adapters.PlacedSensor` and scores it.

Defaults come from the bundled sensor models so the FOV numbers are not re-invented: lidar/livox
templates are empty (the adapter reads the plugin defaults), and the camera templates carry only the
optics an unspawned placement needs (``fovy``/resolution) plus an explicit detection ``far``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .adapters import PlacedSensor


@dataclass
class MountConstraint:
    surfaces: tuple[str, ...]  # e.g. ("wall", "ceiling")
    z_range: tuple[float, float]  # allowed mount height [m]
    can_tilt: bool


@dataclass
class SensorSpec:
    type: str
    fov_template: dict
    cost: float
    mount: MountConstraint
    description: str = ""


#: Built-in catalog. ``far`` on cameras is an analysis detection range (not physics); tune per study.
CATALOG: dict[str, SensorSpec] = {
    "oakd_camera": SensorSpec(
        type="oakd_camera",
        fov_template={"fovy": 55.0, "width": 640, "height": 480, "near": 0.3, "far": 8.0},
        cost=1.0,
        mount=MountConstraint(surfaces=("wall", "ceiling"), z_range=(0.5, 3.5), can_tilt=True),
        description="OAK-D RGB-D camera (narrow-ish FOV, medium range).",
    ),
    "realsense_d435": SensorSpec(
        type="realsense_d435",
        fov_template={"fovy": 42.5, "width": 640, "height": 480, "near": 0.2, "far": 6.0},
        cost=0.8,
        mount=MountConstraint(surfaces=("wall", "ceiling"), z_range=(0.5, 3.5), can_tilt=True),
        description="RealSense D435 RGB camera.",
    ),
    "realsense_d455": SensorSpec(
        type="realsense_d455",
        fov_template={"fovy": 62.0, "width": 640, "height": 400, "near": 0.6, "far": 6.0},
        cost=1.0,
        mount=MountConstraint(surfaces=("wall", "ceiling"), z_range=(0.5, 3.5), can_tilt=True),
        description="RealSense D455 RGB camera (wide 87x62 deg FOV, 0.6-6 m range).",
    ),
    "zivid": SensorSpec(
        type="zivid",
        fov_template={"fovy": 39.0, "width": 704, "height": 704, "near": 1.3, "far": 5.0},
        cost=2.0,
        mount=MountConstraint(surfaces=("wall", "ceiling"), z_range=(1.3, 3.0), can_tilt=True),
        description="Zivid 2+ structured-light RGB-D (narrow FOV, 1.3-5 m working range).",
    ),
    "livox_mid360": SensorSpec(
        type="livox_mid360",
        fov_template={},  # adapter reads the Livox plugin defaults (360 x -7..52 deg, 40 m)
        cost=3.0,
        mount=MountConstraint(surfaces=("ceiling",), z_range=(1.0, 4.0), can_tilt=True),
        description="Livox Mid-360 3D lidar (dome FOV, long range). Invert (roll=pi) to face down; "
        "inverted it has a blind disk directly beneath it of radius ~= (mount_z - target_z)*tan(90-v_max) "
        "(~2.6 m on the floor from a 3.3 m ceiling), so mount near walls/corners, not room centres.",
    ),
    "lidar": SensorSpec(
        type="lidar",
        fov_template={},  # adapter reads the 2D lidar plugin defaults (planar fan)
        cost=0.5,
        mount=MountConstraint(surfaces=("wall",), z_range=(0.2, 2.0), can_tilt=False),
        description="2D lidar (planar fan). Coverage is a thin horizontal slice at the mount height.",
    ),
}


def catalog_as_dict() -> dict:
    """JSON-serialisable view of the catalog (for the CLI ``catalog`` command / LLM planner)."""
    out = {}
    for name, spec in CATALOG.items():
        out[name] = {
            "type": spec.type,
            "fov_template": spec.fov_template,
            "cost": spec.cost,
            "mount": {
                "surfaces": list(spec.mount.surfaces),
                "z_range": list(spec.mount.z_range),
                "can_tilt": spec.mount.can_tilt,
            },
            "description": spec.description,
        }
    return out


def placed_from_proposal(proposal: dict, *, index: int = 0) -> PlacedSensor:
    """Build a hypothetical :class:`PlacedSensor` from a placement proposal dict.

    ``proposal`` = ``{"type": str, "pos": [x,y,z], "rpy": [r,p,y], "config": {..}}``. The catalog's
    ``fov_template`` supplies defaults; ``config`` overrides them per placement.
    """
    stype = proposal["type"]
    if stype not in CATALOG:
        raise KeyError(f"unknown sensor type {stype!r}; catalog has {sorted(CATALOG)}")
    config = dict(CATALOG[stype].fov_template)
    config.update(proposal.get("config") or {})
    return PlacedSensor(
        sensor_type=stype,
        pos=np.asarray(proposal["pos"], dtype=np.float64),
        rpy=np.asarray(proposal.get("rpy", [0.0, 0.0, 0.0]), dtype=np.float64),
        config=config,
        label=proposal.get("label", f"{stype}_{index}"),
    )
