# Copyright (C) 2026 Frederik Pasch
# SPDX-License-Identifier: Apache-2.0
"""The worlds the parity, determinism and drive tests cast in.

Three lidars spawned on their own -- a 2D ``sick_tim571`` (811 rays), a ``velodyne_vlp16`` (898) and a
``mid360`` (20,160) -- and an obstacle box between them and the room's far side, in the default room
or in ``roqsim_scenes``' depot. Each lidar names its backend (``raycaster``), so one world built with
``mjwarp`` holds both backends' answers for the same state: the lidar's own, and the CPU cast of the
same rays.
"""

from __future__ import annotations

from pathlib import Path

HERE = Path(__file__).resolve().parent

LIDARS = (
    ("lidar2d", "sick_tim571", (-1.6, -0.2, 0.45)),
    ("vlp16", "velodyne_vlp16", (-1.6, 0.4, 0.7)),
    ("mid360", "mid360", (-1.6, -0.6, 0.6)),
)


def depot_world() -> str:
    import roqsim_scenes  # noqa: PLC0415

    return str(Path(roqsim_scenes.__file__).parent / "worlds" / "depot" / "depot.xml")


def lidar_world(world: str | None, raycaster: str | None, lidars=LIDARS) -> dict:
    """A world document: the lidars, each casting through *raycaster*, and the obstacle."""
    sim = {"timestep": 0.002, "pacing": "asap"}
    if world:
        sim["world"] = world
    components = []
    for name, model, (x, y, z) in lidars:
        config = {} if raycaster is None else {"raycaster": raycaster}
        components.append(
            {
                "spawn_sensor": {
                    "model": model,
                    "prefix": f"{name}_",
                    "namespace": name,
                    "frame_id": f"{name}_link",
                    "pose": {"position": {"x": x, "y": y, "z": z}},
                },
                "name": name,
                "components": [{_plugin_of(model): config}],
            }
        )
    components.append(
        {
            "spawn_model": {
                "model": str(HERE / "obstacle.xml"),
                "prefix": "obs_",
                "motion": "static",
                "pose": {"position": {"x": -0.7, "y": -0.1, "z": 0.0}},
            },
            "name": "obstacle",
        }
    )
    return {"sim": sim, "components": components}


def _plugin_of(model: str) -> str:
    """The plugin each device model's manifest names for its ray cast."""
    return {"sick_tim571": "lidar", "velodyne_vlp16": "lidar", "mid360": "livox_mid360"}[model]


def build(doc: dict, seed: int = 1):
    from roqsim.config import load_config_from_dict  # noqa: PLC0415
    from roqsim.engine import Engine  # noqa: PLC0415

    engine = Engine(load_config_from_dict(doc))
    engine.ctx.seed = seed
    engine.setup()
    engine.reset()
    return engine


def lidar_plugins(engine) -> list:
    from roqsim_sensors.plugins.lidar_common import RayCastSensorPlugin  # noqa: PLC0415

    return [p for p in engine.plugins if isinstance(p, RayCastSensorPlugin)]


def drive_world(world: str | None, raycaster: str | None, *, noise: bool = True) -> dict:
    """A Jackal driving a circle, its own VLP-16 scan and a Mid-360 on its mid mount.

    Both lidars cast through *raycaster* (``sim.raycaster``); ``None`` leaves it to the default,
    ``auto``, which chooses per lidar. With ``noise=False`` the Mid-360's range noise is off, so its
    cloud is the cast's geometry alone.
    """
    sim = {"timestep": 0.002, "pacing": "asap"}
    if world:
        sim["world"] = world
    if raycaster:
        sim["raycaster"] = raycaster
    mid360 = {} if noise else {"range_stddev": 0.0}
    return {
        "sim": sim,
        "components": [
            {
                "spawn_robot": {
                    "model": "clearpath_jackal",
                    "pose": {"position": {"x": -6.0, "y": 0.0}, "orientation": {"yaw": 0.0}},
                },
                "name": "robot",
                "components": [
                    {"diff_drive": {"test_cmd": [0.6, 0.3]}},
                    {
                        "spawn_sensor": {
                            "model": "mid360",
                            "frame_id": "livox_frame",
                            "parent_frame": "mid_mount",
                            "pose": {"position": {"z": 0.25}},
                        },
                        "name": "mid360",
                        "components": [{"livox_mid360": mid360}],
                    },
                ],
            }
        ],
    }
