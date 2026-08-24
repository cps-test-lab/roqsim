"""Scene plugin: report the sensor coverage of a world, enabled/disabled from the world YAML.

The world-YAML front door to :mod:`roqsim_sensors.coverage`. List it in a world's ``plugins:`` block
and it computes coverage **once** (at ``configure``, after the model is compiled) and writes an
agent-digestible ``report.json`` plus a human render -- then does nothing per step. Omit it and there is
no coverage output. The optimization *search* over hypothetical mounts is the CLI's job
(``roqsim sensors coverage``); this plugin answers "what is the coverage of the sensors in this world?".

Config::

    sensor_coverage_probe:
      sensors: auto            # 'auto' = every MuJoCo camera in the world; or an explicit list of
                               #   {type, pos, rpy, config} placements (for lidars, or hypotheticals)
      camera_far: 10.0         # detection range assumed for 'auto' cameras (metres; not physics)
      target: {k: 1, frac: 0.95}
      sample:
        volume: true
        objects: true
        resolution: 0.25
        heights: [0.3, 1.0, 1.7]
        per_object: 64
      out: coverage            # output directory (report.json + render); relative to the CWD
      render: 3d               # 3d | 2d | both | none
      palette: coverage        # colour encoding: 'coverage' (red 0->green many) | 'density'
                               #   (light 0->dark many, so overlapping-sensor regions read darker)

Rendering needs a GL context, which ``import roqsim`` already selected for this machine (set
``MUJOCO_GL`` only to override it). ``sensors: auto`` discovers
cameras robustly (their FOV is read straight from the MJCF); include lidars/Livox via the explicit
list form or evaluate them with the CLI.
"""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np

from roqsim.context import SimContext
from roqsim.plugin import Plugin

from ..coverage import sampling
from ..coverage.adapters import PlacedSensor, build_fov
from ..coverage.catalog import placed_from_proposal
from ..coverage.engine import coverage
from ..coverage.report import build_report


class SensorCoverageProbePlugin(Plugin):
    parallel_safe = False  # runs a one-shot compute + optional GL render at configure time

    def __init__(self, config=None, *, name=None, entity=None, label=None):
        super().__init__(config, name=name, entity=entity, label=label)
        self.sensors = self.config.get("sensors", "auto")
        self.camera_far = float(self.config.get("camera_far", 10.0))
        self.target = self.config.get("target", {})
        sample = self.config.get("sample", {}) or {}
        self.sample_volume = bool(sample.get("volume", True))
        self.sample_objects = bool(sample.get("objects", True))
        self.resolution = float(sample.get("resolution", 0.25))
        self.heights = tuple(float(h) for h in sample.get("heights", [0.3, 1.0, 1.7]))
        self.per_object = int(sample.get("per_object", 64))
        self.out = Path(self.config.get("out", "coverage"))
        self.render = self.config.get("render", "3d")
        self.palette = self.config.get("palette", "coverage")

    def validate_config(self, config: dict) -> list[str]:
        errors = []
        if config.get("render", "3d") not in ("3d", "2d", "both", "none"):
            errors.append("'render' must be one of 3d|2d|both|none")
        if config.get("palette", "coverage") not in ("coverage", "density"):
            errors.append("'palette' must be one of coverage|density")
        sensors = config.get("sensors", "auto")
        if sensors != "auto" and not isinstance(sensors, list):
            errors.append("'sensors' must be 'auto' or a list of placements")
        if not (
            config.get("sample", {}).get("volume", True)
            or config.get("sample", {}).get("objects", True)
        ):
            errors.append("'sample' must enable at least one of volume/objects")
        return errors

    def _discover_fovs(self, model, data):
        if self.sensors == "auto":
            fovs = []
            for cam_id in range(model.ncam):
                name = (
                    mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_id)
                    or f"camera{cam_id}"
                )
                placed = PlacedSensor(
                    "camera", cam_id=cam_id, config={"far": self.camera_far}, label=name
                )
                fovs.append(build_fov(model, data, placed))
            if not fovs:
                raise RuntimeError(
                    "sensor_coverage_probe: sensors=auto found no cameras in the world. Add cameras, "
                    "or list sensors explicitly (e.g. a livox_mid360 placement)."
                )
            return fovs
        return [
            build_fov(model, data, placed_from_proposal(p, index=i))
            for i, p in enumerate(self.sensors)
        ]

    def _build_points(self, model, data):
        points_list, labels_list, names = [], [], []
        if self.sample_volume:
            vol = sampling.room_volume_points(
                model, data, resolution=self.resolution, heights=self.heights
            )
            if len(vol):
                points_list.append(vol)
                labels_list.append(np.full(len(vol), -1))
        if self.sample_objects:
            surf, lab, names = sampling.object_surface_points(
                model, data, per_object=self.per_object
            )
            if len(surf):
                points_list.append(surf)
                labels_list.append(lab)
        if not points_list:
            raise RuntimeError("sensor_coverage_probe: no sample points produced")
        return np.vstack(points_list), np.concatenate(labels_list), names

    def configure(self, ctx: SimContext) -> None:
        model, data = ctx.model, ctx.data
        mujoco.mj_forward(model, data)  # ensure cam/site/geom world poses are populated
        fovs = self._discover_fovs(model, data)
        points, labels, names = self._build_points(model, data)
        result = coverage(model, data, fovs, points, labels=labels, label_names=names)

        placements = self.sensors if isinstance(self.sensors, list) else "auto"
        report = build_report(
            result,
            world=str(ctx.config.get("sim", {}).get("world", "")),
            target=self.target,
            placements=placements if isinstance(placements, list) else [],
            gap_resolution=self.resolution,
        )
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out / "report.json").write_text(json.dumps(report, indent=2))

        if self.render != "none":
            from ..coverage import viz

            if self.render in ("3d", "both"):
                viz.render_coverage_3d(
                    model, data, result, str(self.out / "coverage_3d.png"), palette=self.palette
                )
            if self.render in ("2d", "both"):
                viz.render_heatmap_2d(
                    result, str(self.out / "heatmap_2d.png"), palette=self.palette
                )

        a = report["achieved"]
        ctx.logger.info(
            "sensor_coverage_probe: %d sensors, %d points, coverage k1=%.3f k2=%.3f -> %s",
            len(fovs),
            a["n_points"],
            a["fraction_covered_k1"],
            a["fraction_covered_k2"],
            self.out / "report.json",
        )
