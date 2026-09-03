"""Sensor plugin: named objects detected relative to a robot, from ground truth plus an error model.

**A stand-in for a perception pipeline, and shaped so a real one replaces it.** It reports where
objects are *relative to the robot*, which is what an object pose estimator produces and what a
manipulator needs -- with the pose read from MuJoCo instead of from a camera. Everything downstream
consumes the standard message, so swapping in a real detector is a change of publisher and nothing
else.

**Why robot-relative, and why not TF.** The pose is computed as ``inv(T_world_base) @ T_world_object``
straight from ``data.xpos``/``data.xquat``. It deliberately does NOT walk the TF tree, because on a
localized robot the route from the robot's own frames to a world-anchored prop runs through AMCL's
``map -> odom`` -- so a "ground truth" object pose fetched over TF arrives with the localization error
added to it. Measured on the TIAGo Pro pick trial: an exact parcel pose reached the arm 43 mm out in x
and 73 mm in y, the gripper closed on air, and the stack reported success while the parcel had never
moved. A grasp with 18 mm of pad clearance cannot absorb that. Reading ground truth in the robot's
frame is also what a real camera does -- it measures relative to itself -- so this is the faithful
shape, not a shortcut around one.

**It is a sensor like the others here**, and it is configured like one: ground truth plus a
configurable error model, with noise drawn from ``ctx.rng_for`` so it is reproducible from a recording
(see ``lidar``'s ``range_stddev``). Everything defaults to zero, i.e. a perfect detector; turning
``position_stddev`` into a campaign factor is how you measure what pose accuracy a given grasp needs.

**ROS-free**, like every plugin: it emits a neutral payload on an ``Endpoint`` and only the bridge
knows the message type.

Config -- a component of the entry that spawns the robot the detections are reported
from, since ownership is where the entry sits rather than a config key::

    object_detector:
      frame: base_footprint           # body whose frame the poses are expressed in
      rate_hz: 10.0
      objects:                        # body name -> what a detector would call it
        - {body: graspable_carton, class_id: parcel, size: [0.040, 0.024, 0.090]}
      position_stddev: 0.0            # m, Gaussian, per axis
      orientation_stddev: 0.0         # rad, small-angle, applied about each axis
      position_bias: [0.0, 0.0, 0.0]  # m, systematic -- calibration error is not zero-mean
      dropout_percent: 0.0            # this percentage of detections go missing
      max_range: 0.0                  # 0 = unlimited; else report only within this range
      confidence: 1.0                 # what the hypothesis score reports

The consumer contract, which a real detector must satisfy to drop in:

* topic ``detections``, type ``vision_msgs/msg/Detection3DArray``
* ``header.frame_id`` is a frame reachable through the robot's own kinematics -- never ``map``
* objects identified by ``results[0].hypothesis.class_id``
* latest-wins; a detection missing from a message means "not detected this cycle"
"""

from __future__ import annotations

import mujoco
import numpy as np

from roqsim.context import Endpoint, SimContext
from roqsim.plugin import Plugin


class ObjectDetectorPlugin(Plugin):
    parallel_safe = True  # post-compile read-only: reads data.xpos/xquat, publishes via endpoint

    def __init__(self, config=None, *, name=None, entity=None, label=None):
        super().__init__(config, name=name, entity=entity, label=label)
        self.robot = self.entity
        self.frame = self.config.get("frame", "base_footprint")
        self.rate_hz = float(self.config.get("rate_hz", 10.0))
        self.objects = list(self.config.get("objects", []))
        self.position_stddev = float(self.config.get("position_stddev", 0.0))
        self.orientation_stddev = float(self.config.get("orientation_stddev", 0.0))
        self.position_bias = [float(v) for v in self.config.get("position_bias", (0.0, 0.0, 0.0))]
        self.dropout_percent = float(self.config.get("dropout_percent", 0.0))
        self.max_range = float(self.config.get("max_range", 0.0))
        self.confidence = float(self.config.get("confidence", 1.0))
        self._frame_bid = -1
        self._objects: list[tuple[str, int, tuple[float, float, float]]] = []
        self._ctx: SimContext | None = None

    def validate_config(self, config: dict) -> list[str]:
        errors = self.validate_topics(config)
        if float(config.get("rate_hz", 10.0)) <= 0:
            errors.append("'rate_hz' must be > 0")
        for key in ("position_stddev", "orientation_stddev", "max_range"):
            if float(config.get(key, 0.0)) < 0:
                errors.append(f"'{key}' must be >= 0")
        if not 0.0 <= float(config.get("dropout_percent", 0.0)) <= 100.0:
            errors.append("'dropout_percent' must be in [0, 100]")
        objects = config.get("objects")
        if not objects:
            errors.append("'objects' must name at least one body to detect")
        else:
            for i, entry in enumerate(objects):
                if not isinstance(entry, dict) or not entry.get("body"):
                    errors.append(f"objects[{i}] needs a 'body'")
                elif not entry.get("class_id"):
                    errors.append(f"objects[{i}] ({entry['body']}) needs a 'class_id'")
        bias = config.get("position_bias", (0.0, 0.0, 0.0))
        if len(list(bias)) != 3:
            errors.append("'position_bias' must be three numbers")
        return errors

    def configure(self, ctx: SimContext) -> None:
        self._ctx = ctx
        entity = ctx.entities.get(self.robot)
        prefix = entity.meta.get("prefix", "") if entity else ""
        ns = self.config.get("namespace") or (entity.meta.get("namespace", "") if entity else "")

        frame_name = prefix + self.frame
        self._frame_bid = mujoco.mj_name2id(ctx.model, mujoco.mjtObj.mjOBJ_BODY, frame_name)
        if self._frame_bid < 0:
            raise RuntimeError(f"object_detector: reporting frame body {frame_name!r} not found")

        # Resolve every named body up front. A typo here would otherwise surface as an object that is
        # simply never detected, which reads like an occlusion rather than a configuration error.
        for entry in self.objects:
            body = str(entry["body"])
            bid = mujoco.mj_name2id(ctx.model, mujoco.mjtObj.mjOBJ_BODY, body)
            if bid < 0:
                raise RuntimeError(f"object_detector: body {body!r} not found in the model")
            size = tuple(float(v) for v in entry.get("size", (0.0, 0.0, 0.0)))
            self._objects.append((str(entry["class_id"]), bid, size))

        ctx.interface.add(
            Endpoint(
                name="detections",
                direction="out",
                owner=self.robot,
                namespace=ns,
                read=self._read,
                rate_hz=self.rate_hz,
                backend={
                    "ros2": {
                        "type": "vision_msgs.msg.Detection3DArray",
                        "topic": self.topic_override("detections") or "detections",
                        "frame_id": self.frame,
                    }
                },
            )
        )

    def _read(self):
        """Endpoint ``read`` (physics thread) -> ``[(class_id, pose7, size3, score), ...]``.

        ``pose7`` is ``(x, y, z, qw, qx, qy, qz)`` in the reporting frame. An empty list is a valid
        reading: it means nothing was detected this cycle.
        """
        ctx = self._ctx
        d = ctx.data
        # The reporting frame's world pose, inverted once for all objects.
        base_pos = np.asarray(d.xpos[self._frame_bid], dtype=float)
        base_mat = np.asarray(d.xmat[self._frame_bid], dtype=float).reshape(3, 3)

        noisy = (
            self.position_stddev > 0.0
            or self.orientation_stddev > 0.0
            or self.dropout_percent > 0.0
            or any(self.position_bias)
        )
        rng = ctx.rng_for(self.name or "object_detector") if noisy else None

        out = []
        for class_id, bid, size in self._objects:
            rel_pos = base_mat.T @ (np.asarray(d.xpos[bid], dtype=float) - base_pos)
            rel_mat = base_mat.T @ np.asarray(d.xmat[bid], dtype=float).reshape(3, 3)

            if self.max_range > 0.0 and float(np.linalg.norm(rel_pos)) > self.max_range:
                continue
            if rng is not None:
                if self.dropout_percent > 0.0 and rng.uniform(0.0, 100.0) < self.dropout_percent:
                    continue
                if self.position_stddev > 0.0:
                    rel_pos = rel_pos + rng.normal(0.0, self.position_stddev, size=3)
                if any(self.position_bias):
                    rel_pos = rel_pos + np.asarray(self.position_bias, dtype=float)
                if self.orientation_stddev > 0.0:
                    rel_mat = _perturb(rel_mat, rng.normal(0.0, self.orientation_stddev, size=3))

            quat = np.empty(4, dtype=float)
            mujoco.mju_mat2Quat(quat, rel_mat.reshape(9))
            out.append(
                (
                    class_id,
                    (*(float(v) for v in rel_pos), *(float(v) for v in quat)),
                    size,
                    self.confidence,
                )
            )
        return out


def _perturb(mat: np.ndarray, rot: np.ndarray) -> np.ndarray:
    """Rotate *mat* by a small (rx, ry, rz). Small-angle: the exact axis order is below the noise."""
    cx, cy, cz = np.cos(rot)
    sx, sy, sz = np.sin(rot)
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return (rz @ ry @ rx) @ mat
