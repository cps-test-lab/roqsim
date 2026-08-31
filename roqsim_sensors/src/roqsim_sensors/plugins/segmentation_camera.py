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

"""Sensor plugin: per-pixel class and instance labels, and the 2D boxes that fall out of them.

The perception ground truth the substrate was missing. ``object_detector`` answers *where is the
parcel, in the robot's frame* -- the input a manipulation stack wants. It cannot answer *which pixels
are the parcel*, and that is what a segmentation or detection experiment is measured against: a
labelled frame is what an IoU, a mask AP or a training set is computed from, and it is the one
observable a camera-based perception paper cannot be reconstructed without.

MuJoCo renders it already. With ``mjRND_SEGMENT`` + ``mjRND_IDCOLOR`` a pass returns, per pixel, the
*geom* it hit -- exact, and free of the depth-thresholding a mask reconstructed from a depth image
would need. This plugin is the mapping from that to an experiment's own vocabulary: a world says
which bodies are a ``parcel``, and gets a class image, an instance image, and tight 2D boxes measured
**from the mask** rather than projected from a bounding volume.

**Visible extent, not projected extent, and the difference is the measurement.** Gazebo's
bounding-box camera offers a *full* box (the whole object, including the part behind a wall) beside a
*visible* one. Only the visible box is derivable from a mask, and it is the one a detector could ever
have produced -- so it is the only one here, and ``min_pixels`` drops an instance that is barely in
frame instead of reporting a one-pixel detection no real detector would emit. An occluded object
therefore shrinks and eventually disappears, which is the honest behaviour: a trial whose metric is
"was the pedestrian detected" has to see the occlusion.

Config (in addition to ``camera_common.CameraPlugin``'s)::

    segmentation_camera:
      camera: camera            # the MJCF <camera> to render through
      classes:                  # REQUIRED: the experiment's vocabulary, in priority order
        - {class_id: 1, name: parcel, bodies: ["graspable_*"]}
        - {class_id: 2, name: person, entities: [walker_1]}   # whole kinematic subtree
      instances: false          # also publish the instance-id image (16UC1)
      detections: true          # publish 2D boxes derived from the mask
      min_pixels: 16            # an instance with fewer visible pixels is not reported
      color: false              # the colour stream, off by default here (see below)
      rate_hz: 10.0
      frame_id: camera_optical_frame
      topics: {}                # hardwire absolute topics, e.g. {labels: /seg/class_image}

**Endpoints.** ``labels`` (``sensor_msgs/Image``, ``mono8``) is the class image: each pixel is the
``class_id`` of the class owning the geom that pixel hit, ``0`` where nothing labelled was hit.
``instances`` (``16UC1``, off by default) is the instance-id image. ``detections``
(``vision_msgs/Detection2DArray``) carries one entry per visible instance, with the class in
``results[0].hypothesis.class_id`` and the instance in ``Detection2D.id``. ``camera_info`` comes from
the base and describes all of them, since they are one render through one camera.

**Instance ids are body ids**, so they are stable across frames, across a reset, and across two runs
of the same world -- a tracker's association is then a fact about the world rather than about the
order this plugin happened to see things in. They are consequently *not* contiguous, which is why the
detections name them rather than leaving a consumer to infer them from the image.

**Class ids are the world's, and 0 is reserved** for "nothing labelled here": a background class with
an id of its own would be indistinguishable from an unlabelled geom, and a mask metric computed over
it would score the wall. Classes are matched in declaration order and the first match wins, so a
world can put a specific body ahead of the glob that would otherwise swallow it.

**The colour stream is off by default here** (``PUBLISHES_COLOR = False``). A label camera normally
shares its MJCF ``<camera>`` with the RGB sensor that already publishes those pixels, and a second
colour topic off the same optics is a duplicate that costs a serialisation. ``color: true`` turns it
on for the case that wants it: a pixel-aligned image/label pair off one camera, which is what a
training set is.

**An absent entity is absent here too**, and the pass says so rather than inheriting it.
``DeleteEntity`` hides geoms by moving them to :data:`roqsim.presence.ABSENT_GEOM_GROUP` and zeroing
their alpha (:mod:`roqsim.presence`). Measured, MuJoCo's *default* view options already exclude that
group, so a deleted obstacle contributes no label pixels either way -- but "a deleted obstacle is not
in the ground truth" is a contract this sensor owes an experiment, not something to leave resting on
a default in another project that no test of ours would notice changing. So the pass is rendered with
the group explicitly masked, and ``test_segmentation_camera.py`` pins the outcome. Alpha is the other
half of hiding and is no help here: it is a colour, and an id pass need not respect it.
"""

from __future__ import annotations

from fnmatch import fnmatchcase

import mujoco
import numpy as np

from roqsim.context import Endpoint, SimContext
from roqsim.presence import ABSENT_GEOM_GROUP

from .camera_common import CameraPlugin, join_topic

#: Reserved: "no labelled geom here". See the module docstring.
BACKGROUND = 0

#: mono8 carries the class image, so this is the largest class id it can express.
MAX_CLASS_ID = 255

#: 16UC1 carries the instance image, and instance ids are body ids -- a model with more bodies than
#: this cannot be labelled per instance (the class image is unaffected).
MAX_INSTANCE_ID = 65535


class SegmentationCameraPlugin(CameraPlugin):
    """See the module docstring."""

    PUBLISHES_COLOR = False
    DEFAULT_RATE_HZ = 10.0
    DEFAULT_TOPIC_PREFIX = "segmentation"

    def __init__(self, config=None, *, name=None, entity=None, label=None):
        super().__init__(config, name=name, entity=entity, label=label)
        self.classes = list(self.config.get("classes") or [])
        self.instances = bool(self.config.get("instances", False))
        self.detections = bool(self.config.get("detections", True))
        self.min_pixels = int(self.config.get("min_pixels", 16))
        self._labels: np.ndarray | None = None
        self._instance_img: np.ndarray | None = None
        self._boxes: list | None = None
        #: geom id -> class id / instance id, built once at configure. Indexed by the segmentation
        #: pass's own object id, so a frame costs one gather rather than a lookup per pixel.
        self._class_of_geom: np.ndarray | None = None
        self._instance_of_geom: np.ndarray | None = None
        #: class id -> declared name, for the hypothesis a detection carries.
        self._class_names: dict[int, str] = {}
        #: The scene options the segmentation pass renders with (absent geoms masked out).
        self._seg_opt: mujoco.MjvOption | None = None

    # -- validation ---------------------------------------------------------------------------

    def validate_config(self, config: dict) -> list[str]:
        errors = super().validate_config(config)
        classes = config.get("classes")
        if not classes:
            # A label camera with no vocabulary renders an all-background image, which looks like a
            # working sensor pointed at nothing -- the one failure this must not have.
            errors.append(
                "'classes' is required: a label image is meaningless without the vocabulary it "
                "labels in, e.g. classes: [{class_id: 1, name: parcel, bodies: ['graspable_*']}]"
            )
            return errors
        if not isinstance(classes, list):
            return errors + [
                "'classes' must be a list of {class_id, name, bodies|entities} entries"
            ]
        seen: set[int] = set()
        for i, entry in enumerate(classes):
            where = f"classes[{i}]"
            if not isinstance(entry, dict):
                errors.append(f"{where} must be a mapping with class_id and bodies or entities")
                continue
            if "class_id" not in entry:
                errors.append(f"{where} needs a 'class_id' (1..{MAX_CLASS_ID}; 0 is background)")
            else:
                class_id = int(entry["class_id"])
                if not 1 <= class_id <= MAX_CLASS_ID:
                    errors.append(
                        f"{where}: 'class_id' must be 1..{MAX_CLASS_ID} -- 0 is reserved for "
                        "'nothing labelled here', and the class image is mono8"
                    )
                elif class_id in seen:
                    # Two entries of one id is almost always a copy-paste, and the damage is silent:
                    # the second's bodies join the first's class and its name is never seen again.
                    errors.append(f"{where}: 'class_id' {class_id} is already declared above")
                seen.add(class_id)
            if not entry.get("bodies") and not entry.get("entities"):
                errors.append(f"{where} must name 'bodies' (names or globs) or 'entities'")
            for key in ("bodies", "entities"):
                if entry.get(key) is not None and not isinstance(entry[key], list):
                    errors.append(f"{where}: '{key}' must be a list")
        if int(config.get("min_pixels", 16)) < 1:
            errors.append("'min_pixels' must be >= 1 (an instance of no pixels is not visible)")
        return errors

    # -- lifecycle ----------------------------------------------------------------------------

    def _configure_extra(self, ctx: SimContext, prefix: str, ns: str) -> None:
        self._build_lookups(ctx, prefix)

        self._seg_opt = mujoco.MjvOption()
        # The one line that makes absence mean absence in an id pass; see the module docstring.
        self._seg_opt.geomgroup[ABSENT_GEOM_GROUP] = 0

        labels_ep = Endpoint(
            name="labels",
            direction="out",
            owner=self.robot,
            namespace=ns,
            read=lambda: self._labels,
            rate_hz=self.rate_hz,
            lazy=True,  # a full frame on the wire, like the colour image
            backend={
                "ros2": {
                    "type": "sensor_msgs.msg.Image",
                    "topic": self.topic_override("labels")
                    or join_topic(self.DEFAULT_TOPIC_PREFIX, "class_image"),
                    "frame_id": self.frame_id,
                    "encoding": "mono8",
                }
            },
        )
        ctx.interface.add(labels_ep)
        self._extra_outputs.append(labels_ep)

        if self.instances:
            instances_ep = Endpoint(
                name="instances",
                direction="out",
                owner=self.robot,
                namespace=ns,
                read=lambda: self._instance_img,
                rate_hz=self.rate_hz,
                lazy=True,
                backend={
                    "ros2": {
                        "type": "sensor_msgs.msg.Image",
                        "topic": self.topic_override("instances")
                        or join_topic(self.DEFAULT_TOPIC_PREFIX, "instance_image"),
                        "frame_id": self.frame_id,
                        "encoding": "16UC1",
                    }
                },
            )
            ctx.interface.add(instances_ep)
            self._extra_outputs.append(instances_ep)

        if self.detections:
            detections_ep = Endpoint(
                name="detections",
                direction="out",
                owner=self.robot,
                namespace=ns,
                read=lambda: self._boxes,
                rate_hz=self.rate_hz,
                backend={
                    "ros2": {
                        "type": "vision_msgs.msg.Detection2DArray",
                        "topic": self.topic_override("detections")
                        or join_topic(self.DEFAULT_TOPIC_PREFIX, "detections"),
                        "frame_id": self.frame_id,
                    }
                },
            )
            ctx.interface.add(detections_ep)
            # Gated too: the boxes come off the same render, so a consumer that wants only boxes
            # must still get frames -- the lesson camera_common's _gate_endpoints records for depth.
            self._extra_outputs.append(detections_ep)

    def _build_lookups(self, ctx: SimContext, prefix: str) -> None:
        """geom -> (class, instance), resolved once against the compiled model.

        Bodies are matched by name or glob, and an entity contributes its whole kinematic subtree --
        a robot means its chassis and its wheels, exactly as ``contact_monitor`` reads a robot. The
        instance is the *matched* body, not the geom's own: a robot is one object, so its wheel
        pixels carry the base's id rather than making every link an object of its own.
        """
        m = ctx.model
        self._class_of_geom = np.zeros(m.ngeom, dtype=np.uint8)
        self._instance_of_geom = np.zeros(m.ngeom, dtype=np.uint16)
        body_names = [
            mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or "" for b in range(m.nbody)
        ]

        for entry in self.classes:
            class_id = int(entry["class_id"])
            self._class_names[class_id] = str(entry.get("name") or class_id)
            matched: list[int] = []
            for pattern in entry.get("bodies") or []:
                wanted = f"{prefix}{pattern}"
                hits = [
                    b
                    for b, name in enumerate(body_names)
                    if name and (name == wanted or fnmatchcase(name, wanted))
                ]
                if not hits:
                    # Loud: a pattern matching nothing produces an all-background class, and a
                    # metric computed over it reads as "the detector never saw it".
                    raise RuntimeError(
                        f"segmentation_camera[{self.label}]: class {class_id} pattern {wanted!r} "
                        f"matches no body in this world"
                    )
                matched.extend(hits)
            for name in entry.get("entities") or []:
                matched.extend(self._entity_subtree(ctx, m, name, class_id))

            for body in matched:
                instance = self._instance_root(m, body, matched)
                if instance > MAX_INSTANCE_ID:
                    raise RuntimeError(
                        f"segmentation_camera[{self.label}]: body id {instance} exceeds the "
                        f"{MAX_INSTANCE_ID} an instance image (16UC1) can carry"
                    )
                start = int(m.body_geomadr[body])
                for geom in range(start, start + int(m.body_geomnum[body])):
                    # First match wins, so a specific body declared above a glob keeps its class.
                    if self._class_of_geom[geom] == BACKGROUND:
                        self._class_of_geom[geom] = class_id
                        self._instance_of_geom[geom] = instance

    def _entity_subtree(self, ctx: SimContext, m, name: str, class_id: int) -> list[int]:
        entity = ctx.entities.get(name)
        if entity is None or not entity.body:
            raise RuntimeError(
                f"segmentation_camera[{self.label}]: class {class_id} names entity {name!r}, which "
                f"is not registered (list this camera after the entry that spawns it) or which "
                f"registered no base body"
            )
        root = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, entity.body)
        if root < 0:
            raise RuntimeError(
                f"segmentation_camera[{self.label}]: entity {name!r} base body "
                f"{entity.body!r} not found"
            )
        return _subtree(m, root)

    @staticmethod
    def _instance_root(m, body: int, matched: list[int]) -> int:
        """The highest ancestor of *body* this class also matched -- the object it is part of.

        A subtree match brings in every link, and a link is not an object: walking up to the topmost
        matched ancestor makes a robot one instance whether the class named the entity or globbed
        its links.
        """
        root = body
        parent = int(m.body_parentid[body])
        while parent != 0 and parent != root and parent in matched:
            root = parent
            parent = int(m.body_parentid[root])
        return root

    # -- capture ------------------------------------------------------------------------------

    def _capture_extra(self, ctx: SimContext, renderer) -> None:
        renderer.enable_segmentation_rendering()
        # The scene options carry the absent-geom mask, so this pass passes them explicitly --
        # `update_scene` otherwise uses MuJoCo's defaults, which include every geom group.
        renderer.update_scene(ctx.data, camera=self._cam_id, scene_option=self._seg_opt)
        seg = renderer.render()
        renderer.disable_segmentation_rendering()

        # (H, W, 2) int32: [..., 0] is the object id and [..., 1] its mjtObj type; background is
        # (-1, -1), and a non-geom object (a decor marker) is not something a world can label.
        objid, objtype = seg[..., 0], seg[..., 1]
        is_geom = objtype == mujoco.mjtObj.mjOBJ_GEOM
        # Clipped before the gather so a background pixel's -1 cannot index from the end of the
        # table; `is_geom` is what actually decides, and it is False for exactly those pixels.
        gathered = np.clip(objid, 0, self._class_of_geom.size - 1)
        self._labels = np.where(is_geom, self._class_of_geom[gathered], BACKGROUND).astype(np.uint8)
        instance_img = np.where(is_geom, self._instance_of_geom[gathered], BACKGROUND).astype(
            np.uint16
        )
        self._instance_img = instance_img if self.instances else None
        self._boxes = self._boxes_from(instance_img) if self.detections else None

    def _boxes_from(self, instance_img: np.ndarray) -> list:
        """Tight 2D boxes per visible instance: ``[(class_id, name, instance, cx, cy, w, h), ...]``.

        Measured from the mask, so a partly occluded object reports the extent that is actually
        visible -- the only extent a detector could have produced. Sizes are in pixels, inclusive of
        both edge rows, so a single-pixel instance is 1x1 rather than 0x0.
        """
        boxes = []
        present, counts = np.unique(instance_img, return_counts=True)
        for instance, count in zip(present.tolist(), counts.tolist(), strict=True):
            if instance == BACKGROUND or count < self.min_pixels:
                continue
            mask = instance_img == instance
            rows = np.flatnonzero(mask.any(axis=1))
            cols = np.flatnonzero(mask.any(axis=0))
            y0, y1 = int(rows[0]), int(rows[-1])
            x0, x1 = int(cols[0]), int(cols[-1])
            # An instance's class is a property of its geoms, so read it off the mask rather than
            # keeping a second table that could disagree with the image being published.
            class_id = int(self._labels[mask][0])
            boxes.append(
                (
                    class_id,
                    self._class_names.get(class_id, str(class_id)),
                    int(instance),
                    (x0 + x1) / 2.0,
                    (y0 + y1) / 2.0,
                    float(x1 - x0 + 1),
                    float(y1 - y0 + 1),
                )
            )
        return boxes


def _subtree(m, root: int) -> list[int]:
    """``root`` and every body descended from it."""
    bodies = [root]
    for body in range(root + 1, m.nbody):
        # MuJoCo numbers a body after its parent, so one forward pass resolves descent.
        if int(m.body_parentid[body]) in bodies:
            bodies.append(body)
    return bodies
