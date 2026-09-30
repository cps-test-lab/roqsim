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

"""Observation plugin: true poses of named frames, on one topic, as a ROS stack expects them.

Gazebo's ``PosePublisher`` for roqsim. A stack that reads simulator ground truth off a topic -- a
vendor simulator's adapter that turns true poses into an optical-flow sensor or a dock's infrared
field -- gets it in the shape it reads, from any entity.

Config::

    pose_publisher:
      poses:
        - {frame: robot}                                  # an entity's root, in the world
        - {frame: robot/mouse, relative_to: robot/base_link}
        - {frame: robot/oakd/oakd_link, relative_to: robot, child: camera}
      rate_hz: 30.0         # publish rate of every pose
      lazy: false           # skip the reads while nobody subscribes
      world_frame: map      # the parent frame of a pose in the world
      topics: {poses: ground_truth}   # the one topic every pose goes out on (default "poses")

Each pose is a frame path (:mod:`roqsim.paths`): an entity (its root), or a body, site, declared
frame or device frame of an entity or a component nested in one, named as TF shows it.
``relative_to`` is another frame path, or ``world`` (the default). Nested under an entity, the
paths are relative to it: ``.`` is the entity itself, ``mouse`` its ``mouse``, and a leading ``/``
starts at the top of the world; at the top of a world they start with an entity's name, and the
entry is declared after the entries that spawn what it names.

One endpoint per pose, ``poses/<child>``: a :class:`~roqsim.types.Transform` from the
``relative_to`` frame's TF name (``world_frame`` for the world) to ``child`` (default the frame's
own name: its body's, site's or frame's, or the entity's name for a root). ROS carries each as a
one-transform ``tf2_msgs/TFMessage``, as ``PosePublisher`` publishes one pose per message. The
poses are read from the core pose data (:func:`roqsim.frames.frame_pose`); a frame of an entity
that is absent publishes nothing.
"""

from __future__ import annotations

from .. import endpoint
from ..context import SimContext
from ..document import refuse_unknown_keys
from ..frames import Frame, frame_pose, resolve_frame
from ..paths import PathError
from ..plugin import Plugin, PluginError
from ..types import Transform

_POSE_KEYS = ("frame", "relative_to", "child")
_WORLD = "world"


class PosePublisherPlugin(Plugin):
    parallel_safe = True  # reads the pose data, writes nothing

    def __init__(self, config=None, *, name=None, entity=None, label=None):
        super().__init__(config, name=name, entity=entity, label=label)
        self.rate_hz = float(self.config.get("rate_hz", 30.0))
        self.lazy = bool(self.config.get("lazy", False))
        self.world_frame = str(self.config.get("world_frame", "map"))
        self.topic = self.topic_override("poses") or "poses"
        #: The ``child`` of each pose, in the order configured; set in ``configure``.
        self.children: list[str] = []
        self._poses: dict[str, tuple[Frame, Frame | None, str]] = {}

    def validate_config(self, config: dict) -> list[str]:
        errors = []
        poses = config.get("poses")
        if not isinstance(poses, list) or not poses:
            return ["'poses' must be a non-empty list of {frame, relative_to, child} entries"]
        for i, entry in enumerate(poses):
            where = f"poses[{i}]"
            if not isinstance(entry, dict):
                errors.append(f"{where}: must be a mapping of frame, relative_to, child")
                continue
            try:
                refuse_unknown_keys(entry, _POSE_KEYS, where)
            except ValueError as exc:
                errors.append(str(exc))
            for key in _POSE_KEYS:
                if key in entry and (not isinstance(entry[key], str) or not entry[key]):
                    errors.append(f"{where}: '{key}' must be a non-empty string")
            if "frame" not in entry:
                errors.append(f"{where}: 'frame' is required")
        if float(config.get("rate_hz", 30.0)) <= 0:
            errors.append("'rate_hz' must be > 0")
        return errors

    def _resolve(self, ctx: SimContext, path: str, where: str) -> Frame:
        try:
            return resolve_frame(ctx, path, within=self.entity or None)
        except PathError as exc:
            scope = f"under {self.entity!r}" if self.entity else "from the top of the world"
            raise PluginError(f"pose_publisher {self.address}: {where} ({scope}): {exc}") from None

    def configure(self, ctx: SimContext) -> None:
        self._ctx = ctx
        self._poses = {}
        for i, entry in enumerate(self.config["poses"]):
            frame = self._resolve(ctx, entry["frame"], f"poses[{i}].frame")
            relative = entry.get("relative_to", _WORLD)
            ref = None if relative == _WORLD else self._resolve(ctx, relative, f"poses[{i}]")
            child = entry.get("child") or frame.name
            if child in self._poses:
                raise PluginError(
                    f"pose_publisher {self.address}: poses[{i}] publishes {child!r} a second time; "
                    "a consumer matches a pose by its child frame, so give one a `child:`"
                )
            self._poses[child] = (frame, ref, self.world_frame if ref is None else ref.name)
        self.children = list(self._poses)

    @endpoint.out(
        name="poses/{item}",
        each="children",
        rate="rate_hz",
        lazy="lazy",
        ros2=lambda self, child: {"topic": self.topic, "frame_id": self._poses[child][2]},
    )
    def pose(self, child: str) -> Transform | None:
        """One configured frame's true pose, relative to its ``relative_to`` frame."""
        frame, ref, parent = self._poses[child]
        found = frame_pose(self._ctx, frame, ref)
        if found is None:
            return None
        return Transform(parent, child, found.translation, found.rotation)
