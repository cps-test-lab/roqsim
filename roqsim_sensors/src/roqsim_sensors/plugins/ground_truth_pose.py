"""Sensor plugin: ground-truth base pose as a TF frame.

Publishes a robot base body's *true* MuJoCo world pose as a TF transform
``<frame_id> -> <child_frame>`` (default ``map -> <model>_base_link_gt``). This is the substrate's
perfect pose, deliberately kept out of the odometry/localization chain: nav2 localizes with AMCL off
the drifting wheel odometry (``diff_drive``), while this frame is the disconnected ground-truth leaf
an evaluator diffs the driven path against.

It mirrors the Gazebo stack's ``<robot>_base_link_gt`` frame (produced there by
``gazebo_tf_publisher`` from ``SceneBroadcaster`` poses), so a rosbag recorded against either
simulator carries the same ground-truth frame and the same analysis applies unchanged -- which is
what lets roqsim stand in for Gazebo.

Family-agnostic: it reads ``data.xpos``/``data.xquat`` of the base body, so the same plugin serves a
TurtleBot, a Husky, a Spot, or a humanoid. Register it on any world (or a robot manifest) that wants
a ground-truth frame.

Config::

    ground_truth_pose:
      robot: robot              # entity name to read the base pose from (default: robot)
      body: ""                  # base body override; default: the entity's registered base body
      frame_id: map             # parent frame of the ground-truth transform
      child_frame: ""           # default: "<model>_base_link_gt" (Gazebo-compatible)
      rate_hz: 30.0             # TF publish rate
      topics: { pose: /tf }     # optional absolute-topic hardwire (default relative "tf" -> /tf)

The transform is published on the relative ``tf`` topic, so it lands on the plain ``/tf`` (matching
Gazebo). Configure the ``ros2_bridge`` ``gt: {prefix}`` block to divert it to ``/gt/tf`` instead.
"""

from __future__ import annotations

import mujoco

from roqsim.context import Endpoint, SimContext
from roqsim.plugin import Plugin


class GroundTruthPosePlugin(Plugin):
    parallel_safe = True  # post-compile read-only: reads data.xpos/xquat, publishes via endpoint

    def __init__(self, config=None, *, name=None):
        super().__init__(config, name=name)
        self.robot = self.config.get("robot", "robot")
        self.body = self.config.get("body", "")
        self.frame_id = self.config.get("frame_id", "map")
        self.child_frame = self.config.get("child_frame", "")
        self.rate_hz = float(self.config.get("rate_hz", 30.0))
        self._bid = -1
        self._ctx: SimContext | None = None

    def validate_config(self, config: dict) -> list[str]:
        errors = self.validate_topics(config)
        if float(config.get("rate_hz", 30.0)) <= 0:
            errors.append("'rate_hz' must be > 0")
        return errors

    def configure(self, ctx: SimContext) -> None:
        self._ctx = ctx
        entity = ctx.entities.get(self.robot)
        prefix = entity.meta.get("prefix", "") if entity else ""
        ns = self.config.get("namespace") or (entity.meta.get("namespace", "") if entity else "")

        # Base body: explicit override (prefixed like every other name), else the entity's registered
        # base body, else the conventional "<prefix>base_link".
        body_name = (
            (prefix + self.body)
            if self.body
            else (entity.body if entity and entity.body else prefix + "base_link")
        )
        self._bid = mujoco.mj_name2id(ctx.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if self._bid < 0:
            raise RuntimeError(f"ground_truth_pose: base body {body_name!r} not found")

        # Default child frame mirrors Gazebo's "<model>_base_link_gt". The frame is published verbatim
        # (the TFMessage converter does not namespace child frames), so multi-robot worlds set it
        # explicitly or rely on the model name being unique.
        model = entity.meta.get("model", self.robot) if entity else self.robot
        child = self.child_frame or f"{model}_base_link_gt"

        ctx.interface.add(
            Endpoint(
                name="pose",
                direction="out",
                owner=self.robot,
                namespace=ns,
                read=lambda child=child: self._read(child),
                rate_hz=self.rate_hz,
                backend={
                    "ros2": {
                        "type": "tf2_msgs.msg.TFMessage",
                        "topic": self.topic_override("pose") or "tf",
                        "frame_id": self.frame_id,
                    }
                },
            )
        )

    def _read(self, child_frame: str):
        """Endpoint ``read`` (physics thread): the base body's true world pose as a one-entry TF
        payload ``[(child_frame, pos[3], quat_wxyz[4])]``. ``quat`` is MuJoCo (w, x, y, z)."""
        d = self._ctx.data
        return [(child_frame, d.xpos[self._bid], d.xquat[self._bid])]
