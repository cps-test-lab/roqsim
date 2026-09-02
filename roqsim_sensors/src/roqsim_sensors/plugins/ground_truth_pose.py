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
      # The base pose read is the entity this entry is NESTED UNDER; ownership is position, not
      # a config key, and it is required -- at the top of a document this entry is refused.
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


def _model_stem(model: str) -> str:
    """The frame-name part of a ``model:`` reference: its bare stem.

    A reference is a bundled name, a filename or an absolute path (``spawn_robot``'s ``model:``), and
    only the first is already a legal TF frame name -- the other two carry separators a frame cannot.
    """
    return str(model).rsplit("/", 1)[-1].rpartition(":")[2].removesuffix(".xml")


class GroundTruthPosePlugin(Plugin):
    #: Attaches to the entity whose pose it publishes, so it must be nested under that entry.
    #: Nothing downstream can catch the unowned case: a robot spawned with the default empty
    #: ``prefix`` leaves an unprefixed ``base_link`` in the world, which an ownerless instance
    #: resolves happily and then labels from a model name it has no entity to ask for. The result is
    #: a well-formed ground-truth transform under a frame name no evaluator matches -- found only
    #: once an analysis of the finished bag reports no ground truth.
    requires_owner = True

    parallel_safe = True  # post-compile read-only: reads data.xpos/xquat, publishes via endpoint

    def __init__(self, config=None, *, name=None, entity=None, label=None):
        super().__init__(config, name=name, entity=entity, label=label)
        self.robot = self.entity
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
        # explicitly or rely on the model name being unique. `model` may be a filename or an absolute
        # path as well as a bundled name, so it is reduced to the stem a TF frame can carry.
        model = _model_stem((entity.meta.get("model") if entity else None) or self.robot)
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
