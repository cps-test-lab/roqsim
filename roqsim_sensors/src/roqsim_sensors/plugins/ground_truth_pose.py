"""Sensor plugin: ground-truth pose of a body or a site as a TF frame.

Publishes an entity's *true* MuJoCo pose as a TF transform ``<frame_id> -> <child_frame>`` (default
``map -> <model>_base_link_gt``). This is the substrate's perfect pose, deliberately kept out of the
odometry/localization chain: nav2 localizes with AMCL off the drifting wheel odometry
(``diff_drive``), while this frame is the disconnected ground-truth leaf an evaluator diffs the
driven path against.

It mirrors the Gazebo stack's ``<robot>_base_link_gt`` frame (produced there by
``gazebo_tf_publisher`` from ``SceneBroadcaster`` poses), so a rosbag recorded against either
simulator carries the same ground-truth frame and the same analysis applies unchanged -- which is
what lets roqsim stand in for Gazebo.

Family-agnostic: it reads ``data.xpos``/``data.xquat`` of a body (or ``site_xpos``/``site_xmat`` of
a site), so the same plugin serves a TurtleBot, a Husky, a Spot, a humanoid, or a prop spawned with
``spawn_model``. Register it on any entity that wants a ground-truth frame; several instances on
one entity publish several frames.

Config::

    ground_truth_pose:
      # The pose read is the entity this entry is NESTED UNDER; ownership is position, not a
      # config key, and it is required -- at the top of a document this entry is refused.
      body: ""                  # base body override; default: the entity's registered base body
      site: ""                  # a SITE of the entity instead of a body: the pose of a sensor
                                #   mount, an emitter, an optical-flow sensor; `body` is then unused
      relative_to: world        # `world`: the true world pose (the default). `base`: the pose
                                #   in the entity's base-body frame -- what a simulator publishes
                                #   for a link of a model, and what a stack that composes link
                                #   poses with the model's own pose expects
      frame_id: map             # parent frame of the transform (`base`: default: the base body)
      child_frame: ""           # default: "<model>_base_link_gt" for a body (Gazebo-compatible),
                                #   the site's own name for a site
      rate_hz: 30.0             # TF publish rate
      lazy: false               # true: publish only while something subscribes (Endpoint.lazy) --
                                #   for a frame only a robot's own stack reads, never for /tf
      topics: { pose: /tf }     # optional absolute-topic hardwire (default relative "tf" -> /tf)

The transform is published on the relative ``tf`` topic, so it lands on the plain ``/tf`` (matching
Gazebo). Configure the ``ros2_bridge`` ``gt: {prefix}`` block to divert it to ``/gt/tf`` instead, or
give an instance its own ``topics: {pose: ...}`` when a stack reads ground truth from a topic of its
own rather than from the TF tree.
"""

from __future__ import annotations

import mujoco
import numpy as np

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
        self.site = self.config.get("site", "")
        self.relative_to = str(self.config.get("relative_to", "world"))
        self.frame_id = self.config.get("frame_id", "")
        self.child_frame = self.config.get("child_frame", "")
        self.rate_hz = float(self.config.get("rate_hz", 30.0))
        self.lazy = bool(self.config.get("lazy", False))
        self._bid = -1  # the base body: what is published, or what a site pose is relative to
        self._sid = -1  # the site, when one is named
        self._ctx: SimContext | None = None
        # Scratch for a site's or a relative pose, so a read allocates nothing.
        self._pos = np.zeros(3)
        self._quat = np.zeros(4)
        self._base_quat = np.zeros(4)
        self._inv = np.zeros(4)
        self._mat = np.zeros(9)

    def validate_config(self, config: dict) -> list[str]:
        errors = self.validate_topics(config)
        if float(config.get("rate_hz", 30.0)) <= 0:
            errors.append("'rate_hz' must be > 0")
        if config.get("relative_to", "world") not in ("world", "base"):
            errors.append("'relative_to' must be 'world' or 'base'")
        if config.get("site") and config.get("body"):
            errors.append("'site' and 'body' name two different things to publish; set one")
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
        if self.site:
            self._sid = mujoco.mj_name2id(ctx.model, mujoco.mjtObj.mjOBJ_SITE, prefix + self.site)
            if self._sid < 0:
                # Fail loudly: a frame published for a site that is not there would be a transform
                # to nowhere that a consumer matches by name and trusts.
                raise RuntimeError(f"ground_truth_pose: site {prefix + self.site!r} not found")

        # Default child frame mirrors Gazebo's "<model>_base_link_gt". The frame is published verbatim
        # (the TFMessage converter does not namespace child frames), so multi-robot worlds set it
        # explicitly or rely on the model name being unique. `model` may be a filename or an absolute
        # path as well as a bundled name, so it is reduced to the stem a TF frame can carry. A site
        # is published under its own name: that is what a stack reading "the mouse", "the emitter"
        # off a simulator's pose stream matches on.
        model = _model_stem((entity.meta.get("model") if entity else None) or self.robot)
        child = self.child_frame or (self.site if self.site else f"{model}_base_link_gt")
        # A relative pose hangs from the base body it is relative to; a world pose from the map.
        parent = self.frame_id or (
            body_name.removeprefix(prefix) if self.relative_to == "base" else "map"
        )

        ctx.interface.add(
            Endpoint(
                name="pose",
                direction="out",
                owner=self.robot,
                namespace=ns,
                read=lambda child=child: self._read(child),
                rate_hz=self.rate_hz,
                lazy=self.lazy,
                backend={
                    "ros2": {
                        "type": "tf2_msgs.msg.TFMessage",
                        "topic": self.topic_override("pose") or "tf",
                        "frame_id": parent,
                    }
                },
            )
        )

    def _read(self, child_frame: str):
        """Endpoint ``read`` (physics thread): the true pose as a one-entry TF payload
        ``[(child_frame, pos[3], quat_wxyz[4])]``. ``quat`` is MuJoCo (w, x, y, z).

        A body's world pose is ``xpos``/``xquat`` verbatim. A site's is ``site_xpos`` and a quaternion
        from ``site_xmat``. Relative to the base, both are ``base^-1 * pose``.
        """
        d = self._ctx.data
        if self._sid >= 0:
            pos = d.site_xpos[self._sid]
            mujoco.mju_mat2Quat(self._quat, d.site_xmat[self._sid])
            quat = self._quat
        else:
            pos, quat = d.xpos[self._bid], d.xquat[self._bid]
        if self.relative_to == "world":
            return [(child_frame, pos, quat)]
        base_pos, base_quat = d.xpos[self._bid], d.xquat[self._bid]
        mujoco.mju_negQuat(self._inv, base_quat)
        # pos_rel = R_base^T (pos - base_pos); quat_rel = base_quat^-1 * quat.
        self._mat[:] = d.xmat[self._bid]
        diff = pos - base_pos
        rel = self._pos
        rel[0] = self._mat[0] * diff[0] + self._mat[3] * diff[1] + self._mat[6] * diff[2]
        rel[1] = self._mat[1] * diff[0] + self._mat[4] * diff[1] + self._mat[7] * diff[2]
        rel[2] = self._mat[2] * diff[0] + self._mat[5] * diff[1] + self._mat[8] * diff[2]
        mujoco.mju_mulQuat(self._base_quat, self._inv, quat)
        return [(child_frame, rel, self._base_quat)]
