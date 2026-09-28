"""The ground-truth pose streams the iRobot Create 3 simulator stack reads.

The Create 3's Gazebo simulation runs a pose publisher on the robot and on its dock, and the stack's
adapter (``irobot_create_gz_toolbox``'s pose republisher) turns those poses into the optical-flow
mouse, the dock's infrared field and the kidnap detection. This plugin publishes the same streams,
so the unchanged stack runs against roqsim:

* on ``topic``, one :class:`FrameTransform` per frame, which ROS carries as a one-transform
  ``tf2_msgs/TFMessage`` (:mod:`roqsim_mobile.ros2_types`);
* the entity's own pose in the world, ``map -> <frame>``, from the core pose endpoint
  (``sim/entities/<entity>/pose``, :mod:`roqsim.entity_pose`);
* each of ``sites`` relative to the entity's body, ``<body> -> <site>``, as ``body^-1 * site``:
  constant for a rigid mount wherever the base stands, which is how the adapter composes it.

Nothing is computed while nothing subscribes: every stream is ``lazy``. It is nested under the
entity it describes -- the TurtleBot 4's manifest carries the robot's, and a world places the dock's
under its ``create3_dock`` prop.

Config::

    create3_pose_publisher:
      topic: _internal/sim_ground_truth_pose   # relative, so the entity's namespace scopes it
      frame: turtlebot4     # child frame of the entity's world pose (the adapter's robot_name);
                            #   empty: no world pose, only the sites
      sites: [mouse, ir_omni]   # sites of the entity, published relative to its body
      rate_hz: 62.0
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from roqsim import endpoint, entity_pose
from roqsim.context import SimContext
from roqsim.plugin import Plugin
from roqsim.types import Point3, Quaternion


@dataclass
class FrameTransform:
    """One frame's pose relative to its parent, the parent being the endpoint's ``frame_id`` hint.

    Attributes:
        child_frame_id: the frame placed, by the name its consumer matches
        translation: the child frame's origin, parent frame
        rotation: quaternion (w, x, y, z), parent frame
    """

    child_frame_id: str
    translation: Point3
    rotation: Quaternion


class Create3PosePublisherPlugin(Plugin):
    #: Describes the entity it is nested under; at the top of a document there is none.
    requires_owner = True

    parallel_safe = True  # reads xpos/xquat/site poses, writes nothing

    def __init__(self, config=None, *, name=None, entity=None, label=None):
        super().__init__(config, name=name, entity=entity, label=label)
        self.topic = str(self.config.get("topic", "_internal/sim_ground_truth_pose"))
        self.frame = str(self.config.get("frame", ""))
        self.sites = [str(s) for s in self.config.get("sites", [])]
        self.rate_hz = float(self.config.get("rate_hz", 62.0))
        self._ctx: SimContext | None = None
        self._pose = None  # the core pose endpoint of the entity
        self._bid = -1
        self._sids: dict[str, int] = {}
        self._body = ""

    def validate_config(self, config: dict) -> list[str]:
        errors = []
        if float(config.get("rate_hz", 62.0)) <= 0:
            errors.append("'rate_hz' must be > 0")
        if not config.get("frame") and not config.get("sites"):
            errors.append("nothing to publish: set 'frame', 'sites' or both")
        return errors

    def configure(self, ctx: SimContext) -> None:
        self._ctx = ctx
        entity = ctx.entities.get(self.entity)
        if entity is None or not entity.body:
            raise RuntimeError(f"create3_pose_publisher: entity {self.entity!r} has no body")
        prefix = entity.meta.get("prefix", "")
        self._bid = mujoco.mj_name2id(ctx.model, mujoco.mjtObj.mjOBJ_BODY, entity.body)
        self._body = entity.body.removeprefix(prefix)
        self._pose = ctx.interface.find(entity_pose.OWNER, entity_pose.endpoint_name(self.entity))
        if self._pose is None:
            raise RuntimeError(f"create3_pose_publisher: entity {self.entity!r} has no pose")
        for site in self.sites:
            sid = mujoco.mj_name2id(ctx.model, mujoco.mjtObj.mjOBJ_SITE, prefix + site)
            if sid < 0:
                # A stream for a site that is not there would be a transform to nowhere that the
                # adapter matches by name and trusts.
                raise RuntimeError(f"create3_pose_publisher: site {prefix + site!r} not found")
            self._sids[site] = sid

    @endpoint.out(
        name="pose",
        rate="rate_hz",
        lazy=True,
        when="frame",
        ros2=lambda self: {"topic": self.topic, "frame_id": "map"},
    )
    def world_pose(self) -> FrameTransform | None:
        """The entity's true world pose as ``<frame>``."""
        pose = self._pose.read()
        if pose is None:
            return None
        return FrameTransform(self.frame, pose.position, pose.orientation)

    @endpoint.out(
        name="pose/{item}",
        each="sites",
        rate="rate_hz",
        lazy=True,
        ros2=lambda self, site: {"topic": self.topic, "frame_id": self._body},
    )
    def site_pose(self, site: str) -> FrameTransform:
        """A site's pose relative to the entity's body, under the site's own name."""
        d = self._ctx.data
        sid, bid = self._sids[site], self._bid
        quat = np.empty(4)
        mujoco.mju_mat2Quat(quat, d.site_xmat[sid])
        base_inv = np.empty(4)
        mujoco.mju_negQuat(base_inv, d.xquat[bid])
        # pos_rel = R_body^T (pos - body_pos); quat_rel = body_quat^-1 * quat.
        rel_pos = d.xmat[bid].reshape(3, 3).T @ (d.site_xpos[sid] - d.xpos[bid])
        rel_quat = np.empty(4)
        mujoco.mju_mulQuat(rel_quat, base_inv, quat)
        return FrameTransform(site, rel_pos, rel_quat)
