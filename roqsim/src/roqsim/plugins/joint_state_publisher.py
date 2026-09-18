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

"""Observation plugin: every joint of an entity as one ``joint_states`` message.

What ros2_control's ``joint_state_broadcaster`` publishes for a robot: the position, velocity and
effort of **all** its joints, in one message, whether a controller drives them or nothing does. A
base's controller publishes the joints it drives and knows nothing about the rest -- a suspension
travel, a caster swivel, a lid hinge -- and a consumer that reads those off ``joint_states`` needs
them in the same message as the driven ones: one that derives a controller-side state from every
message it receives (the Create 3 stack's ``dynamic_joint_states``) is broken by a second publisher
that carries only the passive joints.

So this plugin publishes the whole entity and the base's own ``joint_states`` is switched off
(``diff_drive: {publish_joint_states: false}``), or it is left out and the base's message stands
alone. Two publishers on one topic is not refused -- ros2_control allows several broadcasters --
but a world that needs every joint in every message declares this one and only this one.

Config::

    joint_state_publisher:
      # The entity read is the one this entry is NESTED UNDER; at the top of a document it is
      # refused (`requires_owner`).
      namespace: ""          # transport scope for the endpoint
      joints: []             # joint NAMES to publish, the model's own before any spawn prefix;
                             #   default: every hinge and slide joint of the entity's subtree
      rate_hz: 50.0          # endpoint publish rate

Endpoint ``joint_states`` (out) reads ``(names, positions, velocities, efforts)``; the ROS 2
backend hint publishes it as ``sensor_msgs/JointState`` on ``joint_states`` (relative, so it is
scoped by the entity's namespace). Names are published without the spawn prefix, as every other
producer here names joints, so a description published alongside matches them.

Only hinge and slide joints are published: a ``JointState`` carries one scalar per joint, which a
ball or free joint's quaternion is not. A named joint that is not in the model, or not on this
entity, raises -- a state nobody publishes is a consumer reading zeros as a fact.
"""

from __future__ import annotations

import logging

import mujoco
import numpy as np

from ..contact_scope import resolve_base_body
from ..context import Endpoint, SimContext
from ..plugin import Plugin
from ..presence import entity_body_ids

_log = logging.getLogger(__name__)

_SCALAR_JOINTS = (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE))


class JointStatePublisherPlugin(Plugin):
    parallel_safe = True  # post_step only reads qpos/qvel/qfrc_actuator and writes its own buffers
    requires_owner = True

    def __init__(self, config=None, *, name=None, entity=None, label=None):
        super().__init__(config, name=name, entity=entity, label=label)
        self.robot = self.entity
        self.joints = list(self.config.get("joints", []))
        self.rate_hz = float(self.config.get("rate_hz", 50.0))
        self._names: list[str] = []
        self._qpos: np.ndarray = np.zeros(0, dtype=int)
        self._dof: np.ndarray = np.zeros(0, dtype=int)
        self._pos = np.zeros(0)
        self._vel = np.zeros(0)
        self._eff = np.zeros(0)

    def validate_config(self, config: dict) -> list[str]:
        errors = self.validate_topics(config)
        if float(config.get("rate_hz", 50.0)) <= 0:
            errors.append("'rate_hz' must be > 0")
        joints = config.get("joints", [])
        if not isinstance(joints, list) or not all(isinstance(j, str) for j in joints):
            errors.append("'joints' must be a list of joint names")
        return errors

    def configure(self, ctx: SimContext) -> None:
        model = ctx.model
        entity = ctx.entities.get(self.robot)
        prefix = entity.meta.get("prefix", "") if entity else ""
        ns = self.config.get("namespace") or (entity.meta.get("namespace", "") if entity else "")
        body_name = resolve_base_body(entity)
        bodies = set(entity_body_ids(model, body_name))
        if not bodies:
            raise RuntimeError(f"joint_state_publisher: base body {body_name!r} not found")

        if self.joints:
            jids = []
            for name in self.joints:
                jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, prefix + name)
                if jid < 0:
                    raise RuntimeError(f"joint_state_publisher: joint {prefix + name!r} not found")
                if int(model.jnt_bodyid[jid]) not in bodies:
                    raise RuntimeError(
                        f"joint_state_publisher: joint {prefix + name!r} is not on {body_name!r}"
                    )
                if int(model.jnt_type[jid]) not in _SCALAR_JOINTS:
                    raise RuntimeError(
                        f"joint_state_publisher: joint {prefix + name!r} is not a hinge or slide "
                        f"joint, so it has no scalar state to publish"
                    )
                jids.append(jid)
        else:
            jids = [
                j
                for j in range(model.njnt)
                if int(model.jnt_bodyid[j]) in bodies and int(model.jnt_type[j]) in _SCALAR_JOINTS
            ]
        if not jids:
            raise RuntimeError(
                f"joint_state_publisher: {body_name!r} carries no hinge or slide joint to publish"
            )
        self._names = [
            (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or f"joint{j}").removeprefix(
                prefix
            )
            for j in jids
        ]
        self._qpos = np.array([model.jnt_qposadr[j] for j in jids], dtype=int)
        self._dof = np.array([model.jnt_dofadr[j] for j in jids], dtype=int)
        self._pos = np.zeros(len(jids))
        self._vel = np.zeros(len(jids))
        self._eff = np.zeros(len(jids))

        ctx.interface.add(
            Endpoint(
                name="joint_states",
                direction="out",
                owner=self.robot,
                namespace=ns,
                read=self.read_joint_states,
                rate_hz=self.rate_hz,
                backend={
                    "ros2": {
                        "type": "sensor_msgs.msg.JointState",
                        "topic": self.topic_override("joint_states") or "joint_states",
                    }
                },
            )
        )
        _log.info("joint_state_publisher: %d joints of %r", len(jids), body_name)

    def read_joint_states(self):
        return (self._names, self._pos, self._vel, self._eff)

    def post_step(self, ctx: SimContext) -> None:
        d = ctx.data
        # Written in place so read() is zero-copy, like the base controllers' joint state.
        self._pos[:] = d.qpos[self._qpos]
        self._vel[:] = d.qvel[self._dof]
        # The generalised actuator force on each DOF: what a real driver reports as effort for
        # a driven joint, and zero for a passive one.
        self._eff[:] = d.qfrc_actuator[self._dof]
