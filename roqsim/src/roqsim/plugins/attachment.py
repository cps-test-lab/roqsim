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

"""Component plugin: hold an object to a robot, and let go of it, during a run.

The transport half of manipulation, without the manipulation. A forklift carrying a pallet, a mobile
base with a parcel on its deck, a drone releasing a payload over a target: in every one of them the
question a trial asks is *did the load arrive*, and simulating the grasp that holds it is a different
experiment with a different failure mode. This is Gazebo's ``DetachableJoint`` -- a joint that can be
broken at run time -- expressed the way MuJoCo already offers it, as a weld equality that is switched
on and off.

**It attaches where the object is, not where the model said it was.** Activating a weld alone would
snap the load back to the relative pose the MJCF compiled with, teleporting a parcel the robot had
driven up to. On attach the plugin recomputes the weld's relative pose from the current state, so the
object is held exactly where it stands -- which is what makes "drive up to it, pick it up" work
without the world having to spawn it in the carrying pose. Detaching leaves it where it is, with the
velocity it has: a parcel released from a moving deck slides off it, which is the behaviour that
makes a release worth simulating at all.

**It owns no trigger.** Like :mod:`roqsim.plugins.model_override` and the sensors' ``fault:`` block,
one bit crosses the wire and the timing belongs to the experiment: a scenario calls the service when
its own condition says to. The initial state is config, so "does the robot start loaded" is an
ordinary campaign factor rather than two world files.

Config::

    attachment:
      # The entity is the one this entry is NESTED UNDER (`requires_owner`): the thing that CARRIES.
      body: graspable_carton    # REQUIRED: the body being carried
      to: ""                    # body that holds it (default: the owner entity's base body)
      attached: false           # state at reset -- a campaign factor, not a second world file
      prefix: ""                # name prefix for `body`/`to`, as the spawn plugins use
      namespace: ""             # transport scope for the endpoints

Endpoints, scoped by the component's **address** with dots as slashes (``robot.attachment`` ->
``robot/attachment/attach``), exactly as a sensor's fault switch is:

``attach`` (in)
    a ``std_srvs/SetBool`` service: true holds, false releases. The reply says what the state is
    now, so a scenario can fail a trial on a release that did not happen.
``attached`` (out)
    a ``std_msgs/Bool``, so a stack can watch the load without calling anything.

An :class:`AttachmentHandle` is published on the blackboard under ``attachment:<address>`` with the
same three members the fault handles offer, so an in-process consumer -- or a future scenario action
-- drives this the way it drives the other two switchable channels.

**Two bodies, one weld, and both must be able to move.** A weld between bodies that MuJoCo has
welded to the world is not a constraint it can satisfy -- there is nothing to solve for -- so an
attachment whose carried body has no degrees of freedom is refused at configure rather than left to
hold nothing. The carrier may be fixed (a static arm holding a part is a real setup); the load may
not.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

import mujoco
import numpy as np

from roqsim.context import Endpoint, SimContext
from roqsim.plugin import Plugin

_log = logging.getLogger(__name__)


@dataclass
class AttachmentReport:
    """What the state endpoint carries, and what a service reply is built from."""

    attached: bool
    since: float  # sim time the state last changed; -1.0 if it never has
    changes: int  # how many times it changed since reset


@dataclass
class AttachmentHandle:
    """Published on the blackboard under ``attachment:<address>``.

    The three members :class:`roqsim_sensors.live_config.SensorFaultHandle` and
    ``ModelOverrideHandle`` offer, so every switchable channel is driven through one shape.
    """

    name: str
    set_active: Callable[[bool], None]
    is_active: Callable[[], bool]
    read_state: Callable[[], AttachmentReport]


class AttachmentPlugin(Plugin):
    """See the module docstring."""

    #: A carrier carries something, so this belongs inside the carrier's ``components:`` block.
    requires_owner = True

    def __init__(self, config=None, *, name=None, entity=None, label=None):
        super().__init__(config, name=name, entity=entity, label=label)
        self.carrier = self.entity
        self.body = self.config.get("body", "")
        self.to = self.config.get("to", "")
        self.attached_at_reset = bool(self.config.get("attached", False))
        self._ctx: SimContext | None = None
        self._eq_id = -1
        self._load_bid = -1
        self._carrier_bid = -1
        self._report = AttachmentReport(False, -1.0, 0)
        self._eq_name = ""

    # -- validation ---------------------------------------------------------------------------

    def validate_config(self, config: dict) -> list[str]:
        errors = self.validate_topics(config)
        if not config.get("body"):
            errors.append("'body' is required: name the body being carried")
        if "attached" in config and not isinstance(config["attached"], bool):
            errors.append("'attached' must be true or false")
        return errors

    # -- lifecycle ----------------------------------------------------------------------------

    def build(self, spec, ctx: SimContext) -> None:
        """Add the weld, inactive. Both names are resolved by suffix, as the other build-time
        plugins do -- entities register in ``configure``, after every ``build``, so the owner's
        prefix is not known yet unless the world (or a manifest) states it."""
        prefix = self.config.get("prefix")
        load = self._resolve_body_name(spec, self.body, prefix)
        # The carrier's own body is not knowable here without the entity registry, so an unstated
        # `to:` is resolved in configure and the weld is built against the load only once both are
        # known -- which means the weld itself is added here with both names, and `to` must be
        # resolvable by name too. Defaulting it to the OWNER'S conventional base body keeps the
        # common case ("carry it on the robot") free of configuration.
        carrier = self._resolve_body_name(spec, self.to or "base_link", prefix)

        equality = spec.add_equality()
        # Named off the address: two attachments on one robot (a forklift with two forks) must not
        # collide, and a compile error naming neither of them is a poor way to find that out.
        self._eq_name = f"{self.address.replace('.', '_')}_weld"
        equality.name = self._eq_name
        equality.type = mujoco.mjtEq.mjEQ_WELD
        equality.objtype = mujoco.mjtObj.mjOBJ_BODY
        equality.name1 = load
        equality.name2 = carrier
        # Inactive at build: the state at reset is config, and `on_reset` applies it through the same
        # path a runtime call takes, so there is one code path that establishes a hold.
        equality.active = False

    @staticmethod
    def _resolve_body_name(spec, wanted: str, prefix: str | None) -> str:
        if prefix is not None:
            names = {b.name for b in spec.bodies}
            if f"{prefix}{wanted}" in names:
                return f"{prefix}{wanted}"
            raise RuntimeError(f"attachment: body {prefix}{wanted!r} not found")
        matches = [b.name for b in spec.bodies if b.name == wanted or b.name.endswith(wanted)]
        if len(matches) != 1:
            raise RuntimeError(
                f"attachment: expected exactly one body matching {wanted!r}, found {matches}. Set "
                f"'prefix:' when a world carries more than one of this robot."
            )
        return matches[0]

    def configure(self, ctx: SimContext) -> None:
        self._ctx = ctx
        m = ctx.model
        entity = ctx.entities.get(self.carrier)
        ns = self.config.get("namespace") or (entity.meta.get("namespace", "") if entity else "")

        self._eq_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, self._eq_name)
        if self._eq_id < 0:
            raise RuntimeError(
                f"attachment[{self.label}]: weld {self._eq_name!r} missing after compile"
            )
        self._load_bid = int(m.eq_obj1id[self._eq_id])
        self._carrier_bid = int(m.eq_obj2id[self._eq_id])
        if int(m.body_weldid[self._load_bid]) == 0:
            # A weld to a body that cannot move solves nothing, and the failure is invisible: the
            # service replies "attached" and the object never follows.
            raise RuntimeError(
                f"attachment[{self.label}]: the carried body "
                f"{mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, self._load_bid)!r} is welded to the "
                f"world, so nothing can carry it. Give it a free joint."
            )

        ctx.blackboard.set(
            f"attachment:{self.address}",
            AttachmentHandle(
                name=self.address,
                set_active=lambda on: self.set_attached(on, ctx.sim_time),
                is_active=lambda: bool(ctx.data.eq_active[self._eq_id]),
                read_state=lambda: self._report,
            ),
        )
        scope = self.address.replace(".", "/")
        ctx.interface.add(
            Endpoint(
                name="attach",
                direction="in",
                owner=self.carrier,
                namespace=ns,
                write=lambda payload: self.set_attached(bool(payload), ctx.sim_time),
                backend={
                    "ros2": {
                        # A service, not a topic, for the reason the fault switch is one: picking
                        # something up is a command whose outcome the caller needs.
                        "service": "std_srvs.srv.SetBool",
                        "name": f"{scope}/attach",
                        "state_key": f"attachment:{self.address}",
                    }
                },
            )
        )
        ctx.interface.add(
            Endpoint(
                name="attached",
                direction="out",
                owner=self.carrier,
                namespace=ns,
                read=lambda: self._report,
                rate_hz=5.0,
                backend={
                    "ros2": {
                        "type": "std_msgs.msg.Bool",
                        "topic": self.topic_override("attached") or f"{scope}/attached",
                        # The payload is the report; the bridge publishes the named field, so no
                        # converter needs to know this plugin's dataclass.
                        "field": "attached",
                    }
                },
            )
        )

    def on_reset(self, ctx: SimContext) -> None:
        # Back to the configured state, through the same path a call takes -- so a world that starts
        # loaded is loaded again in trial 2, and one that does not is not still holding trial 1's box.
        ctx.data.eq_active[self._eq_id] = 0
        self._report = AttachmentReport(False, -1.0, 0)
        if self.attached_at_reset:
            self.set_attached(True, ctx.sim_time)
            # The reset state is the starting state, not a change the trial made.
            self._report = AttachmentReport(True, -1.0, 0)

    # -- the switch ---------------------------------------------------------------------------

    def set_attached(self, on: bool, sim_time: float = 0.0) -> None:
        """Hold the load where it currently is, or release it. Physics thread only.

        The bridge marshals every inbound payload through ``ctx.post``, so this always runs on the
        physics thread and the single-writer rule holds without this plugin doing anything about it.
        """
        ctx = self._ctx
        on = bool(on)
        if ctx is None or bool(ctx.data.eq_active[self._eq_id]) == on:
            return
        if on:
            self._hold_current_pose(ctx)
        ctx.data.eq_active[self._eq_id] = 1 if on else 0
        self._report = AttachmentReport(
            attached=on,
            since=float(sim_time),
            changes=self._report.changes + 1,
        )
        _log.info(
            "attachment: %s %s %s at t=%.3f",
            self.address,
            "holds" if on else "releases",
            mujoco.mj_id2name(ctx.model, mujoco.mjtObj.mjOBJ_BODY, self._load_bid),
            float(sim_time),
        )

    def _hold_current_pose(self, ctx: SimContext) -> None:
        """Write the weld's relative pose from where the two bodies are *now*.

        Rewriting it on every attach is what makes this a pick-up rather than a teleport: without it
        the load snaps back to wherever the MJCF happened to declare it, which for a parcel the robot
        drove up to is metres away.

        ``eq_data[3:10]`` is the pose of **body1 in body2's frame** -- here the load in the carrier's
        -- which is the layout the compiler fills from the model's reference configuration, and which
        was confirmed by measurement rather than read off a sign convention: written the other way
        round the solver drives the load to twice its offset, so a parcel 0.5 m to the side is
        snatched to 1.5 m. ``tests/test_attachment.py`` pins the direction with a load the carrier
        both moves AND rotates, because a translation-only test cannot tell the two apart.
        """
        d, m = ctx.data, ctx.model
        rot_carrier = np.array(d.xmat[self._carrier_bid]).reshape(3, 3)
        quat_load, quat_carrier, inverse, relative = (np.zeros(4) for _ in range(4))
        mujoco.mju_mat2Quat(
            quat_load, np.ascontiguousarray(np.array(d.xmat[self._load_bid])).reshape(-1)
        )
        mujoco.mju_mat2Quat(quat_carrier, np.ascontiguousarray(rot_carrier).reshape(-1))
        mujoco.mju_negQuat(inverse, quat_carrier)
        mujoco.mju_mulQuat(relative, inverse, quat_load)
        eq = m.eq_data[self._eq_id]
        eq[3:6] = rot_carrier.T @ (d.xpos[self._load_bid] - d.xpos[self._carrier_bid])
        eq[6:10] = relative
        # Torque scale: how hard the orientation half of the weld is enforced. The compiler's default
        # is 1 (a rigid hold); a spec-built equality starts at 0, which constrains the position and
        # lets the load spin freely in the carrier's grip -- a bug that looks like a physics quirk.
        eq[10] = 1.0
