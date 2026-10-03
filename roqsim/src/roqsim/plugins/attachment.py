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

Its keys are declared in :attr:`AttachmentPlugin.CONFIG_SCHEMA`: the ``body`` carried (required),
the body ``to`` hold it by, whether it is ``attached`` at reset, and a name ``prefix`` for both. The
entity that carries is the one this entry is nested under (``requires_owner``).

Endpoints, scoped by the component's **address** with dots as slashes (``robot.attachment`` ->
``robot/attachment/attach``), exactly as a sensor's fault switch is:

``attach`` (a command)
    true holds, false releases, and the reply is the :class:`AttachmentReport` the call left. Over
    ROS it is a ``std_srvs/SetBool`` service whose reply message is that state (``attached`` or
    ``released``), so a scenario can fail a trial on a release that did not happen; a scenario
    reaches it with ``entity_call(entity: 'robot.attachment', command: 'attach', value: 'true')``.
``attached`` (out)
    the report; ROS carries its ``attached`` field as a ``std_msgs/Bool``, so a stack can watch the
    load without calling anything.

An :class:`AttachmentHandle` is published on the blackboard under ``attachment:<address>`` with the
same three members the fault handles offer, so an in-process consumer drives this the way it drives
the other switchable channels.

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

from roqsim import endpoint
from roqsim.context import SimContext
from roqsim.paths import address_path
from roqsim.plugin import Plugin
from roqsim.schema import Field
from roqsim.types import Duration

_log = logging.getLogger(__name__)


@dataclass
class AttachmentReport:
    """Neutral payload for the ``attached`` endpoint and the ``attach`` command's reply.

    Attributes:
        attached: whether the load is held
        since: sim time the state last changed; -1.0 if it has not since reset
        changes: how many times it changed since reset
    """

    attached: bool
    since: Duration
    changes: int

    @property
    def verified(self) -> str:
        """The state as the ``SetBool`` service handler reports it in its reply's message."""
        return "attached" if self.attached else "released"


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

    #: Declared once, so `roqsim plugins describe attachment` publishes the same keys the checks run on.
    CONFIG_SCHEMA = {
        "body": Field(str, required=True, static=True, doc="the body being carried"),
        "to": Field(
            str,
            default="",
            static=True,
            doc="body that holds it (default: the one body ending in base_link)",
        ),
        "attached": Field(
            bool, default=False, doc="state at reset: whether the trial starts with the load held"
        ),
        "prefix": Field(str, static=True, doc="name prefix for body and to, as the spawns use"),
    }

    def __init__(self, config=None, *, name=None, entity=None, label=None):
        super().__init__(config, name=name, entity=entity, label=label)
        self.carrier = self.entity
        self._ctx: SimContext | None = None
        self._eq_id = -1
        self._load_bid = -1
        self._carrier_bid = -1
        self._report = AttachmentReport(False, -1.0, 0)
        self._eq_name = ""

    def validate_config(self, config: dict) -> list[str]:
        # Required keys and types come from CONFIG_SCHEMA; the topic map is this plugin's own.
        return self.validate_topics(config)

    # -- lifecycle ----------------------------------------------------------------------------

    def build(self, spec, ctx: SimContext) -> None:
        """Add the weld, inactive. Both names are resolved by suffix, as the other build-time
        plugins do -- entities register in ``configure``, after every ``build``, so the owner's
        prefix is not known yet unless the world (or a manifest) states it."""
        settings = self.settings
        load = self._resolve_body_name(spec, settings.body, settings.prefix)
        # The owner's registered body is not known before configure, and the weld needs both names
        # now, so an unstated `to:` means the conventional base body: the one named or ending in
        # base_link. A world with none, or with several, is refused by name.
        carrier = self._resolve_body_name(spec, settings.to or "base_link", settings.prefix)

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
        if not matches:
            raise RuntimeError(
                f"attachment: no body matching {wanted!r} is built yet. The weld names both bodies "
                f"at build, so list the component that spawns it before this one."
            )
        if len(matches) != 1:
            raise RuntimeError(
                f"attachment: expected exactly one body matching {wanted!r}, found {matches}. Set "
                f"'prefix:' when a world carries more than one of this robot."
            )
        return matches[0]

    def configure(self, ctx: SimContext) -> None:
        self._ctx = ctx
        m = ctx.model
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

    # -- endpoints ----------------------------------------------------------------------------

    # A service, not a topic, for the reason the fault switch is one: picking something up is a
    # command whose outcome the caller needs. `state_key` is where the SetBool handler reads it.
    @endpoint.command(
        ros2=lambda self: {
            "service": "std_srvs.srv.SetBool",
            "name": f"{address_path(self.address)}/attach",
            "state_key": f"attachment:{self.address}",
        },
    )
    def attach(self, data: bool) -> AttachmentReport:
        """Hold the load where it is, or release it; replies with the state the call left.

        Args:
            data: true holds, false releases
        """
        self.set_attached(data, self._ctx.sim_time)
        return self._report

    # The report is a structure and ROS carries its verdict alone: `field` names it.
    @endpoint.out(
        rate=5.0,
        ros2=lambda self: {"field": "attached", "topic": f"{address_path(self.address)}/attached"},
    )
    def attached(self) -> AttachmentReport:
        """Whether the load is held, since when, and how often it changed since reset."""
        return self._report

    def on_reset(self, ctx: SimContext) -> None:
        # Back to the configured state, through the same path a call takes -- so a world that starts
        # loaded is loaded again in trial 2, and one that does not is not still holding trial 1's box.
        ctx.data.eq_active[self._eq_id] = 0
        self._report = AttachmentReport(False, -1.0, 0)
        if self.settings.attached:
            self.set_attached(True, ctx.sim_time)
            # The reset state is the starting state, not a change the trial made.
            self._report = AttachmentReport(True, -1.0, 0)

    # -- the switch ---------------------------------------------------------------------------

    def set_attached(self, on: bool, sim_time: float = 0.0) -> None:
        """Hold the load where it currently is, or release it. Physics thread only.

        The ``attach`` command reaches this on the physics thread at the start of a step, so the
        single-writer rule holds without this plugin doing anything about it.
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

        The weld holds two points together and one orientation fixed: ``eq_data[0:3]`` is the weld
        point in body2's frame (the carrier's), ``eq_data[3:6]`` the same point in body1's (the
        load's), and ``eq_data[6:10]`` body2's orientation in body1's frame. The point is the
        carrier's origin. A pose stored from the other side agrees with this one only while both
        bodies are unrotated, and an anchor left at its default only at one offset, so
        ``tests/test_attachment.py`` attaches a tilted load to a turned carrier.
        """
        d, m = ctx.data, ctx.model
        # The poses of the state as it is now, not as the last step's kinematics left them.
        mujoco.mj_kinematics(m, d)
        load, carrier = self._load_bid, self._carrier_bid
        inverse, relative = np.zeros(4), np.zeros(4)
        mujoco.mju_negQuat(inverse, d.xquat[load])
        mujoco.mju_mulQuat(relative, inverse, d.xquat[carrier])
        eq = m.eq_data[self._eq_id]
        eq[0:3] = 0.0
        eq[3:6] = d.xmat[load].reshape(3, 3).T @ (d.xpos[carrier] - d.xpos[load])
        eq[6:10] = relative
        # Torque scale 1: the orientation is held as firmly as the position.
        eq[10] = 1.0
