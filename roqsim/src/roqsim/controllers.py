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

"""Which controllers a robot has, what each claims, and which of them are running.

The single source of truth for that, and deliberately ROS-free: a controller's name, its type, its
lifecycle state and the interfaces it claims are facts about the robot, not about a transport. The
bridge serves ``controller_manager_msgs`` out of this; a plugin asks it whether it is active; an
in-process task switches through it. Three separate places each tracking "is this controller
running" is the failure this exists to prevent.

**The world file is the parameter file.** Under ros2_control a controller_manager is given its
controllers and their types as parameters, and ``load_controller`` instantiates one of THOSE -- a
name it was never given fails. Here the world's ``components:`` is that list, which is why nothing
can be loaded that the world did not declare, and why that is the behaviour rather than a
limitation: ``spawner`` fails the same way against the real robot.

**Claims are command interfaces, and only while active.** ros2_control's ``ControllerState`` says
so in as many words -- ``claimed_interfaces`` is commented "command interfaces currently owned by
controller". State interfaces are shared and arbitrated by nobody, which is how a broadcaster reads
the same joint a trajectory controller drives. Reporting a claim for an inactive controller would
be the tempting shortcut: MoveIt's ros2_control manager derives a controller's joints by splitting
``claimed_interfaces``, so a simulation that filled it in while inactive would let a scenario
select a controller that the real robot would not offer.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

#: Blackboard key of the per-world registry.
SERVICE_KEY = "control:controllers"

#: ros2_control lifecycle states. ``unconfigured`` is where a controller that has been loaded but
#: not configured sits; roqsim's are configured by the time the world is built, so they start at
#: ``inactive`` or ``active``.
UNCONFIGURED, INACTIVE, ACTIVE, FINALIZED = "unconfigured", "inactive", "active", "finalized"

#: ``SwitchController.srv`` strictness. There is no zero: a default-constructed request carries one,
#: and controller_manager treats that as best effort rather than refusing it.
BEST_EFFORT, STRICT, AUTO, FORCE_AUTO = 1, 2, 3, 4


@dataclass
class Controller:
    """One controller, as ros2_control would describe it."""

    name: str
    type: str
    #: COMMAND interfaces this controller drives, e.g. ``shoulder_pan_joint/position``. Empty for a
    #: broadcaster, which claims nothing -- that, and not any special case, is what lets a
    #: broadcaster coexist with the controller driving the same joint.
    claims: tuple[str, ...] = ()
    #: STATE interfaces it reads. Reported, never arbitrated.
    reads: tuple[str, ...] = ()
    state: str = ACTIVE
    namespace: str = ""
    #: The entity this controller belongs to. roqsim allows two arms in ONE namespace, each with a
    #: controller of the same name -- a shape real ros2_control cannot have, and which
    #: `roqsim export moveit` already refuses with advice on how to name them. The registry scopes
    #: by owner so it describes that world rather than refusing it a second time, in worse words.
    owner: str = ""
    #: Called with the new activity when this controller is switched. The controller keeps its own
    #: flag; this registry keeps the lifecycle state and the two are set together.
    apply: Callable[[bool], None] | None = None
    #: Appended with ``(sim_time, previous_state, new_state)`` on every transition, so a run can say
    #: WHEN a hand-over happened rather than inferring it from when the motion changed.
    transitions: list[tuple[float, str, str]] = field(default_factory=list)

    @property
    def claimed_interfaces(self) -> tuple[str, ...]:
        """What ros2_control reports: the claims, and only while active."""
        return self.claims if self.state == ACTIVE else ()


class ControllerRegistry:
    """Every controller in one world, and the switching between them."""

    def __init__(self) -> None:
        # Keyed by (namespace, name): a robot's controllers are scoped to its own manager, so two
        # arms may each have an `arm_controller` exactly as two real robots do -- they answer at
        # /left/controller_manager and /right/controller_manager and never meet. The claims are
        # scoped the same way, which matters more: joint names are unprefixed here, so two arms
        # both claim `shoulder_pan_joint/position` and only a per-robot scope keeps them apart.
        self._by_name: dict[tuple[str, str], Controller] = {}

    # -- population ------------------------------------------------------------------------------

    def register(self, controller: Controller) -> Controller:
        key = (controller.namespace, controller.owner, controller.name)
        existing = self._by_name.get(key)
        if existing is not None and existing is not controller:
            raise RuntimeError(
                f"{controller.owner or 'this robot'} has two controllers both called "
                f"{controller.name!r}. A controller's name is how a scenario and MoveIt address it, "
                f"so it has to be unique -- give one of them a `controller_name:`."
            )
        self._by_name[key] = controller
        return controller

    def all(self, namespace: str | None = None) -> list[Controller]:
        found = [c for (ns, *_), c in self._by_name.items() if namespace is None or ns == namespace]
        return sorted(found, key=lambda c: (c.namespace, c.owner, c.name))

    def get(self, name: str, namespace: str = "") -> Controller | None:
        for controller in self.all(namespace):
            if controller.name == name:
                return controller
        return None

    # -- what is held ----------------------------------------------------------------------------

    def claimed(self, namespace: str = "") -> dict[str, str]:
        """Command interface -> the active controller of this robot holding it."""
        held: dict[str, str] = {}
        for controller in self.all(namespace):
            for interface in controller.claimed_interfaces:
                held[interface] = controller.name
        return held

    def blockers(self, name: str, namespace: str = "") -> list[str]:
        """Active controllers OF THE SAME ROBOT whose claims collide with this one's."""
        this = self.get(name, namespace)
        wanted = set(this.claims) if this is not None else set()
        return [
            other.name
            for other in self.all(namespace)
            if other.name != name
            and other.owner == (this.owner if this is not None else "")
            and wanted.intersection(other.claimed_interfaces)
        ]

    # -- switching -------------------------------------------------------------------------------

    def switch(
        self,
        activate: Iterable[str] = (),
        deactivate: Iterable[str] = (),
        strictness: int = BEST_EFFORT,
        *,
        namespace: str = "",
        sim_time: float = 0.0,
    ) -> tuple[bool, str]:
        """Activate and deactivate, atomically, with ros2_control's own strictness rules.

        ``STRICT`` changes nothing unless everything asked for is possible. ``BEST_EFFORT`` does
        what it can and skips the rest. **Neither deactivates a controller the caller did not
        name** -- a switch is one request carrying both lists, which is why a hand-over names the
        outgoing controller as well as the incoming one. ``FORCE_AUTO`` is the exception and says
        so: it deactivates whatever blocks an activation, the mutually-exclusive-interface rule
        ros2_control documents on that constant.
        """
        if strictness not in (BEST_EFFORT, STRICT, AUTO, FORCE_AUTO):
            # Including 0, which is what a default-constructed request carries.
            strictness = BEST_EFFORT

        activate, deactivate = list(dict.fromkeys(activate)), list(dict.fromkeys(deactivate))

        def known(n: str) -> bool:
            return self.get(n, namespace) is not None

        problems = [n for n in (*activate, *deactivate) if not known(n)]
        plan_off = {n for n in deactivate if known(n)}
        plan_on: list[str] = []

        for name in activate:
            if not known(name):
                continue
            held = [b for b in self.blockers(name, namespace) if b not in plan_off]
            if held and strictness == FORCE_AUTO:
                plan_off.update(held)  # the rule that constant exists for
                held = []
            if held:
                problems.append(f"{name} needs interfaces held by {', '.join(sorted(held))}")
                continue
            plan_on.append(name)

        if problems and strictness == STRICT:
            return False, "; ".join(problems) + " (strict: nothing was changed)"

        for name in sorted(plan_off):
            self._set(self.get(name, namespace), INACTIVE, sim_time)
        for name in plan_on:
            self._set(self.get(name, namespace), ACTIVE, sim_time)

        if problems:
            return (strictness != STRICT), "; ".join(problems)
        return True, "ok"

    def _set(self, controller: Controller, state: str, sim_time: float) -> None:
        if controller.state == state:
            return
        controller.transitions.append((float(sim_time), controller.state, state))
        controller.state = state
        if controller.apply is not None:
            controller.apply(state == ACTIVE)


def registry_for(ctx) -> ControllerRegistry:
    """The world's registry, made by whoever asks first.

    Lazily rather than by a plugin the world has to remember to list: on real hardware a
    controller_manager is there because ros2_control is, and nobody opts into one.
    """
    registry = ctx.blackboard.get(SERVICE_KEY)
    if registry is None:
        registry = ControllerRegistry()
        ctx.blackboard.set(SERVICE_KEY, registry)
    return registry
