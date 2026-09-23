# Copyright (C) 2026 Frederik Pasch
# SPDX-License-Identifier: Apache-2.0

"""``controller_manager_msgs`` over roqsim's controller registry.

Declared by hand on the node rather than through an endpoint, for the reason ``services.py`` gives
for ``sim_interfaces``: this is a standardised message set under one node namespace, and several of
its services are QUERIES with no producer behind them -- which the endpoint model, whose every entry
is a read or a write on one plugin, has no direction for.

No world entry creates this. On real hardware nobody opts into a controller_manager; it is there
because ros2_control is. A robot that registered controllers gets the surface, and one that has
none gets nothing to talk to, which is the same answer.

**The world file is the parameter file.** ``load_controller`` moves a DECLARED controller to
``inactive`` -- it cannot conjure one, both because a controller that publishes cannot add endpoints
after the bridge has bound them and because that is what the real service does: a name absent from
the manager's parameters fails there too.
"""

from __future__ import annotations

import logging

from controller_manager_msgs.msg import ControllerState, HardwareInterface
from controller_manager_msgs.srv import (
    ConfigureController,
    ListControllers,
    ListHardwareInterfaces,
    LoadController,
    SwitchController,
    UnloadController,
)
from lifecycle_msgs.msg import TransitionEvent
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile

from roqsim.controllers import ACTIVE, INACTIVE, UNCONFIGURED, registry_for

from .physics import run_on_physics

_log = logging.getLogger(__name__)


def _join(namespace: str, name: str) -> str:
    return f"/{namespace}/{name}" if namespace else f"/{name}"


class ControllerManagerServices:
    """One controller_manager, for the controllers sharing one namespace."""

    def __init__(self, node, ctx, namespace: str = "") -> None:
        self._ctx = ctx
        self._ns = namespace
        self._registry = registry_for(ctx)
        base = _join(namespace, "controller_manager")

        node.create_service(ListControllers, f"{base}/list_controllers", self._list)
        node.create_service(SwitchController, f"{base}/switch_controller", self._switch)
        node.create_service(LoadController, f"{base}/load_controller", self._load)
        node.create_service(ConfigureController, f"{base}/configure_controller", self._configure)
        node.create_service(UnloadController, f"{base}/unload_controller", self._unload)
        node.create_service(
            ListHardwareInterfaces, f"{base}/list_hardware_interfaces", self._interfaces
        )

        # Every controller announces its own transitions, as a real one does. This is also the only
        # place a run can read the INSTANT of a hand-over rather than inferring it from when the
        # motion changed -- which is usually the thing being measured.
        # Latched, and this is the whole reason the topic is worth having. A transition is a fact
        # about a moment that has already passed, and everything that wants it subscribes
        # afterwards: a trial that switches and then asks when the hand-over happened, a recorder
        # attached after the run began, a scenario waiting on the event after its service call
        # returned. Volatile, every one of those receives nothing -- the topic is there, the
        # publisher is there, and the record is silently unreadable.
        history = QoSProfile(
            depth=20,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
        )
        self._events = {
            c.name: node.create_publisher(
                TransitionEvent, f"{_join(c.namespace, c.name)}/transition_event", history
            )
            for c in self._mine()
        }
        self._announced = {c.name: len(c.transitions) for c in self._mine()}

    def _mine(self):
        return self._registry.all(self._ns)

    # -- queries ---------------------------------------------------------------------------------

    def _list(self, request, response):
        for controller in self._mine():
            state = ControllerState()
            state.name = controller.name
            state.type = controller.type
            state.state = controller.state
            # Only while active, and only command interfaces: the upstream message says as much in
            # its own comment. `required_command_interfaces` is the field that is populated
            # regardless of state, and is where a consumer looks to see what a controller WOULD take.
            state.claimed_interfaces = list(controller.claimed_interfaces)
            state.required_command_interfaces = list(controller.claims)
            state.required_state_interfaces = list(controller.reads)
            state.is_chainable = False
            state.is_chained = False
            response.controller.append(state)
        return response

    def _interfaces(self, request, response):
        held = self._registry.claimed(self._ns)
        seen: dict[str, bool] = {}
        for controller in self._mine():
            for name in controller.claims:
                seen[name] = name in held
        for name, claimed in sorted(seen.items()):
            entry = HardwareInterface()
            entry.name = name
            entry.data_type = "double"
            # Available because the simulated joint is always there -- `is_available` reports
            # whether the HARDWARE exports the interface, which has no way to be false here, and
            # reporting it false would read as a robot that lost an interface.
            entry.is_available = True
            entry.is_claimed = claimed
            response.command_interfaces.append(entry)
        for name in sorted({n for c in self._mine() for n in c.reads}):
            entry = HardwareInterface()
            entry.name = name
            entry.data_type = "double"
            entry.is_available = True
            # State interfaces are shared and claimed by nobody, which is the whole reason a
            # broadcaster can read a joint another controller is driving.
            entry.is_claimed = False
            response.state_interfaces.append(entry)
        return response

    # -- lifecycle -------------------------------------------------------------------------------

    def _load(self, request, response):
        """Move a declared controller to ``inactive``; refuse a name the world never declared."""
        controller = self._registry.get(request.name, self._ns)
        response.ok = controller is not None
        if controller is not None and controller.state == UNCONFIGURED:
            controller.state = INACTIVE
        return response

    def _configure(self, request, response):
        controller = self._registry.get(request.name, self._ns)
        response.ok = controller is not None
        if controller is not None and controller.state == UNCONFIGURED:
            controller.state = INACTIVE
        return response

    def _unload(self, request, response):
        """An active controller is never unloaded, exactly as upstream refuses it."""
        controller = self._registry.get(request.name, self._ns)
        response.ok = controller is not None and controller.state != ACTIVE
        return response

    def _switch(self, request, response):
        """Switch at a physics step, and reply only once it has happened.

        A switch that returned before it took effect would let a scenario command the incoming
        controller in the window where the outgoing one still held the joints.
        """
        activate = list(request.activate_controllers)
        deactivate = list(request.deactivate_controllers)
        outcome: dict = {}

        def apply(ctx) -> None:
            outcome["result"] = self._registry.switch(
                activate=activate,
                deactivate=deactivate,
                strictness=int(request.strictness),
                namespace=self._ns,
                sim_time=ctx.sim_time,
            )

        if not run_on_physics(self._ctx, apply):
            response.ok = False
            if hasattr(response, "message"):
                response.message = "the simulation did not apply the switch (is it paused?)"
            return response

        ok, message = outcome.get("result", (False, "the switch did not run"))
        response.ok = bool(ok)
        if hasattr(response, "message"):
            response.message = message
        self._announce()
        return response

    def _announce(self) -> None:
        """Publish whatever transitions the registry recorded but nobody has announced yet."""
        for controller in self._mine():
            publisher = self._events.get(controller.name)
            seen = self._announced.get(controller.name, 0)
            if publisher is None:
                continue
            for stamp, previous, new in controller.transitions[seen:]:
                event = TransitionEvent()
                # SIM time, in nanoseconds. This is the reason the topic is worth publishing: a run
                # that has to infer the hand-over instant from when the motion changed is inferring
                # it from the thing it is usually trying to measure.
                event.timestamp = int(stamp * 1e9)
                event.goal_state.label = new
                event.start_state.label = previous
                publisher.publish(event)
            self._announced[controller.name] = len(controller.transitions)


def serve(node, ctx) -> list[ControllerManagerServices]:
    """One manager per namespace, as a multi-robot ros2_control deployment has."""
    registry = registry_for(ctx)
    namespaces = sorted({c.namespace for c in registry.all()})
    if not namespaces:
        return []
    return [ControllerManagerServices(node, ctx, ns) for ns in namespaces]
