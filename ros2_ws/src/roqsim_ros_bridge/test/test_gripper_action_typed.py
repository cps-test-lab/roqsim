"""A GripperCommand goal reaches a typed command as its ``position`` parameter.

The handler serves a hand-built gripper endpoint with the bare position, and an endpoint declared
with :mod:`roqsim.endpoint` (a door's ``door``) with the named parameter it declares.
"""

from __future__ import annotations

import threading

from control_msgs.action import GripperCommand

import roqsim  # noqa: F401, I001
from roqsim import endpoint
from roqsim.context import SimContext
from roqsim.endpoint import build
from roqsim.plugin import Plugin
from roqsim_ros_bridge.actions import gripper_command

DT = 0.005


class _Clock:
    def __init__(self) -> None:
        self.time = 0.0


class _Leaf(Plugin):
    """A 1-DOF producer declaring its command the way ``door`` does, and reaching it at once."""

    def __init__(self) -> None:
        super().__init__({}, label="door")
        self.position = 0.0
        self.commands: list[float] = []

    @endpoint.command(
        "door",
        ros2={"action": "control_msgs.action.GripperCommand", "state_key": "door:door:state"},
    )
    def command_door(self, position: float) -> None:
        self.position = position
        self.commands.append(position)

    def read_state(self):
        return (self.position, 0.0)


class _GoalHandle:
    def __init__(self, position: float) -> None:
        self.request = GripperCommand.Goal()
        self.request.command.position = position
        self.status = "executing"
        self.is_cancel_requested = False

    def succeed(self) -> None:
        self.status = "succeeded"

    def abort(self) -> None:
        self.status = "aborted"

    def publish_feedback(self, feedback) -> None:
        pass


def test_a_typed_command_gets_the_goal_position_by_name():
    ctx = SimContext(config={})
    ctx.data = _Clock()
    leaf = _Leaf()
    (ep,) = build(leaf, ctx)
    ctx.blackboard.set("door:door:state", leaf.read_state)
    stop = threading.Event()

    def physics():
        while not stop.is_set():
            ctx.drain_commands()
            ctx.data.time += DT
            stop.wait(0.001)

    thread = threading.Thread(target=physics, daemon=True)
    thread.start()
    try:
        handle = _GoalHandle(0.7)
        # A marshalled endpoint is handed to the handler as its own write (BridgeBase._inbound).
        result = gripper_command(handle, ctx, ep.write, ep)
    finally:
        stop.set()
        thread.join(timeout=2.0)

    assert leaf.commands == [0.7]
    assert handle.status == "succeeded" and result.reached_goal
