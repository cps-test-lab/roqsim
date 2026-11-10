"""Where a goal's single value goes: to the producer's own parameter. Tested without ROS.

The GripperCommand handler sends a goal's ``position`` to whatever drives one position, and a
producer names its parameter for its own quantity -- a door's command is ``openness_goal(openness)``
-- so the value goes under that name, and under ``position`` only where the producer declares
several parameters or none.
"""

from __future__ import annotations

from roqsim import endpoint
from roqsim.context import SimContext
from roqsim.endpoint import build
from roqsim.plugin import Plugin
from roqsim_ros_bridge.params import one_value, payload_for


class _Door(Plugin):
    @endpoint.command
    def openness_goal(self, openness: float) -> None:
        """Move to an openness.

        Args:
            openness: 0 closed to 1 fully open
        """


class _Two(Plugin):
    @endpoint.command
    def grip(self, position: float, force: float) -> None:
        """Grip.

        Args:
            position: where
            force: how hard
        """


class _Untyped:
    params = None


def _endpoint(plugin: Plugin):
    (ep,) = build(plugin, SimContext(config={}))
    return ep


def test_a_door_takes_the_goal_as_its_openness():
    ep = _endpoint(_Door())
    assert one_value(ep, "position", 0.4) == {"openness": 0.4}
    assert payload_for(ep, one_value(ep, "position", 0.4)) == {"openness": 0.4}


def test_a_producer_with_several_parameters_takes_it_by_the_goals_name():
    assert one_value(_endpoint(_Two()), "position", 0.4) == {"position": 0.4}


def test_an_untyped_producer_takes_the_bare_value():
    assert payload_for(_Untyped(), one_value(_Untyped(), "position", 0.4)) == 0.4
