# Copyright (C) 2026 Frederik Pasch
# SPDX-License-Identifier: Apache-2.0

"""What a ROS call says while it waits, and the fact that saying it changes nothing.

A call whose service is not advertised polls ``None`` for as long as that lasts. That is the right
control flow -- **when to stop waiting belongs to the scenario**, to its own ``timeout()`` and the
shape of its tree, and a deadline in here would take the decision away from the author and hide it
in a library constant. The stack and the simulator come up concurrently, and how long that is
allowed to take is the author's to say.

What was missing is not a deadline but an explanation. A run that ended on its timeout said only
that the action was waiting, so a name gone out of sync between a plugin and this package -- a
renamed blackboard key, a namespace the scenario did not expect -- arrived as a slow robot. These
tests pin the reason each call gives, and that giving it does not make anything fail sooner.

ROS-free, like the calls themselves: they are duck-typed over a client, so the interesting
mistakes are catchable without a ROS installation.
"""

from __future__ import annotations

import pytest

from scenario_execution_roqsim.access import ros as ros_access


class _NeverReady:
    """A client for a name the graph does not carry."""

    def service_is_ready(self) -> bool:
        return False

    def server_is_ready(self) -> bool:
        return False


def _offered(*names):
    return lambda: tuple(names)


@pytest.mark.parametrize("call,expected", [
    (
        lambda: ros_access._RosCall(
            _NeverReady(), object(), "grip_fault", True, _offered("/other/override")),
        "grip_fault/override",
    ),
    (
        lambda: ros_access._RosSpawn(
            _NeverReady(), object(), "robot", 1, "spawn_entity", _offered("/other/override")),
        "spawn_entity",
    ),
    (
        lambda: ros_access._RosTeleport(
            _NeverReady(), object(), "robot", 1, _offered("/other/override")),
        "set_entity_state",
    ),
    (
        lambda: ros_access._RosRoute(
            _NeverReady(), object(), "robot/navigate_through_poses", wait=True,
            offered=_offered("/other/navigate_to_pose")),
        "robot/navigate_through_poses",
    ),
])
def test_every_call_says_which_name_is_missing(call, expected):
    """A reason that does not name the thing is a reason nobody can act on.

    All four had to be checked because all four wait the same way, and two of them said nothing
    at all -- so a nav goal or a fault injection that never landed produced a timeout with an
    empty explanation.
    """
    reason = call().pending_reason()
    assert reason and expected in reason


def test_the_reason_lists_what_the_graph_does_carry():
    """The name that is missing says which name; the names present say what went wrong.

    The counterpart of the in-process refusal's "this world offers:" -- most of these are a
    prefix or a namespace that does not match, and seeing the real ones is what shows it.
    """
    call = ros_access._RosCall(
        _NeverReady(), object(), "grip_fault", True,
        _offered("/cell1/grip_fault/override", "/cell1/robot/spawn_entity"),
    )
    reason = call.pending_reason()

    assert "the graph carries" in reason
    assert "/cell1/grip_fault/override" in reason


def test_a_graph_query_that_fails_does_not_become_the_failure():
    """The listing is a nicety; the wait and the name are the message.

    A discovery call can fail for reasons that have nothing to do with the scenario -- a daemon
    restarting, a graph in flux. Letting that surface would replace a useful reason with an
    unrelated traceback, at the exact moment somebody is trying to read the reason.
    """
    class _Access(ros_access.RosAccess):
        def __init__(self):  # noqa: D107 - no ROS node wanted here
            pass

    def _boom():
        raise RuntimeError("no graph here")

    assert _Access()._advertised(_boom, "_svc_cache") == ()


def test_saying_it_does_not_make_the_call_give_up():
    """The load-bearing negative: the scenario keeps the timeout.

    Polled a thousand times with the service still missing, the call is still waiting. A library
    that decided this for itself would fail a scenario that was legitimately waiting out a slow
    bring-up, from a constant its author never chose and cannot see.
    """
    call = ros_access._RosCall(_NeverReady(), object(), "grip_fault", True, _offered())
    assert all(call.poll() is None for _ in range(1000))
    assert call.pending_reason(), "and it is still explaining itself"


def test_the_graph_listing_is_cached_rather_than_asked_every_tick():
    """A reason is rebuilt on every RUNNING tick, and a discovery round-trip per tick is a cost
    paid all through a run for a string read once, at the end."""
    class _Access(ros_access.RosAccess):
        def __init__(self):  # noqa: D107 - no ROS node wanted here
            self.calls = 0

        def _fetch(self):
            self.calls += 1
            return [("/a", ["T"]), ("/b", ["T"])]

    access = _Access()
    for _ in range(50):
        names = access._advertised(access._fetch, "_svc_cache")

    assert names == ("/a", "/b")
    assert access.calls == 1, "one listing, reused"
