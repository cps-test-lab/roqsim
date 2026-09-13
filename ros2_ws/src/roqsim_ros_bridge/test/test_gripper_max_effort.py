"""Which GripperCommand goals may carry a ``max_effort``, and which are aborted before anything moves.

A goal asking for a grip force its producer cannot apply must fail rather than run: executed with the
producer's own force instead, it grips harder or softer than it asked for and still reports success.
The policy is pure (`_max_effort_refusal`) so it is tested without an action server.
"""

from __future__ import annotations

from roqsim_ros_bridge.actions import _max_effort_refusal


class _Effort:
    """Stand-in for a producer's force entry: the one field this policy reads."""

    def __init__(self, rated: float) -> None:
        self.rated = rated


def test_no_force_asked_is_always_honoured():
    assert _max_effort_refusal(0.0, None) == ""
    assert _max_effort_refusal(0.0, _Effort(0.0)) == ""
    assert _max_effort_refusal(-1.0, _Effort(100.0)) == ""


def test_a_position_only_producer_refuses_a_force():
    """A door, or anything without a force entry, takes a position and nothing else."""
    assert "position only" in _max_effort_refusal(10.0, None)


def test_a_gripper_without_a_rating_refuses_a_force():
    assert "no force rating" in _max_effort_refusal(10.0, _Effort(0.0))


def test_a_force_within_the_rating_is_honoured():
    assert _max_effort_refusal(100.0, _Effort(100.0)) == ""
    assert _max_effort_refusal(20.0, _Effort(100.0)) == ""


def test_a_force_above_the_rating_is_refused_not_clamped():
    assert "exceeds" in _max_effort_refusal(100.5, _Effort(100.0))
