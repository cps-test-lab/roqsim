"""The ``field`` backend hint: publishing one member of a structured payload as a primitive message.

Guards the gap this closed. ``contact_monitor`` has always declared
``{"ros2": {"type": "std_msgs.msg.Bool", "topic": "collision"}}``, but nothing had ever bridged it,
and the reflective fallback would have assigned its ``ContactReport`` dataclass straight to a
``bool`` field -- a run with a live collision monitor and no ``/collision`` publisher, which reads
as "nothing was hit". Naming the field on the endpoint keeps the meaning with the producer, so no
primitive type needs a converter in the registry that knows one plugin's attribute names.
"""

from dataclasses import dataclass

import pytest

from roqsim_ros_bridge.registry import get_converter


@dataclass
class _Report:
    """Stand-in for contact_monitor's ContactReport: rich payload, one publishable scalar."""

    in_contact: bool
    first_time: float


class _BoolMsg:
    def __init__(self):
        self.data = False


def _fill(payload, hints):
    msg = _BoolMsg()
    # std_msgs.msg.Bool has no registered converter, so this is the reflective path under test.
    get_converter("std_msgs.msg.Bool")(msg, payload, None, hints)
    return msg.data


def test_field_hint_selects_the_member():
    assert _fill(_Report(True, 1.25), {"field": "in_contact"}) is True
    assert _fill(_Report(False, -1.0), {"field": "in_contact"}) is False


def test_scalar_payload_still_passes_through_unnamed():
    """The pre-existing behaviour: no hint, payload IS the value."""
    assert _fill(True, {}) is True


def test_unknown_field_names_itself():
    """A typo must not publish the whole payload and fail one layer down."""
    with pytest.raises(TypeError, match="field='in_contct'"):
        _fill(_Report(True, 1.25), {"field": "in_contct"})
