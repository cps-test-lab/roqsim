"""The ``diagnostic_msgs/DiagnosticStatus`` converter: a whole report as named readings.

The ``field`` hint publishes one member of a structured payload as a primitive message, which is all
a producer needs whose report reduces to a number. One that does not needs the rest to leave the
process too: ``clearance_monitor`` measures a distance, what the distance was to, and whether it is a
measurement or the query's cutoff, and neither of the last two can be reconstructed by reducing a
recorded series of the first. This is the door for those, and it stays ignorant of any producer's
attribute names -- the endpoint's ``fields`` hint says which, keyed under their own names.
"""

from dataclasses import dataclass

import pytest

from roqsim_ros_bridge.registry import get_converter, to_time_msg


@dataclass
class _Report:
    """Stand-in for roqsim.plugins.clearance_monitor.ClearanceReport."""

    current: float = 1.75
    minimum: float = 0.125
    at_time: float = 12.5
    geom: str = "post_geom"
    saturated: bool = False


_FIELDS = ["current", "minimum", "at_time", "geom", "saturated"]


def _fill(payload, hints=None):
    from diagnostic_msgs.msg import DiagnosticStatus

    msg = DiagnosticStatus()
    get_converter("diagnostic_msgs.msg.DiagnosticStatus")(
        msg, payload, to_time_msg(12.5), {"fields": _FIELDS, **(hints or {})}
    )
    return msg


def _readings(msg) -> dict:
    return {entry.key: entry.value for entry in msg.values}


def test_every_named_field_is_published_under_its_own_name():
    msg = _fill(_Report())
    assert [entry.key for entry in msg.values] == _FIELDS
    assert _readings(msg)["geom"] == "post_geom"


def test_a_number_reads_back_as_the_number_it_was():
    """A recorded reading is compared against a threshold, so it round-trips rather than
    being rounded into a value the trial never measured."""
    msg = _fill(_Report(current=0.1234567890123, minimum=1e-9))
    readings = _readings(msg)
    assert float(readings["current"]) == pytest.approx(0.1234567890123, rel=1e-15)
    assert float(readings["minimum"]) == pytest.approx(1e-9, rel=1e-15)
    assert float(readings["at_time"]) == pytest.approx(12.5)


def test_a_flag_is_the_spelling_a_reader_parses():
    assert _readings(_fill(_Report(saturated=True)))["saturated"] == "true"
    assert _readings(_fill(_Report(saturated=False)))["saturated"] == "false"


def test_a_cutoff_that_was_never_resolved_says_so():
    """What the monitor reports when nothing came inside the query's range: a flagged reading
    that names nothing, rather than a plausible distance to an unnamed thing."""
    msg = _fill(_Report(current=float("inf"), minimum=float("inf"), geom="", saturated=True))
    readings = _readings(msg)
    assert readings["geom"] == ""
    assert readings["saturated"] == "true"
    assert float(readings["current"]) == float("inf")


def test_the_status_carries_who_is_reporting_and_about_what():
    msg = _fill(_Report(), {"name": "clearance_monitor: robot.clearance", "hardware_id": "base"})
    assert msg.name == "clearance_monitor: robot.clearance"
    assert msg.hardware_id == "base"


def test_the_level_is_ok_because_a_reading_is_not_a_verdict():
    """A level above OK would be a threshold on the reading, and what counts as too close is
    the experiment's to state."""
    from diagnostic_msgs.msg import DiagnosticStatus

    assert _fill(_Report(minimum=0.0, saturated=False)).level == DiagnosticStatus.OK


def test_a_report_with_no_fields_named_is_refused():
    """Loudly: the message would otherwise publish a status with no readings in it, which in a
    recorded table is indistinguishable from a trial that measured nothing."""
    from diagnostic_msgs.msg import DiagnosticStatus

    with pytest.raises(TypeError, match="fields"):
        get_converter("diagnostic_msgs.msg.DiagnosticStatus")(
            DiagnosticStatus(), _Report(), to_time_msg(1.0), {}
        )


def test_an_unknown_field_names_itself():
    """A typo must name the field and the payload, not fail one layer down."""
    with pytest.raises(TypeError, match="'minimun'"):
        _fill(_Report(), {"fields": ["current", "minimun"]})
