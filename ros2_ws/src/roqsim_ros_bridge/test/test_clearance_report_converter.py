"""The ``diagnostic_msgs/DiagnosticStatus`` converter: a whole report as named readings.

A primitive message carries one member of a structured payload, which is all a producer needs whose
report reduces to a number. One that does not needs the rest to leave the process too:
``clearance_monitor`` measures a distance, what the distance was to, and whether it is a measurement
or the query's cutoff, and neither of the last two can be reconstructed by reducing a recorded series
of the first. This is the door for those, and it knows no producer's field names -- every field of
the payload dataclass is one reading, keyed under its own name.
"""

from dataclasses import dataclass

import pytest

pytest.importorskip("roqsim")  # selects the GL backend before mujoco is imported
pytest.importorskip("rclpy")

from roqsim_ros_bridge import typemap  # noqa: E402
from roqsim_ros_bridge.registry import get_converter, to_time_msg  # noqa: E402

_FIELDS = ["current", "minimum", "at_time", "geom", "saturated"]


@dataclass
class _Report:
    """Stand-in for roqsim.plugins.clearance_monitor.ClearanceReport."""

    current: float = 1.75
    minimum: float = 0.125
    at_time: float = 12.5
    geom: str = "post_geom"
    saturated: bool = False


def _fill(payload, hints=None):
    from diagnostic_msgs.msg import DiagnosticStatus

    msg = DiagnosticStatus()
    get_converter("diagnostic_msgs.msg.DiagnosticStatus")(
        msg, payload, to_time_msg(12.5), dict(hints or {})
    )
    return msg


def _readings(msg) -> dict:
    return {entry.key: entry.value for entry in msg.values}


def test_every_field_is_published_under_its_own_name_in_declaration_order():
    msg = _fill(_Report())
    assert [entry.key for entry in msg.values] == _FIELDS
    assert _readings(msg)["geom"] == "post_geom"


def test_a_number_reads_back_as_the_number_it_was():
    """A recorded reading is compared against a threshold, so it round-trips rather than
    being rounded into a value the trial never measured."""
    msg = _fill(_Report(current=0.1234567890123, minimum=1e-9, at_time=float("inf")))
    readings = _readings(msg)
    assert float(readings["current"]) == pytest.approx(0.1234567890123, rel=1e-15)
    assert float(readings["minimum"]) == pytest.approx(1e-9, rel=1e-15)
    assert float(readings["at_time"]) == float("inf")


def test_a_flag_is_the_spelling_a_reader_parses():
    assert _readings(_fill(_Report(saturated=True)))["saturated"] == "true"
    assert _readings(_fill(_Report(saturated=False)))["saturated"] == "false"


def test_a_cutoff_that_was_never_resolved_says_so():
    """What the monitor reports when nothing came inside the query's range: a flagged reading
    that names nothing, rather than a plausible distance to an unnamed thing."""
    msg = _fill(_Report(current=3.0, minimum=3.0, geom="", saturated=True))
    readings = _readings(msg)
    assert readings["geom"] == ""
    assert readings["saturated"] == "true"
    assert float(readings["current"]) == float(readings["minimum"]) == 3.0


def test_the_status_carries_who_is_reporting_and_about_what():
    msg = _fill(_Report(), {"name": "clearance_monitor: robot.clearance", "hardware_id": "base"})
    assert msg.name == "clearance_monitor: robot.clearance"
    assert msg.hardware_id == "base"


def test_the_level_is_ok_because_a_reading_is_not_a_verdict():
    """A level above OK would be a threshold on the reading, and what counts as too close is
    the experiment's to state."""
    from diagnostic_msgs.msg import DiagnosticStatus

    assert _fill(_Report(minimum=0.0, saturated=False)).level == DiagnosticStatus.OK


def test_a_payload_that_is_not_a_dataclass_is_refused():
    """Loudly: a status with no readings in it is, in a recorded table, indistinguishable from a
    trial that measured nothing."""
    with pytest.raises(TypeError, match="dataclass"):
        _fill((1.0, 0.5))


def test_the_clearance_report_travels_as_a_diagnostic_status():
    """The report's row in the type table, which the ``clearance_report`` endpoint binds through."""
    from diagnostic_msgs.msg import DiagnosticStatus

    from roqsim.plugins.clearance_monitor import ClearanceReport

    (wire,) = typemap.lookup(ClearanceReport).wires
    msg = DiagnosticStatus()
    wire.fill(msg, ClearanceReport(1.5, 0.25, 3.0, "post_geom", False), to_time_msg(3.0), {})
    assert wire.msg == "diagnostic_msgs.msg.DiagnosticStatus"
    assert _readings(msg) == {
        "current": "1.5",
        "minimum": "0.25",
        "at_time": "3.0",
        "geom": "post_geom",
        "saturated": "false",
    }
