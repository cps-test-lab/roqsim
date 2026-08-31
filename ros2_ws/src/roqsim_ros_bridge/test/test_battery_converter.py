"""The ``sensor_msgs/BatteryState`` converter: unknown is NaN, and a discharge is negative.

Both are conventions of the message rather than choices, and both fail quietly if got wrong: a zero
voltage reads as a dead pack, a `percentage` of 100 reads as a full one, and a positive current on a
discharging robot plots as charging.
"""

from dataclasses import dataclass
from math import isnan

import pytest

from roqsim_ros_bridge.registry import get_converter, to_time_msg


@dataclass
class _Report:
    """Stand-in for roqsim.plugins.energy_monitor.EnergyReport."""

    energy_j: float = 900.0
    power_w: float = 30.0
    mechanical_w: float = 30.0
    charge_fraction: float = -1.0
    depleted: bool = False
    voltage: float = 0.0
    current_a: float = 0.0
    capacity_wh: float = 0.0


def _fill(payload, hints=None):
    from sensor_msgs.msg import BatteryState

    msg = BatteryState()
    get_converter("sensor_msgs.msg.BatteryState")(msg, payload, to_time_msg(3.0), hints or {})
    return msg


def test_a_monitor_with_no_battery_configured_reports_unknown_not_full():
    msg = _fill(_Report())
    assert isnan(msg.percentage) and isnan(msg.voltage) and isnan(msg.capacity)
    assert msg.present is True


def test_a_configured_pack_reports_charge_and_a_negative_current():
    msg = _fill(_Report(charge_fraction=0.75, voltage=24.0, current_a=1.25, capacity_wh=96.0))
    assert msg.percentage == pytest.approx(0.75)
    assert msg.voltage == pytest.approx(24.0)
    # Negative: current is leaving the pack. A consumer plots a drain, not a charge.
    assert msg.current == pytest.approx(-1.25)
    assert msg.capacity == pytest.approx(4.0)  # 96 Wh at 24 V
    assert msg.charge == pytest.approx(3.0)  # 75% of it


def test_depletion_shows_in_the_status_fields_a_consumer_reads():
    from sensor_msgs.msg import BatteryState

    empty = _fill(_Report(charge_fraction=0.0, voltage=24.0, capacity_wh=96.0, depleted=True))
    assert empty.power_supply_status == BatteryState.POWER_SUPPLY_STATUS_NOT_CHARGING
    assert empty.power_supply_health == BatteryState.POWER_SUPPLY_HEALTH_DEAD
    alive = _fill(_Report(charge_fraction=0.5, voltage=24.0, capacity_wh=96.0))
    assert alive.power_supply_status == BatteryState.POWER_SUPPLY_STATUS_DISCHARGING


def test_the_frame_is_namespaced_like_every_other_endpoint():
    msg = _fill(_Report(), {"frame_id": "base_link", "frame_prefix": "robot_b"})
    assert msg.header.frame_id == "robot_b/base_link"
    assert msg.header.stamp.sec == 3
