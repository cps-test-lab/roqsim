"""The core plugins' endpoints, as their classes declare them: names, kinds, payloads and units."""

from __future__ import annotations

import pytest

from roqsim import endpoint
from roqsim.plugins.bumper import BumperPlugin
from roqsim.plugins.clearance_monitor import ClearanceMonitorPlugin
from roqsim.plugins.contact_impulse import ContactImpulsePlugin
from roqsim.plugins.contact_location import ContactLocationPlugin
from roqsim.plugins.contact_monitor import ContactMonitorPlugin
from roqsim.plugins.energy_monitor import EnergyMonitorPlugin
from roqsim.plugins.joint_state_publisher import JointStatePublisherPlugin
from roqsim.plugins.model_override import ModelOverridePlugin
from roqsim.plugins.spawn_model import SpawnModelPlugin

#: plugin class -> {endpoint name: (kind, payload type)}
EXPECTED = {
    BumperPlugin: {"bumper/{item}": ("out", "bool")},
    ClearanceMonitorPlugin: {"clearance": ("out", "ClearanceReport")},
    ContactImpulsePlugin: {"contact_impulse": ("out", "ContactImpulseReport")},
    ContactLocationPlugin: {"contact_location": ("out", "ContactLocation")},
    ContactMonitorPlugin: {"contact": ("out", "ContactReport")},
    EnergyMonitorPlugin: {"battery": ("out", "EnergyReport")},
    JointStatePublisherPlugin: {"joint_states": ("out", "JointState")},
    ModelOverridePlugin: {
        "override": ("command", None),
        "override_state": ("out", "OverrideReport"),
        "override_verified": ("out", "OverrideReport"),
    },
}


def _described(cls) -> dict[str, dict]:
    return {row["name"]: row for row in (s.describe(cls) for s in endpoint.declared(cls))}


@pytest.mark.parametrize("cls", list(EXPECTED), ids=lambda c: c.__name__)
def test_each_endpoint_is_declared_with_its_kind_payload_and_doc(cls):
    rows = _described(cls)
    assert set(rows) == set(EXPECTED[cls])
    for name, (kind, payload) in EXPECTED[cls].items():
        row = rows[name]
        assert row["kind"] == kind, name
        assert row.get("payload") == payload, name
        assert row["doc"], f"{cls.__name__}.{name} has no docstring for `plugins describe`"
        if kind == "out":
            assert row["rate_hz"] == {"from": "rate_hz"}, name


def _units(row) -> dict[str, str]:
    return {f["name"]: f.get("unit", "") for f in row["result"].get("fields", [])}


def test_the_reports_state_their_units():
    assert _units(_described(ClearanceMonitorPlugin)["clearance"]) == {
        "current": "m",
        "minimum": "m",
        "at_time": "s",
        "geom": "",
        "saturated": "",
    }
    impulse = _units(_described(ContactImpulsePlugin)["contact_impulse"])
    assert impulse["impulse_ns"] == "N*s" and impulse["peak_normal_n"] == "N"
    battery = _units(_described(EnergyMonitorPlugin)["battery"])
    assert (battery["energy_j"], battery["power_w"], battery["capacity_wh"]) == ("J", "W", "W*h")
    assert _units(_described(ContactLocationPlugin)["contact_location"])["x"] == "m"
    assert _units(_described(ContactMonitorPlugin)["contact"])["first_time"] == "s"


def test_the_bumper_is_a_lazy_family_over_its_zones():
    row = _described(BumperPlugin)["bumper/{item}"]
    assert row["family"] == "zones" and row["lazy"] is True


def test_override_takes_the_bool_a_setbool_carries():
    row = _described(ModelOverridePlugin)["override"]
    assert [(p["name"], p["type"], p["required"]) for p in row["params"]] == [
        ("data", "bool", True)
    ]
    # Its reply carries the report the step after the change recorded.
    assert row["confirm"] == "override_verified"


def test_spawn_model_declares_the_streamed_and_the_static_pose_on_one_name():
    rows = [s.describe(SpawnModelPlugin) for s in endpoint.declared(SpawnModelPlugin)]
    assert [(r["name"], r["conditional"], r["payload"]) for r in rows] == [
        ("{item}_pose", True, "Transforms"),
        ("{item}_pose", True, "Transforms"),
    ]
    assert rows[0]["rate_hz"] == {"from": "tf_rate"}
