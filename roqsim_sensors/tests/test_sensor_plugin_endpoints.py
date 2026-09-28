"""The sensor plugins' endpoints, as their classes declare them: names, kinds, payloads and units."""

from __future__ import annotations

import pytest
from roqsim_sensors.plugins.force_limit import ForceLimitPlugin
from roqsim_sensors.plugins.gnss import GnssPlugin
from roqsim_sensors.plugins.imu import ImuPlugin
from roqsim_sensors.plugins.lidar import LidarPlugin
from roqsim_sensors.plugins.livox_mid360 import LivoxMid360Plugin
from roqsim_sensors.plugins.oakd_camera import OakDCameraPlugin
from roqsim_sensors.plugins.object_detector import ObjectDetectorPlugin
from roqsim_sensors.plugins.range_sensor import RangeSensorPlugin
from roqsim_sensors.plugins.realsense_d435 import RealsenseD435Plugin
from roqsim_sensors.plugins.segmentation_camera import SegmentationCameraPlugin

from roqsim import endpoint

FAULT = {
    "override": ("command", None),
    "override_state": ("out", "FaultReport"),
    "override_verified": ("out", "FaultReport"),
}
COLOUR = {
    "image": ("out", "Image"),
    "image_compressed": ("out", "Image"),
    "camera_info": ("out", "CameraInfo"),
}
DEPTH = {
    "depth": ("out", "Image"),
    "depth_camera_info": ("out", "CameraInfo"),
    "depth_compressed": ("out", "Image"),
}

#: plugin class -> {endpoint name: (kind, payload type)}
EXPECTED = {
    OakDCameraPlugin: {**COLOUR, **DEPTH},
    RealsenseD435Plugin: {**COLOUR, **DEPTH, "points": ("out", "PointCloud")},
    SegmentationCameraPlugin: {
        **COLOUR,
        "labels": ("out", "Image"),
        "instances": ("out", "Image"),
        "detections": ("out", "Boxes2D"),
    },
    LidarPlugin: {**FAULT, "scan": ("out", "LaserScan")},
    RangeSensorPlugin: {**FAULT, "range": ("out", "LaserScan")},
    LivoxMid360Plugin: {**FAULT, "cloud": ("out", "PointCloud")},
    ImuPlugin: {**FAULT, "imu": ("out", "Imu")},
    GnssPlugin: {"fix": ("out", "GnssFix")},
    ObjectDetectorPlugin: {"detections": ("out", "ObjectDetections")},
    ForceLimitPlugin: {"force_limit": ("out", "LimitReport")},
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


def test_a_sensor_offers_its_fault_switch_only_with_a_fault_block():
    rows = _described(LidarPlugin)
    assert all(rows[name]["conditional"] for name in FAULT)
    assert [(p["name"], p["type"]) for p in rows["override"]["params"]] == [("data", "bool")]


def _units(row) -> dict[str, str]:
    return {f["name"]: f.get("unit", "") for f in row["result"].get("fields", [])}


def test_the_reports_state_their_units():
    fix = _units(_described(GnssPlugin)["fix"])
    assert (fix["lat"], fix["alt"], fix["eph"], fix["vel_n"], fix["cog"]) == (
        "deg",
        "m",
        "m",
        "m/s",
        "deg",
    )
    limit = _units(_described(ForceLimitPlugin)["force_limit"])
    assert (limit["at_time"], limit["force"], limit["torque"]) == ("s", "N", "N*m")
    (boxes,) = _described(SegmentationCameraPlugin)["detections"]["result"]["fields"]
    box = {f["name"]: f.get("unit", "") for f in boxes["items"][0]["fields"]}
    assert (box["cx"], box["width"]) == ("px", "px")
