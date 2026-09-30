"""How ROS carries this package's own payloads, through the ``roqsim.ros2_types`` entry point."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("rclpy")
typemap = pytest.importorskip("roqsim_ros_bridge.typemap")

from builtin_interfaces.msg import Time  # noqa: E402
from roqsim_sensors.plugins.gnss import GnssFix  # noqa: E402
from roqsim_sensors.plugins.object_detector import ObjectDetection, ObjectDetections  # noqa: E402
from roqsim_sensors.plugins.segmentation_camera import Box2D, Boxes2D  # noqa: E402
from roqsim_sensors.ros2_types import TYPES  # noqa: E402

from roqsim_ros_bridge.registry import resolve_type  # noqa: E402

STAMP = Time(sec=2, nanosec=5)


def _fill(value, hints):
    (rostype,) = [t for t in TYPES if t.cls is type(value)]
    (wire,) = rostype.wires
    msg = resolve_type(wire.msg)()
    wire.fill(msg, value, STAMP, hints)
    return wire.msg, msg


def test_a_fix_is_a_navsatfix_with_its_status_and_covariance():
    fix = GnssFix(47.4, 8.5, 488.0, 0.5, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3, 12, True)
    kind, msg = _fill(fix, {})
    assert kind == "sensor_msgs.msg.NavSatFix"
    assert (msg.latitude, msg.longitude, msg.altitude) == (47.4, 8.5, 488.0)
    assert msg.header.frame_id == "gnss_link" and msg.status.status == 0
    assert msg.position_covariance[0] == pytest.approx(0.25)


def test_detections_are_a_detection3darray_in_the_reporting_frame():
    detections = ObjectDetections(
        [
            ObjectDetection(
                "parcel",
                np.array([1.0, 2.0, 3.0]),
                np.array([1.0, 0, 0, 0]),
                np.array([0.1, 0.2, 0.3]),
                0.9,
            )
        ]
    )
    kind, msg = _fill(detections, {"frame_id": "base_link", "frame_prefix": "r1"})
    assert kind == "vision_msgs.msg.Detection3DArray"
    (det,) = msg.detections
    assert msg.header.frame_id == "r1/base_link" and det.id == "parcel"
    assert det.bbox.center.position.y == 2.0 and det.bbox.size.z == 0.3
    assert det.results[0].hypothesis.score == pytest.approx(0.9)


def test_boxes_are_a_detection2darray_with_the_instance_as_id_and_the_class_by_name():
    boxes = Boxes2D([Box2D(1, "parcel", 42, 10.5, 20.0, 5.0, 4.0)])
    kind, msg = _fill(boxes, {"frame_id": "cam"})
    assert kind == "vision_msgs.msg.Detection2DArray"
    (det,) = msg.detections
    assert det.id == "42" and det.results[0].hypothesis.class_id == "parcel"
    assert (det.bbox.center.position.x, det.bbox.size_x) == (10.5, 5.0)


def test_the_bridge_finds_them_through_the_entry_point():
    for rostype in TYPES:
        assert typemap.lookup(rostype.cls) is not None, rostype.cls.__name__
