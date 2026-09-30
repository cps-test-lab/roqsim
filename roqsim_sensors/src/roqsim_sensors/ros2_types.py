# Copyright (C) 2026 Frederik Pasch
# SPDX-License-Identifier: Apache-2.0

"""How ROS carries this package's own payloads: a GNSS fix, 3D object detections and 2D boxes.

Loaded by the ROS 2 bridge through the ``roqsim.ros2_types`` entry point (see
``roqsim_ros_bridge.typemap``), so it is imported only where the bridge is; nothing else in this
package imports ROS or the bridge. Free of ROS imports itself, as the bridge's table is: the
converters are loaded when a message is filled. Each type is published, never taken.
"""

from __future__ import annotations

from dataclasses import asdict

from roqsim_ros_bridge.typemap import RosType, Wire

from .plugins.gnss import GnssFix
from .plugins.object_detector import ObjectDetections
from .plugins.segmentation_camera import Boxes2D


def _published_only(what: str):
    def decode(msg):
        raise TypeError(f"{what} is published, not taken")

    return decode


def _fill_fix(msg, v: GnssFix, stamp, hints) -> None:
    from roqsim_ros_bridge.registry import fill_navsatfix

    fill_navsatfix(msg, asdict(v), stamp, hints)


def _fill_objects(msg, v: ObjectDetections, stamp, hints) -> None:
    from roqsim_ros_bridge.registry import fill_detection3d_array

    fill_detection3d_array(
        msg,
        [(d.class_id, (*d.position, *d.orientation), tuple(d.size), d.score) for d in v.detections],
        stamp,
        hints,
    )


def _fill_boxes(msg, v: Boxes2D, stamp, hints) -> None:
    from roqsim_ros_bridge.registry import fill_detection2d_array

    fill_detection2d_array(
        msg,
        [(b.class_id, b.class_name, b.instance_id, b.cx, b.cy, b.width, b.height) for b in v.boxes],
        stamp,
        hints,
    )


TYPES = (
    RosType(
        GnssFix,
        (Wire("sensor_msgs.msg.NavSatFix", _fill_fix, _published_only("a GNSS fix")),),
    ),
    RosType(
        ObjectDetections,
        (
            Wire(
                "vision_msgs.msg.Detection3DArray",
                _fill_objects,
                _published_only("an object detection"),
            ),
        ),
    ),
    RosType(
        Boxes2D,
        (
            Wire(
                "vision_msgs.msg.Detection2DArray",
                _fill_boxes,
                _published_only("a 2D detection"),
            ),
        ),
    ),
)
