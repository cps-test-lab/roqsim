# Copyright (C) 2026 Frederik Pasch
# SPDX-License-Identifier: Apache-2.0

"""The one Create 3 adapter roqsim has to supply: bumper zones into ``HazardDetection`` events.

``irobot_create_gz_toolbox``'s ``sensors_node`` derives every other Create 3 hazard from plain ROS
types -- cliff and IR intensity from ``LaserScan``, wheel drop from ``JointState``, the dock from
ground-truth ``Odometry`` -- but its bumper reads a Gazebo ``Contacts`` message and zones the
contact positions itself. roqsim's ``bumper`` plugin already zones its contacts, by the same bearing
table, and publishes a ``std_msgs/Bool`` per zone; this node turns each pressed zone into the
``HazardDetection`` the Create 3's ``hazards_vector_publisher`` collects on
``_internal/bumper/event``, one event per zone per tick while it is pressed, exactly as the Gazebo
adapter emits one per contact message. ``frame_id`` is the zone, which is what a consumer that
reacts to *where* it was bumped reads.

Parameters::

    zones: [bump_left, bump_front_left, bump_front_center, bump_front_right, bump_right]
    bumper_topic_prefix: bumper      # roqsim publishes <prefix>/<zone>
    hazard_topic: _internal/bumper/event
    publish_rate: 62.0
"""

from __future__ import annotations

import rclpy
from irobot_create_msgs.msg import HazardDetection
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import Bool

DEFAULT_ZONES = [
    "bump_left",
    "bump_front_left",
    "bump_front_center",
    "bump_front_right",
    "bump_right",
]


class BumperHazard(Node):
    def __init__(self, **kwargs):
        super().__init__("bumper_hazard", **kwargs)
        self.declare_parameter("zones", DEFAULT_ZONES)
        self.declare_parameter("bumper_topic_prefix", "bumper")
        self.declare_parameter("hazard_topic", "_internal/bumper/event")
        self.declare_parameter("publish_rate", 62.0)
        zones = list(self.get_parameter("zones").value)
        prefix = str(self.get_parameter("bumper_topic_prefix").value)
        self._pressed: dict[str, bool] = dict.fromkeys(zones, False)
        self._subs = [
            self.create_subscription(
                Bool, f"{prefix}/{zone}", self._make_callback(zone), qos_profile_sensor_data
            )
            for zone in zones
        ]
        self._pub = self.create_publisher(
            HazardDetection, str(self.get_parameter("hazard_topic").value), qos_profile_sensor_data
        )
        rate = float(self.get_parameter("publish_rate").value)
        self._timer = self.create_timer(1.0 / rate, self._tick)

    def _make_callback(self, zone: str):
        def on_bool(msg: Bool) -> None:
            self._pressed[zone] = bool(msg.data)

        return on_bool

    def pressed_zones(self) -> list[str]:
        return [zone for zone, pressed in self._pressed.items() if pressed]

    def hazards(self) -> list[HazardDetection]:
        """One BUMP event per pressed zone, stamped now, framed by the zone."""
        now = self.get_clock().now().to_msg()
        out = []
        for zone in self.pressed_zones():
            msg = HazardDetection()
            msg.type = HazardDetection.BUMP
            msg.header.stamp = now
            msg.header.frame_id = zone
            out.append(msg)
        return out

    def _tick(self) -> None:
        for msg in self.hazards():
            self._pub.publish(msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = BumperHazard()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
