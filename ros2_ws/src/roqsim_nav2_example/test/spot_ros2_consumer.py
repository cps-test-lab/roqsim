"""A perception front end written against the real Spot's ROS 2 driver, and nothing else.

Everything it expects comes from ``spot_ros2_interface.json``, the interface spot_ros2 documents at
the commit named there: it subscribes to one body camera's image, camera_info and depth, looks the
camera up in tf from ``body``, back-projects the depth image's centre pixel into ``body`` and checks
each message against what the driver publishes. With ``--state`` it also checks the odometry and
joint states. It knows nothing of the simulator: no sim time, no remapping, no roqsim name.

Prints one JSON report and exits 0 when everything it expects arrived as expected.

    python3 spot_ros2_consumer.py [--station frontleft] [--state] [--timeout 60]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image, JointState
from tf2_ros import Buffer, TransformListener

INTERFACE = json.loads((Path(__file__).parent / "spot_ros2_interface.json").read_text())


def _expand(text: str, station: str) -> str:
    return text.format(station=station)


class SpotPerception(Node):
    def __init__(self, station: str, state: bool) -> None:
        super().__init__("spot_ros2_consumer")
        self.station = station
        self.got: dict[str, object] = {}
        cam = INTERFACE["camera"]
        qos = QoSProfile(depth=10)  # an ordinary subscriber: reliable, volatile
        for key, msg_type in (
            ("image", Image),
            ("camera_info", CameraInfo),
            ("depth", Image),
            ("depth_camera_info", CameraInfo),
        ):
            self.create_subscription(
                msg_type, _expand(cam[key]["topic"], station), self._keep(key), qos
            )
        if state:
            st = INTERFACE["state"]
            self.create_subscription(Odometry, st["odometry"]["topic"], self._keep("odometry"), qos)
            self.create_subscription(
                JointState, st["joint_states"]["topic"], self._keep("joint_states"), qos
            )
        self.tf = Buffer()
        self.listener = TransformListener(self.tf, self)

    def _keep(self, key: str):
        def cb(msg) -> None:
            self.got.setdefault(key, msg)

        return cb


def _depth_at_centre(msg: Image) -> float:
    """The centre pixel of a 16UC1 image, in metres (0 = no return)."""
    row, col = msg.height // 2, msg.width // 2
    i = row * msg.step + col * 2
    raw = int.from_bytes(bytes(msg.data[i : i + 2]), "big" if msg.is_bigendian else "little")
    return raw / 1000.0


def check(node: SpotPerception, state: bool) -> dict:
    cam, st = INTERFACE["camera"], INTERFACE["state"]
    s = node.station
    report: dict[str, object] = {"station": s, "source": INTERFACE["source"], "checks": {}}
    checks = report["checks"]

    def expect(name: str, ok: bool, seen) -> None:
        checks[name] = {"ok": bool(ok), "seen": seen}

    image, info = node.got["image"], node.got["camera_info"]
    depth, depth_info = node.got["depth"], node.got["depth_camera_info"]
    expect("image.encoding", image.encoding == cam["image"]["encoding"], image.encoding)
    expect(
        "image.frame_id",
        image.header.frame_id == _expand(cam["image"]["frame_id"], s),
        image.header.frame_id,
    )
    expect(
        "image.size_matches_camera_info",
        (image.width, image.height) == (info.width, info.height),
        [image.width, image.height],
    )
    expect(
        "camera_info.distortion_model",
        info.distortion_model == cam["camera_info"]["distortion_model"],
        info.distortion_model,
    )
    expect("camera_info.d", list(info.d) == cam["camera_info"]["d"], list(info.d))
    expect(
        "camera_info.frame_id",
        info.header.frame_id == _expand(cam["camera_info"]["frame_id"], s),
        info.header.frame_id,
    )
    expect(
        "camera_info.k_is_pinhole",
        info.k[0] > 0 and info.k[4] > 0 and info.k[8] == 1.0,
        list(info.k),
    )
    expect("depth.encoding", depth.encoding == cam["depth"]["encoding"], depth.encoding)
    expect(
        "depth.frame_id",
        depth.header.frame_id == _expand(cam["depth"]["frame_id"], s),
        depth.header.frame_id,
    )
    expect(
        "depth_camera_info.frame_id",
        depth_info.header.frame_id == _expand(cam["depth_camera_info"]["frame_id"], s),
        depth_info.header.frame_id,
    )
    expect(
        "depth.size_matches_camera_info",
        (depth.width, depth.height) == (depth_info.width, depth_info.height),
        [depth.width, depth.height],
    )
    nonzero = sum(1 for i in range(0, len(depth.data), 2) if depth.data[i] or depth.data[i + 1])
    expect("depth.has_returns", nonzero > 0, nonzero)

    root = cam["tf_root_of_cameras"]
    for frame in (image.header.frame_id, depth.header.frame_id):
        t = node.tf.lookup_transform(root, frame, Time())
        p = t.transform.translation
        expect(f"tf.{root}->{frame}", True, [round(p.x, 4), round(p.y, 4), round(p.z, 4)])

    # The centre of the depth image, back-projected through its camera_info into `body`: a range on
    # the optical axis, placed through the tf the driver publishes.
    z = _depth_at_centre(depth)
    if z > 0:
        t = node.tf.lookup_transform(root, depth.header.frame_id, Time()).transform
        q = t.rotation
        # The optical axis (0, 0, z) of the depth frame, rotated into `body`.
        ax = 2 * (q.x * q.z + q.w * q.y) * z
        ay = 2 * (q.y * q.z - q.w * q.x) * z
        az = (1 - 2 * (q.x * q.x + q.y * q.y)) * z
        point = [t.translation.x + ax, t.translation.y + ay, t.translation.z + az]
        expect(
            "depth.centre_in_body",
            all(math.isfinite(v) for v in point),
            {"range_m": z, "point": [round(v, 3) for v in point]},
        )
    else:
        expect("depth.centre_in_body", True, {"range_m": 0.0, "point": None})

    if state:
        odom, joints = node.got["odometry"], node.got["joint_states"]
        expect(
            "odometry.frame_id",
            odom.header.frame_id == st["odometry"]["frame_id"],
            odom.header.frame_id,
        )
        expect(
            "odometry.child_frame_id",
            odom.child_frame_id == st["odometry"]["child_frame_id"],
            odom.child_frame_id,
        )
        expect(
            "joint_states.names",
            sorted(joints.name) == sorted(st["joint_states"]["names"]),
            list(joints.name),
        )
    report["ok"] = all(c["ok"] for c in checks.values())
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--station", default="frontleft", choices=INTERFACE["stations"])
    parser.add_argument("--state", action="store_true", help="also check odometry and joints")
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args(argv)
    rclpy.init()
    node = SpotPerception(args.station, args.state)
    want = {"image", "camera_info", "depth", "depth_camera_info"}
    if args.state:
        want |= {"odometry", "joint_states"}
    deadline = time.monotonic() + args.timeout
    report: dict = {"ok": False}
    try:
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.2)
            if want <= set(node.got) and node.tf.can_transform(
                INTERFACE["camera"]["tf_root_of_cameras"],
                _expand(INTERFACE["camera"]["image"]["frame_id"], args.station),
                Time(),
            ):
                report = check(node, args.state)
                break
        else:
            report = {"ok": False, "missing": sorted(want - set(node.got))}
    finally:
        node.destroy_node()
        rclpy.shutdown()
    print(json.dumps(report, indent=1))
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
