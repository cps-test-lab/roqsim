"""How ROS carries this package's own payload types (the ``roqsim.ros2_types`` entry point).

Loaded by the ROS 2 bridge only, so importing it needs ``roqsim_ros_bridge``; the message modules are
imported when a message is filled.
"""

from __future__ import annotations

import numpy as np

from roqsim_mobile.plugins.create3_pose_publisher import FrameTransform
from roqsim_ros_bridge.typemap import RosType, Wire, frame


def _fill_tf_message(msg, v: FrameTransform, stamp, hints) -> None:
    """One ``TransformStamped``, its parent the endpoint's ``frame_id`` hint and its child verbatim."""
    from geometry_msgs.msg import TransformStamped

    tf = TransformStamped()
    tf.header.stamp = stamp
    tf.header.frame_id = frame(hints, "frame_id", "map")
    tf.child_frame_id = v.child_frame_id
    t, q = tf.transform.translation, tf.transform.rotation
    t.x, t.y, t.z = (float(x) for x in v.translation)
    q.w, q.x, q.y, q.z = (float(x) for x in v.rotation)
    msg.transforms = [tf]


def _decode_tf_message(msg) -> FrameTransform:
    (tf,) = msg.transforms
    t, q = tf.transform.translation, tf.transform.rotation
    return FrameTransform(
        tf.child_frame_id,
        np.array([t.x, t.y, t.z], dtype=np.float64),
        np.array([q.w, q.x, q.y, q.z], dtype=np.float64),
    )


ROS_TYPES = [
    RosType(
        FrameTransform,
        (Wire("tf2_msgs.msg.TFMessage", _fill_tf_message, _decode_tf_message),),
        hints={"frame_id": "map"},
    ),
]
