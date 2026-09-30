# Copyright (C) 2026 Frederik Pasch
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""How ROS carries this package's own payload types.

Loaded by the ROS bridge through the ``roqsim.ros2_types`` entry point, so nothing else in the
package imports the bridge.

:class:`~roqsim_manipulation.plugins.arm_controller.JointVelocities` travels as
``trajectory_msgs/JointTrajectory``, the message a ros2_control velocity command topic and the
streaming controllers that drive it (``moveit_servo`` among them) use: one point, whose
``positions`` carry the joint velocities. Inbound, the last point is the command.
"""

from __future__ import annotations

import numpy as np

from roqsim_ros_bridge.typemap import RosType, Wire, as_f64

from .plugins.arm_controller import JointVelocities


def _fill(msg, value: JointVelocities, stamp, hints) -> None:
    from trajectory_msgs.msg import JointTrajectoryPoint

    msg.header.stamp = stamp
    msg.joint_names = list(value.names)
    point = JointTrajectoryPoint()
    point.positions = as_f64(value.velocities)
    msg.points = [point]


def _decode(msg) -> JointVelocities:
    velocities = list(msg.points[-1].positions) if msg.points else []
    return JointVelocities(list(msg.joint_names), np.array(velocities, dtype=np.float64))


TYPES = RosType(JointVelocities, (Wire("trajectory_msgs.msg.JointTrajectory", _fill, _decode),))
