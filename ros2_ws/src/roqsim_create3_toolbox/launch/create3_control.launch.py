# Copyright (C) 2026 Frederik Pasch
# SPDX-License-Identifier: Apache-2.0

"""The Create 3's ros2_control diffdrive_controller beside a roqsim TurtleBot 4.

On the robot, the Create 3 firmware's motion_control caps and ramps the velocity command; there is
no ros2_control. roqsim models that in the base's diff_drive. The reference simulator models it in
this controller, with irobot_create_control's control.yaml, and this file runs the released
controller and its released parameters between ``motion_control`` and the base:

    motion_control -> diffdrive_controller/cmd_vel -> diffdrive_controller -> diffdrive_controller/cmd_vel_out

A world puts it in the command path by pointing the base at ``diffdrive_controller/cmd_vel_out``;
pointed at ``diffdrive_controller/cmd_vel`` as usual, the limited command is a reference only.

    ros2 launch roqsim_create3_toolbox create3_control.launch.py [namespace:=/robot]

What differs from the reference simulator, and why:

* The controller drives ros2_control's mock hardware (``config/create3_control.urdf``), since the
  simulator owns the wheels. Its description goes out on ``create3_control/robot_description``: the
  TurtleBot 4's description on ``robot_description`` carries no ros2_control block.
* Only ``diffdrive_controller`` is spawned. The joint_state_broadcaster would publish the mock
  wheels on ``joint_states`` beside the simulator's.
* ``config/create3_control.yaml`` turns the limited command on, leaves ``odom -> base_link`` to the
  simulator and runs the controller at 62 Hz. The controller's own odometry, integrated from the
  mock wheels, is on ``diffdrive_controller/odom``.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

ARGUMENTS = [
    DeclareLaunchArgument("namespace", default_value="", description="Robot namespace"),
]

#: Where this file's controller_manager reads its robot description, relative to the namespace.
DESCRIPTION_TOPIC = "create3_control/robot_description"


def generate_launch_description():
    config = os.path.join(get_package_share_directory("roqsim_create3_toolbox"), "config")
    control_yaml = os.path.join(
        get_package_share_directory("irobot_create_control"), "config", "control.yaml"
    )
    with open(os.path.join(config, "create3_control.urdf"), encoding="utf-8") as handle:
        description = handle.read()
    namespace = LaunchConfiguration("namespace")
    sim_time = {"use_sim_time": True}

    nodes = [
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            name="create3_control_description",
            namespace=namespace,
            parameters=[{"robot_description": description}, sim_time],
            remappings=[
                ("robot_description", DESCRIPTION_TOPIC),
                ("/tf", "create3_control/tf"),
                ("/tf_static", "create3_control/tf_static"),
            ],
            output="screen",
        ),
        Node(
            package="controller_manager",
            executable="ros2_control_node",
            namespace=namespace,
            parameters=[control_yaml, os.path.join(config, "create3_control.yaml"), sim_time],
            remappings=[("robot_description", DESCRIPTION_TOPIC)],
            output="screen",
        ),
        Node(
            package="controller_manager",
            executable="spawner",
            namespace=namespace,
            arguments=[
                "diffdrive_controller",
                "-c",
                "controller_manager",
                "--controller-manager-timeout",
                "60",
            ],
            output="screen",
        ),
    ]
    ld = LaunchDescription(ARGUMENTS)
    for n in nodes:
        ld.add_action(n)
    return ld
