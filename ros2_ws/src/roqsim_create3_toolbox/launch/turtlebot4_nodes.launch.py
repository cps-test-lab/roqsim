# Copyright (C) 2026 Frederik Pasch
# SPDX-License-Identifier: Apache-2.0

"""The TurtleBot 4's node graph over roqsim: the Create 3 nodes plus ``turtlebot4_node``.

What turtlebot4_gz_bringup/launch/turtlebot4_spawn.launch.py brings up beside the simulator,
minus its Gazebo GUI plugin (the HMI display and buttons have no window here; their topics exist
through turtlebot4_node and ui_mgr and nothing presses them).

    ros2 launch roqsim_create3_toolbox turtlebot4_nodes.launch.py [namespace:=/robot]
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

ARGUMENTS = [
    DeclareLaunchArgument("namespace", default_value="", description="Robot namespace"),
    DeclareLaunchArgument(
        "model",
        default_value="standard",
        choices=["standard", "lite"],
        description="TurtleBot 4 model",
    ),
    DeclareLaunchArgument(
        "params_file",
        default_value=os.path.join(
            get_package_share_directory("roqsim_create3_toolbox"), "config", "create3_params.yaml"
        ),
        description="motion_control parameters (safety_override); a campaign's varied copy",
    ),
]


def generate_launch_description():
    pkg = get_package_share_directory("roqsim_create3_toolbox")
    create3 = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(pkg, "launch", "create3_nodes.launch.py")),
        launch_arguments=[
            ("namespace", LaunchConfiguration("namespace")),
            ("params_file", LaunchConfiguration("params_file")),
        ],
    )
    turtlebot4_node = Node(
        package="turtlebot4_node",
        executable="turtlebot4_node",
        name="turtlebot4_node",
        namespace=LaunchConfiguration("namespace"),
        parameters=[
            os.path.join(pkg, "config", "turtlebot4_node.yaml"),
            {"model": LaunchConfiguration("model"), "use_sim_time": True},
        ],
        output="screen",
    )
    ld = LaunchDescription(ARGUMENTS)
    ld.add_action(create3)
    ld.add_action(turtlebot4_node)
    return ld
