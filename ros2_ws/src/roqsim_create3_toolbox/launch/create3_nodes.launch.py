# Copyright (C) 2026 Frederik Pasch
# SPDX-License-Identifier: Apache-2.0

"""The Create 3's own nodes over a roqsim TurtleBot 4: the reference simulator's node graph.

Mirrors irobot_create_common_bringup/launch/create3_nodes.launch.py and
irobot_create_gz_bringup/launch/create3_gz_nodes.launch.py, with the same nodes from the same
released packages and their shipped parameters. What differs is only what the simulator supplies:
roqsim's TurtleBot 4 model publishes the raw streams (joint states, cliff and IR ray grids,
ground-truth poses) on the names the Gazebo adapter reads, and one adapter of our own turns
roqsim's bumper zones into the bumper events, since the Gazebo one reads a Gazebo contact message.

    ros2 launch roqsim_create3_toolbox create3_nodes.launch.py [namespace:=/robot] [params_file:=...]

The simulator is launched separately (a world with the ``ros2_bridge``; see worlds/); this file
brings up the stack only, so a campaign runs it beside its Nav2 launch.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

ARGUMENTS = [
    DeclareLaunchArgument("namespace", default_value="", description="Robot namespace"),
    DeclareLaunchArgument(
        "robot_name",
        default_value="turtlebot4",
        description="The child frame the robot's ground-truth pose is published under "
        "(turtlebot4.manifest.yaml: ground_truth_pose child_frame)",
    ),
    DeclareLaunchArgument(
        "dock_name",
        default_value="standard_dock",
        description="The child frame the dock's ground-truth pose is published under",
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
    config = os.path.join(get_package_share_directory("roqsim_create3_toolbox"), "config")
    namespace = LaunchConfiguration("namespace")
    sim_time = {"use_sim_time": True}
    tf_remaps = [("/tf", "tf"), ("/tf_static", "tf_static")]

    def node(package, executable, name, params=(), remappings=(), **extra):
        return Node(
            package=package,
            executable=executable,
            name=name,
            namespace=namespace,
            parameters=[*params, sim_time],
            remappings=list(remappings),
            output="screen",
            **extra,
        )

    nodes = [
        # -- the simulator adapters (irobot_create_gz_toolbox, plain ROS types in) ---------------
        node(
            "irobot_create_gz_toolbox",
            "pose_republisher_node",
            "pose_republisher_node",
            params=[
                os.path.join(config, "pose_republisher_params.yaml"),
                {
                    "robot_name": LaunchConfiguration("robot_name"),
                    "dock_name": LaunchConfiguration("dock_name"),
                },
            ],
        ),
        node(
            "irobot_create_gz_toolbox",
            "sensors_node",
            "sensors_node",
            params=[os.path.join(config, "sensors_params.yaml")],
        ),
        node("irobot_create_gz_toolbox", "interface_buttons_node", "interface_buttons_node"),
        # The bumper: roqsim zones its contacts itself, so its zones become the events here.
        node("roqsim_create3_toolbox", "bumper_hazard", "bumper_hazard"),
        # -- the Create 3 API (irobot_create_nodes) ----------------------------------------------
        node(
            "irobot_create_nodes",
            "hazards_vector_publisher",
            "hazards_vector_publisher",
            params=[os.path.join(config, "hazard_vector_params.yaml")],
        ),
        node(
            "irobot_create_nodes",
            "ir_intensity_vector_publisher",
            "ir_intensity_vector_publisher",
            params=[os.path.join(config, "ir_intensity_vector_params.yaml")],
        ),
        node(
            "irobot_create_nodes",
            "motion_control",
            "motion_control",
            params=[LaunchConfiguration("params_file")],
            remappings=tf_remaps,
        ),
        node(
            "irobot_create_nodes",
            "wheel_status_publisher",
            "wheel_status_publisher",
            params=[os.path.join(config, "wheel_status_params.yaml")],
        ),
        node(
            "irobot_create_nodes",
            "mock_publisher",
            "mock_publisher",
            params=[os.path.join(config, "mock_params.yaml")],
        ),
        node(
            "irobot_create_nodes",
            "robot_state",
            "robot_state",
            params=[os.path.join(config, "robot_state_params.yaml")],
        ),
        node(
            "irobot_create_nodes",
            "kidnap_estimator_publisher",
            "kidnap_estimator_publisher",
            params=[os.path.join(config, "kidnap_estimator_params.yaml")],
        ),
        node(
            "irobot_create_nodes",
            "ui_mgr",
            "ui_mgr",
            params=[os.path.join(config, "ui_mgr_params.yaml"), {"gazebo": "ignition"}],
        ),
    ]
    ld = LaunchDescription(ARGUMENTS)
    for n in nodes:
        ld.add_action(n)
    return ld
