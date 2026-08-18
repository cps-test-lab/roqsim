"""Launch the roqsim TurtleBot 4 world with the ROS 2 bridge.

    ros2 launch roqsim_ros_bridge turtlebot.launch.py
    ros2 launch roqsim_ros_bridge turtlebot.launch.py world:=/abs/path/to/world.yaml headless:=false

Other nodes (nav2, rviz) should run with use_sim_time:=true — the bridge publishes /clock.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    share = get_package_share_directory("roqsim_ros_bridge")
    default_world = os.path.join(share, "worlds", "turtlebot_ros2.yaml")

    world = LaunchConfiguration("world")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "world", default_value=default_world, description="Path to the roqsim world YAML"
            ),
            Node(
                package="roqsim_ros_bridge",
                executable="roqsim_bridge",
                name="roqsim_bridge",
                output="screen",
                arguments=["--world", world],
            ),
        ]
    )
