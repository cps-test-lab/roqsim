"""Launch a roqsim walker world with the ROS 2 bridge, serving NavigateThroughPoses.

    ros2 launch roqsim_walker_ros walker_nav.launch.py
    ros2 launch roqsim_walker_ros walker_nav.launch.py world:=/abs/world.yaml headless:=false

Send the walker through a route:

    ros2 action send_goal /navigate_through_poses nav2_msgs/action/NavigateThroughPoses \
      "{poses: [{header: {frame_id: map}, pose: {position: {x: 2.0, y: 0.0}}}]}" --feedback

Other nodes (nav2, rviz) should run with use_sim_time:=true -- the bridge publishes /clock.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    share = get_package_share_directory("roqsim_walker_ros")
    default_world = os.path.join(share, "worlds", "walker_nav2.yaml")

    world = LaunchConfiguration("world")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "world", default_value=default_world, description="Path to the roqsim world YAML"
            ),
            # The bridge launcher from roqsim_ros_bridge runs any world; this package only adds
            # the NavigateThroughPoses handler (auto-imported via the extensions entry point).
            Node(
                package="roqsim_ros_bridge",
                executable="roqsim_bridge",
                name="roqsim_bridge",
                output="screen",
                arguments=["--world", world],
            ),
        ]
    )
