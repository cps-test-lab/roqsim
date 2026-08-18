#!/usr/bin/env bash
# roqsim-ros entrypoint: source ROS 2 and the colcon-built workspace, then exec the command so the
# roqsim ROS bridge / nav2 example / walker_ros packages are on the path.
set -e

# ROS_DISTRO is exported by the ros:* base image; fall back to jazzy if unset.
source "/opt/ros/${ROS_DISTRO:-jazzy}/setup.bash"

if [ -f /ws/ros2_ws/install/setup.bash ]; then
    source /ws/ros2_ws/install/setup.bash
fi

exec "$@"
