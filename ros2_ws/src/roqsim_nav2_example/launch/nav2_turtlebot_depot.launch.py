"""nav2 + roqsim TurtleBot 4 in the Depot world, with **AMCL** localization.

This is the Gazebo-compatible bring-up: it mirrors nav2's ``tb4_simulation_launch.py`` (Depot world,
AMCL, the same map), but drives the roqsim MuJoCo simulator instead of gz. Together with a ground-truth
``<model>_base_link_gt`` TF (published by the ``ground_truth_pose`` plugin in the world), it lets
roqsim stand in for Gazebo behind the *same* scenario: the ROS graph a nav2 client sees is the same.

Starts: the sim + ROS bridge (``roqsim_ros_bridge``) running the Depot world, the TB4 kinematic tree,
and nav2 itself via nav2_bringup's ``bringup_launch.py`` — the same file the Gazebo bring-up ends up
running. AMCL publishes ``map->odom`` (no static stand-in here — the wheel odometry drifts and AMCL
corrects it, exactly as on real hardware). Everything runs with use_sim_time (the bridge publishes
/clock).

    ros2 launch roqsim_nav2_example nav2_turtlebot_depot.launch.py            # headless (egl)
    ros2 launch roqsim_nav2_example nav2_turtlebot_depot.launch.py headless:=false   # MuJoCo window

``autostart:=False`` brings nav2 up *configured but inactive*, leaving activation to whoever knows
when the simulator is ready — see the node-set comment below.
"""

import os
import sys

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
    OpaqueFunction,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node


def _bridge_process(context, *_args, **_kwargs):
    """Build the sim+bridge process, honouring the ``headless`` launch arg.

    ``headless:=false`` opens a MuJoCo viewer window and selects the on-screen ``glfw`` GL backend;
    otherwise we render offscreen with ``egl``. ``headless`` is the alias tb4_simulation_launch.py
    uses, so the same scenario ``ros_launch`` arguments drive either backend.
    """
    headless = LaunchConfiguration("headless").perform(context).lower() in ("true", "1", "yes")
    world = LaunchConfiguration("world").perform(context)

    # `ros2 launch` runs under the system interpreter, so sys.executable is /usr/bin/python3 even
    # with a venv active -- and that Python can't import roqsim. Prefer the active venv's python.
    bridge_python = (
        os.path.join(os.environ["VIRTUAL_ENV"], "bin", "python3")
        if os.environ.get("VIRTUAL_ENV")
        else sys.executable
    )

    env = dict(os.environ)
    env["MUJOCO_GL"] = env.get("MUJOCO_GL", "egl") if headless else "glfw"

    cmd = [bridge_python, "-m", "roqsim_ros_bridge.run_bridge", "--world", world]
    if headless:
        cmd.append("--headless")

    return [ExecuteProcess(cmd=cmd, output="screen", env=env)]


def generate_launch_description():
    pkg = get_package_share_directory("roqsim_nav2_example")
    default_world = os.path.join(pkg, "worlds", "depot_nav2.yaml")
    default_map = os.path.join(pkg, "maps", "depot.yaml")
    default_params = os.path.join(pkg, "params", "nav2_params_depot.yaml")
    # The description nav2_bringup's tb4_simulation_launch.py uses. Deliberately
    # nav2_minimal_tb4_description and not the fuller turtlebot4_description: a different package
    # means a different TF tree, which is the divergence this is here to remove.
    robot_description_file = os.path.join(
        get_package_share_directory("nav2_minimal_tb4_description"),
        "urdf",
        "standard",
        "turtlebot4.urdf.xacro",
    )
    default_rviz = os.path.join(
        get_package_share_directory("nav2_bringup"), "rviz", "nav2_default_view.rviz"
    )

    map_yaml = LaunchConfiguration("map")
    params_file = LaunchConfiguration("params_file")
    use_sim_time = LaunchConfiguration("use_sim_time")
    autostart = LaunchConfiguration("autostart")

    return LaunchDescription(
        [
            DeclareLaunchArgument("world", default_value=default_world),
            DeclareLaunchArgument("map", default_value=default_map),
            DeclareLaunchArgument("params_file", default_value=default_params),
            DeclareLaunchArgument("use_sim_time", default_value="true"),
            # nav2_bringup's own argument, same name and same default, so this launch and
            # tb4_simulation_launch.py are driven by one flag: the shared campaign scenario passes
            # autostart:=False to BOTH backends and activates nav2 itself once the simulator is up.
            DeclareLaunchArgument(
                "autostart",
                default_value="true",
                description="activate the nav2 lifecycle nodes at launch; False leaves them "
                "configured-but-inactive for an external gate to start",
            ),
            DeclareLaunchArgument(
                "headless",
                default_value="true",
                description="run the sim offscreen (egl); false opens the MuJoCo viewer (glfw)",
            ),
            DeclareLaunchArgument(
                "rviz",
                default_value="false",
                description="launch rviz2 in-process (needs rviz2 installed); off by default so "
                "rviz can run from a separate image",
            ),
            # sim + ROS bridge; built at launch time so it can honour the headless arg (see helper).
            OpaqueFunction(function=_bridge_process),
            # The robot's kinematic tree, from the SAME description Gazebo's
            # tb4_simulation_launch.py uses -- so the TF tree is identical by construction rather
            # than by maintenance. It consumes the bridge's /joint_states and publishes base_link
            # and everything below it (base_footprint, wheels, rplidar_link, ...). Without it the
            # bridge's hand-published subset was the whole tree, and any consumer expecting a
            # standard TB4 frame failed: nav2's collision_monitor defaults to
            # base_frame_id: base_footprint, could not transform its scan, and -- sitting in the
            # cmd_vel path -- failed closed and stopped the robot.
            #
            # The world sets ``publish_static_tf: false`` on the bridge so the sensor-mount
            # transforms come from here only; two publishers for one static transform is a TF
            # conflict, not redundancy.
            Node(
                package="robot_state_publisher",
                executable="robot_state_publisher",
                name="robot_state_publisher",
                output="screen",
                parameters=[
                    {
                        "use_sim_time": use_sim_time,
                        "robot_description": Command(["xacro ", robot_description_file]),
                    }
                ],
            ),
            # nav2 itself: the SAME bring-up Gazebo runs. tb4_simulation_launch.py is
            # gz + robot_state_publisher + rviz + bringup_launch.py; everything above this line is
            # our replacement for the gz half, and this include is the other half, unmodified. It
            # brings up the composed nav2_container with the standard twelve servers under
            # lifecycle_manager_localization (map_server, amcl) + lifecycle_manager_navigation (the
            # rest). This used to be twelve hand-listed Nodes under a single manager -- the node
            # set matched, but the process structure and the manager topology did not, so a
            # campaign comparing the two backends still compared two different nav2 deployments.
            #
            # use_composition / use_respawn / namespace are deliberately NOT passed: taking
            # bringup_launch.py's defaults is what keeps the two backends identical by
            # construction rather than by maintenance -- the same reason the description above
            # comes from nav2_minimal_tb4_description.
            #
            # AMCL owns map->odom, so no static_transform_publisher stand-in here. The scan
            # (rplidar_link, with its static base_link->rplidar_link TF from the sim), /odom and the
            # Depot /map give AMCL everything it needs; the initial pose is set in
            # nav2_params_depot.yaml (identity: the robot spawns at the map origin) and refined by
            # the scenario's init_nav2.
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(
                        get_package_share_directory("nav2_bringup"), "launch", "bringup_launch.py"
                    )
                ),
                launch_arguments={
                    "map": map_yaml,
                    "params_file": params_file,
                    "use_sim_time": use_sim_time,
                    "autostart": autostart,
                }.items(),
            ),
            Node(
                package="rviz2",
                executable="rviz2",
                name="rviz2",
                output="screen",
                condition=IfCondition(LaunchConfiguration("rviz")),
                arguments=["-d", default_rviz],
                parameters=[{"use_sim_time": use_sim_time}],
            ),
        ]
    )
