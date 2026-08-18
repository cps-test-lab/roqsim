"""nav2 + roqsim Unitree G1 humanoid bring-up.

Starts: the sim + ROS bridge (roqsim_ros_bridge) running the g1_nav2.yaml world (the G1 walks via
its RL locomotion policy, driven by /cmd_vel), a static map->odom transform (localization stand-in),
nav2 map_server + planner_server + controller_server + behavior_server + bt_navigator, and a
lifecycle manager to activate them. Everything runs with use_sim_time (the bridge publishes /clock).

    ros2 launch roqsim_nav2_example nav2_g1.launch.py
    ros2 launch roqsim_nav2_example nav2_g1.launch.py gui:=true   # MuJoCo viewer + rviz2
"""

import os
import sys

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from nav2_common.launch import RewrittenYaml


def _bridge_process(context, *_args, **_kwargs):
    """Build the sim+bridge process, honouring the ``gui`` launch arg.

    ``gui:=true`` opens a MuJoCo viewer window and selects the on-screen ``glfw`` GL backend;
    otherwise it appends ``--headless`` and we render offscreen with ``egl``.
    """
    gui = LaunchConfiguration("gui").perform(context).lower() in ("true", "1", "yes")
    world = LaunchConfiguration("world").perform(context)

    # `ros2 launch` runs under the system interpreter, so sys.executable is /usr/bin/python3 even
    # with a venv active -- and that Python can't import roqsim. Prefer the active venv's python.
    bridge_python = (
        os.path.join(os.environ["VIRTUAL_ENV"], "bin", "python3")
        if os.environ.get("VIRTUAL_ENV")
        else sys.executable
    )

    env = dict(os.environ)
    env["MUJOCO_GL"] = "glfw" if gui else env.get("MUJOCO_GL", "egl")

    cmd = [bridge_python, "-m", "roqsim_ros_bridge.run_bridge", "--world", world]
    if not gui:
        cmd.append("--headless")

    return [ExecuteProcess(cmd=cmd, output="screen", env=env)]


def generate_launch_description():
    pkg = get_package_share_directory("roqsim_nav2_example")
    default_world = os.path.join(pkg, "worlds", "g1_nav2.yaml")
    default_map = os.path.join(pkg, "maps", "empty_room.yaml")
    default_params = os.path.join(pkg, "params", "nav2_params_g1.yaml")
    default_rviz = os.path.join(
        get_package_share_directory("nav2_bringup"), "rviz", "nav2_default_view.rviz"
    )

    map_yaml = LaunchConfiguration("map")
    params_file = LaunchConfiguration("params_file")
    use_sim_time = LaunchConfiguration("use_sim_time")

    configured_params = RewrittenYaml(
        source_file=params_file,
        root_key="",
        param_rewrites={"use_sim_time": use_sim_time, "yaml_filename": map_yaml},
        convert_types=True,
    )

    lifecycle_nodes = [
        "map_server",
        "planner_server",
        "controller_server",
        "behavior_server",
        "bt_navigator",
    ]

    def nav2_node(package, executable, name):
        return Node(
            package=package,
            executable=executable,
            name=name,
            output="screen",
            parameters=[configured_params],
        )

    return LaunchDescription(
        [
            DeclareLaunchArgument("world", default_value=default_world),
            DeclareLaunchArgument("map", default_value=default_map),
            DeclareLaunchArgument("params_file", default_value=default_params),
            DeclareLaunchArgument("use_sim_time", default_value="true"),
            DeclareLaunchArgument(
                "gui",
                default_value="false",
                description="show the MuJoCo viewer window (glfw); default headless (egl)",
            ),
            # sim + ROS bridge; built at launch time so it can honour the gui arg (see helper).
            OpaqueFunction(function=_bridge_process),
            # localization stand-in: static map->odom identity.
            Node(
                package="tf2_ros",
                executable="static_transform_publisher",
                name="static_map_odom",
                output="screen",
                arguments=["--frame-id", "map", "--child-frame-id", "odom"],
                parameters=[{"use_sim_time": use_sim_time}],
            ),
            # The base_link->lidar static TF comes from the sim: the lidar plugin declares its mount
            # frame (from the MuJoCo site) and the ros2_bridge publishes it on /tf_static.
            nav2_node("nav2_map_server", "map_server", "map_server"),
            nav2_node("nav2_planner", "planner_server", "planner_server"),
            nav2_node("nav2_controller", "controller_server", "controller_server"),
            nav2_node("nav2_behaviors", "behavior_server", "behavior_server"),
            nav2_node("nav2_bt_navigator", "bt_navigator", "bt_navigator"),
            Node(
                package="nav2_lifecycle_manager",
                executable="lifecycle_manager",
                name="lifecycle_manager_navigation",
                output="screen",
                parameters=[
                    {"use_sim_time": use_sim_time, "autostart": True, "node_names": lifecycle_nodes}
                ],
            ),
            # RViz for visualisation, only when gui:=true (alongside the MuJoCo viewer window).
            Node(
                package="rviz2",
                executable="rviz2",
                name="rviz2",
                output="screen",
                condition=IfCondition(LaunchConfiguration("gui")),
                arguments=["-d", default_rviz],
                parameters=[{"use_sim_time": use_sim_time}],
            ),
        ]
    )
