"""nav2 + roqsim Unitree G1 humanoid bring-up.

Starts: the sim + ROS bridge (roqsim_ros_bridge) running the g1_nav2.yaml world (the G1 walks via
its RL locomotion policy, driven by /cmd_vel), a static map->odom transform (localization stand-in),
pointcloud_to_laserscan projecting the head-mounted Livox Mid-360's cloud into the /scan nav2 reads,
nav2 map_server + planner_server + controller_server + behavior_server + bt_navigator, and a
lifecycle manager to activate them. Everything runs with use_sim_time (the bridge publishes /clock).

Needs the pointcloud_to_laserscan package (ros-jazzy-pointcloud-to-laserscan).

    ros2 launch roqsim_nav2_example nav2_g1.launch.py
    ros2 launch roqsim_nav2_example nav2_g1.launch.py gui:=true   # MuJoCo viewer + rviz2
"""

import math
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
            # The static chain base_link->torso_link->mid360_link comes from the sim: the robot's
            # manifest declares torso_link and the mid360 mount publishes its own frame, and the
            # ros2_bridge sends both on /tf_static.
            #
            # nav2's costmaps read a LaserScan on /scan; the G1's sensor is a Livox Mid-360 point
            # cloud. No Unitree or published G1 navigation setup states a projection, so every value
            # below is this example's assumption, derived from the G1 model and nav2_params_g1.yaml:
            #   target_frame   base_link: the costmaps' robot_base_frame, so the scan's origin is the
            #                  robot and the height band is relative to the pelvis.
            #   min_height     -0.55 m: the floor is 0.77-0.79 m below base_link while the policy
            #                  marches in place, and the body's tilt (about 0.9 deg) lifts floor
            #                  returns by up to 0.13 m at range_max. Obstacles lower than about 0.22 m
            #                  above the floor are not in the scan.
            #   max_height     0.60 m: the top of the head, 1.38 m above the floor; nothing higher can
            #                  touch the robot.
            #   range_min      0.45 m: the robot's own shoulders, arms and hands return up to 0.41 m
            #                  from base_link horizontally (standing, unitree_g1_dex1). They are real
            #                  returns of the sensor, but not obstacles to plan around. The scan's
            #                  (0, 0, 0) points (the Mid-360's no-return value) fall inside it too.
            #   range_max      8.0 m: the costmaps' obstacle_max_range; beyond it the tilt above
            #                  would put the floor inside the band.
            #   angle_*        a full turn at the device model's azimuth step, 360 rays.
            #   scan_time      0.1 s: the Mid-360's 10 Hz frame rate.
            Node(
                package="pointcloud_to_laserscan",
                executable="pointcloud_to_laserscan_node",
                name="pointcloud_to_laserscan",
                output="screen",
                remappings=[("cloud_in", "/livox/lidar"), ("scan", "/scan")],
                parameters=[
                    {
                        "use_sim_time": use_sim_time,
                        "target_frame": "base_link",
                        "min_height": -0.55,
                        "max_height": 0.60,
                        "range_min": 0.45,
                        "range_max": 8.0,
                        "angle_min": -math.pi,
                        "angle_max": math.pi,
                        "angle_increment": 2.0 * math.pi / 360.0,
                        "scan_time": 0.1,
                        "use_inf": True,
                    }
                ],
            ),
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
