"""Nav2 on a car-like roqsim robot: one that steers rather than turning in place.

    ros2 launch roqsim_nav2_example nav2_carlike.launch.py
    ros2 launch roqsim_nav2_example nav2_carlike.launch.py robot:=piracer gui:=true

Starts the simulator with its ROS bridge on ``worlds/<robot>_nav2.yaml``, a static identity
``map -> odom`` (localisation stand-in), and Nav2's map, planner, controller, behaviour and BT
navigator servers under one lifecycle manager. Every Nav2 node reads two parameter files, in order:
``params/nav2_params_carlike.yaml`` (the car-like stack, the same for every robot) and
``params/carlike_<robot>.yaml`` (that robot's footprint, turning radius, speeds and scan sources),
so the second one's values win. The behaviour tree is nav2_bt_navigator's
``navigate_w_replanning_only_if_path_becomes_invalid.xml``; the shared params file says why.

To bring up another car-like robot, add ``worlds/<name>_nav2.yaml`` and
``params/carlike_<name>.yaml`` beside the PiRacer's and pass ``robot:=<name>``, or name the files
with ``world:=`` and ``robot_params:=``.

The launch runs under the system interpreter; the simulator runs under the active virtualenv's
Python (``VIRTUAL_ENV``), which is where roqsim is installed.
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

PACKAGE = "roqsim_nav2_example"


def _existing(path, what):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"nav2_carlike: no {what} at {path}")
    return path


def _bringup(context, *_args, **_kwargs):
    pkg = get_package_share_directory(PACKAGE)
    robot = LaunchConfiguration("robot").perform(context)
    world = LaunchConfiguration("world").perform(context) or os.path.join(
        pkg, "worlds", f"{robot}_nav2.yaml"
    )
    robot_params = LaunchConfiguration("robot_params").perform(context) or os.path.join(
        pkg, "params", f"carlike_{robot}.yaml"
    )
    _existing(world, f"world for robot {robot!r}")
    _existing(robot_params, f"per-robot params for robot {robot!r}")
    gui = LaunchConfiguration("gui").perform(context).lower() in ("true", "1", "yes")
    use_sim_time = LaunchConfiguration("use_sim_time")

    # The simulator and its bridge, under the venv that has roqsim (see the module docstring).
    python = (
        os.path.join(os.environ["VIRTUAL_ENV"], "bin", "python3")
        if os.environ.get("VIRTUAL_ENV")
        else sys.executable
    )
    env = dict(os.environ)
    env["MUJOCO_GL"] = "glfw" if gui else env.get("MUJOCO_GL", "egl")
    cmd = [python, "-m", "roqsim_ros_bridge.run_bridge", "--world", world]
    if not gui:
        cmd.append("--headless")

    bt_xml = os.path.join(
        get_package_share_directory("nav2_bt_navigator"),
        "behavior_trees",
        "navigate_w_replanning_only_if_path_becomes_invalid.xml",
    )
    shared = RewrittenYaml(
        source_file=LaunchConfiguration("params_file"),
        root_key="",
        param_rewrites={
            "yaml_filename": LaunchConfiguration("map"),
            "default_nav_to_pose_bt_xml": bt_xml,
        },
        convert_types=True,
    )

    def nav2_node(package, executable):
        # use_sim_time last, as its own parameter: a file sets only the nodes it names, and a node
        # left on wall time stamps its plans decades ahead of every transform the others publish.
        return Node(
            package=package,
            executable=executable,
            name=executable,
            output="screen",
            parameters=[shared, robot_params, {"use_sim_time": use_sim_time}],
        )

    lifecycle_nodes = [
        "map_server",
        "planner_server",
        "controller_server",
        "behavior_server",
        "bt_navigator",
    ]
    return [
        ExecuteProcess(cmd=cmd, output="screen", env=env),
        # Localisation stand-in: static map -> odom identity. The world spawns the robot at the
        # map origin, so the drive's odometry is its map pose.
        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            name="static_map_odom",
            output="screen",
            arguments=["--frame-id", "map", "--child-frame-id", "odom"],
            parameters=[{"use_sim_time": use_sim_time}],
        ),
        nav2_node("nav2_map_server", "map_server"),
        nav2_node("nav2_planner", "planner_server"),
        nav2_node("nav2_controller", "controller_server"),
        nav2_node("nav2_behaviors", "behavior_server"),
        nav2_node("nav2_bt_navigator", "bt_navigator"),
        Node(
            package="nav2_lifecycle_manager",
            executable="lifecycle_manager",
            name="lifecycle_manager_navigation",
            output="screen",
            parameters=[
                {"use_sim_time": use_sim_time, "autostart": True, "node_names": lifecycle_nodes}
            ],
        ),
    ]


def generate_launch_description():
    pkg = get_package_share_directory(PACKAGE)
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "robot",
                default_value="piracer",
                description="picks worlds/<robot>_nav2.yaml and params/carlike_<robot>.yaml",
            ),
            DeclareLaunchArgument(
                "world", default_value="", description="world file; default from robot"
            ),
            DeclareLaunchArgument(
                "robot_params",
                default_value="",
                description="per-robot params file loaded after params_file; default from robot",
            ),
            DeclareLaunchArgument(
                "params_file", default_value=os.path.join(pkg, "params", "nav2_params_carlike.yaml")
            ),
            DeclareLaunchArgument(
                "map", default_value=os.path.join(pkg, "maps", "empty_room.yaml")
            ),
            DeclareLaunchArgument("use_sim_time", default_value="true"),
            DeclareLaunchArgument(
                "gui",
                default_value="false",
                description="show the MuJoCo viewer window (glfw); default headless (egl)",
            ),
            DeclareLaunchArgument(
                "rviz", default_value="false", description="launch rviz2 (needs rviz2 installed)"
            ),
            OpaqueFunction(function=_bringup),
            Node(
                package="rviz2",
                executable="rviz2",
                name="rviz2",
                output="screen",
                condition=IfCondition(LaunchConfiguration("rviz")),
                arguments=[
                    "-d",
                    os.path.join(
                        get_package_share_directory("nav2_bringup"),
                        "rviz",
                        "nav2_default_view.rviz",
                    ),
                ],
                parameters=[{"use_sim_time": LaunchConfiguration("use_sim_time")}],
            ),
        ]
    )
