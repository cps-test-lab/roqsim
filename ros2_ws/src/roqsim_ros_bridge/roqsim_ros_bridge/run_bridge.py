"""Console entry point: run a roqsim world (with the ROS bridge plugin) as a ROS 2 node.

    ros2 run roqsim_ros_bridge roqsim_bridge --world <world.yaml>

This simply delegates to the standalone :mod:`roqsim.runner`; the world YAML is expected to
include the ``ros2_bridge`` (and optionally ``sim_interfaces``) plugin. Defaults to the bundled
turtlebot ROS world when ``--world`` is omitted.
"""

from __future__ import annotations

import os
import sys

from ament_index_python.packages import get_package_share_directory

from roqsim.runner import main as runner_main


def main(argv=None):
    """Translate ``--world <path>`` into the runner's positional target.

    ``--world`` stays this entry point's interface -- it is what the launch files and the ROS-side
    documentation pass -- while :mod:`roqsim.runner` now takes the world as a positional ``target`` (it
    also accepts scenes and model references, which a flag named ``--world`` would misdescribe).
    """
    argv = list(sys.argv[1:] if argv is None else argv)
    world = None
    rest = []
    it = iter(argv)
    for arg in it:
        if arg == "--world":
            world = next(it, None)
        elif arg.startswith("--world="):
            world = arg.split("=", 1)[1]
        else:
            rest.append(arg)
    if world is None:
        world = os.path.join(
            get_package_share_directory("roqsim_ros_bridge"), "worlds", "turtlebot_ros2.yaml"
        )
    return runner_main([world] + rest)


if __name__ == "__main__":
    raise SystemExit(main())
