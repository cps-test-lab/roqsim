nav2 example
============

``roqsim_nav2_example`` runs `nav2 <https://docs.nav2.org>`_ on roqsim robots. The Depot setup is
the one to build an experiment on: a real building, the stock nav2 map and AMCL, as nav2's Gazebo
simulation runs them. The empty-room setups further down are for testing only.

The Depot world with AMCL (Gazebo-compatible)
---------------------------------------------

``nav2_turtlebot_depot.launch.py`` is the drop-in for Gazebo. It runs the TurtleBot 4
in the **Depot** world (``roqsim_scenes:depot``, baked from the Gazebo/Fuel model) with the stock nav2
Depot map, and localizes with **AMCL** instead of a static ``map->odom`` transform — mirroring nav2's
``tb4_simulation_launch.py`` on gz. The robot spawns at world ``(-8, 0)`` and AMCL seeds at the map
origin, fixing ``map = world + (8, 0)`` exactly as in Gazebo; ``depot_nav2.yaml`` also adds the
:doc:`ground_truth` ``ground_truth_pose`` plugin, so ``/tf`` carries ``turtlebot4_base_link_gt`` just
like the gz stack. A nav2 client — and a scenario-execution ``ros_launch`` of either backend — sees
the same ROS graph.

.. code-block:: bash

   ros2 launch roqsim_nav2_example nav2_turtlebot_depot.launch.py            # headless (egl)
   ros2 launch roqsim_nav2_example nav2_turtlebot_depot.launch.py headless:=false   # MuJoCo window

Pass ``map:=`` / ``params_file:=`` to pin an external map or nav2 params, and ``autostart:=False`` to
bring nav2 up configured-but-inactive. A comparison of this backend against gz passes all
three, so both simulators run byte-identical nav2 config *and* activate on the same condition: wait
for the simulator's first ``/scan``, then call ``manage_nodes`` with ``ManageLifecycleNodes.STARTUP``
on each lifecycle manager. Nothing in nav2 waits for a simulator — ``autostart`` arms a one-shot timer
that activates unconditionally — so with the default ``autostart:=true`` a simulator that is slow to
publish (MJCF, meshes and a GL context still loading) can leave ``collision_monitor`` judging its
scan source dead; sitting in ``cmd_vel_smoothed -> cmd_vel``, it then fails closed at zero velocity.
Raising ``source_timeout`` hides that rather than removing it, and leaves the costmaps briefly
reasoning about transforms that do not exist.

nav2 itself comes from ``nav2_bringup/bringup_launch.py``, included unmodified: the same file
``tb4_simulation_launch.py`` includes, with the same composed ``nav2_container`` and the same
``lifecycle_manager_localization`` / ``lifecycle_manager_navigation`` split. This launch adds only
what replaces the ``gz`` half — the sim + bridge, and ``robot_state_publisher`` fed from the same
``nav2_minimal_tb4_description`` xacro Gazebo uses, so the TF tree is identical by construction. The
bridge owns only what the *simulator* owns (``odom -> base_link`` and the ground-truth frame);
``depot_nav2.yaml`` sets ``publish_static_tf: false`` so the sensor-mount transforms come from the
URDF alone — two publishers for one static transform is a TF conflict, not redundancy.

The Depot world (``roqsim_scenes:depot``) ships **open** (roofless) via the generic ``ceiling`` plugin,
which is nav-neutral (the roof is above the 2D scan plane) but clears overhead sensor line-of-sight
and top-down views. Set ``ceiling.keep: true`` for the roofed warehouse.

Test setups: the empty room
---------------------------

Three launch files run nav2 in the built-in empty room with a static ``map``→``odom`` identity
transform in place of localization. They exist to test roqsim, not to run experiments on: the room has
nothing to navigate around, and the robot's odometry is taken as ground truth.

.. code-block:: bash

   source /opt/ros/jazzy/setup.bash
   source ros2_ws/install/setup.bash
   ros2 launch roqsim_nav2_example nav2_turtlebot.launch.py   # TurtleBot 4, the goal-reaching test's setup
   ros2 launch roqsim_nav2_example nav2_g1.launch.py          # Unitree G1
   ros2 launch roqsim_nav2_example nav2_spot.launch.py        # Boston Dynamics Spot

Each starts the sim + ROS 2 bridge (``roqsim_ros_bridge``) on its world (``worlds/turtlebot_nav2.yaml``,
``g1_nav2.yaml``, ``spot_nav2.yaml``), the static transform, and nav2 ``map_server`` +
``planner_server`` (NavFn) + ``controller_server`` (Regulated Pure Pursuit) + ``behavior_server`` +
``bt_navigator`` with its params file (``params/nav2_params.yaml``, ``nav2_params_g1.yaml``,
``nav2_params_spot.yaml``). Each takes ``world:=``, ``map:=``, ``params_file:=`` and
``use_sim_time:=``; the G1 and Spot ones also ``gui:=true`` (a MuJoCo viewer and rviz2).

* The TurtleBot 4's scan is published in the RPLIDAR's ``rplidar_link`` frame, and the bridge
  publishes the static ``base_link`` → ``shell_link`` → ``rplidar_link`` transforms the robot's model
  declares, so no ``robot_state_publisher`` / URDF TF chain is required.
* ``nav2_g1.launch.py`` projects the head-mounted Livox Mid-360's point cloud into ``/scan`` with
  ``pointcloud_to_laserscan``, which it needs installed (``ros-jazzy-pointcloud-to-laserscan``).
* ``nav2_spot.launch.py`` needs the NVIDIA Spot policy, which is not committed: ``make venv`` tries
  to fetch it and carries on if it cannot, ``python -m roqsim_quadruped.policy.fetch_policy`` fetches
  it, or ``SPOT_POLICY_PATH`` names a copy (see ``roqsim_quadruped/README.md``). Without one,
  ``spot_locomotion`` refuses to load.

The goal-reaching test
----------------------

``test/test_nav2_goal.py`` launches ``nav2_turtlebot.launch.py`` headless, sends one goal via
``nav2_simple_commander.BasicNavigator``, and asserts the robot reaches within a loose radius under a
generous timeout. It runs as part of ``make test`` **when ROS is sourced**:

.. code-block:: bash

   source /opt/ros/jazzy/setup.bash
   make test          # unit tests + this nav2 integration test

The test starts the launch tree with the venv interpreter (so the bridge subprocess can import
``roqsim``) and skips cleanly when ROS/nav2 is unavailable.

