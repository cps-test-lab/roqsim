nav2 example
============

``roqsim_nav2_example`` brings up a minimal `nav2 <https://docs.nav2.org>`_ stack on top of the
roqsim TurtleBot 4 -- and on a car-like base, the PiRacer (below) -- and includes headless
goal-reaching integration tests.

What it starts
--------------

* the sim + ROS 2 bridge (``roqsim_ros_bridge``) running the example world;
* a static ``map``→``odom`` identity transform as a localization stand-in (the robot spawns at the
  map origin and diff-drive odometry is accurate, so no AMCL is needed — a deliberate simplification
  that keeps the example robust);
* nav2 ``map_server`` + ``planner_server`` (NavFn) + ``controller_server`` (Regulated Pure Pursuit)
  + ``behavior_server`` + ``bt_navigator``, activated by a lifecycle manager.

The scan is published in the RPLIDAR's ``rplidar_link`` frame, and the bridge publishes the static
``base_link`` → ``shell_link`` → ``rplidar_link`` transforms the robot's model declares, so no
``robot_state_publisher`` / URDF TF chain is required.

Run it
------

.. code-block:: bash

   source /opt/ros/jazzy/setup.bash
   source ros2_ws/install/setup.bash
   ros2 launch roqsim_nav2_example nav2_turtlebot.launch.py

Then send a goal (e.g. with the RViz "Nav2 Goal" tool, or ``nav2_simple_commander``). Run other
nodes with ``use_sim_time:=true``.

The Depot world with AMCL (Gazebo-compatible)
---------------------------------------------

``nav2_turtlebot_depot.launch.py`` is the drop-in-for-Gazebo variant. It runs the same TurtleBot 4
in the **Depot** world (``roqsim_scenes:depot``, baked from the Gazebo/Fuel model) with the stock nav2
Depot map, and localizes with **AMCL** instead of the static ``map->odom`` stand-in — mirroring nav2's
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

Car-like robots: the PiRacer
----------------------------

``nav2_params.yaml`` rotates the robot to a path's heading before it drives, which a car-like base
cannot do: ``ackermann_drive`` and ``tricycle_drive`` answer ``cmd_vel`` with ``v = 0`` by not moving,
so under those params the robot sits still. ``nav2_carlike.launch.py`` runs the stack for them, on
the PiRacer (``robot:=piracer``, the default):

.. code-block:: bash

   ros2 launch roqsim_nav2_example nav2_carlike.launch.py gui:=true

* ``params/nav2_params_carlike.yaml`` -- the same for every car-like robot: Smac Hybrid-A* over
  Reeds-Shepp motions (a plan may reverse), Regulated Pure Pursuit with ``allow_reversing`` and no
  rotate-to-heading, a behaviour tree that replans only when the path is blocked and never spins
  (``navigate_w_replanning_only_if_path_becomes_invalid.xml``), and no spin behaviour;
* ``params/carlike_<robot>.yaml`` -- loaded after it, so its values win: the footprint polygon about
  ``base_link``, the ``minimum_turning_radius``, speeds, lookahead, inflation, tolerances and which
  costmap layers there are (the PiRacer has no scanner, so its costmaps are the map alone);
* ``worlds/<robot>_nav2.yaml`` -- the robot in the built-in empty room, with the ROS bridge and
  ``sim_interfaces``. The drive comes from the robot's manifest, so the world declares none.

The velocity command's message type is the drive's ``stamped_cmd_vel`` (``geometry_msgs/Twist``
unless it is set) and Nav2's ``enable_stamped_cmd_vel`` (unstamped on Jazzy unless it is set);
``nav2_params_carlike.yaml`` sets neither, so both sides use ``Twist``. A stack that publishes
``TwistStamped`` sets both, because a subscription takes one type and a mismatch leaves the robot
with no command at all.

``base_link`` must be the point the robot turns about, the centre of its fixed axle.
``minimum_turning_radius`` is a planning value, not the physical minimum: the physical minimum is
the geometry's (``wheelbase / tan(max_steer_angle)`` for a car), and the planning value is chosen
above it, so the controller has lock left to correct a tracking error with and Hybrid-A* stays out of
tight many-cusp manoeuvres pure pursuit cannot follow. Each per-robot file states both numbers.

To bring up another car-like robot, copy ``worlds/piracer_nav2.yaml`` and
``params/carlike_piracer.yaml`` to ``<name>_nav2.yaml`` and ``carlike_<name>.yaml``, change what
their comments name, list them in ``setup.py``, and pass ``robot:=<name>``; add an entry to
``ROBOTS`` in ``test/test_nav2_carlike_goals.py`` to test it. A ``tricycle_drive`` base uses the same
setup: its physical minimum turning radius is ``|steer_offset| / tan(max_steer_angle)``, its
``minimum_turning_radius`` is chosen above that, and its ``base_link`` is the centre of the fixed
axle, which ``tricycle_drive`` requires anyway. :doc:`plugins`, "Choosing a drive for a new robot",
says which drive plugin to start from.

The goal-reaching tests
-----------------------

``test/test_nav2_carlike_goals.py`` drives each robot in its ``ROBOTS`` (the PiRacer) to two goals --
ahead and to the left facing left, then straight back from there -- and judges each on ground
truth, not on the odometry Nav2 steers by: the robot entity's world pose from ``sim_interfaces``'
``get_entity_state`` (:doc:`interfaces`), read when Nav2 reports the goal done. It also asserts the
steering swung over on the turn and the odometry ran backwards on the second goal. Each launch gets a ROS domain of its
own (``ROS_DOMAIN_ID`` if set, otherwise one derived from the test's process id) on localhost only,
because a second simulator's ``/clock`` on the same domain breaks every sim-time wait in Nav2.
``CARLIKE_NAV_LOG=<dir>`` keeps each launch's output.

``test/test_nav2_goal.py`` launches the TurtleBot stack headless, sends one goal via
``nav2_simple_commander.BasicNavigator``, and asserts the robot reaches within a loose radius under a
generous timeout. Both run as part of ``make test`` **when ROS is sourced**:

.. code-block:: bash

   source /opt/ros/jazzy/setup.bash
   make test          # unit tests + these nav2 integration tests

Each test starts the launch tree with the venv interpreter (so the bridge subprocess can import
``roqsim``) and skips cleanly when ROS/nav2 is unavailable.
