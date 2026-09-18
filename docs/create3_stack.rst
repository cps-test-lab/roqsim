The Create 3 / TurtleBot 4 stack
================================

``roqsim_create3_toolbox`` runs the iRobot Create 3's own software stack -- and the TurtleBot 4's
node on top of it -- over roqsim's ``turtlebot4`` model, so a stack, a scenario or an evaluator
that talks to a TurtleBot 4 finds the same ROS 2 interface it finds on the robot and in the
robot's Gazebo simulator: ``wheel_vels``, ``hazard_detection``, ``dock_status``, ``kidnap_status``,
the ``safety_override`` parameter, the ``e_stop`` service, the ``dock``/``undock`` actions.

None of that is reimplemented here. The Create 3 API is the released ``irobot_create_nodes``, the
adapters that turn a simulator's raw streams into Create 3 events are the released
``irobot_create_gz_toolbox`` (Gazebo-named, but every adapter in it except the bumper consumes plain
ROS types), and the TurtleBot 4's node is the released ``turtlebot4_node``. What this package adds is
the one adapter that could not be reused, the launch files, and the world; what the model adds is
the raw streams on the names those adapters' shipped parameter files expect.

Run it
------

.. code-block:: bash

   source /opt/ros/jazzy/setup.bash
   source ros2_ws/install/setup.bash
   python -m roqsim_ros_bridge.run_bridge --world ros2_ws/src/roqsim_create3_toolbox/worlds/turtlebot4_create3.yaml --headless
   ros2 launch roqsim_create3_toolbox turtlebot4_nodes.launch.py    # or create3_nodes.launch.py

Then, as on the robot::

   ros2 topic hz /wheel_vels                                    # ~62 Hz
   ros2 param get /motion_control max_speed                     # 0.306; 0.46 after ...
   ros2 param set /motion_control safety_override full
   ros2 service call /e_stop irobot_create_msgs/srv/EStop "{e_stop_on: true}"
   ros2 action send_goal /dock irobot_create_msgs/action/Dock {}

Nav2 runs beside it exactly as in :doc:`nav2_example`, publishing ``cmd_vel``: ``motion_control``
owns that topic, applies the safety features, and hands the clamped command to the base on
``diffdrive_controller/cmd_vel``, which is what the world's ``diff_drive`` listens to.

How it is layered
-----------------

The same three layers the robot's Gazebo simulator has, so the stack's behaviour is the stack's
and only the physics is roqsim's:

.. list-table::
   :header-rows: 1
   :widths: 24 38 38

   * - What the stack consumes
     - The Gazebo TurtleBot 4 supplies it as
     - roqsim supplies it as
   * - ``diffdrive_controller/cmd_vel`` (``TwistStamped``) in; ``odom`` + ``odom -> base_link`` out
       at 62 Hz; 0.46 m/s, 1.9 rad/s, 0.9 m/s²; a 0.5 s command timeout
     - ``ros2_control`` ``diff_drive_controller``
     - ``diff_drive`` with the world's ``stamped_cmd_vel``, ``topics``, limits, ``cmd_vel_timeout``
       and ``odom_rate_hz``
   * - ``joint_states`` carrying every joint in one message, wheels and wheel-drop suspension
     - ``joint_state_broadcaster``
     - ``joint_state_publisher``, the base's own switched off
   * - Bumper events on ``_internal/bumper/event``, one ``HazardDetection`` per pressed zone,
       framed by the zone
     - a contact sensor on the bumper link; the toolbox zones each contact by its bearing
     - the ``bumper`` plugin zones the contacts by the same sector table and publishes a ``Bool``
       per zone; ``bumper_hazard`` (this package) turns those into the events
   * - Cliff ×4 and IR ×7 as ``LaserScan`` on ``_internal/<sensor>/scan``
     - one ray and 5×5 ray sensors in the URDF
     - ``range_sensor`` instances on sites at the URDF's poses
   * - Wheel drop from ``joint_states``, ground-truth poses of the robot, its mouse and IR receiver,
       of the dock and its halo emitter, on ``_internal/sim_ground_truth_*_pose``
     - the URDF's suspension joints; Gazebo's ``PosePublisher``
     - the model's suspension joints; ``ground_truth_pose`` instances (``site:``,
       ``relative_to: base``) on the robot and on the ``create3_dock`` prop
   * - ``hazard_detection``, ``ir_intensity``, ``wheel_vels``, ``wheel_ticks``, ``wheel_status``,
       ``kidnap_status``, ``dock_status``, ``ir_opcode``, ``mouse``, ``slip_status``, ``stop_status``,
       ``battery_state``, ``interface_buttons``; ``safety_override``, ``max_speed``; ``e_stop``,
       ``robot_power``; ``dock``, ``undock``, ``drive_*``, ``wall_follow``, ``led_animation``,
       ``audio_note_sequence``; the HMI topics
     - ``irobot_create_nodes``, ``irobot_create_gz_toolbox``, ``turtlebot4_node``
     - the same binaries, launched by this package with their shipped parameter files

What is the same, and what is not
---------------------------------

**Interfaces are identical** to the Gazebo TurtleBot 4 and to the real robot's republished surface --
the nodes that define them are the same binaries. Every topic the robot's ``republisher.yaml`` lists,
commented out or not, exists.

**Behaviour is identical for the stack layer**: the speed clamp, the backup limit, ``e_stop``, the
hazard vector, kidnap estimation, the IR intensity formula, the cliff threshold, the wheel-drop
hysteresis, the docking geometry and behaviours are the same code. **It differs for the physics
layer**: contact forces, ray hits, wheel slip and suspension travel come from MuJoCo, so a bumper
zone or a cliff fires on the same geometric condition, not at the same millisecond.

**Neither simulator is the robot.** Inherited knowingly from the reference simulator: reflexes are a
stub there (enabling one throws); ``slip_status``, ``interface_buttons``, ``stop_status`` and the
battery are mocks or models; ``motion_control`` is iRobot's re-implementation of firmware
behaviour, not the firmware; the real republisher forwards only what its configuration lists,
whereas in simulation everything is always on; the HMI display and LEDs have no physical
counterpart without the Gazebo GUI plugin; and ``kidnap_status`` never turns true, because the
simulator adapters stamp a wheel-drop event with its joint's name and a cliff event with
``base_link`` while the kidnap estimator counts events framed ``wheel_drop_left`` and
``cliff_<sensor>`` -- the hazards themselves (four ``CLIFF``, two ``WHEEL_DROP``) are on
``hazard_detection`` when the robot is lifted, in both simulators. The firmware's
``wheel_accel_limit`` is not a parameter of the simulated ``motion_control``: the ramp is the base's
``diff_drive: {wheel_accel_limit}``.

On the real robot
-----------------

A topic that is missing on the robot is usually commented out in the republisher's parameter
file: ``turtlebot4_bringup/config/republisher.yaml`` is what ``create3_republisher`` (started by
``turtlebot4_bringup``'s ``robot.launch.py``, argument ``create3_param_file``) forwards from the
Create 3's own DDS domain to the compute board's, and it has nothing to do with ``nav2_params.yaml``.
Uncomment ``wheel_vels`` there (the installed copy, or your own file passed as
``create3_param_file:=``) and restart the ``turtlebot4`` service. The safety parameters are
parameters of the Create 3's ``motion_control`` node; which node name is reachable from the compute
board depends on the discovery-server and namespace setup, so check ``ros2 node list`` there.

Varying it in a campaign
------------------------

``config/create3_params.yaml`` holds the ``motion_control`` parameters as one file, so a campaign
tool that rewrites a parameter file per configuration addresses
``motion_control.ros__parameters.safety_override`` and nothing else; the base's and the sensors'
keys are world keys (``components.robot.diff_drive.wheel_accel_limit``,
``components.robot.cliff_front_left.max_range``) and are varied as any world key is; and the
Create 3 topics are recorded and converted like any other, given ``irobot_create_msgs`` wherever
the bags are read.
