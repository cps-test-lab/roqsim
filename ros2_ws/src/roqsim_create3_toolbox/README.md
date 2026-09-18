# roqsim_create3_toolbox

The iRobot Create 3 / TurtleBot 4 software stack over roqsim's `turtlebot4` model: the simulator
adapter and the launch files, so a stack, a scenario or an evaluator that talks to a TurtleBot 4
finds the same ROS 2 interface it finds on the robot and in the robot's Gazebo simulator.

Nothing Create-3-semantic is reimplemented. The Create 3 API is the released `irobot_create_nodes`,
the adapters that turn a simulator's raw streams into Create 3 events are the released
`irobot_create_gz_toolbox` (every adapter in it but the bumper consumes plain ROS types), and the
TurtleBot 4's node is the released `turtlebot4_node`. This package adds:

- `bumper_hazard` -- the one adapter that could not be reused: roqsim's `bumper` plugin zones the
  contacts itself (by the Create 3 simulator's own bearing table) and publishes a `Bool` per zone;
  this node turns a pressed zone into the `HazardDetection` event `hazards_vector_publisher` collects;
- `launch/create3_nodes.launch.py` -- the Create 3 node graph, the reference simulator's node set
  with its shipped parameter files (`config/`, copied with attribution);
- `launch/turtlebot4_nodes.launch.py` -- the above plus `turtlebot4_node`;
- `config/create3_params.yaml` -- the `motion_control` parameters (`safety_override`) as one file a
  campaign varies;
- `worlds/turtlebot4_create3.yaml` -- the robot with the base's contract to the stack
  (`diffdrive_controller/cmd_vel`, the controller's limits, a command timeout, 62 Hz odometry) and
  its charging dock.

The raw streams themselves -- bumper zones, cliff and IR ray grids, every joint's state, the
ground-truth poses of the robot, its mouse and IR receiver -- come from the `turtlebot4` model's
manifest, on the topic names the adapters' parameter files expect, and are lazy: a world that never
launches this stack publishes none of them.

See `docs/create3_stack.rst` for the layering, what is and is not the same as the robot, and how to
run and vary it.
