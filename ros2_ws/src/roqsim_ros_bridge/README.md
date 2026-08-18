# roqsim_ros_bridge

The ROS 2 bridge for **roqsim**, provided as roqsim plugins:

- `ros2_bridge` — `cmd_vel` in; `odom` + dynamic `tf` (odom→base_link), `scan`, `clock`,
  `joint_states` out. This node is the time source (publishes `/clock`); run every other node with
  `use_sim_time:=true`.
- `sim_interfaces` — a subset of [`simulation_interfaces`](https://github.com/ros-simulation/simulation_interfaces):
  `GetSimulatorFeatures`, `GetEntities`, `Get/SetEntityState`, `Get/SetSimulationState`,
  `StepSimulation`, `ResetSimulation`.

The `roqsim` **core stays ROS-free**; this package adds the ROS dependencies and registers the two
plugins under the `roqsim.plugins` entry-point group.

## Build

```bash
cd ros2_ws
source /opt/ros/jazzy/setup.bash
colcon build --packages-select roqsim_ros_bridge
```

## Interpreter note (important)

`roqsim` and `mujoco` are a plain pip package installed in a Python venv, **not** a ROS package.
colcon installs the console script with a `/usr/bin/python3` shebang, which does not see the venv —
so run the bridge with the **venv interpreter** (the same one that has `roqsim` + `mujoco`), with
both ROS and the colcon install sourced so `rclpy`, `simulation_interfaces`, and `roqsim_ros_bridge` are
on `PYTHONPATH`:

```bash
source /opt/ros/jazzy/setup.bash
source ros2_ws/install/setup.bash
MUJOCO_GL=egl python3 -m roqsim_ros_bridge.run_bridge \
    --world ros2_ws/src/roqsim_ros_bridge/worlds/turtlebot_ros2.yaml
# (python3 here = your venv python with roqsim installed)
```

Alternatively install `roqsim` into the interpreter ROS uses (e.g.
`/usr/bin/python3 -m pip install -e . --break-system-packages`), after which
`ros2 run roqsim_ros_bridge roqsim_bridge` and the launch file work directly.

## Try it

```bash
# terminal 1: the sim + bridge (headless)
MUJOCO_GL=egl python3 -m roqsim_ros_bridge.run_bridge \
    --world ros2_ws/src/roqsim_ros_bridge/worlds/turtlebot_ros2.yaml

# terminal 2 (sourced): drive it and watch sensors
ros2 topic pub /cmd_vel geometry_msgs/msg/Twist '{linear: {x: 0.2}, angular: {z: 0.3}}'
ros2 topic echo /odom
ros2 topic echo /scan
ros2 service call /get_simulator_features simulation_interfaces/srv/GetSimulatorFeatures
```

A viewer window opens by default; pass `--headless` to run without one.
