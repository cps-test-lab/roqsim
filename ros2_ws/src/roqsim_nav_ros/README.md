# roqsim_nav_ros

The ROS 2 goal interface for everything roqsim navigates itself: a pedestrian, an opponent robot, a
driven prop. It serves the navigator's goal endpoints as nav2's `NavigateToPose` and
`NavigateThroughPoses`, and its configured route as `roqsim_nav_interfaces/StartRoute`.

One package serves every mover: the bridge's handler registry holds one handler per action type, so
a second package registering the same type would let install order decide which one runs.

## How it plugs in

1. `roqsim_nav`'s navigator (ROS-free) declares its goal endpoints with the action type named as a
   string.
2. `roqsim_ros_bridge` sees the `action` hint, looks the type up in its handler registry, and serves
   an `ActionServer` at `<namespace>/<name>`.
3. This package supplies the handlers and advertises them in the `roqsim_ros_bridge.extensions`
   entry-point group (`setup.py`), which the bridge imports at start-up.

`nav2_msgs` is a dependency of this package only; the core bridge stays nav2-free.

## Run the walker demo

The demo world holds a `roqsim_walker` pedestrian, a pip package to install beside this one.

```bash
colcon build --packages-select roqsim_nav_ros
source install/setup.bash
ros2 launch roqsim_nav_ros walker_nav.launch.py
```

In another sourced shell:

```bash
ros2 action list                      # -> /navigate_through_poses
ros2 action send_goal /navigate_through_poses nav2_msgs/action/NavigateThroughPoses \
  "{poses: [{header: {frame_id: map}, pose: {position: {x: 2.0, y: 0.0}}},
            {header: {frame_id: map}, pose: {position: {x: 0.0, y: 2.0}}}]}" --feedback
```

The walker leaves its patrol, walks the poses, and the goal succeeds on arrival. Feedback is paced on
sim time. When the route completes the walker resumes its configured patrol, if it had one.

## Semantics

| Situation | Result |
|---|---|
| Goal accepted, mover arrives | `succeed()` |
| Goal cancelled | mover stops; `canceled()` |
| A newer goal arrives first | older goal `abort()`s |
| Empty pose list | `abort()` |

Several movers under one bridge need no configuration: a handler resolves the navigator from its
endpoint's owner. Give each a `namespace:` to scope its action name
(`/<ns>/navigate_through_poses`).
