# roqsim_walker_ros

ROS 2 goal interface for [`roqsim_walker`](../../../roqsim_walker): serves nav2's
**`NavigateThroughPoses`** action so a pedestrian can be commanded through a list of poses.

This package is deliberately tiny — one action handler. It contains no node, no launch-time wiring
and no bridge code, because the bridge already knows how to serve action endpoints.

## How it plugs in

Three decoupled pieces:

1. **`roqsim_walker`** (pure, ROS-free) declares an `in` endpoint whose backend hint *names* the
   action as a string:

   ```python
   backend = {
       "ros2": {"action": "nav2_msgs.action.NavigateThroughPoses", "name": "navigate_through_poses"}
   }
   ```

2. **`roqsim_ros_bridge`** sees an `action` hint, looks the type string up in its handler
   registry, and serves an `ActionServer` at `<namespace>/<name>`.

3. **This package** supplies that handler, and advertises itself in the
   `roqsim_ros_bridge.extensions` entry-point group (`setup.py`), which the bridge imports once at
   start-up. The import runs the `@action_handler` decorator and the handler is registered.

The result: `nav2_msgs` is a dependency of *this* package only. The core bridge stays nav2-free, and
the walker package stays ROS-free.

### Adding your own action (the generic recipe)

Any package extends the bridge the same way — no bridge edits:

```python
# mypkg/actions.py
from roqsim_ros_bridge.actions import action_handler


@action_handler("my_msgs.action.DoTheThing")
def do_the_thing(goal_handle, ctx, on_payload, endpoint):
    # `on_payload` marshals a neutral payload onto the physics thread (ctx.post).
    # `endpoint.owner` is the producing entity -> resolve its handle generically:
    handle = ctx.blackboard.get(f"mykind:{endpoint.owner}")
    ...
```

```python
# mypkg/setup.py
entry_points = {"roqsim_ros_bridge.extensions": ["mypkg = mypkg.actions"]}
```

The same group also picks up `@converter` / `@decoder` registrations for new message types.

## Run it

```bash
colcon build --packages-select roqsim_walker_ros
source install/setup.bash
ros2 launch roqsim_walker_ros walker_nav.launch.py
```

In another sourced shell:

```bash
ros2 action list                      # -> /navigate_through_poses
ros2 action send_goal /navigate_through_poses nav2_msgs/action/NavigateThroughPoses \
  "{poses: [{header: {frame_id: map}, pose: {position: {x: 2.0, y: 0.0}}},
            {header: {frame_id: map}, pose: {position: {x: 0.0, y: 2.0}}}]}" --feedback
```

The walker leaves its patrol, walks the poses, and the goal succeeds on arrival. Feedback carries
`number_of_poses_remaining` and `distance_remaining` (paced on **sim** time). Cancelling stops the
walker where it stands; a newer goal preempts (aborts) an in-flight one.

When the route completes the walker resumes its configured patrol, if it had one.

## Semantics

| Situation | Result |
|---|---|
| Goal accepted, walker arrives | `succeed()` |
| Goal cancelled | walker stops; `canceled()` |
| A newer goal arrives first | older goal `abort()`s (sequence numbers, not guesswork) |
| Empty pose list | `abort()` |

Multiple walkers under one bridge need no configuration: the handler resolves each walker from its
endpoint's `owner` via `ctx.blackboard.get(f"walker:{owner}")`. Give each walker a `namespace:` to
scope its action name (`/<ns>/navigate_through_poses`).
