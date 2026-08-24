# roqsim_webctrl

robotics-site-ops (`rso_web`) integration for roqsim.

The **sim-control** web plugin (a bottom-right play / pause / step / step-many / reset transport with a
sim-time and real-time-factor readout) is generic and ships inside `rso_web`. It calls the standard
`simulation_interfaces` services — `set_simulation_state`, `step_simulation`, `reset_simulation`,
`get_simulation_state` — which `roqsim_ros_bridge` already exposes over rosbridge. So the sim side
needs **no** new code; it just needs the plugin enabled in the deployment's `web.yaml`.

This package ships that reusable fragment so any scene can include it without hand-authoring:

```yaml
# web.yaml (any roqsim scene)
components:
  - id: sim-control
    placement: bottom-right
    # steps: 10          # physics steps per single-step click (default 1)
    # manySteps: 200     # physics steps per step-many click (default steps * 10)
    # namespace: "/r1"   # only if the services are namespaced (e.g. per robot)
```

The same fragment is available programmatically:

```python
from roqsim_webctrl import sim_control_fragment_path
```

Because it is generic, one fragment works across every scene — the reason this lives in roqsim and
not in a task-specific package.
