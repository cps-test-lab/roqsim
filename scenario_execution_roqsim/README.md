# scenario_execution_roqsim — what a scenario can ask an roqsim simulation

The substrate's OpenSCENARIO 2 vocabulary. Three actions:

| action | succeeds when |
| --- | --- |
| `entity_moved(entities, threshold, mode, dwell, require)` | the named entities have been **displaced** from where they were when the action started |
| `entity_rotated(entities, angle, dwell, require)` | ...have **turned** by an angle (geodesic, so axis-free) |
| `set_model_override(instance, active, require_landed)` | a world's `model_override` fault has been applied (or restored) **and the plugin confirms it landed** |

```
import osc.roqsim

do parallel:
    serial:
        drive_somewhere()
        emit end
    serial:
        entity_moved(entities: ['parcel'], threshold: 0.05, mode: displacement_mode!z, dwell: 8.0)
        set_model_override(instance: 'grip_fault')
```

## One action, two transports

An roqsim simulation is driven two ways and these actions work in both, unedited:

- **stepped, in-process** — scenario-execution's own runner owns the loop (`--simulation`). The action is handed the adapter and reads `MujocoSim.context`: entity poses from
  `data.xpos`, the fault through the blackboard handle `model_override:<name>`, writes queued with
  `ctx.post` because only the physics thread may touch `model`/`data`.
- **over ROS** — the simulator is in another container. Poses come from
  `simulation_interfaces/GetEntityState`, the fault from `<instance>/override` (`std_srvs/SetBool`),
  whose reply already *is* the verdict: the bridge's handler barriers on physics twice and answers with
  the plugin's own `verified`.

The transport is chosen from what the runner offered (`simulation` vs `node`), never declared in the
scenario — see [`access/__init__.py`](src/scenario_execution_roqsim/access/__init__.py). It works
because both channels already speak the same vocabulary: `simulation_interfaces` is keyed on **entity
names**, exactly like `ctx.entities`, and time comes from the runner's `Clock` on either path. That is
`Endpoint`'s design (architecture.rst §13) applied on the scenario side instead of the plugin side.

Two consequences, stated rather than hidden:

- Over ROS a pose is a round-trip, so a threshold crossing is resolved at the **tick period**, not at
  the physics step. A dwell shorter than one tick means "the first tick past the threshold" either way.
- **TF is deliberately not the ROS pose source.** It would arrive with `map → odom` localisation error
  folded in — measured at 43 mm in x and 73 mm in y in the tiago world, which is why its
  `object_detector` exists — while `GetEntityState` is ground truth like the in-process read.

`entity_moved` and `entity_rotated` therefore work against **any** simulator serving
`simulation_interfaces`. `set_model_override` is roqsim-specific: the endpoint is that plugin's.

## Things that will bite

- **These actions cannot run under `remote()`.** A remote server is handed neither `simulation` nor
  `node`. The modifier re-instantiates an action by entry-point name on another machine; anything
  reading the simulation must stay where the simulation is.
- **`scenario_execution` is not a declared dependency**, on purpose — the PyPI name is a different,
  older project, and installing it breaks every campaign at parse time. See the note in
  [`pyproject.toml`](pyproject.toml). Every environment that runs these actions already provides it.
- **Net displacement, not path length.** `osc.ros`'s `odometry_distance_traveled` integrates; this
  measures a straight line from the baseline. On a curved approach the two disagree, sometimes a lot.
- **Signed axis thresholds.** `mode: z, threshold: 0.05` is *risen* 5 cm, not `|Δz| ≥ 5 cm`. A campaign
  sweeping `[-0.05, 0.05]` on an axis mode is sweeping two different questions.
- **Do not attach a modifier to a composition** here. `create_decorator` re-parents by append, so a
  decorated `serial:` moves to the end of its parent's children; and `success_is_running` on a Sequence
  makes it *restart* rather than hold (a py_trees `Decorator` always ticks its child), which re-takes an
  `entity_moved` baseline and re-fires a fault every crossing. Use `parallel` + `emit end`, as above.

## Which parts are testable where

`displacement.py` is pure numpy — no MuJoCo, no ROS, no scenario-execution — so the one part with a
right and a wrong answer is a table test in any venv. The actions need `scenario_execution` importable
and their tests skip without it; `access/ros.py` imports `rclpy` only when a ROS runner actually handed
the action a node, which is what keeps the package installable in a plain venv.
