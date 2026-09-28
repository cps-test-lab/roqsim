# scenario_execution_roqsim — what a scenario can ask an roqsim simulation

The substrate's OpenSCENARIO 2 vocabulary. The actions that observe a run and drive one (the
others -- spawning, deleting, placing and navigating entities -- are declared, with their arguments,
in [`lib_osc/roqsim.osc`](src/scenario_execution_roqsim/lib_osc/roqsim.osc)):

| action | succeeds when |
| --- | --- |
| `entity_moved(entities, threshold, mode, dwell, require)` | the named entities have been **displaced** from where they were when the action started |
| `entity_rotated(entities, angle, dwell, require)` | ...have **turned** by an angle (geodesic, so axis-free) |
| `entity_reports(entity, report, expected_value, comparison_operator, dwell, fail_if_bad_comparison)` | a value a **plugin publishes** about the entity compares as expected -- how a scenario ends a run on a trial's outcome |
| `endpoint_call(entity, endpoint, value, require_verified)` | a command a plugin declares -- a `model_override` fault, a sensor's `fault:` block, a tare -- has applied **and, where the endpoint names a confirmation, it says the command landed** |

```
import osc.roqsim

do parallel:
    serial:
        drive_somewhere()
        emit end
    serial:
        entity_moved(entities: ['parcel'], threshold: 0.05, mode: displacement_mode!z, dwell: 8.0)
        endpoint_call(entity: 'grip_fault', endpoint: 'override', value: 'true')
```

## Ending a run on a trial's outcome

The scenario owns when a run ends. A plugin that knows the outcome -- a `force_limit` that tripped,
a `contact` monitor, a trial plugin's own verdict -- publishes it as an `out` endpoint on the entity
it concerns, and the scenario waits on it and ends the run, with a `timeout` as the bound:

```
import osc.helpers
import osc.roqsim

scenario trial:
    timeout(120s)
    do serial:
        entity_reports(entity: 'ur5e', report: 'force_limit.tripped', expected_value: 'True')
        emit end
```

`report` is `<report>.<field>` as the world names it, never a topic; a bare `'force_limit'` means the
field its ROS publication carries (`tripped`), so the short form compares one value wherever it runs.
The comparison arguments are `osc.ros`'s `check_data`'s: `expected_value` is a Python
literal (a string is quoted inside the string, `"'resolved'"`), `comparison_operator` one of
`lt le eq ne ge gt`, and `fail_if_bad_comparison` fails instead of waiting. `dwell` is
`entity_moved`'s: the comparison must hold continuously for that much sim time. An entity, report or
field that does not exist raises, listing the ones that do.

## One action, two transports

An roqsim simulation is driven two ways and these actions work in both, unedited:

- **stepped, in-process** — scenario-execution's own runner owns the loop (`--simulation`). The
  action is handed the adapter and reads `MujocoSim.context`: entity poses from `data.xpos`, commands
  and reports through the world's endpoints, writes queued on the physics thread because only it may
  touch `model`/`data`.
- **over the control socket** — the simulator is another process (`roqsim sim`, under the ROS runner
  or any other). Every action reaches the same endpoints over the socket it serves: a pose is
  `sim/entities/<name>/pose`, a command is a `call` whose reply carries its confirmation, a report is
  a `read` of the whole value, and placement and presence are `sim/entities/set_state` /
  `set_presence`. The simulator is found as `roqsim ls` finds it (`ROQSIM_CONTROL`, the run
  directory, or the only one running), and waited for until it answers.

The transport is chosen from what the runner offered (`simulation`, or not), never declared in the
scenario — see [`access/__init__.py`](src/scenario_execution_roqsim/access/__init__.py). Both routes
resolve a name against the same list of endpoints, and placement goes through the same core
functions (`roqsim.entity_control`), so **a refusal reads the same on both** (tested in
`tests/test_ipc_access.py`). Time comes from the runner's `Clock` on either path.

Two consequences, stated rather than hidden:

- Over the socket a pose is a round-trip, so a threshold crossing is resolved at the **tick period**,
  not at the physics step. A dwell shorter than one tick means "the first tick past the threshold"
  either way.
- Over the socket a pose is an **entity's**; in-process a raw body name is accepted as well.

## Things that will bite

- **These actions cannot run under `remote()`.** A remote server is handed no `simulation` and is
  not where the simulator is. The modifier re-instantiates an action by entry-point name on another machine; anything
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

## Adding an action

Copy the nearest existing one: an abstract method on `WorldAccess`, an implementation in each of
`access/in_process.py` and `access/ipc.py`, the action class, its entry point in
[`pyproject.toml`](pyproject.toml) and its declaration in `lib_osc/roqsim.osc`. Both transports, or
the scenario stops being portable between the two shapes.

**If the in-process side reaches a new blackboard key, pin it.** That key is published by a plugin
in a package this one deliberately does not import (see below), so nothing in the build notices a
rename of either side — the symptom is an action that raises in a campaign cell. Add a case to
`tests/test_blackboard_conventions.py` that builds the publishing plugin and reaches it through the
access layer. You do not have to remember: that file scans this package for the keys it reads and
fails, naming the prefix, until a case exists.

**If a call waits for something that may never come, answer `pending_reason`.** A call that polls `None` with no
reason turns a wiring mistake into a trial that runs out of time saying nothing. When to *stop*
waiting is the scenario's — its own `timeout()` — so a call never fails on a missing name; it only
explains itself while it waits.

## Which parts are testable where

`displacement.py` is pure numpy — no MuJoCo, no ROS, no scenario-execution — so the one part with a
right and a wrong answer is a table test in any venv. The actions need `scenario_execution` importable
and their tests skip without it; `access/ipc.py` imports ZeroMQ only when no in-process simulation
was handed over.

The package depends on `roqsim` and nothing else. Importing a plugin package would pull MuJoCo into
the behaviour-tree build, which happens before any world is compiled, and it would not stop at one:
every package whose plugins a scenario can address would follow. So the coupling to those plugins is
conventional — a blackboard key here, a service name there — and `tests/test_blackboard_conventions.py`
is what keeps the two sides honest, at test time, where importing costs nothing.
