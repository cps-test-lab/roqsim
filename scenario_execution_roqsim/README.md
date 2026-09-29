# scenario_execution_roqsim — what a scenario can ask an roqsim simulation

The substrate's OpenSCENARIO 2 vocabulary. The actions that observe a run and drive one (the
others -- spawning, deleting, placing and navigating entities -- are declared, with their arguments,
in [`lib_osc/roqsim.osc`](src/scenario_execution_roqsim/lib_osc/roqsim.osc)):

| action | what it does |
| --- | --- |
| `entity_moved(entities, threshold, mode, dwell, require)` | succeeds once the named entities have been **displaced** from where they were when the action started |
| `entity_rotated(entities, angle, dwell, require)` | ...once they have **turned** by an angle (geodesic, so axis-free) |
| `entity_near(entity, target, distance, mode)` | succeeds once the entity is **within a distance** of another entity |
| `entity_near_position(entity, position, distance, mode)` | ...of a point |
| `entity_in_region(entity, region, outside)` | succeeds once the entity is **inside** a box or polygon of the floor plan (or, with `outside`, outside it) |
| `entity_monitor(entity, value, target_variable)` | keeps a scenario variable equal to a value a **plugin publishes** about the entity, every tick; never succeeds on its own |
| `entity_call(entity, command, value, require_verified)` | sends a command a plugin declares -- a `model_override` fault, a sensor's `fault:` block, a tare -- and succeeds once it has applied **and, where the command names a confirmation, it says the command landed** |

```
import osc.roqsim

do parallel:
    serial:
        drive_somewhere()
        emit end
    serial:
        entity_moved(entities: ['parcel'], threshold: 0.05, mode: displacement_mode!z, dwell: 8.0)
        entity_call(entity: 'grip_fault', command: 'override', value: 'true')
```

`value` is JSON; a command declared with typed parameters takes a mapping of them, or a bare value
for its only one (`value: 'true'` for `override(data: bool)`). `entity_call` fails the trial
(FAILURE, not an exception) when the producer refuses the command, when no outcome arrives in time,
and with `require_verified` (the default) when the confirmation reports `no_effect` or could not be
read. `command` is the command's name, or
`'<component>/<name>'` to pick one of two with that name (`entity_call(entity: 'ur5e', command:
'force_torque/tare')`); the parameter is not called `call` because that is an OpenSCENARIO keyword.

## Conditions on what a plugin reports

A plugin that knows something about an entity -- a `force_limit` that tripped, a `clearance`
distance, a trial plugin's own verdict -- publishes it as an `out` endpoint on that entity.
`entity_monitor` keeps a variable equal to one field of it, the way `osc.ros`'s `topic_monitor` does
for a topic, and every condition is then plain OpenSCENARIO over that variable:

```
import osc.helpers
import osc.roqsim

scenario trial:
    timeout(120s)
    min_clearance: float = 0.3
    var tripped: bool = false
    var clearance: float = 10.0
    do parallel:
        entity_monitor(entity: 'ur5e', value: 'force_limit.tripped', target_variable: tripped)
        entity_monitor(entity: 'robot', value: 'clearance.current', target_variable: clearance)
        serial:
            wait tripped == true
            emit end
        serial:
            wait clearance < 0.2
            emit fail
```

| pattern | how it reads |
| --- | --- |
| end the run on an outcome | `wait tripped == true` then `emit end` |
| threshold | `wait clearance < 0.3` |
| assertion: fail if it ever goes bad | a parallel branch: `wait clearance < 0.2` then `emit fail` |
| bound an action | `entity_navigate(entity: 'cart', goal_poses: [...]) with:` then `until docked == true` |
| combined conditions | `wait tripped or clearance < 0.1` |
| comparison with a parameter | `wait clearance < min_clearance` |
| event and condition | `wait @fault_on if clearance < 0.3` |

A flag is compared explicitly (`tripped == true`): a condition is a comparison or a logical
expression, not a bare variable. A condition that must hold for a length of time is composed in the
language itself; scenario-execution's language documentation describes that pattern.

`value` is `<endpoint>.<field>` as the world names it, never a topic; a bare `'force_limit'` means the
field its ROS publication carries (`tripped`), so the short form reads one value wherever it runs.
`target_variable` names a `var` of the scenario or of an actor.

- **It never succeeds.** Put it in a `parallel` branch beside the ones that decide; it runs until
  the scenario ends or its branch is ended (`until`, `one_of`).
- **Until the first reading arrives the variable keeps its declared default**, so declare one the
  condition does not hold for (`false`, a clearance larger than any threshold).
- **Refused, with the same text on both transports:** an entity, endpoint or field that does not
  exist (listing the ones that do), and a field that is not a single number, flag or string (naming
  it). These are authoring errors and raise; a field that holds no value yet leaves the variable
  unchanged.
- In a stepped run the variable is written on every tick. Over the control socket a read is a
  round-trip, so it follows at the rate replies arrive.

## Conditions on where an entity is

Whether an entity is near something, or inside an area, is geometry, which an OpenSCENARIO
expression cannot compute; so these two are actions that succeed once their condition holds, rather
than monitors.

```
import osc.helpers
import osc.roqsim

scenario delivery:
    timeout(120s)
    event door_open
    do parallel:
        serial:
            entity_near(entity: 'robot', target: 'shelf', distance: 0.6)
            entity_near_position(entity: 'robot', position: position_3d(x: 4m, y: 2m), distance: 0.3)
            emit end
        serial:
            entity_in_region(entity: 'robot', region: [position_3d(x: 2m, y: 0m),
                                                       position_3d(x: 3m, y: 1m)])
            emit fail
        with:
            until @door_open
        serial:
            entity_near(entity: 'person', target: 'robot', distance: 1.0)
            emit fail
```

| pattern | how it reads |
| --- | --- |
| reached a place | `entity_near(entity: 'robot', target: 'shelf', distance: 0.6)` |
| reached a point | `entity_near_position(entity: 'robot', position: position_3d(x: 4m, y: 2m), distance: 0.3)` |
| came within reach, in 3D | `entity_near(entity: 'gripper', target: 'parcel', distance: 0.05, mode: distance_mode!spatial)` |
| entered an area | `entity_in_region(entity: 'robot', region: [<corner>, <corner>])` |
| left an area | `entity_in_region(entity: 'robot', region: [...], outside: true)` |
| must never enter | a parallel branch: `entity_in_region(...)` then `emit fail` |
| ...only for a while | that branch `with: until @door_open` (or `until elapsed(20s)`) |
| must arrive in time | `entity_near(...) with: timeout(60s)`, or the scenario's `timeout()` |

- **Reference points.** The distance is between the entities' reference points, and a region
  contains an entity when its reference point is inside: the origin of the entity's body, as the
  core's `sim/entities/<name>/pose` reports it, not the nearest point of its geometry. A robot
  "within 0.6 m of a shelf" has its base origin within 0.6 m of the shelf's origin.
- **Planar by default.** `entity_near` and `entity_near_position` measure in the floor plane (`distance_mode!planar`), because
  "the robot reached the shelf" and "a person came within 1 m" are statements about the floor plan,
  and the reference points of two entities of different heights never share a z. `distance_mode!spatial`
  measures in 3D.
- **Another entity or a point**: `entity_near` takes the other entity's name in `target`,
  `entity_near_position` a `position_3d` in `position` (world frame).
- **A region is an area of the floor plan.** Two points are an axis-aligned box's opposite corners,
  three or more a polygon's vertices in order (concave is fine); z is ignored, and a point on the
  boundary is inside.
- **Checked every tick, ends on success.** The ground truth is read on every tick, as `entity_moved`
  reads a pose; the action succeeds on the first tick the condition holds and never fails on its own.
  Bounding it is the scenario's: `timeout()` fails the trial when it never holds; `until` ends the
  wait as a success, so it bounds a watch (a "must never enter" branch), not a goal.
- **Absent is not unknown**, for every `entity_*` condition (`entity_moved`, `entity_rotated`,
  `entity_near`, `entity_near_position`, `entity_in_region`). An entity that exists but is absent
  (deleted, or not spawned yet) is nowhere: the condition does not hold, and the action waits,
  saying so in its feedback, until it is spawned. A name the world never had is refused at once
  (on the first tick), naming the closest entity names, with the same text on both transports.
- **Entities, not bodies.** Every condition reads the core's `sim/entities/<name>/pose`, the same
  endpoint on both routes, so it names entities (the world's `name:`), and a welded entity (a
  shelf) is a valid target on both.
- **Waiting for a welded entity to move is refused.** `entity_moved` and `entity_rotated` refuse an
  entity welded to the world at once, with the same text on both routes: the pose endpoint reports
  `movable: false`, its pose never changes, and a condition that can never be met must not look
  like a slow timeout.

## One action, two transports

An roqsim simulation is driven two ways and these actions work in both, unedited:

- **stepped, in-process** — scenario-execution's own runner owns the loop (`--simulation`). The
  action is handed the adapter and reads `MujocoSim.context`: entity poses, commands and reports
  through the world's endpoints, writes queued on the physics thread because only it may
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

One consequence, stated rather than hidden:

- Over the socket a pose is a round-trip, so a threshold crossing is resolved at the **tick period**,
  not at the physics step. A dwell shorter than one tick means "the first tick past the threshold"
  either way.

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
