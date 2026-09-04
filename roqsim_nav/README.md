# roqsim_nav

2D navigation for roqsim: **one navigator, for everything that moves under its own control.**

A trial usually needs more than the robot it measures — a second robot in the aisle, a pedestrian
crossing, a cart that goes somewhere rather than along a fixed polyline. Those are *apparatus*, and
this package drives them from inside the simulator: A\* over a grid rasterized from the world's own
wall geoms, so there is no map file, no localisation, and nothing on the ROS graph but the robot
under test.

```bash
roqsim sim roqsim_nav:nav_opponents      # a TurtleBot 4 nothing drives, and three opponents
```

```yaml
- spawn_robot: {model: turtlebot4, pos: [4, -3]}     # an opponent whose wheels really turn
  name: cart
  components: [ {navigator: {speed: 0.4, goals: [[4, 3]]}} ]
```

## What varies, and what does not

Everything above the output line is shared — the plan, the follower, the caution probe, the goal
interface, local avoidance. The only thing that differs per entity is how motion reaches the physics:

| `output` | moves | cost |
|---|---|---|
| `drive` | a `spawn_robot` | Calls the `RobotHandle.drive(vx, vy, w)` its controller published — the *same* entry the ROS bridge writes `/cmd_vel` into. `diff_drive` still does its own inverse kinematics, acceleration ramp and odometry. A full robot in the solver. |
| `mocap` | `spawn_model: {mocap: true}` | Writes the pose. **Zero solver DOFs**: collision geometry the robot sees, nothing to integrate. The cheap default. |
| `walker` | a `walker` | Seventeen mocap bodies and a gait, also zero DOFs. Registered by `roqsim_walker`, not listed here. |

Outputs resolve from the **`roqsim_nav.outputs`** entry-point group (or `module:Class`, or
`file.py:Class` beside a world), so an out-of-tree embodiment needs no edit here — and nothing in
this package branches on the name of one. The base geometry a `drive` output shapes its command for
is declared by the drive itself (`RobotHandle.kinematics`), never keyed on a robot's name.

## Local avoidance is a second registry

ORCA is *an* answer, not the answer. A world declares one model, once:

```yaml
- avoidance: {model: orca, neighbor_dist: 4.0, time_horizon: 3.0}
```

resolved from **`roqsim_nav.avoidance`** the same three ways. The interface is
*preferred velocity in, achievable velocity out* — forces, accelerations and sampled rollouts stay
inside an implementation. `submit`/`solve`/`result` are three phases rather than one call, so a
batched model can compute every agent at once and the order plugins appear in a world cannot matter.
`test_avoidance_contract.py` runs against a **stub** and must pass with `rvo2` uninstalled: if it
needed ORCA, the interface would have been shaped around ORCA.

**Who yields is derived, not configured.** An entity with a navigator is apparatus and gives way; an
entity without one is the subject, and joins as a non-yielding agent whose state is overwritten from
ground truth — so the others go round it and it is never pushed by them.

`orca` needs `rvo2`, built from source: `pip install 'roqsim_nav[avoidance]'`. Without a model
declared, everyone simply executes what they wanted.

Known limit, stated rather than smuggled: an agent is a **disc**. A long cart is its circumscribed
circle. Widening to footprints is a real interface change and should be made deliberately.

## Routes, and stopping

`route_mode: plan` (default) runs A\* between the given points — they are goals. `route_mode: exact`
makes the path *be* the polyline: straight legs, no planner, for replaying a scripted trajectory.
`autostart: false` plans at load and holds the mover until something starts it, so a world can own
the trajectory while a scenario owns its timing (`entity_navigate_start`).

**Caution stops; it never re-routes.** The planner's grid holds static walls only, so each mover also
looks ahead: a blocker is a body with DOFs or a mocap body — exactly the complement of what
`wall_polygons` rasterizes, so a wall at a corner is never caution's problem. A stopped mover is
still on its path; a re-routing one has changed a trajectory the experiment was holding fixed.
`recovery` is the separate knob that *can* change a path (on by default, as for walkers); its draws
come from `ctx.rng_for`, so opponent variability is a seed you can sweep rather than noise.

`tracker: pure_pursuit` follows the path rather than chasing goal endpoints. It is **not** an
improvement for a pose-written body and the numbers say so — round a right-angle corner it cuts by
about its lookahead where the default follower cuts by its arrival radius. It is there for a base
whose steering is constrained.

## The loop closes on ground truth

Not on `read_odom`, for two independent reasons: it is the wrong frame (odom, zeroed each reset,
against a world-frame grid, with no map→odom transform to bridge them) and the wrong instrument (an
opponent's trajectory must not become a function of wheel slip, hence of contacts with the robot
under test). A mover with realistic localisation error is a different experiment.

## Commanding one at runtime

Every navigator publishes a `NavHandle` on the blackboard under `nav:<entity>:handle`, and routes are
stamped with a **sequence number** so a caller can tell its own arrival from a stale one — a
navigator that finished its previous route is already "finished" when a new one is queued.

* in a scenario: `entity_navigate(entity: 'cart', goal_poses: [...])`, or `entity_navigate_start`
* over ROS 2: nav2's `NavigateToPose` / `NavigateThroughPoses`, served by `roqsim_nav_ros`
* in process: `ctx.blackboard.get("nav:cart:handle").send_goals([(2.0, 0.0)])`

## Tests

```bash
pytest roqsim_nav -q
```
