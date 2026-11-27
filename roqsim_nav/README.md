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
- spawn_robot: {model: turtlebot4, pose: {position: {x: 4, y: -3}}}   # an opponent whose wheels really turn
  name: cart
  components: [ {navigator: {speed: 0.4, goals: [[4, 3]]}} ]
```

## What varies, and what does not

Everything above the output line is shared — the plan, the follower, the caution probe, the goal
interface, local avoidance. The only thing that differs per entity is how motion reaches the physics:

| `output` | moves | cost |
|---|---|---|
| `drive` | a `spawn_robot` | Calls the `RobotHandle.drive(vx, vy, w)` its controller published — the *same* entry the ROS bridge writes `/cmd_vel` into. `diff_drive` still does its own inverse kinematics, acceleration ramp and odometry. A full robot in the solver. |
| `mocap` | `spawn_model: {motion: driven}` | Writes the pose. **Zero solver DOFs**: collision geometry the robot sees, nothing to integrate. The cheap default. |
| `walker` | a `walker` | Seventeen mocap bodies and a gait, also zero DOFs. Registered by `roqsim_walker`, not listed here. |

Outputs resolve from the **`roqsim_nav.outputs`** entry-point group (or `module:Class`, or
`file.py:Class` beside a world), so an out-of-tree embodiment needs no edit here — and nothing in
this package branches on the name of one. The base geometry a `drive` output shapes its command for
is declared by the drive itself (`RobotHandle.kinematics`), never keyed on a robot's name.

## One block says how a mover behaves around what its plan did not contain

Three capabilities, each one question with one answer, and no combination table to learn:

```yaml
avoidance:
  stop: true            # look ahead and hold until the way is clear      (default)
  steer: give_way       # which shared model gives way for it, or `none`  (default: none)
  reroute: false        # remember what stopped it and plan around it     (needs `stop`)
  lookahead: 0.6        # ...and the probe's tuning, in the same block
  params: {neighbor_dist: 4.0}   # anything the chosen model accepts
```

They are deliberately **not** a ladder. A walker steers without ever stopping — that is how every
existing pedestrian world behaves — and an ordered scale from "ignore" to "reroute" cannot say it.
Keeping them separate also means each key stands alone: `stop` is the forward probe, `steer` is the
local model, `reroute` is the only one that can change the planned path.

Not stopping is a decision rather than an oversight. An opponent that must be in the same place at
the same time in every repetition should not stop, because stopping for the robot under test makes
its trajectory a function of that robot's behaviour.

## Local avoidance is a second registry

ORCA is *an* answer, not the answer. `steer:` names one, resolved from
**`roqsim_nav.avoidance`** the same three ways as a plugin. The interface is
*preferred velocity in, achievable velocity out* — forces, accelerations and sampled rollouts stay
inside an implementation. `submit`/`solve`/`result` are three phases rather than one call, so a
batched model can compute every agent at once and the order plugins appear in a world cannot matter.
`test_avoidance_contract.py` runs against a **stub** and must pass with `rvo2` uninstalled: if it
needed ORCA, the interface would have been shaped around ORCA.

**Who yields is derived, not configured.** An entity with a navigator is apparatus and gives way; an
entity without one is the subject, and joins as a non-yielding agent whose state is overwritten from
ground truth — so the others go round it and it is never pushed by them.

`orca` needs `rvo2`, built from source. An extra cannot express its build dependency, so the
install is two commands: `pip install Cython`, then
`pip install --no-build-isolation git+https://github.com/sybrenstuvel/Python-RVO2.git`. Without a model
declared, everyone simply executes what they wanted.

Known limit, stated rather than smuggled: to local avoidance an agent is a **disc**. A long cart is
its circumscribed circle there. The global plan of a body that holds something is not
(below).

## A way for a body holding something

`roqsim_nav.lattice` plans over position **and heading** for a body and what it holds, so what is
checked is what the body will walk. It is the planner a carry follows; nothing in it knows a walker,
a robot or a task.

- **Footprint** (`roqsim_nav.footprint`): convex pieces in the body frame, each with the height band
  it occupies -- the body's collision geoms, what it holds, the arms that hold it -- built once per
  carry and rasterized once per heading bin (72 bins of 5 degrees, on a 2.5 cm grid), swept over the
  bin. A check errs toward "blocked" only, by about two cells at the root plus the bin's sweep at a
  piece's reach (`Footprint.slack`). `margin` grows every piece, for a body that follows its way with
  an error.
- **Height layers**: each piece meets only what stands in its own band -- `wall_polygons` over that
  band, the static walls and the bodies with DOFs that stand still (a door's leaf as it stands),
  except what is held. A load at 1.0 m passes over a 0.75 m bench and meets a shelf at its height;
  the legs meet what stands on the floor.
- **Moves**: arcs forward (a step of the mover's `Pace`, 0.1 m for a walker, the heading changing
  by 0 or one bin), a turn on the spot to any
  heading it sweeps clear (not twice running), a step straight back up to `back_max` (once between
  moves forward), a turn past half a turn only with `long_way`, and a shot onto the goal. A* picks
  among them by time at the pace's speed and turn rate, plus a cost for each kind and for passing
  near something.
- **Near its ends** a move the raster says is blocked is checked exactly, polygon on polygon and
  across its sweep (each piece posed at samples no point of it moves more than a centimetre
  between, checked with half that and how far the move's path bows from its chord to spare): a body stands close to what it works at, closer than the raster's
  slack. There the body's own
  pieces (`Piece.body`) may stand against what they already stand against at the start, and at
  the goal against the target it goes to work at (`dock`); what it holds may not. A goal the body
  would stand in is refused, naming what is there. Every way found is checked so once more before it
  is returned, and one that meets something raises a `RuntimeError`: a fault of the planner's.
- **Search**: Hybrid-A*'s: a state's pose stays where its moves took it and is checked there, and
  two states count as one on a 10 cm, 10 degree grid. The heuristic is, for the first state, a
  disc's cost to go over what stands on the floor; past it, the cost to go over where the root may
  stand at some heading, so what the body holds meets what stands at its height in the estimate
  too. It is weighted (`heuristic_weight`, 2 by default; 1 for the cheapest way): far fewer
  states, ways a little longer than the shortest.
- **Corridor**: the search keeps to the disc's way (`GridPlanner`, a disc of `disc_radius`),
  widened by the footprint's reach and `widen`, and once more at twice that where nothing is found;
  the way says `widened`. A search gives up after `max_states` states.
- **No way**: `NoWay`, naming where it got to, how wide the way is there and between what, and what
  the footprint meets there at every heading, with `back_max` and `long_way`.

```python
from roqsim_nav.footprint import Footprint, geom_piece
from roqsim_nav.lattice import NoWay, Pace, plan_way

way = plan_way(
    model,
    data,
    Footprint(pieces),
    (x, y, yaw),
    (gx, gy, gyaw),
    back_max=0.5,
    long_way=False,
    skip_roots=[held_root],
    pace=Pace(step=0.1, back_step=0.05, speed=0.8, turn_rate=1.0),  # a walker's, the default
)
way.segments  # walk (poses), turn (angle), back (distance)
way.summary()  # seconds, expanded, length, reverses, long_turns, margin, widened
```

`check_way` checks every pose of a way with the exact footprint, at every centimetre and degree.
The test worlds of the cases it was built for are in `tests/data/carry_*.xml`.

## Routes, and stopping

`route_mode: plan` (default) runs A\* between the given points — they are goals. `route_mode: exact`
makes the path *be* the polyline: straight legs, no planner, for replaying a scripted trajectory.
`autostart: false` holds the mover until something starts it, so a world can own the trajectory
while a scenario owns its timing (`entity_navigate_start`).

A goal the planner cannot reach ends the route as **failed** where the mover stands, rather than
sending it along a straight leg through the wall; a goal given up on after `max_recovery` recoveries
is skipped and reported when the route ends. The route's status then carries an `error` naming the
goal, and every caller below sees a failure, never an arrival.

**`stop` holds; `reroute` is how it plans around.** The planner's grid holds static walls only, so
each mover also looks ahead: a blocker is a body with DOFs or a mocap body — exactly the complement
of what `wall_polygons` rasterizes, so a wall at a corner is never the probe's problem. A stopped
mover is still on its path; a re-routing one has changed a trajectory the experiment may have been
holding fixed, which is why stopping is the default and re-routing is asked for.

What is in the way is judged against the corridor the body is about to occupy, filled in and
looked at every tick (`CautionProbe._ahead`): `width` across, from its body's front to `lookahead`
beyond it, from the bottom of its body up (the floor, for a mover standing on it), laid along the
path ahead. A thing that moves counts when any of its geoms, visual ones too, reaches into the
mover's `obstacle_height` band, and is where its collidable geoms are, by their extent; what is in the way is what reaches past the
body's front, so one already inside the body counts at once and one beside it does not; nor does
what goes along the way at least 0.8 times as fast as the mover, its velocity measured from where its
root body has been. The same corridor decides holding and setting off. An embodiment declares
`avoidance:` keys for what it moves (`NavOutput.avoidance_defaults`; a walker a 0.46 m `width`), and
a world's own key wins.

`reroute` remembers where it was stopped, as a disc that expires after `forget_after`, and plans
around it. The memory is what makes it work at all: the grid is static, so a mover that only
re-planned would compute the same path and drive into the same obstacle again. It is deliberately
not a costmap — a handful of discs, stamped onto a copy of the raster at plan time and nowhere else.

`recovery` is separate again, for a mover that is wedged rather than merely blocked: after
`stuck_time` without progress it backs away from the blocker and re-plans. Progress is `stuck_eps`,
or for a mover whose speed covers less than that in `stuck_time`, half what it does cover: a slow
mover walking its route is not wedged, and one at speed 0 is never stuck. **Nothing in the
navigator's path is random** — give_way, the probe, recovery and the planner are all deterministic
functions of the world, and the only draw anywhere here is a walker's dwell at a waypoint. Two runs
of the same world are identical to the bit. What an opponent's path is *not* independent of is the
robot under test: a mover that stops will stop for it, and one that steers will steer around it.
That coupling is the point of having opponents at all, and
`avoidance: {stop: false, steer: none}` + `route_mode: exact` is the configuration for a mover that
must ignore the subject entirely.

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
`status()` is `(seq, finished, goals_left, dist_left, error)`, and `error` is empty for an arrival.

* in a scenario: `entity_navigate(entity: 'cart', goal_poses: [...])`, or `entity_navigate_start`
* over ROS 2: nav2's `NavigateToPose` / `NavigateThroughPoses`, served by `roqsim_nav_ros`, and the
  configured route as `roqsim_nav_interfaces/StartRoute` at `<entity>/start_route` — served only by
  a mover that has one. A nav2 goal with no poses is aborted, never read as "start".
* in process: `ctx.blackboard.get("nav:cart:handle").send_goals([(2.0, 0.0)])`, or `.start()`;
  `send_goals(goals, exact=True)` walks one route as given, unplanned and unsteered, with no
  recovery, no corner cut, and its last point stopped at within 1 cm rather than at
  `arrival_radius`

## Tests

```bash
pytest roqsim_nav -q
```
