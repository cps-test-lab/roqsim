# roqsim_walker

Kinematic pedestrian **walkers** for roqsim. Pure Python + MuJoCo: **no ROS dependency**.

A walker is a character mesh skinned onto a 17-joint skeleton of MuJoCo **mocap bodies**.
Blueprints are discovered across every installed `roqsim.models` provider, so a downstream
package can ship its own characters and a world still names only the walker. Body
pose comes from motion clips. Because it is fully kinematic it adds **zero DOFs** to the solver, yet
its per-limb collision capsules are what the robot's lidar and contacts see — and, because the solver
treats a mocap body as immovable, a walker will shove anything free that it walks into.

**Navigation is not here.** A walker is one *embodiment* of `roqsim_nav`'s navigator — the `walker`
output this package registers — so a pedestrian, a robot and a driven prop are moved by the same
plugin and differ only in what the motion is written into. This package owns the body: the skeleton,
the skin, the blueprints and the motion clips.

Its navigator either **patrols** a configured route, or **drives it to goals** through a backend-neutral
endpoint — which the ROS 2 bridge serves as `nav2_msgs/NavigateToPose` and
`NavigateThroughPoses` (see `ros2_ws/src/roqsim_nav_ros`).

## Quick start

```bash
pip install -e 'roqsim_walker[test]'
roqsim sim roqsim_walker/src/roqsim_walker/worlds/walker_patrol.yaml
```

## Configuration

```yaml
components:
  - walker:
      walker: MaleVisitorWalk  # blueprint folder under models/people/ (required)
      namespace: ""            # transport scope of the walker's endpoints and its navigator's
      outfit: B                # clothing variant: a letter, or {pants: C, jacket: A}
      skin: true               # false -> capsule visuals (fast; no mesh load)
      pose:                    # where the walker stands at the start and after every reset:
        position: {x: -2.0, y: -2.0}  # a world pose with no z (a walker stands on the floor)
        orientation: {yaw: 0.0}       # and a heading only; `pos` is refused
      motion: {walk: /abs/walk.npz}   # override a resolved locomotion clip
      body_lag: 0.0            # s; > 0 draws the body smoothly behind its nav root (see below)
    name: pedestrian           # entity name (a sibling of the plugin ref, not config)
    components:
      - navigator:             # where it goes: roqsim_nav's navigator, as for any mover
          output: walker
          speed: 1.2           # m/s; past ~1.7 the run clip blends in
          goals: [[2.0, -2.0], [2.0, 2.0], [-2.0, 2.0]]   # the route after the start
          loop: true           # cycle the patrol forever
          dwell: [[0, 0], [2, 4], [0, 0], [1, 3]]   # per route point, the start first
          radius: 0.26         # the walker's disc to avoidance and the planner
          avoidance: {steer: none, stop: false}
```

Navigation is stated on the nested `navigator` only; a walker's own block refuses `speed`,
`waypoints`, `loop`, `dwell`, `avoidance`, `action_name`, `orca` and the rest by name, saying where
each lives on the navigator. A walker with no nested `navigator` gets one that is goal-driven only:
`output: walker`, `speed: 1.0`, `radius: 0.26` and `avoidance: {stop: false}`.

The nav root starts, stops and turns at once. With `body_lag` set (0.15 s is a person's pace), the
body is drawn on a point following it as a critically damped spring of that time constant, turns
speed up and slow down over twice it, and the foot grounding eases too, so a walker seen close up,
or bent over something in its hands, does not jolt; it comes to rest where the nav root is. While
it moves, the drawn body trails the nav root by about `2 * body_lag * speed` (0.36 m at 0.15 s and
1.2 m/s), so what is drawn is not where the navigator, its avoidance and a pose reading of the nav
root place the walker. The default, 0, draws the body on the root.

### Navigation layers

| Layer | Module | What it does |
|---|---|---|
| Global plan | `roqsim_nav`'s `planner.py`, `occupancy.py` | 8-connected A\* over an inflated occupancy grid rasterized from the model's **wall geoms**, string-pulled to sparse waypoints |
| Behaviour | `roqsim_nav`'s `behavior.py` | py-trees `Selector[recovery, navigate]`: follow path, advance goals, back-up-and-replan when stuck |
| Local avoidance | `roqsim_nav`'s `avoidance/` | Gives way to the robot and other movers |

Walls are read straight from the compiled model (`roqsim_nav`'s `obstacles.py`). The default
`empty_room` is a **walled** room, so A\* engages on its perimeter walls; **with no wall geoms** (a wall-less MJCF via `sim.world`) the grid is skipped and walkers follow straight-line
legs. A `floorplan` mesh adds its own walls the same way.

### Avoidance

`avoidance: {steer: give_way, stop: false}` on the walker's navigator gives it `roqsim_nav`'s default
local model (pure Python, no extra): it steers round the robot and other walkers, and never stops for
them. `stop: true` makes it look ahead and halt as a robot does. For ORCA, use `steer: orca` and
install `rvo2` the way `roqsim_nav`'s README gives it.

### Goals at runtime

Every walker publishes a `WalkerHandle` on the blackboard under `walker:<name>`:

```python
handle = ctx.blackboard.get("walker:pedestrian")
seq = handle.send_route([(2.0, 0.0), (0.0, 2.0)])  # thread-safe; returns a sequence number
seq_applied, finished, goals_left, dist_left, error = handle.status()
handle.cancel_route()
```

A route overrides the patrol; on arrival the walker resumes patrolling from its nearest patrol
waypoint (or stands, if it had no patrol). `status()` latches `finished` under the route's own
sequence number, so a caller can distinguish *its* completion from a stale or preempted one — this is
exactly what the `NavigateThroughPoses` action handler polls.

### Hands on something

A walker can put one hand or both on world points -- a box by its handles, a part on a bench, a
door handle:

```python
handle.reach(left=left_point, right=right_point)  # re-set it every step as the points move
handle.reach(right=right_point)                   # one hand: the left goes back to the walk
left, right = handle.hands()                      # where the wrists are now
handle.reach()                                    # let go with both
```

Each point is three finite numbers (x, y, z, world); anything else raises `ValueError` at the
call, and neither hand changes. A hand given `None`, or not given, goes back to the walk.

Each arm reaches its point by two-bone IK on the walker's own bone lengths and eases onto it and
back on its own, over the gait, which goes on underneath; an arm not reaching keeps the gait's swing.
The body bends forward (most of it at the hips, the legs kept standing) only as far as the reaching
hands need, blended by the stronger hand's weight. The bend follows points coming up at once, so a
box lifted does not rise into a body still bent over it; points going down, and a hand let go, it
follows at a bounded rate (`_BEND_RATE`, 1.2 rad/s, in `roqsim_walker.animation`), speeding up and
slowing down, so for a moment the hands are above a thing set down quickly. A hand let go fades
from the pose its arm last had, in the body's frame, to the gait's -- it does not reach back to
its point; once both hands are back on the gait the body is upright and the next reach bends down
from there. Whatever is in the hands is the caller's to move: a controller that carries a box
drives the box and puts the hands on its handles (`roqsim_walker.reach`). Call `reach` on the
physics thread.

## Hands

A walker's hands are one switch, off by default:

```yaml
- walker: {walker: MaleVisitorWalk, hands: true}
```

**Off**, a walker is the seventeen-body walker described above, byte for byte: the same bodies,
geoms and skin (a test holds a digest of them), and nothing new computed per step. Every existing
world keeps its look and its timing unedited.

**On**, a walker gets what handling things needs, from the motion core (`roqsim_motion`):

- wrists that turn (three rotations each) and five fingers of three bones a hand, mocap bodies
  **with no geom**, so neither a ray nor a contact pays for them;
- named hand poses, blended: `open`, `power`, `pinch`, `flat`, `point`; a grip closes finger by
  finger until each meets what it holds;
- a `grip_l`/`grip_r` site on each wrist, at the hand's grip frame: +x out of the palm, +z along the
  fingers, the frame an object's grip site states so that holding makes the two one;
- a press proxy per hand: a fingertip sphere that pushes what has a degree of freedom (a button),
  posed on the index fingertip while a move presses and parked below the floor otherwise;
- its posing through the motion core, published as `humanoid_body:<name>`: each hand's goal (a pose
  with a rotation, or a point), the posture -- turning toward a target, crouching for a low one,
  bending forward as `reach` always has, leaning for one to the side -- and gaze, blended over the
  gait by each hand's weight. A nested `humanoid` component (`roqsim_motion`) starts the moves on it.

`reach` keeps working with hands on; the core places the wrist on its point as the two-bone reach
does with them off.

### Moves

A walker with hands takes moves from a `humanoid` component (`roqsim_motion`): each is one
program on the machine contract -- `start(move, **args)` returns a sequence number, `status()`
says when that start finished and with what error, `busy`, `cancel` -- and each step of it is an
event. Which moves a worker makes, in which order and for what, is a scenario's (Python or
OpenSCENARIO); a move never decides what comes next.

```yaml
- walker: {walker: MaleVisitorWalk, hands: true, pose: {position: {x: 0.0, y: 0.0}}}
  name: worker
  components:
    - navigator: {output: walker, speed: 1.0, arrival_radius: 0.05, avoidance: {steer: none, stop: false}}
    - humanoid: {graspable: [cabinet, tote_*]}   # what its hands may hold
```

```python
worker = ctx.blackboard.get("humanoid:worker")
seq = worker.grasp("both", "tote_1")             # = worker.start("grasp", hand="both", target="tote_1")
status = worker.status()                         # MachineStatus(seq, program, finished, error)
```

A target is a site or body of the model by name (`cabinet/drawer_0/grip`), an entity by its
`name:` (its body), or a world pose; a hand is `left`, `right` or `both`.

| move | parameters | events | fails with |
|---|---|---|---|
| `approach` | `target`, `standoff` 0.6 m, `facing` (world yaw; default: a grip site's or a face's front, a tote's long side, or from where it stands) | `walking`, `arrived` | `unknown target`, `no route to …`, `cannot face …` |
| `reach` | `hand`, `target`, `offset` (x, y, z[, r, p, y] in its frame), `orient` true, `pose` `open`, `speed` 0.8 m/s | `reached` | `out of reach: … approach first` |
| `press` | `hand`, `target` (a site on the button's face), `max_force`, `travel`, `pose` `point` | `contact`, `pressed`, `released` | `button did not move: … at … N`, out of reach |
| `grasp` | `hand`, `target` (an entity's grip sites, or one grip site), `grip` `power` | `closing`, `held` | `nothing to grasp: … not graspable`, `… has no grip site`, `hold not made`, out of reach |
| `release` | `hand` `both` | `opened` | `not holding anything` |
| `pull` / `push` | `hand`, `distance` m, `speed` 0.2 m/s, `axis` (default: the held body's slide, toward / away from the walker) | `moving`, `pulled` / `pushed` | `not holding …`, `stuck: … moved … of …`, `lost hold: …` |
| `carry` | `to`, `standoff` 0.6 m, `height` (default: under the pelvis, or clear of the surface it goes to) | `lifted`, `walking`, `arrived` | `not holding anything`, `lost hold`, `no route` |
| `place` | `at`, `release` true, `speed` 0.15 m/s | `lowering`, `set_down`, `released` | `not holding anything`, `out of reach: under the hands' reach is …`, `nothing under it` |

**Scenes.** Three worlds show a worker handling things, each with its scenario in Python
(`roqsim_walker.scenes`) and in OpenSCENARIO (the `*_osc.yaml` twin and its `.osc`), which give one
timeline:

| world | the scene |
|---|---|
| `handling_start_machine` | walks to a machine, presses its start button, and the machine runs |
| `handling_drawer_to_bench` | opens a drawer, takes a tote out by its handles, carries it to a bench and sets it down |
| `handling_box_swap` | at a cell's drawer, swaps the full tote for an empty one once the cell reports it full |

`walker_pedestrians` is ten walkers with hands off beside a robot and its lidar: the world
`bench-hands` measures.

**Contact.** A press pushes with the hand's press proxy, so a button moves because it is pushed;
the press stops at the button's travel or at `max_force`, since a kinematic hand would push through
a joint's stop. A hold is a weld between the hand's grip site and the object's, added when the world
is built for every `graspable` grip site and switched on and off while it runs; it carries a tote
and pulls a drawer by the physics. A free object is held whole and turns with the hand; one on a
joint (a drawer) is held by its position only. Two hands on one object: one weld carries it, the
other hand rides on its own handle. A hold that pulls past `hold.max_force` or slips past
`hold.slip` (each low-passed over 0.15 s, as the walker poses its hands at its own rate) lets go,
and the move fails `lost hold`. What the walker may hold does not collide with its own body. A
cancelled move stops where it is: a hold stays held. A reset drops the move and every hold.

**The skin.** Where a blueprint has finger weights (`<obj>.fingers.npz`), its hands deform with the
fingers. The bundled characters' source rigs weight a thumb chain and *one* chain for the four
fingers, so the thumb follows its bones and the four fingers' chain is split between the four
fingers by the bone nearest each vertex; `roqsim walker import-actor --fingers-only` writes the
sidecar for a blueprint from its source. A blueprint without one binds each hand to its wrist as one
rigid part: its hand follows the wrist, and its fingers are not drawn from weights it does not have.

**What hands on costs**, measured with `roqsim walker bench-hands` (a robot with its 360-ray lidar
and 1 or 10 walkers patrolling; `off`, `on` idle, and `moving`: both hands on goals that move every
step). One machine, rendering in software, so compare the rows, not the machines; run it for yours:

| walkers | case | MuJoCo step | posing in Python, per step | lidar scan | camera frame | skin bones |
|---|---|---|---|---|---|---|
| 1 | off | 7.0 µs | 60 µs | 1.56 ms | 40.0 ms | 17 |
| 1 | on | 8.7 µs | 66 µs | 1.56 ms | 38.6 ms | 46 |
| 1 | moving | 8.8 µs | 147 µs | 1.56 ms | 41.2 ms | 46 |
| 10 | off | 20.3 µs | 583 µs | 1.80 ms | 44.4 ms | 170 |
| 10 | on | 36.1 µs | 642 µs | 1.89 ms | 46.4 ms | 465 |
| 10 | moving | 36.1 µs | 1465 µs | 1.87 ms | 44.0 ms | 465 |

The posing is what a walker world pays most, with hands or without: about half a millisecond per
walker update for the gait. Hands on adds about a tenth while the hands hold their pose, the
fingers' frames being recomputed only when a pose changes; a hand on a moving goal runs the solver on
every update (up to about 0.7 ms for two hands bending). The thirty geom-less bones add about 1.6 µs
a walker to MuJoCo's step, and the frame and the scan barely move: the skin is the same vertices
split over more bones, and the fingers carry nothing a ray hits.

## Assets

`models/people/<Walker>/` is a **blueprint**: a textured OBJ, a `*.walker.json` sidecar (materials +
outfit variants, per-rig bone table, per-limb collision radii, measured shoe-sole offsets) and the
authored skin weights. The OBJ's corners that share a position and a texture coordinate load as one
skin vertex, so its normals are smooth wherever its UV map has no seam (MuJoCo sums a skin vertex's
normal from that vertex's faces only). Locomotion clips live in `models/anims/<set>/` and are picked by body type +
gender, falling back to `adult`.

Bundled blueprints (a blueprint can also come from any other installed `roqsim.models` provider):

| blueprint | source | licence | notes |
|---|---|---|---|
| `FemaleVisitorWalk` | [Open-RMF (Fuel: Luca)](https://fuel.gazebosim.org/1.0/Luca/models/FemaleVisitorWalk) | CC-BY 4.0 | handbag group dropped at import; T-posed, +X-facing (`flip: false`) |
| `MaleVisitorWalk` | [Open-RMF (Fuel: OpenRobotics)](https://fuel.gazebosim.org/1.0/OpenRobotics/models/Male%20visitor) | CC0 1.0 | purple shirt; single-texture |
| `WorkerHiVisWalk` | [Open-RMF (Fuel: Luca), *OpScrubsWalk*](https://fuel.gazebosim.org/1.0/Luca/models/OpScrubsWalk), derived | CC-BY 4.0 | an industrial worker: high-visibility vest (a texture region), work shirt and trousers, safety shoes; `roqsim walker derive-hivis-worker` |

Each carries its licence + attribution in a `CREDITS.txt` beside the files (CC0 / CC-BY / CC-BY-SA
only — the same bar as `roqsim_assets`). **The locomotion clips are CARLA-derived as well** — both
`anims/adult/` and `anims/female/` are retargets of CARLA `AS_*` animations, so a world using *any*
walker, including the Open-RMF ones, carries CARLA's CC-BY attribution. Full derivation chain, per
file: [`THIRD_PARTY.md`](THIRD_PARTY.md). Attribution is a condition of redistribution, so keep that
file with the package.

### Importing another Open-RMF actor

`roqsim walker import-actor` converts a rigged Gazebo/Open-RMF `<actor>` `.dae` into a blueprint — the
skin (mesh, textures, skeleton, skin weights) is the source's, the locomotion comes from this
package's clips rather than the source's own `<animation>` tracks (and those clips are CARLA-derived
CC-BY — see `THIRD_PARTY.md`). One importer covers the whole Open-RMF actor family (they share a rig): the hospital set
`DoctorFemaleWalk`, `NurseFemaleWalk`, `OpScrubsWalk`, `PatientWalkingCane`, `VisitorKidWalk`. See
`tools/README.md`.

```bash
pip install 'roqsim_walker[import]'   # pycollada
roqsim walker import-actor <actor.dae> --name NurseFemaleWalk --anim-set female \
    --credits-source <fuel-url> --credits-licence "CC-BY 4.0" --credits-author "..."
```

## Tools

`roqsim walker import-actor` (above) and `roqsim walker bench-hands`, which measures what hands on
costs (`--smoke --robot none` for a quick run).

## Tests

```bash
pytest roqsim_walker -q
```
