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
      motion: {walk: /abs/walk.npz}   # a locomotion clip of its own: a clip's name or a .npz
      layers: []               # clips played over the gait from the start (see below)
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
root place the walker. A walker with hands poses them on the drawn body, so what its hands carry
moves with the body a viewer sees. The default, 0, draws the body on the root.

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

Where a walker goes is its navigator's, as for every mover, so a route goes to the navigator's
handle, `nav:<name>:handle`:

```python
nav = ctx.blackboard.get("nav:pedestrian:handle")
seq = nav.send_goals([(2.0, 0.0), (0.0, 2.0)])  # thread-safe; returns a sequence number
seq_applied, finished, goals_left, dist_left, error = nav.status()
nav.cancel()
```

A route overrides the patrol; on arrival the walker resumes patrolling from its nearest patrol
waypoint (or stands, if it had no patrol). `status()` latches `finished` under the route's own
sequence number, so a caller can distinguish *its* completion from a stale or preempted one -- this is
what the `NavigateThroughPoses` action handler polls.

The walker's own handle, `walker:<name>`, is its body: its hands (`reach`, `hands`, below), the clips
it plays over the gait and where it looks (`play`, `stop_clip`, `clip_ended`, `playing`, `look_at`).

### Clips over the gait, and where it looks

The gait poses the whole body from where the navigator moves it. A **motion layer** plays a clip
over it on the joints it names, so a worker waves with one arm while it walks, talks on the phone
with its head and right arm, or a clip of the whole body plays while it stands:

```yaml
- walker:
    walker: MaleVisitorWalk
    layers:                                     # played from the start of every episode
      - {clip: phone_call, joints: [arm_r, head], loop: true}
```

```python
walker = ctx.blackboard.get("walker:pedestrian")
layer = walker.play("wave", joints=["arm_r"])  # once; returns the layer's id
walker.clip_ended(layer)  # True once it has played and faded out
walker.play("/abs/clips/carry_box.npz", joints="upper", loop=True, weight=0.7)
walker.stop_clip("/abs/clips/carry_box.npz")  # fades it out; stop_clip() stops every layer
walker.look_at("tote_1")  # the head follows an entity, a site or a body
walker.look_at([2.0, 1.0, 1.6])  # or a world point; look_at(None) turns it back
```

| key | | |
|---|---|---|
| `clip` | required | a clip of the walker's anim set by name (`anims/<set>/<clip>.npz` in any `roqsim.models` provider, the walker's set before the shared one), or a path to a `.npz` (a relative one from the working directory) |
| `joints` | `[all]` | groups -- `all`, `upper` (spine, head, arms), `arms`, `arm_l`, `arm_r`, `head`, `spine`, `pelvis`, `legs`, `leg_l`, `leg_r` -- and joint names, mixed |
| `weight` | 1.0 | 0..1: how much of the pose under it the clip takes |
| `loop` | `false` (`true` under `layers:`) | round and round until stopped, or once |
| `fade` | 0.3 s | in and out; a clip played once fades within its own length |
| `speed` | 1.0 | times the clip's own pace |

Layers stack in the order they were started, each over what is under it. A hand's reach and the
moves of a walker with hands come over them, and the look over everything: a worker plays a clip on
its arms and still presses the button with its hand. A layer that includes `pelvis` carries the
clip's height too, so a crouch in a clip lowers the body; the feet are grounded after it, as for
the gait. The clip's travel and heading are not played: where the walker goes is its navigator's.

`look_at` turns the head toward a target within its limits (70 degrees to a side, 40 up, 50 down),
at a bounded rate (3 rad/s), and follows the target as it moves; `look_at(None)` eases back to the
gait's. The `look_at` move ends once the head is on its target, so one aimed at a target crossing
the view faster than the head turns fails after 4 s, naming the head turning.

The same from outside the process: the walker's commands `play`, `stop_clip` and `look_at`, and its
`clips` output (on the control socket, and from OpenSCENARIO through `entity_call`; not on ROS). A
walker with hands moves through its `humanoid`'s `play`, `stop_clip` and `look_at` moves, which wait
for the clip or the head.

Everything a walker plays is checked when the world loads: a clip it cannot find or read, a joint
it does not have, a weight out of range is refused, naming it. Clips come from BVH captures through
`roqsim walker import-clip` (Tools, below).

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
- named hand poses, eased: `open`, `power`, `pinch`, `flat`, `point`; a grip closes finger by
  finger, the thumb over them, until each meets what it holds;
- a `grip_l`/`grip_r` site on each wrist, at the hand's grip frame: a handle's radius in front of
  the palm across the knuckles, +x out of the palm, +z along the fingers, the frame an object's grip site states so that holding makes the two one;
- a press proxy per hand: a fingertip sphere that pushes what has a degree of freedom (a button),
  posed on the drawn index fingertip while a move presses and parked below the floor otherwise;
- its posing through the motion core, published as `humanoid_body:<name>`: each hand's goal (a pose
  with a rotation, or a point), the posture -- turning toward a target, crouching for a low one,
  bending forward as `reach` does with hands off, leaning for one to the side -- blended over the
  gait by each hand's weight. Where it looks is the walker's look, as with hands off (below). A
  nested `humanoid` component (`roqsim_motion`) starts the moves on it.

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
| `approach` | `target`, `standoff` 0.6 m, `facing` (world yaw; default: a grip site's or a face's front, a tote's long side, or from where it stands), `via` (floor points `[x, y]` walked through first) | `walking`, `arrived` | `unknown target`, `no route to …`, `cannot face …`, `cannot stand at …`, `via: a point is [x, y] …` |
| `reach` | `hand`, `target`, `offset` (x, y, z[, r, p, y] in its frame), `orient` true, `pose` `open`, `speed` 0.8 m/s | `reached` | `out of reach: … approach first` |
| `press` | `hand`, `target` (a site on the button's face), `max_force`, `travel`, `pose` `point` | `contact`, `pressed`, `released` | `button did not move: … at … N`, out of reach |
| `grasp` | `hand`, `target` (an entity's grip sites, or one grip site), `grip` `power` | `closing`, `held` | `nothing to grasp: … not graspable`, `… has no grip site`, `hold not made`, out of reach |
| `release` | `hand` `both` | `opened` | `not holding anything` |
| `pull` / `push` | `hand`, `distance` m, `speed` 0.2 m/s, `axis` (default: the held body's slide, toward / away from the walker) | `moving`, `pulled` / `pushed` | `not holding …`, `stuck: … moved … of …`, `lost hold: …` |
| `carry` | `to`, `standoff` 0.6 m, `height` (default: under the pelvis, or clear of the surface it goes to), `via` and `facing` (as `approach`'s), `back_max` 0.5 m (how far it may step back before a turn; 0: never), `long_way` false (may it turn the long way round where no step back clears) | `lifted`, `stepping_back`, `turning_long_way`, `walking`, `arrived` | `not holding anything`, `lost hold`, `no route`, `cannot turn to …` (naming what the load would sweep through, what a step back would meet, and whether the long way was allowed), `cannot stand at …`, `via: a point is [x, y] …`, `back_max: a distance …` |
| `place` | `at`, `release` true, `speed` 0.15 m/s | `lowering`, `set_down`, `released` | `not holding anything`, `out of reach: under the hands' reach is …`, `out of reach: … it would stand on …` (half off an edge, on a wall and a deck), `out of reach: the hands bring it to … and no nearer`, `caught on …` (held up above the place), `nothing under it` |
| `play` | `clip`, `joints` all, `weight` 1.0, `loop` false, `fade` 0.3 s, `speed` 1.0 (as a layer's keys) | `playing`, then `played` (once) or `looping` (loop) | `no clip …`, `no joint or joint group …`; ends with the clip, a looping one once faded in; cancelled, it fades the clip out |
| `stop_clip` | `clip` (empty: every clip) | `stopped` | -- |
| `look_at` | `target` (a site, body or entity, or a world point `[x, y, z]`; empty: back) | `looking` | `unknown target …` |

A move that walks (`approach`, `carry`) goes straight to its stand-off on the navigator, planned
round the walls by its planner if it has one; `via` sends it through floor points of the scene's
first, in order -- a way laid out round a fence or a bench, for a navigator that walks its points as
given (`route_mode: exact`). The turn before a carry faces the first of them, and a carry walks
through a point where its way bends by 45 degrees or less, turning at its carrying pace as it walks
on; at a sharper corner it comes to rest and turns on the spot at its carrying pace, so what it
carries in front stays over the way and does not swing out of it at the corner.

`approach` to something a hand takes by its handles (a grip site, a tote) stands its `standoff` from
the handles, then steps back until nothing solid is within the body's clearance in front of it,
foot to knee (a walker keeps 0.3 m, as a crouch brings its knees forward): a tote in the middle of a
rack is reached over the rack, not from inside it. An approach to a surface stands where its
`standoff` says.

`grasp` of something it will carry (a free object, its handles above the knees) takes each
handle with the hand turned on it, fingers forward and down along the forearm that will carry it
(`carry_turn`, 50 degrees for a walker): the hold keeps the hand so, and the wrists stay straight
while it carries. A handle off the floor is taken as its frame says.

`carry` lifts what the hands hold straight up off what it stood on first (6 cm), then in toward
the body; carrying, the body turns on the spot the short way where its load sweeps clear of what stands
round it. Where it would not, the body first steps straight back, keeping its heading, by as little
as clears the sweep (in steps of 5 cm, and 5 cm more so the load does not graze what it clears, at
most `back_max`); before it does, rays from its back over its breadth, foot to shoulder, and from
the load look behind it, and a step back that would meet something is not taken. Where no step back
clears it, it fails naming what is in the way: the long way round is a choice the scenario states,
`long_way: true`, and taken only if it is clear, with a `turning_long_way` event naming what the
short way would have met. A turn of half a turn, within 5 degrees,
goes round on the side the body walks on to next (toward the following point of its `via`, or the
way it faces at the end), and right where that is straight on; the sign of a rounding error never
picks it. `place` takes the load over the highest
thing between where it is and the place -- read on a grid under its footprint, past what rides in
it -- then leads it on until it is within millimetres of the place, square to the place's heading
(a site's, a body's, or a world pose that states one; at most 20 degrees), and lowers it until it
rests at the place's height. It fails, holding the load, where the hands cannot bring it there,
where it would stand half off a surface or on a rim, and where something holds it up above the
place.

`pull` and `push` move what the hand holds by `distance`: a hand that lags its goal (a robot's
servo under the load) goes on past it until the object has, by at most half the distance; a lagging
hand's stroke that would leave a drawer within a hand's width of its stop pushes it to the stop, and
at a stop a posed hand's goal comes back onto it while a lagging hand's stays, so neither pulls the
drawer back.

**Scenes.** Three worlds show a worker handling things, each with its scenario in Python
(`roqsim_motion.scenes`, which the robot's box swap in `roqsim_humanoid` runs too) and in OpenSCENARIO (the `*_osc.yaml` twin and its `.osc`), which give one
timeline:

| world | the scene |
|---|---|
| `handling_start_machine` | walks to a machine, presses its start button, and the machine runs |
| `handling_drawer_to_bench` | opens a drawer, takes a tote out by its handles, carries it to a bench and sets it down |
| `handling_box_swap` | at a cell's drawer, swaps the full tote for an empty one once the cell reports a tote in its drawer |

`walker_pedestrians` is ten walkers with hands off beside a robot and its lidar: the world
`bench-hands` measures.

**Contact.** A press pushes with the hand's press proxy, so a button moves because it is pushed. The
hand is turned so its pointing index lies along the button's axis, its palm turned no more than that needs (a hand at rest points thumb up), and the proxy's front is
the drawn fingertip, its contact stiff enough not to sink into the button: what pushes the button is
the finger a viewer sees. The press stops at the button's travel or at `max_force`, since a kinematic hand would push through
a joint's stop. A hold is a weld between the hand and the object, one per hand and `graspable` grip
site, added when the world is built and switched on and off while it runs; it carries a tote and
pulls a drawer by the physics. Switched on, it welds the object at the pose it has in the hand at
that moment, not grip site onto grip site: it keeps the object as it was in the hand when taken, so a hand a
few degrees off a handle does not turn it; a free object is held whole, one on a joint (a drawer) by
its position only. Two hands on one object: one hold carries it, the other hand rides on its own
handle (two holds would squeeze it between hands posed at the walker's own rate). A hold that pulls past `hold.max_force` or slips past
`hold.slip` (each low-passed over 0.15 s, as the walker poses its hands at its own rate) lets go,
and the move fails `lost hold`. What the walker may hold does not collide with its own body. A
cancelled move stops where it is: a hold stays held. A reset drops the move and every hold.

The fingers do not collide. A grip closes finger by finger until a ray from each fingertip finds what
it holds within a few millimetres, so the hand takes the shape of a handle; contact between the
fingers and what they hold is not modelled, and the hold carries the object. Kinematic fingers
squeezing a free body would push it with unbounded force, so that is left for a measured step of its
own.

**The skin.** Where a blueprint has finger weights (`<obj>.fingers.npz`), its hands deform with the
fingers, and the hand's bones are measured on its mesh (`roqsim_walker.skin.fit_hands`): each finger's
knuckle, joints, line and tip where the mesh has them, the palm's surface and the grip frame in front
of it. The bones then turn the skin about its own knuckles, the index tip a press aims and the proxy
sits on is the drawn one, and the hand keeps the size the character was modelled with (a generic hand
on another mesh's fingers sweeps the skin about the wrong pivots: a fist deeper than the palm is
broad, a fingertip centimetres off the one drawn). The bundled characters' source rigs weight a thumb
chain and *one* chain for the four fingers, so the thumb follows its bones and the four fingers' chain
is split along the palm's breadth, each vertex to the finger whose bones it is nearest.
`roqsim walker import-actor` writes the sidecar with the blueprint, and `--fingers-only` adds it to
an existing blueprint from its source. A
blueprint without one binds each hand to its wrist as one rigid part, with the default hand's
geometry: its hand follows the wrist, and its fingers are not drawn from weights it does not have.

**What hands on costs.** With hands off a walker is the walker without hands, so it adds nothing a
step. With hands on, the posing in Python, which is what a walker world pays most for, grows by
about a tenth while the hands hold their pose: the fingers' frames are recomputed only when a pose
changes. A hand on a moving goal (`moving`) runs the solver on every update. The lidar scan and the
camera frame are unchanged, since the fingers carry nothing a ray hits and the skin is the same
vertices split over more bones: 17 skin bones a walker with hands off, 46 with them on.
`roqsim walker bench-hands` measures a machine's own numbers (a robot with its 360-ray lidar and 1
or 10 walkers patrolling; `off`, `on` idle, and `moving`: both hands on goals that move every step).

## Assets

`models/people/<Walker>/` is a **blueprint**: a textured OBJ, a `*.walker.json` sidecar (materials +
outfit variants, per-rig bone table, per-limb collision radii, measured shoe-sole offsets) and the
authored skin weights. The OBJ's corners that share a position and a texture coordinate load as one
skin vertex, so its normals are smooth wherever its UV map has no seam (MuJoCo sums a skin vertex's
normal from that vertex's faces only). Locomotion clips live in `models/anims/<set>/` and are picked by body type +
gender, falling back to `adult`; a walker without its `idle`, `walk`, `run` and `short` clip, or
whose `motion:` names one that cannot be read, is refused when the world loads.

Every tick the body is lowered onto its lowest shoe-sole point. A blueprint whose sole carries a
`hull` per shoe (the convex-hull corners of the shoe's vertices, each with its bind position and its
skin weights) is grounded on those points, skinned as the mesh, so its skin touches the floor
whichever way a foot is turned; `import-actor` writes it, and every bundled blueprint has one. One
without is grounded on a heel and a toe point -- the lowest of the shoe's vertices the ankle carries
and of those the toe carries, each riding its joint -- which stand it on the floor, but a foot turned
far, as at toe-off, can put another part of the shoe under the floor.

Everything `walker.json` says of the mesh's shape (height, collision radii, sole points and hull) is
measured against the bones MuJoCo poses the skin with (`skin._bind_bones`), not the source rig's
own, which the walker's straight limbs do not keep.

Bundled blueprints (a blueprint can also come from any other installed `roqsim.models` provider):

| blueprint | source | licence | notes |
|---|---|---|---|
| `FemaleVisitorWalk` | [Open-RMF (Fuel: Luca)](https://fuel.gazebosim.org/1.0/Luca/models/FemaleVisitorWalk) | CC-BY 4.0 | handbag group dropped at import; T-posed, +X-facing (`flip: false`) |
| `MaleVisitorWalk` | [Open-RMF (Fuel: OpenRobotics)](https://fuel.gazebosim.org/1.0/OpenRobotics/models/Male%20visitor) | CC0 1.0 | purple shirt; single-texture |

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

`roqsim walker import-actor` (above); `roqsim walker bench-hands`, which measures what hands on
costs (`--smoke --robot none` for a quick run); and `roqsim walker import-clip`, which turns a BVH
motion capture into a clip:

```bash
roqsim walker import-clip wave.bvh --out my_pkg/models/anims/adult/wave.npz
roqsim walker import-clip walk_cycle.bvh --out walk.npz --loop --map spine=Chest
```

Each walker bone points where its source bone does, frame by frame: the source's rest pose (a
T-pose or an A-pose) is matched to the walker's arms-down rest bone by bone, so the capture moves the
walker's limbs as it moves its own, whatever the proportions. Bones are found by the Mixamo, CMU and
common game-rig names; one not found is an error naming the capture's bones, and `--map joint=Bone`
names it. `--up` and `--forward` read a capture in other axes; `--loop` marks a cycle, whose stride
is measured from its travel. The capture's travel is dropped, and its heading at the first frame
(its later turns are kept); its height is kept.
`python -m pydoc roqsim_walker.cli.import_clip` has the details.

## Tests

```bash
pytest roqsim_walker -q
```
