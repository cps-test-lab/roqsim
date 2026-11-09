# roqsim_motion

One motion core for roqsim's humanoids: it turns a motion clip's pose and targets -- each hand's
pose, gaze, a posture -- into joint rotations on a canonical humanoid, and a retargeting table maps
those onto a body. Two bodies use it: the walker's mocap skeleton (`roqsim_walker`, kinematic, no
degrees of freedom) and a humanoid robot's MJCF joints (the `humanoid_robot` plugin, below).

**The line.** A humanoid's moves are one entity's procedures, as a machine's programs are: each is
something the humanoid does by itself, started through its handle and waited on by its status. Which
moves a worker makes, in which order and for what, is a scenario's (`docs/architecture.rst`,
"Machines and scenarios"). A move never decides what comes next.

## The core

| module | what it holds |
|---|---|
| `description.py` | the canonical humanoid (the walker's seventeen joints), rest offsets, limits, forward kinematics |
| `ik.py` | the whole-body solver: turn, crouch, bend and lean for the targets; two-bone arms and legs; the wrist on the target's rotation, its twist about the forearm shared with the forearm; gaze |
| `hands.py` | five fingers of three bones a hand, each bone on its own axis (the thumb's first opposes it); a hand measured on a mesh or the default one; the named poses `open`, `power`, `pinch`, `flat`, `point`, each finger eased whole; a grip that closes finger by finger, index to little, the thumb over them |
| `targets.py` | a hand's target: a world pose, or a site, body or entity with an offset, resolved every tick |
| `retarget.py` | retargeting tables: a body mapped onto the canonical humanoid |

The solver is analytic, since it runs in Python on every update of every handling humanoid. A target
left out of reach after the posture it allows is reported by its residual and never stretched to.
Nothing is drawn and no iteration count depends on time, so a run is deterministic.

**The grip frame** is where a hand holds: in front of the palm across the knuckles, +x out of the
palm toward what it holds, +z along the fingers. An object's grip site states the same frame, so holding is making the
two frames one.

## The moves

A `humanoid` component, nested under a walker with `hands: true` or under a humanoid robot, gives it
the moves `approach`, `reach`, `press`, `grasp`, `release`, `pull`, `push`, `carry` and `place`, and
`play`, `stop_clip` and `look_at` for a body that plays clips and turns its head (a walker's): the
programs of a `MachinePlugin`, so each is started through the machine handle (`machine:<name>`, and
`humanoid:<name>` with a method per move), waited on by its status and recorded as events. Their
parameters, events and failures: `roqsim_walker/README.md`, "Moves". The scenes that compose them --
in Python here (`roqsim_motion.scenes`), in OpenSCENARIO with `scenario_execution_roqsim`'s
`humanoid_*` actions -- are worlds of `roqsim_walker` and, for a robot, of `roqsim_humanoid`.

## A humanoid robot

`humanoid_robot`, nested under the `spawn_model` that places a humanoid robot, makes the robot a body
of the core with its retargeting table (`<model>.retarget.yaml` beside the model). `spawn_model`
reads no manifest, so none of the robot's own controller software runs: no locomotion policy, no arm
controller.

```yaml
- spawn_model: {model: unitree_g1_dex1, prefix: g1_, motion: static}   # the model has its own free base
  name: g1
  components:
    - humanoid_robot: {drive: physical}          # kinematic | physical
    - navigator: {output: humanoid, speed: 0.5, arrival_radius: 0.05}
    - humanoid: {graspable: [cell, tote_*]}
```

- **kinematic**: the joints are set to their targets and the base to the nav root, every step; the
  robot pushes what it touches and is never pushed.
- **physical**: every mapped joint is tracked by a plain position servo -- the model's own, servos
  this plugin adds where the model has none (`servos: own`), or added beside the model's on every
  joint (`servos: add`, for an arm whose own servos are too soft to hold a load) -- and the base hangs
  on a soft weld to an anchor the navigator moves. The body pushes and is pushed.

**No locomotion or balance controller is written, and balance is not simulated.** The base hangs on
its anchor, the legs play a simple gait while it walks, the feet may slide or press into the floor,
and the contact forces under them are not those of a standing robot.

Its hands are solved on its own arm and waist joints by damped least squares with MuJoCo's site
Jacobian, drawn back toward the table's `rest` pose where a goal leaves them free; a physical arm's
sag under a load is trimmed out against the hand's actual pose. A table's hand names its grip frame
(a site, or a body and an offset) and turns it into the grip-frame convention; a gripper closes
through its actuator. Two hands on one object both hold it (the second by its position only), where
a walker's second hand rides on its handle: the robot's servos sag, and one hand alone would let the
object tilt.

| model | table | tested |
|---|---|---|
| `unitree_g1_dex1` (BSD-3-Clause) | `roqsim_humanoid/.../unitree_g1_dex1.retarget.yaml` | physical, scene 4 |
| `oli` (Apache-2.0) | `roqsim_humanoid/.../oli.retarget.yaml` | kinematic |

## Tests

```bash
pytest roqsim_motion -q
```
