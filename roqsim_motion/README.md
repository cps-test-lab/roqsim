# roqsim_motion

One motion core for roqsim's humanoids: it turns a motion clip's pose and targets -- each hand's
pose, gaze, a posture -- into joint rotations on a canonical humanoid, the walker's mocap skeleton
(`roqsim_walker`, kinematic, no degrees of freedom), which drives its body through one interface
(`body.py`).

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

The solver is analytic, since it runs in Python on every update of every handling humanoid. A target
left out of reach after the posture it allows is reported by its residual and never stretched to.
Nothing is drawn and no iteration count depends on time, so a run is deterministic.

**The grip frame** is where a hand holds: in front of the palm across the knuckles, +x out of the
palm toward what it holds, +z along the fingers. An object's grip site states the same frame, so holding is making the
two frames one.

## The moves

A `humanoid` component, nested under a walker with `hands: true`, gives it
the moves `approach`, `reach`, `press`, `grasp`, `release`, `pull`, `push`, `carry` and `place`, and
`play`, `stop_clip` and `look_at` for a body that plays clips and turns its head (a walker's): the
programs of a `MachinePlugin`, so each is started through the machine handle (`machine:<name>`, and
`humanoid:<name>` with a method per move), waited on by its status and recorded as events. Their
parameters, events and failures: `roqsim_walker/README.md`, "Moves". The scenes that compose them --
in Python here (`roqsim_motion.scenes`), in OpenSCENARIO with `scenario_execution_roqsim`'s
`humanoid_*` actions -- are worlds of `roqsim_walker`.

## Tests

```bash
pytest roqsim_motion -q
```
