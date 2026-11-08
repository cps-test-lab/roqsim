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
| `ik.py` | the whole-body solver: turn, crouch, bend and lean for the targets; two-bone arms and legs; the wrist on the target's rotation; gaze |
| `hands.py` | five fingers of three bones a hand, the named poses `open`, `power`, `pinch`, `flat`, `point`, blended; a grip that closes finger by finger |
| `targets.py` | a hand's target: a world pose, or a site, body or entity with an offset, resolved every tick |
| `retarget.py` | retargeting tables: a body mapped onto the canonical humanoid |

The solver is analytic, since it runs in Python on every update of every handling humanoid. A target
left out of reach after the posture it allows is reported by its residual and never stretched to.
Nothing is drawn and no iteration count depends on time, so a run is deterministic.

**The grip frame** is where a hand holds: at the palm's centre, +x out of the palm toward what it
holds, +z along the fingers. An object's grip site states the same frame, so holding is making the
two frames one.

## Tests

```bash
pytest roqsim_motion -q
```
