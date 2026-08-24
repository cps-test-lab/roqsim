# roqsim_walker

Kinematic pedestrian **walkers** for roqsim. Pure Python + MuJoCo: **no ROS dependency**.

A walker is a character mesh skinned onto a 17-joint skeleton of MuJoCo **mocap bodies**.
Blueprints are discovered across every installed `roqsim.models` provider, so a downstream
package can ship its own characters and a world still names only the walker. Body
pose comes from motion clips; the nav root comes from an A\* planner + a py-trees behaviour tree +
(optionally) ORCA local avoidance. Because it is fully kinematic it adds **zero DOFs** to the solver,
yet its per-limb collision capsules are what the robot's lidar and contacts see.

It either **patrols** a configured route, or is **driven to goals** through a backend-neutral
endpoint — which the ROS 2 bridge serves as `nav2_msgs/NavigateThroughPoses`
(see `ros2_ws/src/roqsim_walker_ros`).

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
      name: pedestrian         # entity name
      namespace: ""            # transport scope for the goal endpoint
      outfit: B                # clothing variant: a letter, or {pants: C, jacket: A}
      skin: true               # false -> capsule visuals (fast; no mesh load)
      speed: 1.2               # m/s; past ~1.7 the run clip blends in
      pos: [0.0, 0.0]          # spawn, used when `waypoints` is empty (goal-driven only)
      waypoints:               # patrol route; the walker starts at waypoints[0]
        - [-2.0, -2.0]
        - [ 2.0, -2.0, [2, 4]] # optional dwell: seconds, or [lo, hi] random pause
        - [ 2.0,  2.0]
      loop: true               # cycle the patrol forever
      arrival_radius: 0.25
      avoidance: false         # ORCA local avoidance (needs the [avoidance] extra)
      robot_body: base_link    # body to yield to (default: the robot entity's base)
      action_name: navigate_through_poses
      orca:     {neighbor_dist: 4.0, time_horizon: 3.0, radius: 0.26, max_speed: 1.6}
      planner:  {inflation_radius: 0.3, waypoint_radius: 0.3}
      recovery: {stuck_time: 1.5, backup_time: 0.5, max_recovery: 4}
      motion:   {walk: /abs/walk.npz}   # override a resolved locomotion clip
```

### Navigation layers

| Layer | Module | What it does |
|---|---|---|
| Global plan | `nav/planner.py`, `nav/occupancy.py` | 8-connected A\* over an inflated occupancy grid rasterized from the model's **wall geoms**, string-pulled to sparse waypoints |
| Behaviour | `nav/behavior.py` | py-trees `Selector[recovery, navigate]`: follow path, advance goals, back-up-and-replan when stuck |
| Local avoidance | `nav/controller.py` (ORCA) | Yields to the robot, other walkers and mocap props; walls are static obstacles |

Walls are read straight from the compiled model (`nav/obstacles.py`), so the planner and ORCA always
agree. The default `empty_room` is a **walled** room, so A\* engages on its perimeter walls; **with no
wall geoms** (a wall-less MJCF via `sim.world`) the grid is skipped and walkers follow straight-line
legs. A `floorplan` mesh adds its own walls the same way.

### Avoidance

`avoidance: true` turns on ORCA for that walker. It needs the optional extra (built from source):

```bash
pip install -e 'roqsim_walker[avoidance]'
```

`rvo2` publishes no wheel, so the extra is a git direct reference and needs **git + a compiler**. Two
places that bites: a PyPI upload of this package cannot carry the extra (PyPI rejects direct-URL
metadata), and a wheel-only or air-gapped build — a campaign image — cannot resolve it. Install the
base package in those, and enable avoidance only where the toolchain exists.

Without `rvo2` installed the walker logs a warning once and navigates without collision avoidance.
The shared ORCA simulation is created when *any* walker enables it; a walker with `avoidance: false`
still occupies an ORCA agent (so peers steer around it) but is never pushed off its own path.

### Goals at runtime

Every walker publishes a `WalkerHandle` on the blackboard under `walker:<name>`:

```python
handle = ctx.blackboard.get("walker:pedestrian")
seq = handle.send_route([(2.0, 0.0), (0.0, 2.0)])  # thread-safe; returns a sequence number
seq_applied, finished, goals_left, dist_left = handle.status()
handle.cancel_route()
```

A route overrides the patrol; on arrival the walker resumes patrolling from its nearest patrol
waypoint (or stands, if it had no patrol). `status()` latches `finished` under the route's own
sequence number, so a caller can distinguish *its* completion from a stale or preempted one — this is
exactly what the `NavigateThroughPoses` action handler polls.

Multiple `walker` plugins share one controller (one ORCA simulation sees every walker, the robot and
any mocap props). The first to initialise owns the per-step tick.

## Assets

`models/people/<Walker>/` is a **blueprint**: a textured OBJ, a `*.walker.json` sidecar (materials +
outfit variants, per-rig bone table, per-limb collision radii, measured shoe-sole offsets) and the
authored skin weights. Locomotion clips live in `models/anims/<set>/` and are picked by body type +
gender, falling back to `adult`.

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

## Tests

```bash
pytest roqsim_walker -q
```
