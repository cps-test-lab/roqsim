# roqsim — notes for Claude

Lightweight, plugin-driven MuJoCo simulation framework. **Read [docs/architecture.rst](docs/architecture.rst)
first** — it is the source of truth for architecture, the plugin lifecycle, and the porting playbook.

## Layout
- `roqsim/` — ROS-free core pip package (engine, plugin base, registry, config, drivers, the `roqsim`
  command tree in `commands.py`, and the URDF/SRDF/web exporters; the world-agnostic `dummy`,
  `spawn_model`, `ceiling`, `contact_monitor`, `clearance_monitor` and
  `model_override` plugins; state recording and
  `roqsim render` are driver-level, in `capture.py` / `recording.py` / `render.py`, not plugins). Sources in
  `roqsim/src/roqsim/`, tests in `roqsim/tests/`. Depends on no sibling — keep it that way.
  `health.py` (`roqsim health`) is not even driver-level: it is a **reader**, a separate process that
  tails the two CSVs `capture.py` streams and touches nothing in a run. Keep it that way too — a
  health check that ran inside the simulator would share the simulator's failure modes, and one that
  went through a transport bridge could not diagnose a broken bridge.
- `roqsim_sensors/` — generic (robot-family-agnostic) sensor plugins + assets: `lidar`,
  `oakd_camera` (RGB-D), `realsense_d435`/`realsense_d455` (RGB + opt-in depth/`PointCloud2`; depth
  in `32FC1` metres or `16UC1` millimetres), `realsense_d415` (RGB only), `force_torque` (six-axis
  wrench at a site — the one contact-force observable), `fiducial_marker` (ArUco/AprilTag,
  OpenCV-generated; optional `markers` extra). Models are one folder per device
  (`models/<name>/<name>.xml` + its own `meshes/`). Depends on `roqsim`.
- `roqsim_mobile/` — mobile-robot plugins + assets (floorplan, spawn_robot, diff_drive, omni_drive,
  wheeled base models, demo worlds). Depends on `roqsim` + `roqsim_sensors`. Wheeled **bases only**.
- `roqsim_manipulation/` — manipulator **plugins only** (spawn_arm, arm_controller,
  cartesian_admittance). No geometry, and no experiment logic. Depends on `roqsim`.
- `roqsim_manipulation_assets/` — the arm and gripper **models** (UR10e, UR5e, Panda, Gen3,
  OpenManipulator-X, xArm 7, Robotiq 2F-85, Schunk PG+70) + demo worlds. Depends on
  `roqsim_manipulation`, never the reverse. Real robots only — no workpieces.
- `roqsim_mobile_manipulation/` — robots that are a base **and** an arm (frankie, tiago_pro). The one
  package that may depend on both `roqsim_mobile` and `roqsim_manipulation`.
- `roqsim_aerial/` — aerial vehicles (`crazyflie_2`) + `quadrotor_controller`, cascaded position and
  attitude control over collective thrust and three body moments. Depends on `roqsim` only. Two
  things differ from every ground family and both fail silently: a world with no `density`/
  `viscosity` is a **vacuum**, so the drone hovers but nothing damps it; and a quadrotor MJCF has no
  stabiliser, so an uncommanded drone is not a robot standing still but a falling brick — its
  manifest pulls the controller in rather than offering it.
- `roqsim_walker/` — kinematic pedestrian **walkers**: the `walker` plugin, the 17-joint humanoid, A\* +
  behaviour-tree navigation with optional ORCA, character blueprints (`models/people/`) and CARLA
  locomotion clips (`models/anims/`). Depends on `roqsim` only — it is a *dynamic obstacle*, not a robot
  family, so no robot package depends on it and it depends on none. Zero DOFs (mocap bodies), but its
  per-limb capsules are what a robot's lidar and contacts see. **Its assets are CC-BY** (CARLA), so
  attribution travels with anything that ships them — see `roqsim_walker/THIRD_PARTY.md`.

**Plugins and geometry are separate packages.** Every family with an actuated limb needs
`arm_controller` — a humanoid's arms, a mobile manipulator's arm, a gantry — and almost none needs
the arm *models*. While the two shared one package, wanting the plugin installed 115 MB of meshes.
The dependency runs assets → plugins, which is what keeps it acyclic: a model's manifest names the
plugins intrinsic to it, and the plugins know nothing about any particular model.

**The substrate ships mechanism; an experiment ships what it is measuring.** The test is reuse, not
file type. `arm_controller` serves every arm and `ur10e` is a robot anyone can mount — those stay.
Four things did not, and each was a *what* rather than a *how*:

| left | to | because |
| --- | --- | --- |
| `peg_in_hole` | a downstream experiment | a bored block with a swept clearance (+7.7 MB of meshes) |
| `insertion_task` | a downstream experiment | one paper's trial protocol — its defaults were that paper's constants |
| `pipe_weldment`, `welding_torch` | a downstream experiment | a workpiece we sized ourselves |
| `pick_place_metrics` | a downstream experiment | a rule for what counts as success in one trial |

Before adding a plugin here, ask whether a **second** experiment would use it unchanged. If the answer
needs a story, it belongs downstream. Nothing is lost by that: an experiment registers models with an
`roqsim.models` entry point, or loads a plugin by path from beside its world — see `docs/plugins.rst`,
"where a workpiece lives". Both are ordinary doors, not special cases.

**Family packages are siblings, not a chain.** A robot belonging to two families goes in a package
that depends on both, never in one of them. Keeping the two mobile manipulators in `roqsim_mobile`
forced it to depend on `roqsim_manipulation`, which inverted its own contract (a base package requiring
the arm package), made a plain TurtleBot install pull in every arm and gripper, and reordered the
wheel install graph for deployments — it broke a campaign image build with `No matching distribution
found for roqsim_manipulation`. If a new robot does not fit an existing family, add a sibling; do not
widen a family's dependencies to accommodate it.
- `scenario_execution_roqsim/` — the OpenSCENARIO 2 vocabulary (`import osc.roqsim`): what a
  scenario can ask a running simulation (`entity_moved`, `entity_rotated`) and what it can break in one
  (`set_model_override`). **The only package here that may import `scenario_execution`**, and it must
  never be imported by one that does not. Named for that project's convention (`scenario_execution_*`),
  not ours, which is why the Makefile globs two name shapes. Depends on `roqsim` only. Each action works in
  a stepped run and in a ROS run unedited — the transport is chosen from what the runner offered, and
  the names it uses (entities, plugin instances) are identical on both.
- `ros2_ws/src/roqsim_ros_bridge/` — colcon package: ROS 2 bridge + `simulation_interfaces` (plugins).
- `ros2_ws/src/roqsim_nav2_example/` — colcon package: minimal nav2 example + headless goal test.
- `docs/` — Sphinx user docs (roqsim) incl. `architecture.rst` (architecture + porting playbook).

## Golden rules
- **Single-writer:** only the physics thread (the one calling `engine.step()`) touches `model`/`data`.
  External input (ROS callbacks, services) must go through `ctx.post(cmd)`, never mutate `data` directly.
- **No mid-run recompile:** modify the `MjSpec` only in `build()`; at runtime use mocap/qpos writes or
  the entity pool.
- Plugins implement any subset of the lifecycle hooks: `build`, `configure`, `on_reset`, `pre_step`,
  `post_step`, `shutdown`. A plugin can be both an init and a tick plugin.
- Each plugin validates its own config via `validate_config`.
- **A capability is declared by the plugin, never listed in the core.** When a consumer must treat some
  plugins differently — a geometry-only render or export skipping a transport plugin — it asks a class
  attribute (`transport_only`, `provides_world`, `parallel_safe`), so an out-of-tree plugin gets the
  same behaviour without anyone editing `roqsim`. A name list in core would silently serve only our own.
- Keep the core ROS-free. The ROS bridge is just another plugin.
- **`select_offscreen_gl()` runs first in `roqsim/__init__.py`, and nothing may be imported above
  it.** MuJoCo reads `MUJOCO_GL` once, during `import mujoco`, and binds `GLContext` there; unset is
  not an error but a choice, and it resolves to **glfw**, which aborts on a headless node with
  `mujoco.FatalError: gladLoadGL error`. The package `__init__` is the only place that runs before
  every `roqsim.*` submodule and therefore before every `import mujoco` of ours. Do not "tidy" that
  call into a driver's `main` (it lived in `runner.main` and was inert for every headless run), do
  not let isort merge it into the import block below it (the `E402` per-file-ignore in
  `pyproject.toml` is what keeps it separable), and do not let `roqsim/gl.py` import mujoco even
  transitively. This class of bug is invisible in testing: without a camera no `mujoco.Renderer` is
  ever constructed, so the wrong backend is never instantiated and everything passes.
- **Docs split:** keep user-facing docs (how to use it) and internal docs (how it works) separate —
  in `docs/`, the `index.rst` toctree has a "User guide" section and an "Internals & development"
  section. Put new content in the right one (and split a page if it mixes both).
- **Docs follow every change:** after any change, check whether user-facing or internal
  documentation needs updating — `docs/` (especially `architecture.rst` and `plugins.rst`), the
  package READMEs, module/config docstrings, and the commented example worlds — and update what the
  change made stale.
- Sensor noise is per-sensor config (e.g. lidar `range_stddev`); there is no generic error-model
  framework (it was removed on purpose — see architecture.rst §9).
- **Draw randomness from `ctx.rng_for(name)`, never from a module-level `np.random` or a stateful
  generator.** It is counter-based (Philox) and keyed on `(seed, episode, sim_time, name)`, so a draw is a
  pure function of the world rather than of how many draws happened before it. A shared stateful
  stream's position depends on sensor rates, step count, and whether anyone was subscribed to a
  camera — which makes a value at t = 12.5 unreproducible without replaying the whole run, and breaks
  re-running a sensor from a recording. Call it once per (sensor, step), not once per value; draws
  are vectorised anyway. The run's seed comes from `sim.seed` in the world or `roqsim sim --seed`
  (which wins), and a run without either draws and logs a seed so it can be replayed. The `episode`
  is in the key because `reset()` restarts `sim_time`: without it every trial after the first in one
  process re-draws the first one's noise, so repetitions are duplicates rather than samples.
- **End a trial with `ctx.request_stop(reason)`, not by padding `--seconds`.** A wall-clock limit has
  to be guessed high enough for the slowest cell and is then wasted on every faster one. It is a
  request, not a kill switch: the driver polls it and exits cleanly, so `shutdown` runs and files
  flush, and an embedding driver may ignore it. Physics-thread only, like every other write on `ctx`.
- Global contact tuning is `sim.contact_override` (`solref` / `solimp` / `friction`, MuJoCo's
  `o_*` overrides), validated at load time — global, and applied *before compile*, so it is in the
  compiled model and in the run's provenance. Per-geom values belong in the model, not here. Changing
  a value on NAMED objects, DURING a run, is the `model_override` plugin instead: aimed and triggered,
  where this key is neither. Do not add `opt.*` to that plugin's allowlist — one owner per knob.

## Tools
- **Everything runnable is a subcommand of `roqsim`** — the simulator is `roqsim sim <world>`, and every
  tool is `roqsim <group> <tool>`. Start at `roqsim --help`, then `roqsim <group> --help` for one line per
  tool, then `roqsim <group> <tool> --help` for its options; `python -m pydoc <module>` has the
  reasoning behind one. Do not go looking for scripts to invoke by path.
- Adding a tool: write it standalone, then link it into that tree in the same commit. The recipe is
  `docs/developer_guide.rst` → "Adding a tool"; `make test` fails until it is linked.

## Dev (via the Makefile)
- `make venv` (creates `.venv` with `--system-site-packages`), then `make test`
  (unit tests; also the nav2 integration test when ROS is sourced).
- `make format` / `make lint` (ruff), `make doc` / `make view-doc` (Sphinx).
- New plugin? Subclass `roqsim.plugin.Plugin`, add hooks, register via the `roqsim.plugins`
  entry-point group (or reference it by `module:Class` / `file.py:Class` in the world YAML).
