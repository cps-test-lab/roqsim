# roqsim — notes for Claude

Plugin-driven MuJoCo simulation framework. **Read [docs/architecture.rst](docs/architecture.rst)
first**: it is the source of truth for the plugin lifecycle, the API contracts, the invariants and the
porting playbook. This file holds the rules a change must keep; it lists no plugins, models or tools.

## What is installed
Ask the installation, not this file:
- `roqsim --help` — every command; then `roqsim <group> --help` and `roqsim <group> <tool> --help`.
- `roqsim plugins list` / `roqsim plugins describe <name>` — every registered plugin, as JSON.
- `roqsim catalog models|worlds [--refs]` / `roqsim catalog model <name>` — what can be spawned and run.
- `roqsim ls` / `roqsim endpoints` / `roqsim describe <path>` — a running simulation's endpoints,
  over the control socket `roqsim sim` serves (`docs/control.rst`).
- OSC actions: `docs/quickstart.rst` and `scenario_execution_roqsim/src/scenario_execution_roqsim/lib_osc/roqsim.osc`.

## Package layout
Role, then allowed roqsim dependencies (each `pyproject.toml` is authoritative).
- `roqsim/` — ROS-free core: engine, plugin API, config, drivers, the `roqsim` command tree. No sibling.
- `roqsim_sensors/` — robot-family-agnostic sensor plugins and device models. `roqsim`.
- `roqsim_assets/` — props, textures, the conveyor. `roqsim`.
- `roqsim_scenes/` — imported and generated scene worlds. `roqsim`, `roqsim_assets`.
- `roqsim_mobile/` — wheeled bases only. `roqsim`, `roqsim_sensors`.
- `roqsim_manipulation/` — arm plugins only, no geometry. `roqsim`.
- `roqsim_manipulation_assets/` — arm and gripper models; real robots, no workpieces. `roqsim_manipulation`, never the reverse.
- `roqsim_mobile_manipulation/` — robots that are a base and an arm. `roqsim_mobile`, `roqsim_manipulation(_assets)`, `roqsim_sensors`.
- `roqsim_humanoid/` — humanoids + RL locomotion. `roqsim_mobile`, `roqsim_manipulation`, `roqsim_sensors`.
- `roqsim_quadruped/` — quadrupeds + RL locomotion. `roqsim_mobile`, `roqsim_sensors`.
- `roqsim_aerial/` — aerial vehicles, flight control, wind, PX4 SITL (`[px4]`). `roqsim`, `roqsim_sensors`.
- `roqsim_nav/` — shared 2D navigation and the `navigator`; no geometry, so any family may use it. `roqsim`.
  Embodiments and local planners plug in via `roqsim_nav.outputs` / `roqsim_nav.avoidance`; nothing branches on their names.
- `roqsim_walker/` — kinematic pedestrians, a `roqsim_nav` output. `roqsim`, `roqsim_nav`; no robot package depends on it.
  Licences differ per character (`CREDITS.txt` beside each) and the clips are CC-BY: `roqsim_walker/THIRD_PARTY.md`.
- `roqsim_mcp/`, `roqsim_scene_builder/` — MCP servers (introspection; scene windows and renders). `roqsim`.
- `roqsim_webctrl/` — web-control plugin fragment. No roqsim dependency.
- `scenario_execution_roqsim/` — the OSC vocabulary (`import osc.roqsim`). `roqsim`; `scenario_execution` via `[osc]`.
- `ros2_ws/src/` (colcon): `roqsim_ros_bridge` (transport + `simulation_interfaces`, as plugins), `roqsim_nav_interfaces`,
  `roqsim_nav_ros` (nav2 goal actions for roqsim's own movers), `roqsim_walker_ros`, `roqsim_nav2_example`,
  `roqsim_create3_toolbox` (`docs/create3_stack.rst`).

## Golden rules
- **Single writer.** Only the physics thread (the one calling `engine.step()`) touches `model`/`data`;
  outside input goes through `ctx.post(cmd)`. Architecture §7.
- **No mid-run recompile.** Change the `MjSpec` only in `build()`; at runtime write mocap/qpos or use the
  entity pool. Architecture §7, anti-patterns.
- A plugin implements any subset of `build`, `configure`, `on_reset`, `pre_step`, `post_step`, `shutdown`
  and validates its own config in `validate_config`. Architecture §2–3.
- **A capability is declared by the plugin, never listed in the core.** Consumers ask a class attribute
  (`transport_only`, `provides_world`, `parallel_safe`); a name list in core would silently serve only
  our own plugins. Architecture §3.
- **Keep the core ROS-free**; the ROS bridge is just another plugin.
- **`select_offscreen_gl()` runs first in `roqsim/src/roqsim/__init__.py`, with nothing imported above it.** MuJoCo
  binds its GL backend at `import mujoco`, and unset means glfw, which aborts headless. Do not move it
  into a driver, do not let isort merge it (the root `pyproject.toml` E402 per-file-ignore), do not let
  `roqsim/src/roqsim/gl.py` import mujoco. Tests without a camera cannot catch a regression. Architecture §8.
- **Draw randomness from `ctx.rng_for(name)`**, once per (sensor, step), never from `np.random` or a
  held generator. It is keyed on `(seed, episode, sim_time, name)`, so a draw is a function of the world
  and not of how many draws came before; that keeps a value reproducible and a replay exact.
- **A seed is driver-owned; an unresolved one raises** (`roqsim.seed.SeedError`), never defaults to 0.
  A driver resolves it with `roqsim.seed.resolve_seed` and sets `ctx.seed` before `setup()`; a preview
  pins `roqsim.seed.PREVIEW_SEED`. `python -m pydoc roqsim.seed`.
- **A scenario ends its own run; `ctx.request_stop(reason)` ends a standalone one.** A trial plugin
  publishes its outcome as observable state and never ends a scenario-driven run. Architecture §4,
  "Who ends a run".
- **`sim.contact_override` is global and pre-compile; `model_override` is aimed and at runtime.**
  Per-geom values belong in the model; one owner per knob, so never add `opt.*` to `model_override`'s
  allowlist. A flex's material is `flex_material`, before compile. Architecture §4 and §9.2.
- **Aerial worlds fail silently in two ways.** A world with no `density`/`viscosity` is a vacuum, so
  nothing damps a drone; a multirotor MJCF has no stabiliser, so an uncommanded drone falls, which is
  why its manifest pulls the controller in (unless an external flight stack flies it). Architecture §4,
  the `sim.density` note.
- Sensor noise is per-sensor config; there is no generic error-model framework, on purpose. Architecture §9.
- `roqsim health` is a reader: a separate process tailing the run's CSVs, never inside the simulator or
  behind a bridge, so it does not share the failure modes it diagnoses. `docs/quickstart.rst`.
- **Plugins and geometry are separate packages**, assets → plugins: a model's manifest names its
  plugins and a plugin knows no model, so wanting a plugin never installs meshes.
- **Family packages are siblings, not a chain.** A robot in two families goes in a package depending
  on both; never widen a family's dependencies, which drags every install along and inverts its contract.
- **The substrate ships mechanism; an experiment ships what it measures.** Add a plugin here only if a
  second experiment would use it unchanged; workpieces, trial protocols and success rules live
  downstream (`docs/plugins.rst`, "where a workpiece lives").
- **Only `scenario_execution_roqsim` imports `scenario_execution`**, and no package that does not
  import `scenario_execution` may import it. Its actions work unedited in a stepped run and a ROS run.
- **Docs split:** user guide and internals are separate toctree sections in `docs/index.rst`; put new
  content in the right one and split a page that mixes both.
- **Docs follow every change:** update what the change made stale in `docs/` (especially
  `docs/architecture.rst`, `docs/plugins.rst`), package READMEs, docstrings and the commented example worlds.

## Tools and dev
Everything runnable is a `roqsim` subcommand; never invoke a script by path. A new tool is written
standalone and linked into the tree in the same commit (`docs/developer_guide.rst`, "Adding a tool");
`make test` fails until it is linked. A new plugin registers under the `roqsim.plugins` entry point or
is loaded by `module:Class` / `file.py:Class` (Architecture §12).

```
make venv            # .venv with --system-site-packages
make test            # unit tests (+ ros2_ws tests when ROS is sourced); make test-<package> for one
make lint | format   # ruff
make smoke           # headless-run every shipped world
make check | doc     # publication hygiene | Sphinx build
```
