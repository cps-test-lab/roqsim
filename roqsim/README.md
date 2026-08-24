# roqsim

The **core** of [roqsim](../README.md): a MuJoCo step loop, a plugin lifecycle, a YAML config layer, and
the two drivers that run them. This is the package everything else depends on, and the only one that
depends on no sibling — installing it gets you a simulator and the `roqsim` command tree, and nothing
that assumes a particular robot.

**No robots, sensors, scenes or props live here.** A world is assembled from sibling packages —
`roqsim_mobile` (wheeled bases), `roqsim_manipulation` + `roqsim_manipulation_assets` (arms and grippers),
`roqsim_humanoid`, `roqsim_quadruped`, `roqsim_sensors` (lidar, RGB-D, force-torque, fiducials), `roqsim_assets`
(props and surface textures), `roqsim_scenes` (imported worlds), `roqsim_walker` (pedestrians). The core
knows none of their names: it discovers them through entry points, so a package outside this
repository extends it without a change here.

**The core is ROS-free**, on purpose. ROS 2 coupling is a plugin in its own colcon package
(`ros2_ws/src/roqsim_ros_bridge`), so a campaign that does not need ROS does not install it.

## What it ships

| module | role |
|--------|------|
| `engine.py` | The step loop: compile the `MjSpec`, run the plugin lifecycle, step physics. |
| `plugin.py` | The `Plugin` base class and its hooks — `validate_config`, `build`, `configure`, `on_reset`, `pre_step`, `post_step`, `shutdown`. A plugin implements any subset. |
| `config.py` | The single-YAML config layer: `extends` inheritance, `--set` dotted overrides, per-plugin `validate_config`, and validation of the run-level `sim:` block. |
| `context.py` | `SimContext` — the one object plugins share: time, the command queue (`ctx.post`), the interface registry, the blackboard, deterministic RNG, and the stop request. |
| `registry.py`, `models.py`, `world.py`, `textures.py` | Resolution of plugins, models, worlds and textures by name across installed packages. |
| `runner.py` | The standalone driver (`roqsim sim`): loop, pacing, recording, and the optional viewer. |
| `scenario_adapter.py` | The `scenario-execution` `SimulationInterface` driver. |
| `capture.py`, `recording.py`, `state.py`, `render.py` | State recording and everything read or drawn from it afterwards — driver-level, not plugins. |
| `export_web.py`, `export_capture.py`, `export_urdf.py`, `export_srdf.py` | A compiled world or a recorded run out to a browser scene descriptor, a run capture, a URDF, or a MoveIt SRDF. |
| `commands.py` | The `roqsim` command tree. |

Five built-in plugins, all world-agnostic:

- `dummy` — adds one free-floating box and counts its own hook invocations on the blackboard. It
  validates the framework end-to-end with no assets at all, which is what the test suite asserts on.
- `spawn_model` — place any `roqsim.models` entry in a world as a prop.
- `ceiling` — a `with_ceiling` switch that *deletes* every geom lying entirely above a height cut at
  build time. Hiding it is not enough: contact still happens, and a roof made transparent to open the
  view stops being a roof for the lidar too — `mj_ray` skips a geom exactly when its resolved alpha is
  0, so "invisible" and "unsensed" are the same setting and neither can be had alone. Deletion is what
  lets an overhead sensor and a top-down view see in while everything else keeps its physics.
- `contact_monitor` — turn the contacts that matter into an observable, so a trial ends on a real
  contact force instead of on a tunable clearance threshold.
- `model_override` — change named model values (friction, contact masks, actuator force limits, mass)
  while a run is in progress, on an external trigger, and restore them exactly. It is what makes "the
  gripper loses the object here" a property of the world rather than of whatever drives the robot.
  Curated allowlist: fields MuJoCo cannot take at runtime are refused by name, with the reason.

## Run it

```bash
roqsim sim roqsim_mobile:husky_ros2          # a demo world from a sibling package (a viewer opens)
roqsim sim world.yaml --headless --pacing asap --steps 1000 --profile
roqsim sim world.yaml --seed 7 --record run.npz --video run.webm
```

`roqsim` is the only name to know: `roqsim --help` lists the groups (one per installed package that ships
tools), `roqsim <group> --help` gives one line per tool, and `roqsim <group> <tool> --help` is that tool's
own options. `python -m pydoc <module>` has the reasoning behind one.

A world is one YAML file — a `sim:` block of run-level settings and a `components:` list, where each
entry names a plugin and configures it:

```yaml
sim:
  pacing: realtime          # realtime | asap | {factor: N}

components:
  - spawn_robot: {model: husky_a200, pos: [0, 0], yaw: 0}
    name: robot                              # names the entry, and so the entity it spawns
    components:                              # what belongs to that robot
      - diff_drive: {test_cmd: [0.5, 0.4]}
```

A component's owner is **the entry it is nested under** — there is no `robot:` key to write, and so
no way to write it wrong. A robot's own sensors and controllers come from its model manifest and need
no mention here at all; nest an entry only to add something the model does not ship.

Plugins are referenced by registered name, by `module:Class`, or by `file.py:Class` beside the world —
which is how an experiment loads its own plugin without registering anything.

> **Two changes to the world format.**
>
> `plugins:` was renamed to **`components:`**. The former spelling still loads, so existing worlds and
> model manifests keep working; a document carrying *both* keys is refused, because two spellings of
> one key in one file is a merge nobody can predict. Anything that reads a loaded world back — `roqsim
> scenes describe`, the exporters, `roqsim scenes floorplan-to-world` — now emits `components:`.
>
> **Ownership is nesting, and `name:` is a sibling.** A sensor or controller belongs to the entry it
> is nested under, so the per-family `robot:` / `arm:` config keys are gone. An entry's `name:` moved
> out of the plugin's config to sit beside the plugin ref, and it is now the *one* name an entry has:
> it labels the entry, names the entity a `spawn_*` or prop creates, and is what `disable:` and an
> override address.
>
> An entry's **address** is the dotted path of labels from the top of the document — `robot`,
> `robot.lidar` — and it is what an entity is registered under, so two robots may each carry an
> `arm` without one hiding the other. A top-level entry's address is just its label, so no existing
> world's entity names change. Three things are refused at load, because each would otherwise make
> an address mean two things: two components of one owner sharing a label; a `name:` containing
> `. * # [ ] = : /` or whitespace; and a component whose label is also one of its owner's config
> keys.

## Reproducibility

`sim.seed` in the world -- or `roqsim sim --seed N`, which wins -- sets the run's seed and announces it (a run without either draws a seed and logs it,
so it can be replayed). Plugins draw noise from `ctx.rng_for(name)`, which is **counter-based** rather
than stateful: its draws are a pure function of `(seed, episode, sim_time, name)`, so a value is reproducible
without replaying the stream that preceded it — and a sensor re-run from a recording produces the same
noise the live run published. See the docstring on `SimContext.rng_for` for why a shared stateful
generator cannot do this.

`ctx.request_stop(reason)` ends a run when the trial is actually over, instead of padding it out to a
wall-clock `--seconds` guessed high enough for the slowest cell. It is a request: `shutdown` still
runs and files still flush, and an embedding driver may ignore it.

## Extend it

Four entry-point groups, each read by the core and declared by the provider:

| group | contributes |
|-------|-------------|
| `roqsim.plugins` | a plugin, usable by name in any world |
| `roqsim.models` | a model (robot, prop, sensor), resolvable as `<pkg>:<name>` |
| `roqsim.worlds` | a world, runnable as `roqsim sim <pkg>:<name>` |
| `roqsim.commands` | a group of tools under `roqsim <group>` |

Two rules the engine relies on, and a plugin that breaks either produces nondeterminism that is very
hard to trace:

- **Single writer.** Only the physics thread (the one calling `engine.step()`) touches `model` /
  `data`. External input — ROS callbacks, services, a UI — goes through `ctx.post(cmd)`.
- **No mid-run recompile.** Modify the `MjSpec` only in `build()`. At runtime use mocap writes, qpos
  writes, or the entity pool.

## Install and test

```bash
pip install -e "roqsim[test]"
python -m pytest roqsim/tests
```

Runtime dependencies are `mujoco`, `numpy`, `pyyaml`, `click` and `pillow`. Video output additionally
needs `ffmpeg` on `PATH` (checked, with a message, rather than failing mid-render); rendering a raw
mesh needs `roqsim_assets`. From the repository root, `make venv` installs the whole family and
`make test` runs every package's tests.

## Docs

`docs/` (Sphinx, built with `make doc`) is the source of truth: `architecture.rst` for the engine,
the plugin lifecycle and the porting playbook; `plugins.rst` for what each plugin does;
`interfaces.rst` for the endpoint model the ROS bridge serves; `developer_guide.rst` for adding a
plugin or a tool.
