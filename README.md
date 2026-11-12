<p align="center">
  <img src="docs/_static/roqsim-horizontal-github.png" alt="roqsim" width="560">
</p>

<p align="center">
  <a href="https://github.com/cps-test-lab/roqsim/actions"><img src="https://github.com/cps-test-lab/roqsim/actions/workflows/ci.yml/badge.svg" alt="ci"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-blue.svg" alt="Apache-2.0"></a>
  <img src="https://img.shields.io/badge/python-3.12%2B-blue.svg" alt="Python 3.12+">
  <img src="https://img.shields.io/badge/MuJoCo-4.0-orange.svg" alt="MuJoCo 4.0">
</p>

**roqsim** — **ro**bots, **q**uickly **sim**ulated (in [MuJoCo](https://mujoco.org)) — is a
plugin-driven simulation framework where a world and everything in it, robots, sensors, props and
scene, is declared in a **single YAML file**.

```yaml
sim:
  world: empty_room
components:
  - spawn_robot: {model: turtlebot4, pose: {position: {x: 0, y: 0}}}
```

That is a driving, sensing robot: the TurtleBot 4 brings its own differential drive, lidar and RGB-D
camera, because a model's manifest names the plugins intrinsic to it. No C++, no scene graph to
hand-assemble.

## Features

- **One YAML file per world.** Robots, sensors, props and scene declared together; plugins hook
  a MuJoCo step loop at well-defined lifecycle points. Write your own in a file next to the world.
- **43 robot models across 6 families** — 17 wheeled bases (TurtleBot 4 and 3 Waffle, Husky A200,
  Jackal, Ridgeback, Warthog, Panther, ROSbot, MP-400, MPO-500/700, ROX-Diff, LGDXRobot2,
  MakerSpet Mini, Raspimouse, OOMWOO ONE, PiRacer), 14 arms and 3 grippers (UR10e, UR5e, UR30,
  Panda, Gen3, xArm7, LBR iiwa 7 and 14, M1013, FANUC M-20iD/25 and CRX-25iA, OpenManipulator-X,
  ViperX 300s, WidowX 250s; Robotiq
  2F-85 and Hand-E, Schunk PG+70), 2 mobile manipulators (TIAGo Pro, Frankie), 4 humanoids (Unitree G1,
  G1 + Dex1, LimX Oli, AgiBot G2), Boston Dynamics Spot, the Crazyflie 2 and an X500 — each
  vendored with pinned upstream provenance, the X500 authored from PX4's own airframe definition.
- **Sensors, and where to put them.** Lidar, RGB-D, IMU, force-torque and fiducial markers, with
  19 bundled sensor device models. The IMU reports proper acceleration, true attitude (or none,
  marked as such) and covariances built from its declared noise, so a `robot_localization` stack
  has the input it expects; a segmentation camera adds per-pixel class and instance labels with
  tight 2D boxes, measured from the mask, so a perception experiment has ground truth to be scored
  against. Coverage analysis answers the question that actually blocks you: *how many cameras, and
  where?*
- **Scenes from what you already have.** Import Gazebo SDF, USD or CAD — or draw a floorplan in a window
  and get a world back.
- **Light that looks real.** Baked lighting: a world's bounced light, soft contact shadows and ambient
  occlusion path-traced once with Blender (`roqsim bake`, minutes on a CPU), kept with the world and
  drawn by MuJoCo's renderer, while robots are lit live and throw their shadows into it.
- **People as dynamic obstacles.** Kinematic pedestrians with A\* and behaviour-tree navigation, plus
  optional ORCA local avoidance, for the case your robot has to share a corridor.
- **Runs where you need it.** Viewer by default, headless for CI and containers; real-time, scaled, or
  as-fast-as-possible pacing.
- **Speaks to your stack.** A ROS 2 bridge exposing standard `simulation_interfaces`, a working nav2
  example, and a `SimulationInterface` for
  [scenario-execution](https://github.com/cps-test-lab/scenario-execution). The core itself is
  ROS-free and pip-installable.
- **Answers questions afterwards.** Record a run, then pull poses, joints, contacts and sensor series
  out of it — or export the scene to the browser.
- **Extensible without forking.** Third-party packages register plugins, models, worlds and textures by
  entry point. The core never learns their names.

## Quick start

```bash
make venv     # create .venv and install everything
make help     # list all targets

.venv/bin/roqsim sim roqsim_mobile:turtlebot4_demo      # a viewer opens
```

**Ready-to-run worlds** ship in the box, named by a `<package>:<world>` ref that `roqsim sim`
takes. Every wheeled base, mobile manipulator and aerial vehicle, and most arms, also register a
`<robot>_demo` world that shows that one robot in an empty room, which is how the commands above
run; `roqsim catalog worlds --refs` lists every world the installation offers. `roqsim --help` lists the core's commands and the command groups; `roqsim <group> --help` gives one line per tool.

Headless, as fast as the machine allows, with timings:

```bash
.venv/bin/roqsim sim roqsim_mobile:turtlebot4_demo --headless --pacing asap --steps 1000 --profile
```

## Documentation

The documentation is published at **<https://cps-test-lab.github.io/roqsim/>**, rebuilt from
`main` — read the model catalog there, where each model shows its preview. The sources:
start with [getting started](docs/getting_started.rst), then:

| | |
| --- | --- |
| [Installation](docs/installation.rst) | packages, the venv, ROS 2 |
| [Quickstart](docs/quickstart.rst) · [Plugins](docs/plugins.rst) | writing a world; every built-in plugin |
| [Models](docs/models.rst) · [Textures](docs/textures.rst) | the robot/prop catalog; surfaces and floors |
| [Rendering](docs/rendering.rst) · [Baked lighting](docs/baked_lighting.rst) | renderers; realistic light and shadows |
| [Interfaces](docs/interfaces.rst) | ROS 2, scenario-execution, your own code |
| [Scene builder](docs/scene_builder.rst) · [Coverage](docs/coverage.rst) | building worlds; sensor placement |
| [nav2 example](docs/nav2_example.rst) · [Ground truth](docs/ground_truth.rst) | navigation; getting numbers out |
| [Architecture](docs/architecture.rst) | how it fits together, and the porting playbook |

Build them locally with `make doc` (or `make view-doc`).

## Licensing

roqsim's own code is **Apache-2.0** (see [LICENSE](LICENSE)).

Vendored third-party assets keep their own terms, and some of those terms **require attribution when
you redistribute**. The authoritative records are the `THIRD_PARTY.md` file in each package, the
`CREDITS.txt` beside each asset and the licence file beside each model; [NOTICE](NOTICE) summarises
them by licence and says where they are.

| license | what is under it |
| --- | --- |
| BSD (3-Clause, 2-Clause) | robot, arm, gripper and sensor models; locomotion policies |
| Apache-2.0 | robot, arm, gripper and sensor models; a locomotion policy; a scene |
| MIT | robot, aerial vehicle and sensor models |
| MPL-2.0 | a robot model -- file-level copyleft, the notice travels with the files |
| CC-BY-4.0 | props, a scene, pedestrian characters, every locomotion clip -- attribution required |
| CC0-1.0 | surface textures, pedestrian characters |

An asset's own record carries its licence and the attribution that licence asks for, and ships
with it in its package.

Nothing under a non-commercial (`CC-*-NC`) or no-derivatives (`CC-*-ND`) license is included.
The splat renderer (`roqsim_render_splat`) draws through gsplat (Apache-2.0) and PyTorch
(BSD-3-Clause), installed as its dependencies; the reference 3DGS rasteriser, whose licence is
non-commercial, is neither a dependency nor used.
The GPU ray-cast backend (`roqsim_raycast_mjwarp`) casts through MuJoCo Warp and NVIDIA Warp (both
Apache-2.0), installed only with its `[mjwarp]` extra.
