<p align="center">
  <img src="docs/_static/roqsim-horizontal-github.png" alt="roqsim" width="560">
</p>

<p align="center">
  <a href="https://github.com/cps-test-lab/roqsim/actions"><img src="https://github.com/cps-test-lab/roqsim/actions/workflows/ci.yml/badge.svg" alt="ci"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-blue.svg" alt="Apache-2.0"></a>
  <img src="https://img.shields.io/badge/python-3.10%2B-blue.svg" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/MuJoCo-3.0%2B-orange.svg" alt="MuJoCo 3.0+">
</p>

**roqsim** — **ro**bots, **q**uickly **sim**ulated (in [MuJoCo](https://mujoco.org)) — is a
plugin-driven simulation framework where a world and everything in it, robots, sensors, props and
scene, is declared in a **single YAML file**.

```yaml
sim:
  world: empty_room
components:
  - spawn_robot: {model: turtlebot4, pos: [0, 0]}
```

That is a driving, sensing robot: the TurtleBot 4 brings its own differential drive, lidar and RGB-D
camera, because a model's manifest names the plugins intrinsic to it. No C++, no scene graph to
hand-assemble.

## Features

- **One YAML file per world.** Robots, sensors, props and scene declared together; **48 plugins** hook
  a MuJoCo step loop at well-defined lifecycle points. Write your own in a file next to the world.
- **18 robot models across 5 families** — 4 wheeled bases (TurtleBot 4, TurtleBot 3 Waffle, Husky A200,
  Jackal), 5 arms and 2 grippers (UR10e, UR5e, Panda, Gen3, OpenManipulator-X; Robotiq 2F-85, Schunk
  PG+70), 2 mobile manipulators (Tiago Pro, Frankie), 4 humanoids (Unitree G1, G1 + Dex1, LimX Oli,
  AgiBot G2), and Boston Dynamics Spot — each vendored with pinned upstream provenance.
- **Sensors, and where to put them.** Lidar, RGB-D, force-torque and fiducial markers, with six
  vendor-CAD sensor models. Coverage analysis answers the question that actually blocks you: *how many
  cameras, and where?*
- **Scenes from what you already have.** Import Gazebo SDF, USD or CAD — or draw a floorplan in a window
  and get a world back.
- **People as dynamic obstacles.** Kinematic pedestrians with A\* and behaviour-tree navigation, plus
  optional ORCA local avoidance, for the case your robot has to share a corridor.
- **Runs where you need it.** Viewer by default, headless for CI and Kubernetes; real-time, scaled, or
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

.venv/bin/roqsim sim roqsim_mobile:turtlebot_ros2      # a viewer opens
```

**21 ready-to-run worlds** ship in the box — robot demos, a warehouse scene, sensor rigs and nav2
setups. `roqsim --help` lists the command groups; `roqsim <group> --help` gives one line per tool.

Headless, as fast as the machine allows, with timings:

```bash
.venv/bin/roqsim sim roqsim_mobile:turtlebot_ros2 --headless --pacing asap --steps 1000 --profile
```

## Documentation

Start with [getting started](docs/getting_started.rst), then:

| | |
| --- | --- |
| [Installation](docs/installation.rst) | packages, the venv, ROS 2 |
| [Quickstart](docs/quickstart.rst) · [Plugins](docs/plugins.rst) | writing a world; every built-in plugin |
| [Models](docs/models.rst) · [Textures](docs/textures.rst) | the robot/prop catalog; surfaces and floors |
| [Interfaces](docs/interfaces.rst) | ROS 2, scenario-execution, your own code |
| [Scene builder](docs/scene_builder.rst) · [Coverage](docs/coverage.rst) | building worlds; sensor placement |
| [nav2 example](docs/nav2_example.rst) · [Ground truth](docs/ground_truth.rst) | navigation; getting numbers out |
| [Architecture](docs/architecture.rst) | how it fits together, and the porting playbook |

Build them locally with `make doc` (or `make view-doc`).

## Licensing

roqsim's own code is **Apache-2.0** (see [LICENSE](LICENSE)).

Vendored third-party assets keep their own terms, and some of those terms **require attribution when
you redistribute**. The authoritative records are the `THIRD_PARTY.md` file in each package and the
`CREDITS.txt` beside each asset; [NOTICE](NOTICE) summarises them.

| license | assets |
| --- | --- |
| BSD-3-Clause | Spot and xArm 7 (MuJoCo Menagerie), Unitree G1 ×2, UR5e/UR10e/Robotiq (ROS-Industrial), Jackal, Husky |
| Apache-2.0 | LimX Oli, Panda, OpenManipulator-X, TurtleBot 3/4, Tiago Pro, Husarion ROSbot |
| MPL-2.0 | AgiBot G2 meshes — file-level copyleft, the notice travels with the files |
| MIT | Frankie, Crazyflie 2 (MuJoCo Menagerie) |
| CC0-1.0 | surface textures (ambientCG, Poly Haven) |
| CC-BY-4.0 | the warehouse scene (Gazebo Fuel), pedestrian characters and locomotion clips (CARLA, Fuel) |

Nothing under a non-commercial (`CC-*-NC`) or no-derivatives (`CC-*-ND`) license is included.
