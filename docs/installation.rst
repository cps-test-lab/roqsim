Installation
============

roqsim consists of multiple packages:

* ``roqsim`` — the ROS-free framework core (engine, plugins API, drivers, the ``roqsim`` command tree).
* ``roqsim_sensors`` — generic sensor plugins and models (lidar, OAK-D RGB-D, RealSense, force/torque,
  fiducial markers).
* ``roqsim_assets`` — the prop library (furniture, containers, fittings) and the scene tools.
* ``roqsim_scenes`` — baked MJCF scenes and the world/floorplan generators.
* ``roqsim_mobile`` — wheeled **bases**: floorplan, ``spawn_robot``, diff-drive, omni-drive, and the
  base models (TurtleBot 3/4, Husky, Jackal).
* ``roqsim_manipulation`` — manipulator **plugins** (``spawn_arm``, ``arm_controller``,
  ``cartesian_admittance``); ``roqsim_manipulation_assets`` — the arm and gripper **models**.
* ``roqsim_mobile_manipulation`` — robots that are a base **and** an arm (``frankie``, ``tiago_pro``);
  the one package depending on both families.
* ``roqsim_humanoid`` / ``roqsim_quadruped`` — legged robot families; ``roqsim_walker`` — kinematic
  pedestrians (dynamic obstacles, not a robot family).
* ``roqsim_scene_builder`` / ``roqsim_webctrl`` — the human-in-the-loop scene windows and the web control UI.
* ``scenario_execution_roqsim`` — the OpenSCENARIO 2 vocabulary (``import osc.roqsim``): what a
  scenario can ask a running simulation and what it can break in one. Named for
  scenario-execution's own convention rather than ours, because that is the project it plugs into.
* ``ros2_ws/`` — colcon packages: ``roqsim_ros_bridge`` (the ROS 2 bridge) and
  ``roqsim_nav2_example`` (a nav2 example + test).

Install what you need: every family package pulls its own dependencies, and none of them depends on
another family. ``make venv`` installs them all in editable mode.

The Makefile (recommended)
--------------------------

.. code-block:: bash

   make venv     # create .venv and install both packages + docs/format tooling
   make help     # list all targets

The virtual environment is created with ``--system-site-packages`` **on purpose**: when you later
``source`` a ROS 2 distribution, its Python packages (``rclpy``, nav2, ``simulation_interfaces``)
become importable by the *same* interpreter that has MuJoCo and the roqsim packages. This avoids
the common footgun where ``ros2 run`` uses ``/usr/bin/python3``, which cannot see your venv.

Manual install
--------------

.. code-block:: bash

   python3 -m venv --system-site-packages .venv
   .venv/bin/pip install -e "roqsim[test]" -e "roqsim_sensors[test]" -e "roqsim_mobile[test]"

ROS 2 (optional)
----------------

For the ROS 2 bridge and the nav2 example, source a ROS 2 distribution (Jazzy) and build the
workspace:

.. code-block:: bash

   source /opt/ros/jazzy/setup.bash
   make build-ros          # colcon build ros2_ws (only runs when ROS is sourced)
   source ros2_ws/install/setup.bash

External assets (fetched + converted, not committed)
----------------------------------------------------

Some vendor CAD/mesh sources have unclear redistribution terms, so instead of committing derived files
roqsim **regenerates them locally** from their sources and git-ignores both. They are declared in
``external/external_assets.yaml`` — per resource: sources with URLs, a conversion script and its
dependencies, and the generated target paths — and driven by:

.. code-block:: bash

   make external-list                 # show resources, their sources and generated targets
   make external-resources            # fetch every source + run its conversion into the targets
   make external-resources RESOURCE=livox_mid360_meshes BLENDER=/path/to/blender   # one resource
   make external-sync-gitignore       # rewrite the managed .gitignore block from the manifest
   make add-external-resource ARGS="--name X --source URL::PATH[::manual] --target PATH ..."

Two resources are declared today, spanning the two shapes the schema supports:

``livox_mid360_meshes``
   Two Livox STEP files (placed under ``external/sources/livox/``) are tessellated with Open CASCADE
   (``cascadio``) and processed in Blender into the housing / dome / FOV meshes. Its sources are
   ``manual`` — they sit behind Livox's product page — so the runner tells you where to put the file
   rather than downloading it.

``spot_locomotion_policy``
   The NVIDIA-licensed Boston Dynamics Spot policy (``spot_policy.pt`` + ``spot_env.yaml``). A
   **download-only, optional** resource: fetched anonymously from NVIDIA's public bucket, and because it
   is only needed to *run* Spot the fetch is fail-soft — a network error warns and skips instead of
   breaking ``make venv``. ``make venv`` fetches it through this system, and
   ``python -m roqsim_quadruped.policy.fetch_policy`` is a thin wrapper around the same runner.

The tool venv (``external/.venv-tools/``), the fetched sources and the generated outputs are all in the
managed ``.gitignore`` block; only the manifest, the runner and the conversion scripts under
``external/convert/`` are tracked.

Interpreter note
----------------

Because the venv uses ``--system-site-packages``, run ROS entry points with the **venv interpreter**
so ``roqsim`` is importable, e.g.:

.. code-block:: bash

   source /opt/ros/jazzy/setup.bash
   source ros2_ws/install/setup.bash
   .venv/bin/python -m roqsim_ros_bridge.run_bridge \
       --world ros2_ws/src/roqsim_ros_bridge/worlds/turtlebot_ros2.yaml

Container images
----------------

Two images are published to the GitHub Container Registry on every push to ``main``, each as a
multi-architecture index covering **linux/amd64 and linux/arm64** — so the same reference works on
an x86 node and on an arm64 machine (an Apple Silicon laptop, a Graviton/Ampere node, an arm64
robot host) with nothing to select by hand:

``ghcr.io/cps-test-lab/roqsim``
   The lean core: ROS-free, headless MuJoCo, with the sensor / mobile / manipulation / scenes /
   walker packages. ``ENTRYPOINT`` is the ``roqsim`` command tree, so the image is used the way the
   CLI is:

   .. code-block:: bash

      docker run --rm ghcr.io/cps-test-lab/roqsim --help
      docker run --rm -v "$PWD:/work" -w /work ghcr.io/cps-test-lab/roqsim \
          sim my_world.yaml --headless --seconds 10

``ghcr.io/cps-test-lab/roqsim-ros``
   ROS 2 Jazzy, nav2, MoveIt and rviz2, every ``roqsim_*`` package, and a colcon-built ``ros2_ws``
   (bridge, nav2 example, walker_ros). Its entrypoint sources ROS and the workspace, then execs
   what you pass:

   .. code-block:: bash

      docker run --rm ghcr.io/cps-test-lab/roqsim-ros ros2 pkg list

Neither image sets ``MUJOCO_GL``: which offscreen backend works is a property of the node the image
lands on, not of the image, so both backends are installed and
:func:`roqsim.gl.select_offscreen_gl` picks at import. Baking a value in would override that choice
with a guess — see :doc:`architecture`.

Building them yourself
~~~~~~~~~~~~~~~~~~~~~~

``container/build.sh`` wraps ``docker build`` with the repo root as its context. It builds for the
host by default, which is what a local test wants:

.. code-block:: bash

   ./container/build.sh --image roqsim
   ./container/build.sh --platform linux/arm64 --image roqsim   # one explicit architecture

``--multiarch`` instead builds every architecture the image is published for, reading the list from
``container/platforms.env``. It requires ``--push`` and ``--project``: a multi-platform build
produces an index, and since the local daemon can hold only one image there is nowhere but a
registry for the result to go.

.. code-block:: bash

   ./container/build.sh --multiarch --project ghcr.io/cps-test-lab --push

The architecture policy
~~~~~~~~~~~~~~~~~~~~~~~

``container/platforms.env`` is the single source of truth for which architectures each image is
built for, read by both ``container/build.sh`` and ``.github/workflows/image.yml`` — so a local
build and a CI build cannot disagree about what an image is. It also maps each platform to the
GitHub runner native to it, because CI builds each architecture on its own hardware and merges the
results into one manifest rather than emulating under QEMU.

That is a correctness measure before it is a speed one. Asking buildx for an architecture the base
image does not publish does not fail — it takes the base's only architecture, labels it with the
one requested, and pushes an image that dies with ``Invalid ELF image for this architecture`` the
first time anything execs inside it, at run time and on someone else's machine. A native build has
no second architecture present to mislabel, and CI additionally runs each freshly built image on
its own native hardware before pushing, which is the only check that catches this class of fault:
a mislabelled image builds, pushes and inspects perfectly.

Adding an architecture is therefore one edit in ``container/platforms.env`` (its platform line plus
a ``RUNNER_`` mapping) and nothing in the workflow. ``make check`` fails if a platform has no
runner mapped, or if the workflow starts naming an architecture of its own. The ceiling is what the
**base** image publishes: ``python:*-slim`` and the official ``ros:*-ros-base`` are both
multi-architecture indexes, which is what makes arm64 available here at all.
