Developer guide
===============

The user-facing interfaces are covered in :doc:`interfaces` and :doc:`plugins`. For the full
architecture and — importantly — the **porting playbook** for reworking existing MuJoCo code into
plugins, see the :doc:`architecture` document: the plugin lifecycle in depth, the concurrency model
(single-writer + ``ctx.post`` command queue), rendering, the foreseen synchronous/lockstep mode,
performance/benchmarking, conventions, and the porting playbook (a decision tree and recipes for
turning monolithic MuJoCo code into cooperating plugins).

.. toctree::
   :maxdepth: 2

   architecture

Repository layout
-----------------

.. code-block:: text

   roqsim/            core pip package (framework, ROS-free; depends on no sibling)
   roqsim_sensors/    generic sensor plugins + models + the coverage analysis layer (depends on roqsim)
   roqsim_assets/     prop library; roqsim_scenes/  baked scenes + world generators
   roqsim_mobile/     wheeled BASE plugins + base models (depends on roqsim + roqsim_sensors)
   roqsim_manipulation/         arm plugins only;  roqsim_manipulation_assets/  arm + gripper models
   roqsim_mobile_manipulation/  base AND arm robots — the one package depending on both families
   roqsim_humanoid/ roqsim_quadruped/   legged families;  roqsim_walker/  pedestrians (dynamic obstacles)
   roqsim_scene_builder/ roqsim_webctrl/   human-in-the-loop scene windows, web control UI
   scenario_execution_roqsim/     OSC actions (entity_moved, entity_rotated, set_model_override);
                                    the ONLY package here that may import scenario_execution
   ros2_ws/src/
     roqsim_ros_bridge/            ROS 2 bridge + simulation_interfaces (plugins)
     roqsim_nav2_example/   minimal nav2 example + goal test
   docs/                 this documentation (+ architecture.rst)

Golden rules
------------

* **Single-writer:** only the physics thread (the one calling ``engine.step()``) touches
  ``model``/``data``. External input goes through ``ctx.post(cmd)``.
* **No mid-run recompile:** modify the ``MjSpec`` only in ``build()``; at runtime use mocap/qpos
  writes or a pre-compiled entity pool.
* Keep the core ROS-free. The ROS bridge is just another plugin.
* **Read a model field through ``int()`` before matching it against an ``mjt*`` enum.**
  ``model.jnt_type[j]`` is a numpy scalar, and ``x in (mjJNT_HINGE, mjJNT_SLIDE)`` puts the
  *enum* on the left of ``==`` — which MuJoCo 3.12 answers ``False`` where 3.11 answered
  ``True``. A bare ``value == enum`` still works, so the break is silent and partial: the
  filter just returns nothing, and an arm reports no joints instead of raising. ``int()`` on
  the model value (or plain-``int`` members, as in ``export_capture._SCALAR_JOINTS``) is
  correct on every version.

Developer workflow
------------------

.. code-block:: bash

   make venv      # create .venv and install everything
   make test      # unit tests (+ nav2 integration when ROS is sourced)
   make format    # ruff format + autofix
   make lint      # ruff check (no changes)
   make doc       # build HTML docs into build/html
   make view-doc  # build + open in a browser

.. _adding-a-tool:

Adding a tool
-------------

Every tool is a subcommand of ``roqsim``, so ``roqsim --help`` is the whole inventory and nobody has to know
a path or which package a tool lives in:

.. code-block:: bash

   roqsim --help                       # the groups, one per package that ships tools
   roqsim scenes --help                # one line per tool in that group
   roqsim scenes sdf-to-scene --help   # that tool's own options
   python -m pydoc roqsim_scenes.cli.sdf_to_scene   # the reasoning behind it

Two tools are top-level rather than in a group, because they are the two verbs the substrate exists for:
``roqsim sim`` runs a world and ``roqsim render`` draws one. Everything else is ``roqsim <group> <tool>``.

**Write it standalone, then link it in — in the same commit.**

#. Draft it as a script under ``<pkg>/tools/<name>.py`` with a ``main(argv)`` and a ``__main__`` block.
#. Move the logic to ``<pkg>/src/<pkg>/cli/<name>.py`` and register it with **one line** in that
   package's group (``src/<pkg>/cli/__init__.py``)::

      group.add_command(tool("<pkg>.cli.<name>"))

#. Leave ``<pkg>/tools/<name>.py`` as a three-line wrapper, so running it from the folder still works::

      #!/usr/bin/env python3
      """Run-from-the-folder wrapper for `roqsim scenes <name>`; the logic is in roqsim_scenes.cli.<name>."""
      from roqsim_scenes.cli.<name> import main

      raise SystemExit(main())

A package that ships its first tool also declares the group itself, in its ``pyproject.toml``::

   [project.entry-points."roqsim.commands"]
   scenes = "roqsim_scenes.cli:scenes"

That is the whole mechanism: any installed package can contribute a group this way, including one
maintained outside this repository. It never edits the core.

**You write no help text.** All three levels come from the docstring you already wrote — click takes
the listing line from its first line, ``--help`` is your own ``argparse``, and ``python -m pydoc``
prints the rest. There is no ``short_help`` to fill in and no flag to add. Two conventions carry it,
both ordinary Python:

* the docstring's **first line is a one-line summary** — it is what a listing shows, so keep it plain
  prose (a terminal prints ``\`\`markup\`\``` verbatim);
* the parser says ``ArgumentParser(description=__doc__.split("\n")[0])``. Handing it the *whole*
  docstring is what once made a single ``--help`` cost more than the rest of the tree together.

``make test`` **fails** while a tool with a ``__main__`` block is unregistered, while a wrapper carries
logic, or while a ``--help`` grows an essay. These are checks (``roqsim/tests/test_command_registry.py``),
not conventions — the previous convention decayed silently until a checked-in Makefile was calling a
binary that had never existed.

A tool that runs inside Blender is registered with ``tool(..., blender=True)``: it cannot be imported
here, so the command locates ``blender`` and runs the module inside it.

Sensor coverage (analysis layer)
--------------------------------

``roqsim_sensors.coverage`` is an **offline analysis** layer over a compiled world — not part of the
tick pipeline. It answers "how much of the room / which objects do these sensors see, and by how
many?" (user docs: :doc:`coverage`). Design worth knowing when extending it:

* **One shared FOV, per-type extraction.** ``coverage/fov.py`` defines a single ``SensorFov`` (a posed
  angular sector: camera ``FRUSTUM`` via pinhole intrinsics, lidar ``CONE_BAND`` via azimuth/elevation
  bands) and the ``in_fov`` membership test — it knows nothing about specific sensors. ``coverage/
  adapters.py`` is the **only** module that does: a per-type registry (``@register_adapter``) that builds
  a ``SensorFov`` from a sensor's own parameters, reusing each plugin's resolution logic (a camera's
  MJCF ``fovy``/resolution; a lidar plugin's resolved attributes). A new sensor gets a new adapter
  there; the sensor plugins are never edited. This is the extension seam — keep every other coverage
  module dependent on ``SensorFov`` alone.
* **Engine.** ``coverage/engine.py`` gates each point range → angular FOV (numpy) → line of sight
  (one batched :func:`roqsim.raycast.cast` per sensor — the same seam every raycaster in the tree
  uses). The raycast ``geomgroup`` mask **excludes group 4**, so an absent entity cannot occlude;
  that is the seam's default, so it holds without each call site remembering it. A model's FOV-visualisation mesh is kept
  out by a different axis — it is alpha 0, and ``mj_ray`` skips a geom exactly when its resolved alpha
  is 0, whatever the mask says (its group is 2, not 4/5). Masking by group is still required, because
  raycasts hit *visible* geometry regardless of contact flags.
* **Sampling** (``coverage/sampling.py``) is model-based (works for any world source) and uses only
  numpy + MuJoCo raycasts (via ``raycast.cast_many``: ``mj_multiRay`` casts from *one* origin, so six
  axis rays from each of P grid points is irreducibly P calls — what the helper buys is the flat
  buffers, so the classification vectorises over all points at once instead of per point) — **no
  scipy/trimesh** (they are broken under numpy 2 in the system venv, and
  the layer must not depend on them). Free-space classification is deliberately conservative (drops are
  safe: they only make coverage look worse).
* **Two front doors, one core:** the ``sensor_coverage_probe`` plugin (world-YAML toggle,
  ``plugins/sensor_coverage_probe.py``) and the ``roqsim sensors coverage`` CLI (``coverage/cli.py``). The
  ``coverage`` extra (matplotlib, for the 2D heatmap) is optional; the 3D render needs only MuJoCo.
