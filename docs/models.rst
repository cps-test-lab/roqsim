Models & worlds
===============

Robots, arms, sensors, and props ship as **models** (registered in the ``roqsim.models``
entry-point group); the static environment they sit in is a **world** (a built-in world definition,
or a baked MJCF scene registered in ``roqsim.worlds``). A spawn plugin references a model by short
name — e.g. ``spawn_robot: {model: turtlebot4}`` for a robot, or ``spawn_model: {model:
industrial_table}`` to place a static **prop** — and ``sim.world`` selects the world; see
:doc:`plugins` and :doc:`architecture`. Props (from ``roqsim_assets``) are ordinary models: the
same MJCF can be placed from a world YAML with ``spawn_model`` or ``<include>``-d into a baked scene.

.. The catalogs below are generated at build time from the roqsim.models / roqsim.worlds
   registries (see docs/_ext/). Preview thumbnails are rendered once with `make thumbnails` and
   committed beside each model as <name>.thumb.png; a model with no thumbnail simply shows no image.
   This note is a source comment and is intentionally not rendered.

Asking from a shell (or a container)
------------------------------------

The catalogs on this page are rendered at docs-build time, which is no help inside a runtime image.
The same two registries answer there::

    roqsim catalog models              # every model, with the ref that resolves it
    roqsim catalog worlds              # every world YAML, baked scene and built-in definition
    roqsim catalog models --refs       # just the refs, one per line, for piping
    roqsim catalog model turtlebot4    # one model: its file, its manifest, where its meshes come from

Both print JSON, and every row carries ``use`` -- the line to put in a world file or on the command
line -- because "what do I type" is the question being asked. ``python -m roqsim.catalog models``
is the same tool for a caller that has a shell in the image but not the console script, and
``roqsim mcp serve`` exposes ``list_models``, ``get_model_details``
and ``list_worlds`` to an MCP client as the same three functions.

A name printed there is a name that resolves: ``roqsim/tests/test_catalog.py`` hands every ref back
to the loader. What is listed is what each package *offers* -- its ``roqsim.worlds`` entry point --
rather than every YAML it ships: a package's debugging worlds are deliberately unregistered and are
run by path, so a world absent from the listing is a decision rather than an omission.

Models
------

.. roqsim-models::

Worlds
------

.. roqsim-worlds::
