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

Models
------

.. roqsim-models::

Worlds
------

.. roqsim-worlds::
