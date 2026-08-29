roqsim
=========

**roqsim** is a lightweight, plugin-driven `MuJoCo <https://mujoco.org>`_ simulation framework for
mobile robots, robot arms, and mobile manipulators. A MuJoCo step loop plus **plugins** that hook
into well-defined lifecycle points, all loaded and configured from a **single YAML file**.

It runs **standalone** (a viewer by default, headless for CI and containers, real-time / scaled /
as-fast-as-possible pacing) and as a `scenario-execution
<https://github.com/IntelLabs/scenario_execution>`_ ``SimulationInterface``. The core is ROS-free and
pip-installable; a ROS 2 bridge is provided as a separate package.

If you just want to see it run, start with :doc:`getting_started`.

.. toctree::
   :maxdepth: 2
   :caption: User guide

   getting_started
   installation
   quickstart
   interfaces
   plugins
   models
   textures
   coverage
   scene_builder
   nav2_example
   ground_truth

.. toctree::
   :maxdepth: 2
   :caption: Internals & development

   developer_guide
   architecture
   profiling
   future_work
