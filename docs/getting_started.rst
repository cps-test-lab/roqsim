Getting started
===============

The shortest path from zero to a running simulation. No ROS, no concepts — just get it on screen.

Prerequisites
-------------

* Linux with Python 3.10+ and ``git`` (MuJoCo is installed automatically into the venv).

Five steps
----------

.. code-block:: bash

   # 1. get the code
   git clone <roqsim-repo-url>
   cd roqsim

   # 2. create the virtual environment and install everything
   make venv

   # 3. run the TurtleBot 4 demo world (a viewer window opens)
   .venv/bin/roqsim sim roqsim_mobile:turtlebot4_demo

A MuJoCo window opens and the TurtleBot 4 drives in a circle. That's it.

No display? Run it headless
---------------------------

.. code-block:: bash

   .venv/bin/roqsim sim roqsim_mobile:turtlebot4_demo \
       --headless --pacing asap --steps 1000 --profile

``--profile`` prints a per-plugin timing table so you can see where the time goes.

``roqsim_mobile:turtlebot4_demo`` is a world resolved by name from an installed package; a path to a
world YAML works just as well. ``roqsim`` is the only command name to learn — ``roqsim --help`` lists the
groups, one per installed package that ships tools.

Where to next
-------------

* :doc:`quickstart` — run headless, change pacing, drive from scenario-execution or ROS 2.
* :doc:`interfaces` — the world YAML schema and the plugin/context APIs.
* :doc:`plugins` — the built-in plugins and their config keys.
* :doc:`nav2_example` — navigate the TurtleBot with nav2 over ROS 2.
