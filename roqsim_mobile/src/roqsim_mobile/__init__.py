"""roqsim mobile-robot plugins and assets: wheeled bases only.

Plugins: ``floorplan`` (a room built from walls), ``spawn_robot``, ``diff_drive`` (differential and
skid-steer), ``omni_drive`` (holonomic/mecanum). Models: ``turtlebot4``, ``husky_a200``,
``clearpath_jackal``, ``turtlebot3_waffle``, each with a demo world under ``worlds/``.

A robot that is a base **and** an arm does not belong here -- it goes in a package that depends on
both this and ``roqsim_manipulation`` (``roqsim_mobile_manipulation``). Keeping one here would force
this package to depend on the arm package, inverting the contract in its first line.
"""

import os

__version__ = "0.1.0"

#: Dir holding the shipped demo world YAMLs; handy for tools and tests.
WORLDS_DIR = os.path.join(os.path.dirname(__file__), "worlds")
