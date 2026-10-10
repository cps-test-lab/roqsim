"""roqsim mobile-robot plugins and assets: wheeled bases only.

The plugins are the ``roqsim.plugins`` entry points in ``pyproject.toml``; the models are the folders
under ``models/``, each robot with a demo world under ``worlds/`` (``roqsim catalog models`` lists them).

A robot that is a base **and** an arm does not belong here -- it goes in a package that depends on
both this and ``roqsim_manipulation`` (``roqsim_mobile_manipulation``). Keeping one here would force
this package to depend on the arm package, inverting the contract in its first line.
"""

import os

__version__ = "0.1.0"

#: Dir holding the shipped demo world YAMLs; handy for tools and tests.
WORLDS_DIR = os.path.join(os.path.dirname(__file__), "worlds")
