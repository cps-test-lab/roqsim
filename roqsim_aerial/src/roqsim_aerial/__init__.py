"""roqsim aerial-vehicle plugins and assets.

Plugins: ``quadrotor_controller`` (cascaded position + attitude control over collective thrust and
three body moments). Models: ``crazyflie_2``, with a demo world under ``worlds/``.

Two things separate an aerial world from a ground one, and both bite silently. MuJoCo defaults
``density`` and ``viscosity`` to 0, so a world that does not set ``density``/``viscosity`` flies the drone
through a vacuum -- it hovers, but nothing damps it. And a quadrotor MJCF has no stabiliser: an
uncommanded drone is not a robot standing still, it is a falling brick, which is why
``crazyflie_2``'s manifest pulls the controller in rather than offering it.
"""

import os

__version__ = "0.1.0"

#: Dir holding the shipped demo world YAMLs; handy for tools and tests.
WORLDS_DIR = os.path.join(os.path.dirname(__file__), "worlds")
