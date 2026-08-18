"""roqsim_scenes: static-scene worlds imported from CAD, USD or Gazebo SDF."""

import os

#: World-provider dir for the ``roqsim.worlds`` entry point: baked ``<world>/<world>.xml`` scenes
#: (e.g. ``depot/depot.xml``), so ``sim.world: roqsim_scenes:depot`` resolves here.
WORLDS_DIR = os.path.join(os.path.dirname(__file__), "worlds")
