"""roqsim_manipulation_assets: the arm and gripper models for roqsim.

The ASSET half of what used to be one `roqsim_manipulation`. The plugins (``spawn_arm``,
``arm_controller``, ...) stayed there; the geometry -- 115 MB of it -- is here.

Why they are separate: every robot family with an actuated limb needs ``arm_controller``, and almost
none needs these models. `roqsim_humanoid` reuses it for the G1's arms; `roqsim_mobile_manipulation` reuses
it for Frankie's and TIAGo Pro's. While plugins and models shared a package, each of those dragged in
every arm and gripper it never loads. The name follows ``roqsim_assets``, which is the same
idea for scene props.

The dependency runs THIS -> ``roqsim_manipulation``, never the reverse: a model's manifest names the
plugins intrinsic to it (its ``arm_controller`` with that arm's joints, gains and gripper mapping),
while the plugins know nothing about any particular model. That asymmetry is what keeps it acyclic.
"""

import os

#: Dir holding the shipped demo world YAMLs; handy for tools and tests.
WORLDS_DIR = os.path.join(os.path.dirname(__file__), "worlds")
