"""roqsim_mobile_manipulation: robots that are a mobile base AND an arm.

Composite platforms only. `roqsim_mobile` holds wheeled bases, `roqsim_manipulation` holds arms and their
controllers, and a robot that is both belongs to neither -- so it lives here, in the one package that
may legitimately depend on both. Keeping these two models in `roqsim_mobile` is what previously forced
that package to depend on `roqsim_manipulation`, inverting its own documented contract.

Shipped models (one folder each, ``models/<name>/``):

* ``frankie``    -- a Franka Emika Panda with a Franka Hand on an Omron LD-60 differential-drive AGV
  (the QUT Centre for Robotics "Frankie" platform).
* ``tiago_pro``  -- PAL Robotics TIAGo Pro: holonomic base, lifting torso, two 7-DOF arms.

Neither adds a plugin: both are assembled from ``spawn_robot`` + ``diff_drive``/``omni_drive`` +
``arm_controller`` + ``lidar``, with the wiring in each model's manifest. That is the point their port
logs measure -- a composite robot that needs a new plugin means the composition mechanism is missing
something.

The recurring failure mode with these robots is **actuator ownership**: ``arm_controller``'s default
prefix scan claims every actuator sharing the entity's prefix, which on a mobile manipulator includes
the wheel motors. Both manifests therefore name their ``joints:`` explicitly, and both test suites
assert the controllers own disjoint actuator sets.
"""

import os

#: Dir holding the shipped demo world YAMLs; handy for tools and tests.
WORLDS_DIR = os.path.join(os.path.dirname(__file__), "worlds")
