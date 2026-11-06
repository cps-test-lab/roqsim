# Copyright (C) 2026 Frederik Pasch
#
# SPDX-License-Identifier: Apache-2.0

"""Motion planning for a fixed arm in a cluttered cell, in process and on the world's own geometry.

A cell planned with MoveIt describes its obstacles a second time, as a planning scene of boxes, and
runs the planner in another process on another clock. Here the planner checks the compiled MuJoCo
world itself -- the presses, the doors where they stand now, the boxes, the parts a gripper holds --
and hands the arm a trajectory its ``arm_controller`` follows on simulated time, so a stepped run is
deterministic and needs no ROS.

- :class:`ArmKinematics` (:mod:`.kinematics`): forward kinematics, the Jacobian and inverse kinematics
  of a site on the arm, seeded and staying on the branch nearest the seed.
- :class:`CollisionChecker` (:mod:`.collision`): whether a configuration of the arm, with what it
  holds, touches anything it should not, on a copy of the model and data.
- :func:`plan_path` (:mod:`.rrt`): a collision-free joint-space path, RRT-Connect then shortcutting;
  :func:`cartesian_path` a straight line of the site, checked.
- :func:`time_parameterize` (:mod:`.timing`): the fastest timing of a path under per-joint velocity
  and acceleration limits (reachability analysis on the path parameter, as TOPP-RA does), retimed
  slower where the joint torques it needs, by inverse dynamics with the payload, exceed a fraction of
  the drives' limits.
- :mod:`.margins`: how far a pose is from each joint's limits, from a singularity and from the
  torque limits.
- :mod:`.hold`: whether a gripper holds a part, by its pads' contacts and the part's drift.

Everything draws randomness from the generator it is given, so a seeded run plans the same paths.
"""

from .collision import CollisionChecker, CollisionError
from .hold import HoldMonitor, pads_touch
from .kinematics import ArmKinematics, IKError
from .margins import PoseMargins, pose_margins
from .rrt import PlanningError, cartesian_path, plan_path
from .timing import TimedPath, TimingError, time_parameterize

__all__ = [
    "ArmKinematics",
    "CollisionChecker",
    "CollisionError",
    "HoldMonitor",
    "IKError",
    "PlanningError",
    "PoseMargins",
    "TimedPath",
    "TimingError",
    "cartesian_path",
    "pads_touch",
    "plan_path",
    "pose_margins",
    "time_parameterize",
]
