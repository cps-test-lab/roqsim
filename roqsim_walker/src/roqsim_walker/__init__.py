"""Kinematic pedestrian walkers for roqsim (ROS-free).

Ported from our earlier in-house nav prototype. A walker is a flat set of MuJoCo mocap bodies posed every step by forward
kinematics from a motion clip, with a CARLA character mesh skinned onto them. It patrols a configured
route or is driven to goals through a backend-neutral endpoint.
"""

from roqsim_walker.blueprint import available_walkers, resolve_walker

__all__ = ["available_walkers", "resolve_walker"]
