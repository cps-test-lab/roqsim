"""The robot's side of a Franka Research 3, held to Franka's releases.

``roqsim arm franka-upstream`` pins the releases. The limits here are ``rate_limiting.h``'s and
``joint_limits.yaml``'s; the tests read those files and fail when a number here differs.

- :mod:`.limits` -- the joint and Cartesian limits the robot enforces.
- :mod:`.robot_model` -- the robot's own model (nominal arm + the load it was told).
"""
