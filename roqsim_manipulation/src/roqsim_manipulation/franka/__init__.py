"""The robot's side of a Franka Research 3, held to Franka's releases.

``roqsim_manipulation/tools/franka_upstream.py`` pins the releases. The limits here are
``rate_limiting.h``'s and ``joint_limits.yaml``'s, the wire format ``libfranka-common``'s; the tests
read those files and fail when a number here differs.

- :mod:`.limits` -- the joint and Cartesian limits the robot enforces.
- :mod:`.protocol` -- the FCI messages, byte for byte, and the 41 error flags.
- :mod:`.robot_model` -- the robot's own model (nominal arm + the load it was told).
- :mod:`.server` -- the FCI on the network: commands over TCP, the state stream over UDP, lock-step.
"""
