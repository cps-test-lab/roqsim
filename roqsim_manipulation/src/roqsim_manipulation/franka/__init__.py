"""The robot's side of a Franka Research 3: what its controller does with a command, and its FCI.

A simulated FR3 is held to Franka's releases (``external/franka_upstream.py`` pins them): the limits
here are ``rate_limiting.h``'s and ``joint_limits.yaml``'s, the wire format ``libfranka-common``'s,
and the tests read those files and fail when a number here differs.

- :mod:`.limits` -- the joint and Cartesian limits the robot enforces.
- :mod:`.protocol` -- the FCI messages, byte for byte, and the 41 error flags.
- :mod:`.robot_model` -- the robot's own model (nominal arm + the load it was told).
- :mod:`.core` -- the controller: gravity, the internal impedance controllers, the checks, the
  external-torque estimate and the reflexes, one tick per millisecond.
- :mod:`.server` -- the FCI on the network, in lock-step with the physics.

The plugin that puts them on an arm is :mod:`roqsim_manipulation.plugins.franka_fci`.
"""
