"""roqsim quadruped family: the Boston Dynamics Spot robot + its RL locomotion controller.

Provides a ``roqsim.models`` provider (``spot``) and a ``roqsim.plugins`` controller
(``spot_locomotion``) that turns a body-frame velocity command (``cmd_vel``) into smooth walking via
a pretrained flat-terrain policy -- the quadruped analogue of ``roqsim_humanoid``'s
``g1_locomotion`` and ``roqsim_mobile``'s ``diff_drive``.
"""
