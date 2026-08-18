"""roqsim humanoid family: the Unitree G1 legged robot + its RL locomotion controller.

Provides a ``roqsim.models`` provider (``unitree_g1``) and a ``roqsim.plugins`` controller
(``g1_locomotion``) that turns a body-frame velocity command (``cmd_vel``) into smooth walking via a
pretrained policy + PD loop -- the legged analogue of ``roqsim_mobile``'s ``diff_drive``.
"""
