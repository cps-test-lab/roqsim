# Copyright (C) 2026 Frederik Pasch
# SPDX-License-Identifier: Apache-2.0
"""roqsim's lidar ray cast on a CUDA GPU, through MuJoCo Warp: the ``mjwarp`` backend.

``sim.raycaster: mjwarp`` in a world, or ``raycaster: mjwarp`` on one lidar; see
:mod:`roqsim_raycast_mjwarp.backend`. Importing this package imports neither Warp nor MuJoCo Warp:
only a backend that opens does.
"""
