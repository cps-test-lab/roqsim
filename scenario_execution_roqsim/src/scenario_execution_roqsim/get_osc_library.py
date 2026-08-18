# Copyright (C) 2026 Frederik Pasch
# SPDX-License-Identifier: Apache-2.0

"""Expose this package's ``lib_osc`` to scenario-execution's ``import osc.roqsim``."""


def get_osc_library():
    """Return ``(package, filename)`` -- NOT a path.

    scenario-execution resolves the pair as ``files(package)/lib_osc/<filename>`` via
    ``importlib.resources``, which is what makes this work from a pip install as well as from a colcon
    share directory. Returning a path instead fails far from here, as ``too many values to unpack
    (expected 2)`` while *parsing the scenario*, because the caller unpacks the result before it ever
    touches the filesystem.
    """
    return "scenario_execution_roqsim", "roqsim.osc"
