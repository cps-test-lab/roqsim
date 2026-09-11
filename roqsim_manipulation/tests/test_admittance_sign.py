# Copyright (C) 2026 Frederik Pasch
# SPDX-License-Identifier: Apache-2.0
"""A contact controller must push back less as the contact pushes back more.

The law and the sensor disagreed about which direction a wrench means. Both conventions are
plausible and they are negatives of each other, so the mistake did not read as a sign error -- it
read as the contact getting away from the controller.
"""

from __future__ import annotations

import numpy as np
import pytest
from roqsim_manipulation.plugins.cartesian_admittance import CartesianAdmittancePlugin
from roqsim_sensors.plugins.force_torque import WrenchReader


def _law(measures="environment_on_tool", pressed_newtons=0.0):
    """The force-controller law alone, with a sensor reporting a steady contact.

    Built rather than run in a world: the question is which way the law drives for a given reading,
    and a physics run answers it only after the divergence has already happened.
    """
    plugin = CartesianAdmittancePlugin.__new__(CartesianAdmittancePlugin)
    plugin.M = np.array([1, 1, 1, 0.6, 0.6, 0.6], float)
    plugin.D = np.array([80, 80, 80, 160, 160, 160], float)
    plugin.C = np.zeros(6)
    plugin.w_d = np.array([0, 0, -10, 0, 0, 0], float)  # press DOWN with 10 N
    plugin._twist = np.zeros(6)
    plugin.max_twist = None
    plugin.axes = np.ones(6)
    # The force controller: no stiffness, so no equilibrium pose takes part in the law.
    plugin._uses_stiffness = False

    # A tool pressing down on a plate: the plate's reaction is UP, which is what a sensor in the
    # default convention reports. The other convention is the same contact, negated.
    reaction = np.array([0.0, 0.0, +pressed_newtons])
    wrench = reaction if measures == "environment_on_tool" else -reaction
    plugin._ft = WrenchReader(
        name="ft", frame="base", read=lambda: (wrench, np.zeros(3)), measures=measures
    )
    plugin.read_pose = lambda: (np.zeros(3), np.eye(3))
    plugin._clamp = lambda t: t
    return plugin


@pytest.mark.parametrize("measures", ["environment_on_tool", "tool_on_environment"])
def test_the_target_contact_force_is_an_equilibrium(measures):
    """At exactly the target, the law commands nothing. That is what "regulates" means, and the
    reading it is computed from differs by a sign between the two conventions -- so a law that
    assumed one of them had an equilibrium in one and a runaway in the other."""
    twist = _law(measures, pressed_newtons=10.0)._wrench_twist(0.01)
    assert twist[2] == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("measures", ["environment_on_tool", "tool_on_environment"])
def test_pressing_too_hard_commands_a_retreat(measures):
    """Above the target the tool must back off. Before the fix this drove further in, which is the
    positive feedback that ran a 10 N target to about 300 N."""
    twist = _law(measures, pressed_newtons=20.0)._wrench_twist(0.01)
    assert twist[2] > 0.0, "must retreat (+z) when the contact exceeds the target"


@pytest.mark.parametrize("measures", ["environment_on_tool", "tool_on_environment"])
def test_free_space_still_seeks_the_contact(measures):
    """With nothing touching, a downward target still descends -- the fix must not merely invert
    the runaway into a controller that never makes contact."""
    twist = _law(measures, pressed_newtons=0.0)._wrench_twist(0.01)
    assert twist[2] < 0.0, "must descend (-z) toward the contact it is asked to make"


def test_the_response_is_monotonic_in_the_contact_force():
    """The property underneath all three: more contact, less push. A controller whose command grows
    with the thing it is regulating has no equilibrium at any target."""
    commands = [_law(pressed_newtons=n)._wrench_twist(0.01)[2] for n in (0, 5, 10, 15, 20)]
    assert commands == sorted(commands), f"must rise monotonically toward retreat, got {commands}"


def test_the_reader_states_which_convention_it_is_in():
    """The fix rests on the sensor saying so rather than the controller assuming."""
    reader = WrenchReader(name="ft", frame="base", read=lambda: (np.zeros(3), np.zeros(3)))
    assert reader.measures == "environment_on_tool"
