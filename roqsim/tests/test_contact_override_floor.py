# Copyright (C) 2026 Frederik Pasch
#
# SPDX-License-Identifier: Apache-2.0

"""``sim.contact_override.solref``: the floor MuJoCo enforces without saying so.

A contact time constant below ``2 * timestep`` is clamped there by the solver. Nothing reports it:
the world records the value it asked for, the model compiles, the run finishes, and the contact
behaves as though the floor had been requested. A world tightening a fit that way reads the tighter
number back out of its own configuration and concludes the solver does not respond to tuning.

The first test measures the clamp rather than asserting it from the docs, so the refusal below stays
tied to MuJoCo's actual behaviour and a version that moved the floor would show up here first.
"""

from __future__ import annotations

import pathlib
import tempfile

import mujoco
import pytest

from roqsim.config import load_config
from roqsim.engine import Engine
from roqsim.plugin import PluginError

#: A 5 kg box resting on a plate: its steady penetration is what a contact time constant buys.
_PROBE = """<mujoco><option timestep="{ts}"/><worldbody>
<geom type="plane" size="5 5 .1" solref="{a} 1"/>
<body pos="0 0 0.0999"><freejoint/><geom type="box" size=".05 .05 .05" mass="5" solref="{a} 1"/>
</body></worldbody></mujoco>"""


def _penetration_nm(timestep: float, timeconst: float) -> float:
    model = mujoco.MjModel.from_xml_string(_PROBE.format(ts=timestep, a=timeconst))
    data = mujoco.MjData(model)
    for _ in range(int(2.0 / timestep)):
        mujoco.mj_step(model, data)
    return (0.05 - data.qpos[2]) * 1e9


def test_mujoco_clamps_a_time_constant_below_two_timesteps_and_says_nothing():
    """The behaviour the refusal exists for. Below the floor every value is the floor."""
    timestep = 0.002
    at_floor = _penetration_nm(timestep, 2.0 * timestep)
    assert _penetration_nm(timestep, 0.0039) == pytest.approx(at_floor, rel=1e-9)
    assert _penetration_nm(timestep, 0.0005) == pytest.approx(at_floor, rel=1e-9)
    # Above it the value is honoured, so the floor is a floor and not a fixed value.
    assert _penetration_nm(timestep, 0.0045) > at_floor


def _build(timestep: float, solref: str) -> None:
    world = pathlib.Path(tempfile.mkdtemp()) / "w.yaml"
    world.write_text(
        f"sim: {{timestep: {timestep}, seed: 1, contact_override: {{solref: {solref}}}}}\n"
        f"components:\n- dummy: {{}}\n",
        encoding="utf-8",
    )
    Engine(load_config(world)).setup()


@pytest.mark.parametrize(
    ("timestep", "solref"),
    [
        (0.002, "[0.004, 1.0]"),  # exactly the floor
        (0.002, "[0.0045, 1.0]"),  # above it
        (0.0002, "[0.0005, 1.0]"),  # a smaller step reaches a value a larger one cannot
    ],
)
def test_a_time_constant_at_or_above_the_floor_builds(timestep, solref):
    _build(timestep, solref)


@pytest.mark.parametrize(
    ("timestep", "solref", "floor"),
    [(0.002, "[0.0005, 1.0]", "0.004"), (0.0005, "[0.0005, 1.0]", "0.001")],
)
def test_a_time_constant_below_the_floor_is_refused_naming_it(timestep, solref, floor):
    """The floor moves with the step, so the message has to carry the one that applies."""
    with pytest.raises(PluginError, match="below MuJoCo's floor"):
        _build(timestep, solref)
    with pytest.raises(PluginError, match=floor):
        _build(timestep, solref)


def test_a_negative_solref_is_not_a_time_constant_and_is_left_alone():
    """MuJoCo reads a negative pair as ``(-stiffness, -damping)``. No floor applies, and refusing it
    would reject a world that never asked for a time constant."""
    _build(0.002, "[-1000.0, -100.0]")
