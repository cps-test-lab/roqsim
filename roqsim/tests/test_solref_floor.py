# Copyright (C) 2026 Frederik Pasch
#
# SPDX-License-Identifier: Apache-2.0

"""The ``solref`` floor, measured per integrator, and ``sim.contact_override`` held to it.

The measurement reads the stiffness MuJoCo's solver actually uses for a contact row
(``data.efc_KBIP[row, 0]``) and turns it back into the time constant that stiffness belongs to,
``1 / (dmax * dampratio * sqrt(K))``. A stated time constant that runs as stated reads back
unchanged; one MuJoCo raised reads back as the floor. The probe sphere sits deeper than the
``solimp`` width, so its impedance is ``solimp[1]`` -- the depth at which the ``discrete`` floor is
highest, and the one :func:`roqsim.solref.solref_floor` states.
"""

from __future__ import annotations

import math
import pathlib

import mujoco
import pytest

from roqsim.config import load_config
from roqsim.engine import Engine
from roqsim.plugin import PluginError
from roqsim.solref import floor_text, solref_floor

INTEGRATORS = ("euler", "rk4", "implicit", "implicitfast", "discrete")
#: ``sim.integrator`` name -> the keyword MuJoCo's XML uses.
_XML = {"euler": "Euler", "rk4": "RK4"}
_TIMESTEP = 0.002
_DEFAULT_SOLIMP = (0.9, 0.95, 0.001, 0.5, 2.0)


def _probe(integrator, timeconst, *, dampratio=1.0, solimp=_DEFAULT_SOLIMP, refsafe=True):
    """A sphere 2 mm into a plane, both at (*timeconst*, *dampratio*) and *solimp*, forwarded once."""
    flag = "" if refsafe else '<flag refsafe="disable"/>'
    imp = " ".join(repr(v) for v in solimp)
    model = mujoco.MjModel.from_xml_string(
        f"""
        <mujoco>
          <option integrator="{_XML.get(integrator, integrator)}" timestep="{_TIMESTEP}">{flag}</option>
          <worldbody>
            <geom type="plane" size="1 1 .1" solref="{timeconst!r} {dampratio!r}" solimp="{imp}"/>
            <body pos="0 0 .008">
              <freejoint/>
              <geom type="sphere" size=".01" mass=".1" solref="{timeconst!r} {dampratio!r}"
                    solimp="{imp}"/>
            </body>
          </worldbody>
        </mujoco>
        """
    )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    assert data.ncon == 1
    return model, data


def _time_constant_used(integrator, timeconst, **kw) -> float:
    """The time constant the solver's stiffness for the contact's normal row belongs to."""
    _, data = _probe(integrator, timeconst, **kw)
    stiffness = float(data.efc_KBIP[0][0])
    dmax = float(data.contact.solimp[0][1])
    return 1.0 / (dmax * float(data.contact.solref[0][1]) * math.sqrt(stiffness))


def _floor(integrator, **kw) -> float:
    model, data = _probe(integrator, 0.01, **kw)
    return solref_floor(model.opt, data.contact.solref[0], data.contact.solimp[0])


@pytest.mark.parametrize("integrator", INTEGRATORS)
def test_the_floor_is_the_one_mujoco_applies(integrator):
    """Below the floor every time constant runs as the floor; from it upward, as stated."""
    floor = _floor(integrator)
    if integrator == "discrete":
        assert floor == pytest.approx(_TIMESTEP / math.sqrt(0.95), rel=1e-12)
    else:
        assert floor == 2 * _TIMESTEP
    for below in (0.01 * floor, 0.5 * floor, floor * (1 - 1e-6)):
        assert _time_constant_used(integrator, below) == pytest.approx(floor, rel=1e-9)
    for stated in (floor, floor * (1 + 1e-6), 1.5 * floor):
        assert _time_constant_used(integrator, stated) == pytest.approx(stated, rel=1e-9)


@pytest.mark.parametrize("integrator", INTEGRATORS)
@pytest.mark.parametrize(
    ("dampratio", "solimp"),
    [(0.5, _DEFAULT_SOLIMP), (2.0, _DEFAULT_SOLIMP), (1.0, (0.5, 0.99, 0.001, 0.5, 2.0))],
)
def test_under_discrete_the_floor_follows_the_damping_ratio_and_solimp(
    integrator, dampratio, solimp
):
    """Only ``discrete`` caps the stiffness rather than the time constant, so only its floor moves."""
    floor = _floor(integrator, dampratio=dampratio, solimp=solimp)
    if integrator == "discrete":
        assert floor == pytest.approx(_TIMESTEP * math.sqrt(solimp[1]) / (solimp[1] * dampratio))
    else:
        assert floor == 2 * _TIMESTEP
    used = _time_constant_used(integrator, 0.01 * floor, dampratio=dampratio, solimp=solimp)
    assert used == pytest.approx(floor, rel=1e-9)
    stated = floor * (1 + 1e-6)
    used = _time_constant_used(integrator, stated, dampratio=dampratio, solimp=solimp)
    assert used == pytest.approx(stated, rel=1e-9)


@pytest.mark.parametrize("integrator", INTEGRATORS)
def test_with_refsafe_disabled_there_is_no_floor(integrator):
    model, _ = _probe(integrator, 0.01, refsafe=False)
    assert solref_floor(model.opt, (0.01, 1.0), _DEFAULT_SOLIMP) is None
    assert _time_constant_used(integrator, 1e-5, refsafe=False) == pytest.approx(1e-5, rel=1e-9)


def test_the_floor_reads_a_spec_before_compile_as_it_reads_a_model():
    spec = mujoco.MjSpec.from_string('<mujoco><option integrator="implicitfast"/></mujoco>')
    spec.option.timestep = _TIMESTEP
    assert solref_floor(spec.option, (0.02, 1.0), _DEFAULT_SOLIMP) == 2 * _TIMESTEP
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_DISCRETE
    model = spec.compile()
    assert solref_floor(spec.option, (0.02, 1.0), _DEFAULT_SOLIMP) == solref_floor(
        model.opt, (0.02, 1.0), _DEFAULT_SOLIMP
    )


def test_the_floor_as_text_clears_the_floor():
    for floor in (_TIMESTEP / math.sqrt(0.95), 0.0010259783520851542, 2 * _TIMESTEP, 1 / 3):
        assert float(floor_text(floor)) >= floor


# -- sim.contact_override is held to the floor of the integrator it compiles with ----------------
def _build(tmp_path: pathlib.Path, integrator: str, override: str, world: str = "") -> None:
    path = tmp_path / "w.yaml"
    path.write_text(
        f"sim: {{timestep: {_TIMESTEP}, integrator: {integrator}, seed: 1, {world}"
        f"contact_override: {override}}}\ncomponents:\n- dummy: {{}}\n",
        encoding="utf-8",
    )
    Engine(load_config(path)).setup()


@pytest.mark.parametrize("integrator", INTEGRATORS)
def test_the_override_accepts_exactly_down_to_the_floor_and_refuses_below(tmp_path, integrator):
    """The boundary is the one MuJoCo applies under this integrator, measured, not a rule of thumb."""
    measured = _time_constant_used(integrator, 1e-6)
    assert _floor(integrator) == pytest.approx(measured, rel=1e-12)
    _build(tmp_path, integrator, f"{{solref: [{measured * (1 + 1e-9)!r}, 1.0]}}")
    with pytest.raises(
        PluginError, match=f"below MuJoCo's floor of {floor_text(measured)} s"
    ) as caught:
        _build(tmp_path, integrator, f"{{solref: [{measured * (1 - 1e-9)!r}, 1.0]}}")
    assert f"under {integrator}" in str(caught.value)
    # The value the message tells the author to state is one the override then accepts.
    _build(tmp_path, integrator, f"{{solref: [{floor_text(measured)}, 1.0]}}")


def test_under_discrete_a_sub_step_time_constant_is_accepted_where_it_is_honoured(tmp_path):
    """At a damping ratio of 2 the discrete floor is about half a step; under implicitfast it is two."""
    stated = 0.6 * _TIMESTEP
    assert _time_constant_used("discrete", stated, dampratio=2.0) == pytest.approx(stated, rel=1e-9)
    _build(tmp_path, "discrete", f"{{solref: [{stated!r}, 2.0]}}")
    with pytest.raises(PluginError, match=r"2 \* timestep under implicitfast"):
        _build(tmp_path, "implicitfast", f"{{solref: [{stated!r}, 2.0]}}")


def test_under_discrete_the_override_judges_the_solimp_it_puts_in_force(tmp_path):
    """A higher ``solimp[1]`` lowers the discrete floor (1.026 steps at 0.95, 1.005 at 0.99)."""
    stated = 1.02 * _TIMESTEP
    with pytest.raises(PluginError, match="under discrete"):
        _build(tmp_path, "discrete", f"{{solref: [{stated!r}, 1.0]}}")
    _build(tmp_path, "discrete", f"{{solref: [{stated!r}, 1.0], solimp: [0.9, 0.99]}}")


def test_an_override_without_solref_is_judged_on_the_o_solref_it_puts_in_force(tmp_path):
    """Enabling the override makes the model's ``o_solref`` every contact's, stated or not."""
    path = tmp_path / "w.yaml"
    path.write_text(
        "sim: {timestep: 0.02, integrator: implicitfast, seed: 1, "
        "contact_override: {friction: [1.0]}}\ncomponents:\n- dummy: {}\n",
        encoding="utf-8",
    )
    with pytest.raises(PluginError, match="the model's o_solref that the override puts in force"):
        Engine(load_config(path)).setup()


def test_with_refsafe_disabled_the_override_has_no_floor(tmp_path):
    world = tmp_path / "base.xml"
    world.write_text(
        '<mujoco><option><flag refsafe="disable"/></option><worldbody/></mujoco>', encoding="utf-8"
    )
    _build(tmp_path, "implicitfast", "{solref: [1.0e-5, 1.0]}", world=f"world: {world}, ")
