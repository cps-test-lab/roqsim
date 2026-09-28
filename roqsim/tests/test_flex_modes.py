"""What ``roqsim check`` says a flex will do, each claim measured on a stepped model.

:func:`roqsim.flex_modes.first_modes` predicts frequencies from a finite-difference stiffness; the
block here is then excited in each predicted mode shape and left to ring under the ``discrete``
integrator, and the frequency and decay it actually shows are read off its free vibration. That is
also where :mod:`roqsim.flex_modes`' numerical damping under ``discrete`` is pinned -- the
integrator's own damping is one timestep's worth of Rayleigh damping -- and its resolution limit,
the ``omega * timestep`` past which neither that damping ratio nor the frequency is the one a run
shows. The contact ``solref`` floor is measured on a resting contact under two integrators. The last part is ``roqsim check`` and ``roqsim scenes describe`` on a
world with a flex.
"""

from __future__ import annotations

import json
import math
import textwrap

import mujoco
import numpy as np
import pytest
from test_flex_rules import CASES, _mjcf

from roqsim import flex_modes as flexlib
from roqsim.check import _render_text, check_world, main
from roqsim.flex_modes import (
    MAX_OMEGA_DT,
    FlexTooLarge,
    describe_flexes,
    explain_flex,
    first_modes,
    is_elastic,
    solref_floor,
)

#: The bottom face of a 3x3x5 grid: MuJoCo numbers a grid's vertices z fastest.
_BOTTOM = " ".join(str(i) for i in range(0, 45, 5))


def _block(
    *,
    timestep: float = 0.0005,
    damping: float = 0.0,
    young: float = 1e5,
    dof: str = "full",
    pin: str = f'<pin id="{_BOTTOM}"/>',
    solref: str = "0.02 1",
) -> str:
    """A 4 x 4 x 8 cm block standing on its pinned bottom face -- a stubby cantilever -- in no gravity."""
    return textwrap.dedent(
        f"""
        <mujoco>
          <option integrator="discrete" timestep="{timestep}" gravity="0 0 0"/>
          <worldbody>
            <body name="holder" pos="0 0 0.3">
              <flexcomp name="blk" type="grid" count="3 3 5" spacing=".02 .02 .02" dim="3"
                        mass=".1" radius=".002" dof="{dof}">
                <elasticity young="{young}" poisson="0.2" damping="{damping}"/>
                {pin}
                <contact contype="0" conaffinity="0" selfcollide="none" solref="{solref}"/>
              </flexcomp>
            </body>
          </worldbody>
        </mujoco>
        """
    )


def _ring(model: mujoco.MjModel, mode: int, seconds: float = 2.0) -> tuple[float, float]:
    """``(hz, zeta)`` of the free vibration the block shows when released from mode *mode*'s shape.

    The displacement is projected on that shape (the modal coordinate) and read until it has decayed
    to 1 % of its start -- past that, zero crossings are rounding noise. Frequency from the zero
    crossings, damping ratio from a log-linear fit to the peaks.
    """
    modes = first_modes(model, 0, mode + 1, shapes=True)
    shape = modes.shapes[:, mode]
    data = mujoco.MjData(model)
    full = np.zeros((model.nv, model.nv))
    mujoco.mj_forward(model, data)
    mujoco.mj_fullM(model, data, full)
    mass = full[np.ix_(modes.dofs, modes.dofs)]
    qpos_of = model.jnt_qposadr[model.dof_jntid[modes.dofs]]
    rest = model.qpos0[qpos_of]
    data.qpos[qpos_of] = rest + 1e-4 * shape / np.abs(shape).max()

    times, coordinate = [], []
    for _ in range(int(seconds / model.opt.timestep)):
        mujoco.mj_step(model, data)
        times.append(data.time)
        coordinate.append(shape @ mass @ (data.qpos[qpos_of] - rest))
    times, coordinate = np.array(times), np.array(coordinate)
    last = np.flatnonzero(np.abs(coordinate) > 0.01 * np.abs(coordinate).max())[-1]
    times, coordinate = times[: last + 1], coordinate[: last + 1]

    sign = np.sign(coordinate)
    at = np.flatnonzero(sign[:-1] * sign[1:] < 0)
    crossings = times[at] - coordinate[at] * (times[at + 1] - times[at]) / (
        coordinate[at + 1] - coordinate[at]
    )
    hz = (len(crossings) - 1) / 2 / (crossings[-1] - crossings[0])
    halves = list(zip(at[:-1], at[1:], strict=True))
    peaks = [np.abs(coordinate[a:b]).max() for a, b in halves]
    peak_times = [times[a + np.argmax(np.abs(coordinate[a:b]))] for a, b in halves]
    decay = -np.polyfit(peak_times, np.log(peaks), 1)[0]
    return hz, decay / (2 * math.pi * hz)


def _poles(model: mujoco.MjModel, mode: int = 0, seconds: float = 2.0) -> tuple[float, float]:
    """``(omega, zeta)``, natural frequency and damping ratio, of the ring-down of mode *mode*.

    Read from the poles of the sampled motion rather than from crossings and peaks: a single mode
    under a one-step integrator is a two-state linear recurrence, so its modal coordinate obeys
    ``x[k+2] = a1 x[k+1] + a2 x[k]`` exactly, whose roots ``z`` give ``s = log(z) / timestep``.
    That stays exact at a coarse step and a heavy damping, where a ring-down shows few crossings.
    """
    modes = first_modes(model, 0, mode + 1, shapes=True)
    shape = modes.shapes[:, mode]
    data = mujoco.MjData(model)
    full = np.zeros((model.nv, model.nv))
    mujoco.mj_forward(model, data)
    mujoco.mj_fullM(model, data, full)
    mass = full[np.ix_(modes.dofs, modes.dofs)]
    qpos_of = model.jnt_qposadr[model.dof_jntid[modes.dofs]]
    rest = model.qpos0[qpos_of]
    data.qpos[qpos_of] = rest + 1e-4 * shape / np.abs(shape).max()

    coordinate = [shape @ mass @ (data.qpos[qpos_of] - rest)]
    for _ in range(int(seconds / model.opt.timestep)):
        mujoco.mj_step(model, data)
        coordinate.append(shape @ mass @ (data.qpos[qpos_of] - rest))
    x = np.array(coordinate)
    x = x[: np.flatnonzero(np.abs(x) > 0.01 * np.abs(x).max())[-1] + 1]
    (a1, a2), *_ = np.linalg.lstsq(np.stack([x[1:-1], x[:-2]], axis=1), x[2:], rcond=None)
    s = np.log(np.roots([1.0, -a1, -a2])[0].astype(complex)) / model.opt.timestep
    return float(abs(s)), float(-s.real / abs(s))


def _block_at(omega_dt: float, zeta: float | None = None) -> mujoco.MjModel:
    """The block at the timestep that puts its first mode at *omega_dt*, damped at *zeta* (by
    default one timestep's worth of damping, so ``zeta == omega_dt``)."""
    omega = first_modes(mujoco.MjModel.from_xml_string(_block()), 0).omega[0]
    timestep = omega_dt / omega
    damping = timestep if zeta is None else 2 * zeta / omega - timestep
    return mujoco.MjModel.from_xml_string(_block(timestep=timestep, damping=damping))


# -- first_modes ----------------------------------------------------------------------------------
@pytest.mark.parametrize("mode", [0, 1, 2])
def test_each_predicted_mode_rings_at_its_predicted_frequency(mode):
    model = mujoco.MjModel.from_xml_string(_block())
    predicted = first_modes(model, 0).hz[mode]
    measured, _ = _ring(model, mode)
    assert measured == pytest.approx(predicted, rel=0.01)


@pytest.mark.parametrize("timestep", [0.001, 0.0005])
def test_the_discrete_integrator_damps_by_one_timestep(timestep):
    """Numerical damping at damping 0: all of the decay is the integrator's, timestep*omega/2.

    Two timesteps, so the proportionality is measured rather than one value matched.
    """
    model = mujoco.MjModel.from_xml_string(_block(timestep=timestep))
    omega = first_modes(model, 0).omega[0]
    _, zeta = _ring(model, 0)
    assert zeta == pytest.approx(timestep * omega / 2, rel=0.03)
    derived, _ = explain_flex(model, 0)
    assert derived["modes"][0]["zeta"] == pytest.approx(zeta, rel=0.03)
    assert derived["numerical_share"] == 1.0


def test_the_stated_damping_adds_to_the_integrators():
    """Numerical damping, with a damping stated: zeta = (damping + timestep) * omega / 2, as check
    reports it.
    """
    model = mujoco.MjModel.from_xml_string(_block(timestep=0.0005, damping=0.001))
    _, zeta = _ring(model, 0)
    derived, warnings = explain_flex(model, 0)
    assert derived["modes"][0]["zeta"] == pytest.approx(zeta, rel=0.03)
    assert derived["numerical_share"] == pytest.approx(1 / 3)
    # The timestep at or below which the integrator's share is at most half is the damping itself.
    assert derived["max_timestep_for_half"] == pytest.approx(0.001)
    assert warnings == []


# The resolution limit, measured: (omega * timestep, how far the damping ratio and the natural
# frequency fall short of what check reports) for the block's first mode damped by one timestep.
_SHORTFALL = [
    (0.1, 0.005, 0.005),
    (0.2, 0.020, 0.018),
    (0.3, 0.044, 0.038),
    (0.35, 0.058, 0.050),
    (0.5, 0.107, 0.092),
    (0.85, 0.228, 0.199),
]


@pytest.mark.parametrize(("omega_dt", "zeta_short", "omega_short"), _SHORTFALL)
def test_check_warns_where_its_figures_stop_being_the_ones_that_run(
    omega_dt, zeta_short, omega_short
):
    """The resolution limit: past MAX_OMEGA_DT the reported damping ratio or frequency is more than
    5 % off the ring-down -- and exactly there the mode is marked and flex-timestep fires."""
    model = _block_at(omega_dt)
    # One reported mode, so that the warning is about the mode measured here and no other.
    derived, warnings = explain_flex(model, 0, n=1)
    first = derived["modes"][0]
    omega, zeta = _poles(model)
    assert first["omega_dt"] == pytest.approx(omega_dt)
    assert 1 - zeta / first["zeta"] == pytest.approx(zeta_short, abs=0.003)
    assert 1 - omega / (2 * math.pi * first["hz"]) == pytest.approx(omega_short, abs=0.003)

    within = max(1 - zeta / first["zeta"], 1 - omega / (2 * math.pi * first["hz"])) <= 0.05
    assert within is (omega_dt <= MAX_OMEGA_DT)
    assert first["resolved"] is within
    timestep_warnings = [w for w in warnings if w["check"] == "flex-timestep"]
    assert len(timestep_warnings) == (0 if within else 1)
    if not within:
        (warning,) = timestep_warnings
        assert set(warning) == {"check", "message", "hint", "flex"}
        assert warning["message"].startswith("flex 'blk': mode 1 (11.4 Hz) is under-resolved")
        assert f"omega * timestep = {omega_dt:.2g}, above 0.3" in warning["message"]
        assert "not the ones that run" in warning["message"]
        assert f"sim.timestep <= {derived['max_timestep_resolved']:g} s" in warning["hint"]


def test_the_suggested_timestep_resolves_every_reported_mode():
    derived, _ = explain_flex(_block_at(0.85), 0)
    fine = mujoco.MjModel.from_xml_string(_block(timestep=derived["max_timestep_resolved"]))
    derived, warnings = explain_flex(fine, 0)
    assert all(mode["resolved"] for mode in derived["modes"])
    assert max(mode["omega_dt"] for mode in derived["modes"]) <= MAX_OMEGA_DT
    assert [w for w in warnings if w["check"] == "flex-timestep"] == []


@pytest.mark.parametrize(("zeta", "short"), [(0.15, 0.024), (0.5, 0.068), (0.9, 0.109)])
def test_at_the_limit_the_shortfall_grows_with_the_damping(zeta, short):
    """At MAX_OMEGA_DT the 5 % holds up to zeta 0.3 only; a more heavily damped mode is further off."""
    model = _block_at(MAX_OMEGA_DT, zeta)
    derived, _ = explain_flex(model, 0)
    assert derived["modes"][0]["zeta"] == pytest.approx(zeta)
    assert derived["modes"][0]["resolved"] is True
    _, measured = _poles(model)
    assert 1 - measured / zeta == pytest.approx(short, abs=0.003)


def test_a_free_flex_drops_its_six_rigid_modes():
    free = first_modes(mujoco.MjModel.from_xml_string(_block(pin="")), 0)
    assert free.rigid == 6
    assert len(free.omega) == 3 and free.omega[0] > 1.0


@pytest.mark.parametrize(("dof", "nodes"), [("trilinear", 8), ("quadratic", 27)])
def test_an_interpolated_flex_has_modes_over_its_nodes(dof, nodes):
    model = mujoco.MjModel.from_xml_string(_block(dof=dof, pin=""))
    modes = first_modes(model, 0)
    assert modes.ndof == 3 * nodes
    assert modes.rigid == 6 and len(modes.omega) == 3
    measured, _ = _ring(model, 0)
    assert measured == pytest.approx(modes.hz[0], rel=0.01)


def test_a_flex_above_the_dof_cap_is_refused_with_a_way_out(monkeypatch):
    model = mujoco.MjModel.from_xml_string(_block())
    monkeypatch.setattr(flexlib, "MODES_DOF_CAP", 50)
    with pytest.raises(FlexTooLarge, match=r"flex 'blk' has 108 degrees of freedom") as caught:
        first_modes(model, 0)
    assert 'dof="quadratic"' in caught.value.hint
    derived, _ = explain_flex(model, 0)
    assert derived["modes"] is None
    assert "108 degrees of freedom" in derived["modes_skipped"]
    assert "coarser grid" in derived["hint"]


# -- the contact floor ----------------------------------------------------------------------------
def _penetration(timeconst: float, integrator: str, timestep: float = 0.001) -> float:
    """How deep a resting sphere sits in a plane, both at solref *timeconst*."""
    model = mujoco.MjModel.from_xml_string(
        f"""
        <mujoco>
          <option integrator="{integrator}" timestep="{timestep}"/>
          <worldbody>
            <geom type="plane" size="1 1 .1" solref="{timeconst} 1"/>
            <body name="ball" pos="0 0 .03">
              <freejoint/>
              <geom type="sphere" size=".01" mass=".1" solref="{timeconst} 1"/>
            </body>
          </worldbody>
        </mujoco>
        """
    )
    data = mujoco.MjData(model)
    for _ in range(int(1.0 / timestep)):
        mujoco.mj_step(model, data)
    return float(-data.efc_pos[: data.ncon].min())


@pytest.mark.parametrize("integrator", ["discrete", "implicitfast"])
def test_the_solref_floor_is_the_one_mujoco_applies(integrator):
    """The solref floor: below it every time constant rests alike; above it the contact softens."""
    model = mujoco.MjModel.from_xml_string(
        f'<mujoco><option integrator="{integrator}" timestep="0.001"/></mujoco>'
    )
    floor = solref_floor(model)
    assert floor == (0.001 if integrator == "discrete" else 0.002)
    at_floor = _penetration(floor, integrator)
    assert _penetration(0.5 * floor, integrator) == pytest.approx(at_floor, rel=0.01)
    assert _penetration(1.5 * floor, integrator) > 1.5 * at_floor


def test_a_solref_below_the_floor_is_a_warning():
    model = mujoco.MjModel.from_xml_string(_block(timestep=0.001, solref="0.0005 1"))
    derived, warnings = explain_flex(model, 0)
    assert derived["below_floor"] is True and derived["solref_floor"] == 0.001
    floor_warnings = [w for w in warnings if w["check"] == "flex-solref"]
    assert len(floor_warnings) == 1
    assert set(floor_warnings[0]) == {"check", "message", "hint", "flex"}
    assert floor_warnings[0]["message"].startswith("flex 'blk': contact solref")
    assert "one timestep under discrete" in floor_warnings[0]["message"]


# -- the inventory --------------------------------------------------------------------------------
@pytest.mark.parametrize("case", sorted(CASES))
def test_elastic_is_whether_the_flex_pushes_back(case):
    """``is_elastic`` reads compiled fields; the behaviour it stands for is a passive restoring force."""
    model = mujoco.MjModel.from_xml_string(_mjcf(CASES[case][0], option='integrator="discrete"'))
    data = mujoco.MjData(model)
    data.qpos[:] = model.qpos0 + 1e-4 * np.random.default_rng(0).standard_normal(model.nq)
    mujoco.mj_forward(model, data)
    pushes_back = model.nv > 0 and float(np.abs(data.qfrc_passive).max()) > 0
    assert is_elastic(model, 0) is pushes_back


def test_each_of_several_flexes_is_read_from_its_own_rows():
    """Per-flex arrays are addressed per flex, and a flex with no rows has address -1."""
    xml = textwrap.dedent(
        f"""
        <mujoco>
          <option integrator="discrete"/>
          <worldbody>
            <body name="a">{CASES["plain_solid"][0].replace('"blk"', '"plain"')}</body>
            <body name="b">{CASES["shell_bending"][0].replace('"blk"', '"sheet"')}</body>
            <body name="c">{CASES["elastic_solid"][0]}</body>
          </worldbody>
        </mujoco>
        """
    )
    model = mujoco.MjModel.from_xml_string(xml)
    rows = describe_flexes(model)
    assert [(r["name"], r["parent"], r["elastic"]) for r in rows] == [
        ("plain", "a", False),
        ("sheet", "b", True),
        ("blk", "c", True),
    ]


def test_a_flex_is_described_by_what_it_compiled_into():
    model = mujoco.MjModel.from_xml_string(_block())
    (row,) = describe_flexes(model, {"thing": "holder"})
    assert row == {
        "name": "blk",
        "dim": 3,
        "vertices": 45,
        "elements": row["elements"],
        "dof": "full",
        "nodes": 0,
        "pinned": 9,
        "parent": "holder",
        "entity": "thing",
        "rigid": False,
        "elastic": True,
        "passive_contact": False,
        "ndof": 108,
    }
    assert row["elements"] > 0
    assert describe_flexes(model)[0]["entity"] is None


def test_an_interpolated_flex_pins_its_nodes():
    model = mujoco.MjModel.from_xml_string(_block(dof="trilinear", pin='<pin id="0 1"/>'))
    (row,) = describe_flexes(model)
    assert (row["dof"], row["nodes"], row["pinned"], row["ndof"]) == ("trilinear", 8, 2, 18)


# -- roqsim check and scenes describe ---------------------------------------------------------------
def _world(tmp_path, xml: str, sim: str = "") -> str:
    (tmp_path / "block.xml").write_text(xml)
    world = tmp_path / "world.yaml"
    world.write_text(f"sim: {{world: block.xml{sim}}}\ncomponents: []\n")
    return str(world)


def test_check_explains_a_flex_and_warns_without_failing(tmp_path):
    report = check_world(_world(tmp_path, _block(timestep=0.001, solref="0.0005 1")))
    assert report["ok"] is True and report["problems"] == []
    assert report["world"]["model"]["nflex"] == 1
    (flex,) = report["world"]["flexes"]
    assert (flex["name"], flex["pinned"], flex["parent"]) == ("blk", 9, "holder")
    (derived,) = report["derived"]["flexes"]
    assert len(derived["modes"]) == 3
    assert [(w["check"], w["flex"]) for w in report["warnings"]] == [
        ("flex-damping", "blk"),
        ("flex-solref", "blk"),
    ]

    text = _render_text(report)
    assert "WARN  [flex-damping] flex 'blk': numerical damping is 100%" in text
    assert "WARN  [flex-solref] flex 'blk': contact solref" in text
    assert "blk  dim 3, 45 vertices" in text
    assert "modes " in text and "Hz" in text


def test_check_marks_an_under_resolved_mode(tmp_path):
    report = check_world(_world(tmp_path, _block(damping=0.004), ", timestep: 0.004"))
    assert report["ok"] is True
    (derived,) = report["derived"]["flexes"]
    assert [m["resolved"] for m in derived["modes"]] == [True, False, False]
    assert [(w["check"], w["flex"]) for w in report["warnings"]] == [("flex-timestep", "blk")]

    text = _render_text(report)
    assert "WARN  [flex-timestep] flex 'blk': mode 2 (13 Hz) is under-resolved" in text
    assert "modes 11.4, 13*, 22.4* Hz; damping ratio 0.288, 0.326*, 0.562*" in text
    assert "* under-resolved (omega * timestep above 0.3): a run damps" in text


def test_a_stated_timestep_changes_the_verdict(tmp_path):
    """Damping stated, timestep at or below it: the integrator's share is at most half, no warning."""
    xml = _block(damping=0.001)
    report = check_world(_world(tmp_path, xml, ", timestep: 0.0005"))
    assert report["ok"] and report["warnings"] == []
    assert report["derived"]["flexes"][0]["numerical_share"] == pytest.approx(1 / 3)


def test_a_world_without_a_flex_has_nothing_to_warn_about(tmp_path, capsys):
    world = tmp_path / "empty.yaml"
    world.write_text("sim: {}\ncomponents: []\n")
    assert main([str(world), "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["warnings"] == [] and report["derived"] == {"flexes": []}
    assert report["world"]["model"]["nflex"] == 0 and report["world"]["flexes"] == []
