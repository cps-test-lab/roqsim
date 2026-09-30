# Copyright (C) 2026 Frederik Pasch
# SPDX-License-Identifier: Apache-2.0

"""A drive that holds a mechanism up pushes on what it is mounted on, so the weight reaches the floor.

MuJoCo's ``body_gravcomp`` is an external force at each body's centre of mass, projected onto every
degree of freedom above it -- a mobile base's free joint included, where it lifts the whole robot.
:func:`roqsim.actuators.apply_gravity_compensation` routes the holding joints' rows through their
drives, and :class:`roqsim.actuators.GravityReaction` takes the rest off the base every step.

The rover here is the smallest machine with each kind of held joint a real one has: a lift carriage
on a vertical slide (a fork carriage), a link on a horizontal hinge (an arm), and a link on a
vertical hinge (a steered wheel's fork, which holds no weight at all).
"""

from __future__ import annotations

import copy

import mujoco
import numpy as np
import pytest

from roqsim.actuators import apply_gravity_compensation
from roqsim.actuators import resolve as resolve_actuators
from roqsim.config import load_config_from_dict
from roqsim.context import Entity
from roqsim.engine import Engine
from roqsim.plugin import Plugin, PluginError
from roqsim.presence import set_present

BASE_KG, CARRIAGE_KG, LINK_KG, STEER_KG = 100.0, 50.0, 20.0, 30.0
FEET = ("foot_fl", "foot_fr", "foot_rl", "foot_rr")

_MECHANISM = {
    "lift": """
      <body name="carriage" pos=".6 0 .3">
        <joint name="lift" type="slide" axis="0 0 1" range="-.2 .5"/>
        <geom type="box" size=".05 .2 .05" mass="{carriage}" contype="0" conaffinity="0"/>
      </body>""",
    "shoulder": """
      <body name="link" pos="-.3 0 .2">
        <joint name="shoulder" axis="0 1 0"/>
        <geom type="capsule" fromto="0 0 0 .4 0 0" size=".03" mass="{link}" contype="0"
              conaffinity="0"/>
      </body>""",
    "steer": """
      <body name="steer" pos="0 .2 .15">
        <joint name="steer" axis="0 0 1"/>
        <geom type="box" size=".1 .03 .03" pos=".1 0 0" mass="{steer}" contype="0"
              conaffinity="0"/>
      </body>""",
}
_DRIVE = {
    "lift": '<position name="lift" joint="lift" kp="20000" dampratio="1" forcerange="-{lift_n} {lift_n}"/>',
    "shoulder": '<position name="shoulder" joint="shoulder" kp="2000" dampratio="1" forcerange="-200 200"/>',
    "steer": '<position name="steer" joint="steer" kp="2000" dampratio="1" forcerange="-200 200"/>',
}


def _rover_xml(parts, lift_n: float, free: bool) -> str:
    feet = "".join(
        f'<geom name="{name}" type="sphere" size=".05" pos="{x} {y} -.15" mass="0"/>'
        for name, (x, y) in zip(
            FEET, [(0.45, 0.25), (0.45, -0.25), (-0.45, 0.25), (-0.45, -0.25)], strict=True
        )
    )
    body = "".join(_MECHANISM[p] for p in parts).format(
        carriage=CARRIAGE_KG, link=LINK_KG, steer=STEER_KG
    )
    drives = "".join(_DRIVE[p] for p in parts).format(lift_n=lift_n)
    joint = '<freejoint name="base_free"/>' if free else ""
    return f"""
<mujoco>
  <worldbody>
    <body name="base" pos="0 0 .2">
      {joint}
      <geom type="box" size=".5 .3 .1" mass="{BASE_KG}" contype="0" conaffinity="0"/>
      {feet}
      {body}
    </body>
  </worldbody>
  <actuator>{drives}</actuator>
</mujoco>"""


class Rover(Plugin):
    """A rover built the way a spawn plugin builds a robot: resolve, compensate, attach."""

    def build(self, spec, ctx):
        cfg = self.config
        child = mujoco.MjSpec.from_string(
            _rover_xml(
                cfg.get("parts", list(_MECHANISM)), cfg.get("lift_n", 2000), cfg.get("free", True)
            )
        )
        if cfg.get("compensate", True):
            apply_gravity_compensation(child, resolve_actuators(child, None, model_name="rover"))
        spec.attach(child, prefix="rover_", frame=spec.worldbody.add_frame())


def _engine(**rover) -> Engine:
    sim = {"integrator": rover.pop("integrator")} if "integrator" in rover else {}
    cfg = load_config_from_dict(
        {"sim": sim, "components": [{f"{__name__}:Rover": rover, "name": "rover"}]}
    )
    engine = Engine(cfg)
    engine.setup()
    engine.reset()
    return engine


def _hold(engine: Engine, seconds: float = 2.0) -> None:
    """Command every drive to the pose it stands in, and let the rover settle on its feet."""
    m, d = engine.ctx.model, engine.ctx.data
    for i in range(m.nu):
        d.ctrl[i] = d.qpos[m.jnt_qposadr[m.actuator_trnid[i, 0]]]
    for _ in range(int(seconds / m.opt.timestep)):
        engine.step()


def _foot_loads(engine: Engine) -> dict[str, float]:
    """The vertical force the floor puts into each foot."""
    m, d = engine.ctx.model, engine.ctx.data
    loads = dict.fromkeys(FEET, 0.0)
    force = np.zeros(6)
    for i in range(d.ncon):
        contact = d.contact[i]
        for gid in (contact.geom1, contact.geom2):
            name = m.geom(gid).name.removeprefix("rover_")
            if name in loads:
                mujoco.mj_contactForce(m, d, i, force)
                loads[name] += abs(float(force[0] * contact.frame[2]))
    return loads


def _weight(engine: Engine) -> float:
    m = engine.ctx.model
    return float(m.body_subtreemass[m.body("rover_base").id]) * float(-m.opt.gravity[2])


def _free_rows(engine: Engine) -> np.ndarray:
    m, d = engine.ctx.model, engine.ctx.data
    adr = int(m.jnt_dofadr[m.joint("rover_base_free").id])
    return np.array(d.qfrc_gravcomp[adr : adr + 6])


def test_the_floor_carries_the_whole_robot():
    """The measured defect: compensated as an external force, the floor carried the base alone."""
    engine = _engine()
    _hold(engine)
    carried, weight = sum(_foot_loads(engine).values()), _weight(engine)
    rows = _free_rows(engine)
    engine.shutdown()

    assert carried == pytest.approx(weight, rel=2e-3), (
        f"the floor carries {carried:.1f} of {weight:.1f} N"
    )
    assert np.abs(rows).max() < 1e-6 * weight, f"gravity compensation still lifts the base: {rows}"


def test_each_foot_carries_what_it_would_with_the_mechanism_rigid():
    """The weight lands where it hangs: the carriage out front loads the front feet.

    Summing to the right total is not enough -- a reaction applied at the base's own centre of mass
    would pass that and still get tipping wrong. The reference is the same rover with nothing
    compensated, held by the same drives: then every newton goes through the joints by construction.
    """
    compensated = _engine()
    _hold(compensated)
    loads = _foot_loads(compensated)
    compensated.shutdown()
    rigid = _engine(compensate=False)
    _hold(rigid)
    reference = _foot_loads(rigid)
    rigid.shutdown()

    weight = (BASE_KG + CARRIAGE_KG + LINK_KG + STEER_KG) * 9.81
    for foot in FEET:
        assert loads[foot] == pytest.approx(reference[foot], abs=5e-3 * weight), foot
    assert loads["foot_fl"] > loads["foot_rl"], "the carriage is out front"


def test_a_vertical_axis_joint_adds_no_upward_force():
    """A steered wheel's fork holds no weight, so compensating what hangs behind it may lift nothing."""
    engine = _engine(parts=["steer"])
    _hold(engine)
    carried, weight = sum(_foot_loads(engine).values()), _weight(engine)
    rows = _free_rows(engine)
    m, d = engine.ctx.model, engine.ctx.data
    steer = float(d.qfrc_gravcomp[m.jnt_dofadr[m.joint("rover_steer").id]])
    engine.shutdown()

    assert carried == pytest.approx(weight, rel=2e-3)
    assert abs(rows[2]) < 1e-6 * weight
    # Up to the base's tilt on its feet: level, gravity has no moment about a vertical axis, where
    # the same link on a horizontal one would need STEER_KG * 9.81 * 0.1 = 29 N*m.
    assert abs(steer) < 0.05, f"the steer drive holds {steer:.3f} N*m of a link it only turns"


@pytest.mark.parametrize(("lift_n", "holds"), [(2000.0, True), (200.0, False)])
def test_a_drive_too_weak_for_its_load_sags(lift_n, holds):
    """The term is the drive's own, so it counts against the drive's force limit.

    The carriage weighs 490 N. A 200 N drive cannot hold it, and as an external force the
    compensation used to hold it anyway -- the sizing of the motor did not matter.
    """
    engine = _engine(parts=["lift"], lift_n=lift_n)
    _hold(engine)
    m, d = engine.ctx.model, engine.ctx.data
    jid = m.joint("rover_lift").id
    q = float(d.qpos[m.jnt_qposadr[jid]])
    effort = float(d.qfrc_actuator[m.jnt_dofadr[jid]])
    engine.shutdown()

    if holds:
        assert abs(q) < 1e-3
        assert effort == pytest.approx(CARRIAGE_KG * 9.81, rel=1e-2)
    else:
        assert q < -0.19, (
            f"a {lift_n:g} N drive held a {CARRIAGE_KG * 9.81:.0f} N carriage at {q:.3f} m"
        )
        assert abs(effort) <= lift_n + 1e-9


def test_a_fixed_base_is_stepped_exactly_as_before():
    """Nothing lies above a mechanism bolted to the world, so there is nothing to take off.

    The engine then steps with one ``mj_step``, and moving the term from ``qfrc_passive`` to
    ``qfrc_actuator`` changes the arm's motion by rounding only.
    """
    engine = _engine(free=False)
    assert engine._gravity_reaction is None
    m, d = engine.ctx.model, engine.ctx.data
    d.ctrl[:] = [0.1, 0.5, 1.0]
    for _ in range(500):
        engine.step()
    driven = np.array(d.qpos)

    # The same compiled model with the term routed the way MuJoCo does by default: passively.
    passive = copy.copy(m)
    passive.jnt_actgravcomp[:] = 0
    passive.jnt_actfrclimited[:] = 0
    reference = mujoco.MjData(passive)
    mujoco.mj_resetData(passive, reference)
    reference.ctrl[:] = d.ctrl
    for _ in range(500):
        mujoco.mj_step(passive, reference)
    engine.shutdown()

    np.testing.assert_allclose(driven, reference.qpos, atol=1e-9)


def test_an_absent_rover_stays_where_it_was():
    """Presence freezes an entity by compensating all of it, the base included.

    Then nothing stands on anything, and the reaction must not pull the arm's weight back onto a
    base the floor no longer touches.
    """
    engine = _engine()
    _hold(engine, 0.5)
    ctx = engine.ctx
    z0 = float(ctx.data.qpos[2])
    set_present(ctx, Entity(name="rover", kind="robot", body="rover_base"), False)
    for _ in range(1000):
        engine.step()
    z = float(ctx.data.qpos[2])
    engine.shutdown()

    assert z == pytest.approx(z0, abs=1e-3)


def test_rk4_is_refused_rather_than_run_as_euler():
    """``mj_step2`` integrates RK4 as Euler, so the two-half step cannot honour it."""
    with pytest.raises(PluginError, match="rk4"):
        _engine(integrator="rk4")
