"""A site force/torque sensor does not see a flex's contacts, so one that could is refused.

The first half measures MuJoCo 3.14 itself (:mod:`roqsim.flex`, rule 6), on the smallest scenes that
show it: a rigid probe pressed into a block, and a soft cantilever hanging off a sensed mount. A
contact with a flex never reaches the sensor; the flex's weight, its elastic reaction and a force
applied to one of its vertices do. A MuJoCo that starts transmitting the contact fails here, and the
refusal in the second half can go.
"""

from __future__ import annotations

import mujoco
import numpy as np
import pytest
from roqsim_sensors.plugins.force_torque import ForceTorquePlugin

from roqsim.config import load_config_from_dict
from roqsim.context import SimContext
from roqsim.engine import Engine
from roqsim.plugin import Plugin

G = 9.81
_OPTION = '<option integrator="discrete" timestep="0.0005" solver="Newton"/>'
_MATERIAL = (
    '<edge equality="false"/><elasticity young="5e4" poisson="0.3" damping="0.002"/>'
    '<contact condim="3" solref="0.005 1" selfcollide="none"/>'
)

_FLEX_BLOCK = f"""<body name="block" pos="0 0 0.021">
  <flexcomp name="soft" type="grid" count="4 4 3" spacing=".02 .02 .02" dim="3" radius=".001"
            mass="0.2">{_MATERIAL}</flexcomp></body>"""
_RIGID_BLOCK = """<body name="block" pos="0 0 0.021"><freejoint/>
  <geom type="box" size=".031 .031 .021" mass="0.2" solref="0.005 1"/></body>"""


def _probe_world(block: str) -> str:
    """A 0.3 kg plate on a damped, position-servoed slide, commanded 9 mm into a block's top."""
    return f"""<mujoco>{_OPTION}<worldbody>
      <geom type="plane" size="1 1 .1"/>
      {block}
      <body name="probe" pos="0 0 0.10">
        <joint name="z" type="slide" axis="0 0 1" damping="5"/>
        <geom type="box" size=".05 .05 .01" mass="0.3"/>
        <site name="ft"/>
      </body></worldbody>
      <actuator><position joint="z" kp="300"/></actuator>
      <sensor><force site="ft"/></sensor></mujoco>"""


def _press(block: str) -> tuple[float, float, float]:
    """(sensor Fz, contact normal force on the probe, probe weight), once the press has settled."""
    model = mujoco.MjModel.from_xml_string(_probe_world(block))
    data = mujoco.MjData(model)
    probe = model.body("probe").id
    data.ctrl[0] = -0.057
    for _ in range(6000):
        mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)
    force, on_probe = np.zeros(6), 0.0
    for i in range(data.ncon):
        bodies = [int(model.geom_bodyid[g]) if g >= 0 else -1 for g in data.contact[i].geom]
        if probe in bodies:
            mujoco.mj_contactForce(model, data, i, force)
            on_probe += force[0]
    return float(data.sensordata[2]), on_probe, float(model.body_mass[probe]) * G


def test_a_probe_pressing_a_rigid_block_reads_the_contact():
    fz, contact, weight = _press(_RIGID_BLOCK)
    assert contact > 4.0
    assert fz == pytest.approx(weight - contact, abs=0.01)


def test_a_probe_pressing_a_flex_block_reads_its_weight_alone():
    """The same press into a flex: a contact of several newtons, and the sensor does not see it."""
    fz, contact, weight = _press(_FLEX_BLOCK)
    assert contact > 4.0
    assert fz == pytest.approx(weight, abs=0.01)


def _cantilever(support: bool) -> str:
    """A pinned soft beam off a sensed 0.2 kg mount; optionally propped up near its tip."""
    prop = (
        '<geom name="support" type="box" size=".01 .05 .01" pos=".10 0 .472"/>' if support else ""
    )
    return f"""<mujoco>{_OPTION}<worldbody>{prop}
      <body name="mount" pos="0 0 0.5">
        <geom type="box" size=".01 .03 .03" mass="0.2" contype="0" conaffinity="0"/>
        <site name="ft"/>
        <flexcomp name="beam" type="grid" count="6 3 3" spacing=".02 .02 .02" pos=".06 0 0" dim="3"
                  radius=".002" mass="0.2"><pin gridrange="0 0 0 0 2 2"/>{_MATERIAL}</flexcomp>
      </body></worldbody>
      <sensor><force site="ft"/></sensor></mujoco>"""


def _hang(support: bool, push: float = 0.0):
    model = mujoco.MjModel.from_xml_string(_cantilever(support))
    data = mujoco.MjData(model)
    tip = [b for b in range(model.nbody) if model.body_parentid[b] == model.body("mount").id][-1]
    for _ in range(8000):
        data.xfrc_applied[tip, 2] = push
        mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)
    return model, data


def test_a_hanging_flex_loads_the_sensor_with_its_weight_and_an_applied_force():
    """Its weight and its elastic reaction reach the sensor, and so does a force on a vertex."""
    model, data = _hang(support=False)
    weight = float(model.body_mass[1:].sum()) * G
    assert data.sensordata[2] == pytest.approx(weight, abs=0.005)
    _, pushed = _hang(support=False, push=1.0)
    assert pushed.sensordata[2] == pytest.approx(weight - 1.0, abs=0.005)


def test_a_support_under_the_flex_does_not_reach_the_sensor():
    model, data = _hang(support=True)
    weight = float(model.body_mass[1:].sum()) * G
    force, carried = np.zeros(6), 0.0
    for i in range(data.ncon):
        mujoco.mj_contactForce(model, data, i, force)
        carried += force[0]
    assert carried > 1.0
    assert data.sensordata[2] == pytest.approx(weight, abs=0.005)  # not weight - carried


def test_in_motion_the_reading_is_the_vertices_momentum_balance():
    """Released straight, the beam swings: every step's reading matches what its vertices do."""
    model = mujoco.MjModel.from_xml_string(_cantilever(support=False))
    data = mujoco.MjData(model)
    mount = model.body("mount").id
    free = [b for b in range(model.nbody) if model.body_parentid[b] == mount]
    zdof = [int(model.body_dofadr[b]) + 2 for b in free]
    dt, worst, swing = model.opt.timestep, 0.0, []
    for _ in range(300):
        v0 = data.qvel[zdof].copy()
        mujoco.mj_step(model, data)
        accel = (data.qvel[zdof] - v0) / dt
        balance = float(model.body_mass[mount]) * G + float(
            np.sum(model.body_mass[free] * (G + accel))
        )
        worst = max(worst, abs(float(data.sensordata[2]) - balance))
        swing.append(float(data.sensordata[2]))
    assert max(swing) - min(swing) > 0.5  # it really is moving
    assert worst < 1e-6


# -- the plugin: refused where a flex contact could fall into the reading -------------------------


class _Scene(Plugin):
    """Attaches :attr:`XML` into the world; subclasses pick the scene."""

    XML = ""

    def build(self, spec: mujoco.MjSpec, ctx: SimContext) -> None:
        spec.attach(
            mujoco.MjSpec.from_string(self.XML), prefix="", frame=spec.worldbody.add_frame()
        )


def _tool(flex_below: str = "", contype: int = 1) -> str:
    return f"""<mujoco><worldbody><body name="link" pos="0 0 0.5">
      <joint type="slide" axis="0 0 1" damping="100"/>
      <geom type="box" size=".02 .02 .02" mass="0.5"/>
      <site name="fts_site"/>
      <body name="tool" pos="0 0 -.05"><geom type="box" size=".02 .02 .02" mass="0.2"/>{flex_below}</body>
    </body>
    <body name="elsewhere" pos="1 0 .05">
      <flexcomp name="blob" type="grid" count="3 3 3" spacing=".02 .02 .02" dim="3" radius=".002"
                mass=".1"><contact contype="{contype}" conaffinity="{contype}" selfcollide="none"/>
        <edge equality="true"/></flexcomp></body>
    </worldbody></mujoco>"""


class _TouchableFlex(_Scene):
    XML = _tool()


class _CarriedFlex(_Scene):
    XML = _tool(
        flex_below='<flexcomp name="pad" type="grid" count="3 3 2" spacing=".01 .01 .01" dim="3" '
        'radius=".001" mass=".02" pos="0 0 -.03"><pin gridrange="0 0 1 2 2 1"/>'
        '<edge equality="true"/><contact selfcollide="none"/></flexcomp>',
        contype=0,
    )


class _NoFlexContact(_Scene):
    XML = _tool(contype=0)


def _setup(scene: str, **ft):
    cfg = load_config_from_dict(
        {
            "sim": {},
            "components": [
                {f"{__name__}:{scene}": {}},
                {"force_torque": {"site": "fts_site", **ft}, "name": "ft"},
            ],
        }
    )
    engine = Engine(cfg)
    engine.ctx.seed = 0
    engine.setup()
    return engine


def test_a_sensed_tool_that_can_touch_a_flex_is_refused():
    with pytest.raises(RuntimeError, match=r"'blob' can collide.*flex_reaction: excluded"):
        _setup("_TouchableFlex")


def test_a_flex_carried_below_the_sensor_is_refused():
    with pytest.raises(RuntimeError, match=r"'pad' hangs below it"):
        _setup("_CarriedFlex")


@pytest.mark.parametrize("scene", ["_TouchableFlex", "_CarriedFlex"])
def test_stating_the_exclusion_accepts_the_reading(scene):
    engine = _setup(scene, flex_reaction="excluded")
    assert any(isinstance(p, ForceTorquePlugin) for p in engine.plugins)


def test_a_flex_that_makes_no_contact_needs_no_statement():
    _setup("_NoFlexContact")


def test_flex_reaction_takes_one_value():
    errors = ForceTorquePlugin({"site": "s"}).validate_config({"site": "s", "flex_reaction": "on"})
    assert any("flex_reaction" in e for e in errors)
