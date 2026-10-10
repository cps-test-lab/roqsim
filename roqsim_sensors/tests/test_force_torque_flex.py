"""A flex's contacts reach the ``force_torque`` reading, which MuJoCo's own site sensor leaves out.

The first half measures MuJoCo 3.14 itself, on the smallest scenes that show it: a rigid probe
pressed into a block, and a soft cantilever hanging off a sensed mount. A contact with a flex never
reaches MuJoCo's site sensor; the flex's weight, its elastic reaction and a force applied to one of
its vertices do. These are the guard on the correction: a MuJoCo that starts transmitting the contact
fails here, and :class:`roqsim.flex.FlexContactWrench` must then go, or the plugin counts it twice.

The second half is the plugin, which adds the missing contacts: a flex tool and a rigid tool of the
same size, pressed to the same contact force, read the same wrench; a support under a hanging flex
and a probe pressing a flex balance; a flex across the cut is refused.
"""

from __future__ import annotations

import mujoco
import numpy as np
import pytest
from roqsim_sensors.plugins.force_torque import ForceTorquePlugin

from roqsim.config import load_config_from_dict
from roqsim.context import SimContext
from roqsim.engine import Engine
from roqsim.plugin import Plugin, PluginError

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


# -- the plugin: the reading carries the flex's contacts ---------------------------------------


class _Scene(Plugin):
    """Attaches :attr:`XML` into the world; subclasses pick the scene."""

    XML = ""

    def build(self, spec: mujoco.MjSpec, ctx: SimContext) -> None:
        spec.attach(
            mujoco.MjSpec.from_string(self.XML), prefix="", frame=spec.worldbody.add_frame()
        )


_TOOL_MASS = 0.2
_TOOL_OFFSET_X = 0.03  # the tool hangs off-axis, so the contact also has a moment about the site
_RIGID_TOOL = f'<geom type="box" size=".03 .03 .03" pos="0 0 -.03" mass="{_TOOL_MASS}"/>'
_FLEX_TOOL = (
    f'<flexcomp name="pad" type="grid" count="4 4 4" spacing=".02 .02 .02" dim="3" '
    f'radius=".001" mass="{_TOOL_MASS}" pos="0 0 -.03"><pin gridrange="0 0 3 3 3 3"/>'
    '<edge equality="false"/><elasticity young="2e5" poisson="0.3" damping="0.002"/>'
    '<contact condim="3" solref="0.005 1" selfcollide="none"/></flexcomp>'
)


def _pressing(tool: str) -> str:
    """A sensed link on a damped slide, a tool below it, a motor pressing it onto a ground plane.

    The ground stands 0.3 m above the world's own floor, so the tool touches one plane only.
    """
    return f"""<mujoco><option integrator="discrete" timestep="0.0005" solver="Newton"/>
    <worldbody><geom name="ground" type="plane" size="1 1 .1" pos="0 0 .3"/>
      <body name="link" pos="0 0 .42">
        <joint name="z" type="slide" axis="0 0 1" damping="40"/>
        <geom type="box" size=".02 .02 .01" mass="0.5" contype="0" conaffinity="0"/>
        <site name="fts_site"/>
        <body name="tool" pos="{_TOOL_OFFSET_X} 0 -.02">
          <geom type="box" size=".005 .005 .005" mass="0.01" contype="0" conaffinity="0"/>
          {tool}
        </body>
      </body></worldbody>
      <actuator><motor joint="z"/></actuator></mujoco>"""


class _RigidPress(_Scene):
    XML = _pressing(_RIGID_TOOL)


class _FlexPress(_Scene):
    XML = _pressing(_FLEX_TOOL)


class _ProbeOnFlex(_Scene):
    XML = _probe_world(_FLEX_BLOCK)


class _ProbeOnRigid(_Scene):
    XML = _probe_world(_RIGID_BLOCK)


class _HangingOnSupport(_Scene):
    XML = _cantilever(support=True)


class _Straddling(_Scene):
    """One flex strung between the sensed body and a sibling outside its subtree."""

    XML = """<mujoco><worldbody>
      <body name="a" pos="0 0 .5"><freejoint/><geom type="box" size=".01 .01 .01" mass=".1"/>
        <body name="b" pos="0 0 0"><joint type="slide" axis="0 0 1"/><site name="fts_site"/>
          <geom type="box" size=".01 .01 .01" mass=".1" contype="0" conaffinity="0"/></body>
        <body name="c" pos=".05 0 0"><joint type="slide" axis="0 0 1"/>
          <geom type="box" size=".01 .01 .01" mass=".1" contype="0" conaffinity="0"/></body>
      </body></worldbody>
      <deformable><flex name="rope" dim="1" body="b c" vertex="0 0 0 0 0 0" element="0 1"
                        radius=".002"/></deformable></mujoco>"""


def _setup(scene: str, **ft):
    # An attached model's <option> is dropped (the world's wins), so the step is stated here.
    cfg = load_config_from_dict(
        {
            "sim": {"timestep": 0.0005, "integrator": "discrete", "solver": "newton"},
            "components": [
                {f"{__name__}:{scene}": {}},
                {"force_torque": {"site": ft.pop("site", "fts_site"), **ft}, "name": "ft"},
            ],
        }
    )
    engine = Engine(cfg)
    engine.ctx.seed = 0
    engine.setup()
    return engine


def _plugin(engine) -> ForceTorquePlugin:
    return next(p for p in engine.plugins if isinstance(p, ForceTorquePlugin))


def _floor_contact(engine) -> np.ndarray:
    """The summed world force the ground applies to whatever it touches."""
    m, d = engine.ctx.model, engine.ctx.data
    floor = m.geom("ground").id
    total, f6 = np.zeros(3), np.zeros(6)
    for i in range(d.ncon):
        c = d.contact[i]
        if floor not in (int(c.geom[0]), int(c.geom[1])):
            continue
        mujoco.mj_contactForce(m, d, i, f6)
        world = c.frame.reshape(3, 3).T @ f6[:3]  # on the second side
        total += world if int(c.geom[0]) == floor else -world
    return total


def _press_with(scene: str, push: float = 10.0, steps: int = 8000):
    engine = _setup(scene, frame="world")
    engine.ctx.data.ctrl[0] = -push
    for _ in range(steps):
        engine.step()
    mujoco.mj_forward(engine.ctx.model, engine.ctx.data)
    return engine


def test_a_flex_tool_and_a_rigid_tool_pressed_alike_read_the_same_wrench():
    """Same size, same mass, same contact force: the same wrench within 5 %, and the static case
    balances against the floor's contact."""
    readings = {}
    for scene in ("_RigidPress", "_FlexPress"):
        engine = _press_with(scene)
        force, torque = _plugin(engine).read()
        contact = _floor_contact(engine)
        below = float(engine.ctx.model.body_subtreemass[engine.ctx.model.body("link").id]) * G
        # Environment on tool, world frame: the floor's push minus what the cut carries of weight.
        assert contact[2] > 15.0
        assert force[2] == pytest.approx(contact[2] - below, rel=0.05)
        readings[scene] = (force, torque)
    (f_rigid, t_rigid), (f_flex, t_flex) = readings["_RigidPress"], readings["_FlexPress"]
    assert f_flex[2] == pytest.approx(f_rigid[2], rel=0.05)
    # The contact sits off the site's axis, so it carries a moment the reading must also see.
    assert abs(t_rigid[1]) > 0.1
    assert t_flex[1] == pytest.approx(t_rigid[1], rel=0.05)


def test_the_raw_mujoco_sensor_still_misses_the_flex_press():
    """The correction's guard, on the scene the plugin is tested on."""
    engine = _press_with("_FlexPress")
    m, d = engine.ctx.model, engine.ctx.data
    raw_fz = float(d.sensordata[m.sensor("fts_site_force").adr[0] + 2])
    below = float(m.body_subtreemass[m.body("link").id]) * G
    assert raw_fz == pytest.approx(below, rel=0.01)


def test_a_probe_pressing_a_flex_reads_like_one_pressing_a_rigid_block():
    reads = {}
    for scene in ("_ProbeOnRigid", "_ProbeOnFlex"):
        engine = _setup(scene, site="ft", frame="world")
        engine.ctx.data.ctrl[0] = -0.057
        for _ in range(6000):
            engine.step()
        mujoco.mj_forward(engine.ctx.model, engine.ctx.data)
        force, _ = _plugin(engine).read()
        weight = float(engine.ctx.model.body_mass[engine.ctx.model.body("probe").id]) * G
        reads[scene] = force[2] + weight  # the block's push on the probe
    assert reads["_ProbeOnRigid"] > 4.0 and reads["_ProbeOnFlex"] > 4.0


def test_a_support_under_a_hanging_flex_is_in_the_reading():
    engine = _setup("_HangingOnSupport", site="ft", frame="world")
    for _ in range(8000):
        engine.step()
    mujoco.mj_forward(engine.ctx.model, engine.ctx.data)
    m, d = engine.ctx.model, engine.ctx.data
    weight = float(m.body_mass[1:].sum()) * G
    carried, f6 = 0.0, np.zeros(6)
    for i in range(d.ncon):
        mujoco.mj_contactForce(m, d, i, f6)
        carried += f6[0]
    force, _ = _plugin(engine).read()
    assert carried > 1.0
    assert force[2] == pytest.approx(-(weight - carried), abs=0.005)


def test_a_tare_in_contact_zeroes_the_corrected_reading():
    engine = _press_with("_FlexPress")
    plugin = _plugin(engine)
    plugin.tare()
    force, torque = plugin.read()
    assert np.allclose(force, 0.0, atol=1e-9) and np.allclose(torque, 0.0, atol=1e-9)


def test_a_flex_across_the_cut_is_refused():
    with pytest.raises(PluginError, match=r"'rope' lies partly inside"):
        _setup("_Straddling")


def test_flex_reaction_is_refused_now_that_the_reading_carries_the_contacts():
    errors = ForceTorquePlugin({"site": "s"}).validate_config(
        {"site": "s", "flex_reaction": "excluded"}
    )
    assert any("includes the contacts" in e for e in errors)
