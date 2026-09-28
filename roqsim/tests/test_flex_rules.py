"""A flex picks its integrator, and a combination MuJoCo would reject or run wrongly is refused by name.

Two halves. The first pins :mod:`roqsim.flex`'s rules to MuJoCo's: every case is compiled, and the
verdict roqsim reads off the spec must match what MuJoCo does with it -- so a MuJoCo release that
moves a rule fails here, not in someone's run. The second is the engine's use of them: ``auto``
resolves per world, the resolution reaches the record and ``roqsim check``, and each refusal names
the ``sim`` key that fixes it.
"""

from __future__ import annotations

import math
import textwrap

import mujoco
import pytest

from roqsim.check import _render_text, check_world
from roqsim.config import load_config_from_dict
from roqsim.engine import Engine
from roqsim.flex import discrete_flexes, needs_discrete
from roqsim.plugin import PluginError

# One small lattice, varied only in what decides the rules. 3x3x3 vertices, 2 cm apart.
_GRID = 'type="grid" count="3 3 3" spacing=".02 .02 .02" dim="3" mass=".1" radius=".002"'
_SHEET = 'type="grid" count="4 4 1" spacing=".02 .02 .02" dim="2" mass=".1" radius=".002"'
_ROPE = 'type="grid" count="5 1 1" spacing=".02 .02 .02" dim="1" mass=".1" radius=".002"'

ELASTIC = f'<flexcomp name="blk" {_GRID}><elasticity young="1e5" poisson="0.2"/><contact selfcollide="none"/></flexcomp>'
PLAIN = f'<flexcomp name="blk" {_GRID}><contact selfcollide="none"/></flexcomp>'

#: name -> (flexcomp, whether MuJoCo 3.14 integrates it under `discrete` only)
CASES = {
    "elastic_solid": (ELASTIC, True),
    "elastic_solid_pinned": (
        f'<flexcomp name="blk" {_GRID}><elasticity young="1e5"/><pin id="0 1 2"/></flexcomp>',
        True,
    ),
    "elastic_solid_trilinear": (
        f'<flexcomp name="blk" {_GRID} dof="trilinear"><elasticity young="1e5"/><contact selfcollide="none"/></flexcomp>',
        True,
    ),
    "elastic_solid_rigid": (
        f'<flexcomp name="blk" {_GRID} rigid="true"><elasticity young="1e5"/></flexcomp>',
        False,
    ),
    "passive_contact": (
        f'<flexcomp name="blk" {_GRID}><contact passive="true" selfcollide="none"/></flexcomp>',
        True,
    ),
    "passive_contact_rigid": (
        f'<flexcomp name="blk" {_GRID} rigid="true"><contact passive="true"/></flexcomp>',
        False,
    ),
    "plain_solid": (PLAIN, False),
    "damping_only": (
        f'<flexcomp name="blk" {_GRID}><elasticity damping="0.01"/></flexcomp>',
        False,
    ),
    "shell_bending": (
        f'<flexcomp name="blk" {_SHEET}><elasticity young="1e5" thickness=".002" elastic2d="bend"/></flexcomp>',
        True,
    ),
    "shell_stretch": (
        f'<flexcomp name="blk" {_SHEET}><elasticity young="1e5" thickness=".002" elastic2d="stretch"/></flexcomp>',
        True,
    ),
    "shell_no_elastic2d": (
        f'<flexcomp name="blk" {_SHEET}><elasticity young="1e5" thickness=".002"/></flexcomp>',
        False,
    ),
    "rope_young": (f'<flexcomp name="blk" {_ROPE}><elasticity young="1e5"/></flexcomp>', False),
    "rope_edge_stiffness": (
        f'<flexcomp name="blk" {_ROPE}><edge stiffness="10"/></flexcomp>',
        False,
    ),
}


def _mjcf(flex: str, *, option: str = "", holder: str = "") -> str:
    return textwrap.dedent(
        f"""
        <mujoco>
          <option {option}/>
          <worldbody>
            <geom name="floor" type="plane" size="1 1 0.1"/>
            <body name="holder" pos="0 0 0.3" {holder}>
              {flex}
            </body>
          </worldbody>
        </mujoco>
        """
    )


def _compiles(xml: str) -> bool:
    try:
        mujoco.MjModel.from_xml_string(xml)
    except ValueError:
        return False
    return True


# -- the rules agree with MuJoCo ------------------------------------------------------------------
@pytest.mark.parametrize("case", sorted(CASES))
def test_the_rule_matches_what_mujoco_compiles(case):
    """Rule 1 and rule 2 of `roqsim.flex`, checked case by case against MuJoCo itself."""
    flex, expected = CASES[case]
    spec = mujoco.MjSpec.from_string(_mjcf(flex))
    assert needs_discrete(spec) is expected

    passive = "passive=" in flex and expected
    # Rule 1: implicit/implicitfast refuse every discrete flex; euler/rk4 refuse only passive contact.
    assert _compiles(_mjcf(flex, option='integrator="implicitfast"')) is not expected
    assert _compiles(_mjcf(flex, option='integrator="Euler"')) is not passive
    assert _compiles(_mjcf(flex, option='integrator="discrete"'))
    # Rule 2: under discrete, PGS and noslip are refused exactly for a discrete flex.
    pgs = _mjcf(flex, option='integrator="discrete" solver="PGS"')
    noslip = _mjcf(flex, option='integrator="discrete" noslip_iterations="3"')
    assert _compiles(pgs) is not expected
    assert _compiles(noslip) is not expected


def test_the_verdict_names_the_flex_and_why():
    spec = mujoco.MjSpec.from_string(_mjcf(ELASTIC))
    assert [str(f) for f in discrete_flexes(spec)] == ["flex 'blk' (elasticity)"]


def test_a_flex_under_a_mocap_body_never_deforms():
    """Rule 3, measured: why roqsim refuses what MuJoCo accepts.

    The same pinned flex, shaken by a mocap body and by a servoed slide joint. Under the joint its
    vertices lag the holder; under the mocap they do not move relative to it at all.
    """
    flex = f'<flexcomp name="blk" {_GRID}><elasticity young="1e4" damping="0.001"/><pin id="0 1 2 3 4 5 6 7 8"/><contact contype="0" conaffinity="0"/></flexcomp>'

    def deformation(mocap: bool) -> float:
        holder = 'mocap="true"' if mocap else ""
        joint = (
            ""
            if mocap
            else '<joint name="x" type="slide" axis="1 0 0"/><inertial pos="0 0 0" mass="5" diaginertia=".01 .01 .01"/>'
        )
        actuator = "" if mocap else '<actuator><position joint="x" kp="20000" kv="400"/></actuator>'
        xml = _mjcf(
            joint + flex,
            option='integrator="discrete" timestep="0.001" gravity="0 0 0"',
            holder=holder,
        )
        model = mujoco.MjModel.from_xml_string(xml.replace("</mujoco>", actuator + "</mujoco>"))
        data = mujoco.MjData(model)
        holder_id = model.body("holder").id
        mujoco.mj_forward(model, data)
        rest = data.flexvert_xpos - data.xpos[holder_id]
        worst = 0.0
        for _ in range(1000):
            x = 0.05 * math.sin(2 * math.pi * 2 * data.time)
            if mocap:
                data.mocap_pos[0] = [x, 0.0, 0.3]
            else:
                data.ctrl[0] = x
            mujoco.mj_step(model, data)
            worst = max(worst, float(abs(data.flexvert_xpos - data.xpos[holder_id] - rest).max()))
        return worst

    assert deformation(mocap=False) > 1e-4
    assert deformation(mocap=True) < 1e-9


# -- the engine -----------------------------------------------------------------------------------
def _engine(
    tmp_path,
    flex: str = "",
    *,
    sim: dict | None = None,
    option: str = "",
    holder: str = "",
    components=(),
):
    world = tmp_path / "flex_world.xml"
    world.write_text(_mjcf(flex, option=option, holder=holder))
    config = load_config_from_dict(
        {"sim": {"world": str(world), **(sim or {})}, "components": list(components)},
        base_dir=tmp_path,
    )
    engine = Engine(config)
    engine.ctx.seed = 0
    return engine


def _integrator(engine) -> mujoco.mjtIntegrator:
    return mujoco.mjtIntegrator(engine.ctx.model.opt.integrator)


def test_auto_resolves_to_discrete_for_an_elastic_flex_and_records_it(tmp_path):
    engine = _engine(tmp_path, ELASTIC)
    engine.setup()
    try:
        assert _integrator(engine) == mujoco.mjtIntegrator.mjINT_DISCRETE
        assert engine.integrator.requested == "auto"
        assert engine.integrator.reason == "auto: flex 'blk' (elasticity)"
        # What ran, in the provenance's sim block -- not the delegation.
        assert engine.config.as_record()["sim"]["integrator"] == "discrete"
        assert "integrator" not in engine.config.sim
        engine.reset()
        for _ in range(50):
            engine.step()
        assert all(abs(v) < 10 for v in engine.ctx.data.qvel)
    finally:
        engine.shutdown()


@pytest.mark.parametrize(
    "flex",
    ["", PLAIN, CASES["elastic_solid_rigid"][0]],
    ids=["no_flex", "plain_flex", "rigid_flex"],
)
def test_auto_keeps_implicitfast_for_a_model_without_a_discrete_flex(tmp_path, flex):
    engine = _engine(tmp_path, flex)
    engine.setup()
    try:
        assert _integrator(engine) == mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        assert engine.config.as_record()["sim"]["integrator"] == "implicitfast"
    finally:
        engine.shutdown()


def test_an_existing_world_still_runs_under_implicitfast(make_engine):
    """The default room with nothing in it: every world written before `auto` existed."""
    engine = make_engine([])
    engine.setup()
    try:
        assert _integrator(engine) == mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        assert engine.integrator.reason == "auto: no flex that needs discrete"
    finally:
        engine.shutdown()


def test_auto_sees_a_flex_a_plugin_builds(tmp_path):
    """Resolved after the build hooks, so a flex no world file mentions still decides it."""
    (tmp_path / "add_flex.py").write_text(
        textwrap.dedent(
            f"""
            import mujoco
            from roqsim.plugin import Plugin

            class AddFlex(Plugin):
                def build(self, spec, ctx):
                    child = mujoco.MjSpec.from_string('''<mujoco><worldbody><body name="tool">{ELASTIC}</body></worldbody></mujoco>''')
                    spec.worldbody.add_frame(pos=[0, 0, 0.5]).attach_body(child.body("tool"), "ee_", "")
            """
        )
    )
    engine = _engine(tmp_path, components=[{"./add_flex.py:AddFlex": {}}])
    engine.setup()
    try:
        assert _integrator(engine) == mujoco.mjtIntegrator.mjINT_DISCRETE
        assert "ee_blk" in engine.integrator.reason
    finally:
        engine.shutdown()


def test_a_stated_timestep_wins_over_a_build_hook(tmp_path):
    (tmp_path / "set_step.py").write_text(
        textwrap.dedent(
            """
            from roqsim.plugin import Plugin

            class SetStep(Plugin):
                def build(self, spec, ctx):
                    spec.option.timestep = 0.01
            """
        )
    )
    engine = _engine(tmp_path, sim={"timestep": 0.001}, components=[{"./set_step.py:SetStep": {}}])
    engine.setup()
    try:
        assert engine.ctx.model.opt.timestep == pytest.approx(0.001)
    finally:
        engine.shutdown()


@pytest.mark.parametrize("integrator", ["implicit", "implicitfast"])
def test_an_implicit_integrator_with_an_elastic_flex_is_refused(tmp_path, integrator):
    engine = _engine(tmp_path, ELASTIC, sim={"integrator": integrator})
    with pytest.raises(
        PluginError,
        match=rf"sim\.integrator: {integrator} cannot run flex 'blk'.*sim\.integrator: auto",
    ):
        engine.setup()


def test_passive_contact_is_refused_under_any_integrator_but_discrete(tmp_path):
    engine = _engine(tmp_path, CASES["passive_contact"][0], sim={"integrator": "euler"})
    with pytest.raises(
        PluginError, match=r"sim\.integrator: euler cannot run flex 'blk' \(passive contact\)"
    ):
        engine.setup()


def test_an_explicit_integrator_with_an_elastic_flex_is_allowed(tmp_path):
    """MuJoCo integrates it explicitly under euler: a choice a world may make, not an error."""
    engine = _engine(tmp_path, ELASTIC, sim={"integrator": "euler"})
    engine.setup()
    try:
        assert _integrator(engine) == mujoco.mjtIntegrator.mjINT_EULER
    finally:
        engine.shutdown()


def test_pgs_with_a_discrete_flex_is_refused(tmp_path):
    engine = _engine(tmp_path, ELASTIC, sim={"solver": "pgs"})
    with pytest.raises(
        PluginError, match=r"sim\.solver: pgs cannot solve flex 'blk'.*newton or cg"
    ):
        engine.setup()


def test_pgs_from_the_world_mjcf_is_refused_by_the_same_key(tmp_path):
    """The world's own <option> counts: the sim key is what overrides it, so that is what is named."""
    engine = _engine(tmp_path, ELASTIC, option='solver="PGS"')
    with pytest.raises(PluginError, match=r"sim\.solver: pgs"):
        engine.setup()


def test_noslip_with_a_discrete_flex_is_refused(tmp_path):
    engine = _engine(tmp_path, ELASTIC, sim={"noslip_iterations": 5})
    with pytest.raises(
        PluginError, match=r"sim\.noslip_iterations: 5 cannot be used with flex 'blk'"
    ):
        engine.setup()


def test_pgs_and_noslip_stay_allowed_beside_a_flex_that_does_not_need_discrete(tmp_path):
    engine = _engine(tmp_path, PLAIN, sim={"solver": "pgs", "noslip_iterations": 5})
    engine.setup()
    engine.shutdown()


@pytest.mark.parametrize("pin", ['<pin id="0 1 2"/>', ""], ids=["pinned", "unpinned"])
def test_a_flex_in_a_mocap_body_is_refused(tmp_path, pin):
    flex = ELASTIC.replace('<contact selfcollide="none"/>', pin + '<contact selfcollide="none"/>')
    engine = _engine(tmp_path, flex, holder='mocap="true"')
    with pytest.raises(PluginError, match=r"flex 'blk' is attached to the mocap body 'holder'"):
        engine.setup()


# -- config and check -----------------------------------------------------------------------------
@pytest.mark.parametrize(("key", "value"), [("integrator", "verlet"), ("solver", "gauss")])
def test_an_unknown_enum_value_is_refused_at_load(key, value):
    with pytest.raises(PluginError, match=rf"sim\.{key}: unknown value '{value}'; one of"):
        load_config_from_dict({"sim": {key: value}, "components": []})


def test_check_reports_the_resolved_integrator_and_why(tmp_path):
    (tmp_path / "block.xml").write_text(_mjcf(ELASTIC))
    world = tmp_path / "world.yaml"
    world.write_text("sim: {world: block.xml}\ncomponents: []\n")
    report = check_world(str(world))
    assert report["ok"], report["problems"]
    assert report["world"]["integrator"] == "discrete"
    assert "integrator discrete (auto: flex 'blk' (elasticity))" in _render_text(report)


def test_check_reports_a_refusal_at_the_build_stage(tmp_path):
    (tmp_path / "block.xml").write_text(_mjcf(ELASTIC))
    world = tmp_path / "world.yaml"
    world.write_text("sim: {world: block.xml, integrator: implicitfast}\ncomponents: []\n")
    report = check_world(str(world))
    assert not report["ok"]
    assert report["problems"][0]["stage"] == "build"
    assert "sim.integrator: auto" in report["problems"][0]["message"]
