"""A flex's material is a world key, for a flex of any origin, and four of its fields move at run time.

``flex_material`` sets a flex's material on the spec before compile, because MuJoCo bakes Young's
modulus into the compiled stiffness and keeps no field a later write could reach. What is measured
here is the physics, not the model: a stiffer modulus bends a cantilever less, whichever model the
flex came from, with the component declared after it; a ``--set`` of the modulus is a sweep axis;
and each ``model_override`` flex row changes what a run does when it is written at run time.
"""

from __future__ import annotations

import textwrap

import mujoco
import numpy as np
import pytest

from roqsim.config import load_config, load_config_from_dict, overrides_from_dotlist
from roqsim.engine import Engine
from roqsim.plugin import Plugin, PluginError
from roqsim.plugins.flex_material import FlexMaterialPlugin

# A 10 cm x 2 cm x 2 cm beam of 24 vertices, its x=0 face pinned to the body it is declared in.
# Contact off, so gravity and elasticity are all that act on it.
_BEAM = """
<flexcomp name="beam" type="grid" count="6 2 2" spacing=".02 .02 .02" dim="3" mass=".05"
          radius=".001">
  <elasticity young="{young}" poisson="0.3" damping="{damping}"/>
  <contact contype="0" conaffinity="0" selfcollide="none"/>
  <pin gridrange="0 0 0 0 1 1"/>
</flexcomp>
"""


def _beam(young="1e5", damping="0.01") -> str:
    return _BEAM.format(young=young, damping=damping)


def _world_mjcf(tmp_path, body: str, name: str = "w.xml") -> str:
    path = tmp_path / name
    path.write_text(f"<mujoco><worldbody>{body}</worldbody></mujoco>", encoding="utf-8")
    return str(path)


def _engine(tmp_path, components, *, world=None, plugins=None, sim=None) -> Engine:
    cfg = load_config_from_dict(
        {
            "sim": {"timestep": 0.001, **({"world": world} if world else {}), **(sim or {})},
            "components": components,
        },
        base_dir=tmp_path,
    )
    engine = Engine(cfg, plugins=plugins)
    engine.ctx.seed = 0
    return engine


def _run(engine: Engine, seconds: float) -> np.ndarray:
    """Every vertex position at every step: ``(steps, nflexvert, 3)``."""
    engine.setup()
    engine.reset()
    out = []
    for _ in range(int(round(seconds / engine.ctx.dt))):
        engine.step()
        out.append(np.array(engine.ctx.data.flexvert_xpos))
    return np.array(out)


def _deflection(engine: Engine) -> float:
    """How far the beam's tip hangs below its rest height once it has settled (m)."""
    engine.setup()
    engine.reset()
    rest = float(engine.ctx.data.flexvert_xpos[:, 2].min())
    for _ in range(1000):
        engine.step()
    return rest - float(engine.ctx.data.flexvert_xpos[:, 2].min())


# -- a stated modulus changes what the flex does ---------------------------------------------------
def test_young_changes_the_measured_static_deflection(tmp_path):
    """Linear elasticity: three times the modulus, a third of the deflection."""
    world = _world_mjcf(tmp_path, f'<body name="holder" pos="0 0 .5">{_beam()}</body>')
    nominal = _deflection(_engine(tmp_path, [], world=world))
    stiffer = _deflection(
        _engine(tmp_path, [{"flex_material": {"flex": "beam", "young": 3.0e5}}], world=world)
    )
    assert nominal > 0.005, f"the nominal beam should visibly sag, got {nominal * 1e3:.2f} mm"
    assert stiffer / nominal == pytest.approx(1 / 3, rel=0.1), (nominal, stiffer)


def test_a_flex_made_elastic_here_gets_the_discrete_integrator(tmp_path):
    """The integrator is resolved after this plugin has built, so `auto` sees the new modulus."""
    world = _world_mjcf(tmp_path, f'<body name="holder" pos="0 0 .5">{_beam(young="0")}</body>')
    plain = _engine(tmp_path, [], world=world)
    plain.setup()
    assert plain.integrator.resolved == "implicitfast"
    elastic = _engine(tmp_path, [{"flex_material": {"flex": "beam", "young": 1.0e5}}], world=world)
    assert _deflection(elastic) > 0.005
    assert elastic.integrator.resolved == "discrete"


def test_young_is_a_sweep_axis_through_set(tmp_path):
    """`--set components.<address>.young=...`, the path a campaign's factor takes."""
    world = _world_mjcf(tmp_path, f'<body name="holder" pos="0 0 .5">{_beam()}</body>')
    doc = tmp_path / "world.yaml"
    doc.write_text(
        textwrap.dedent(
            f"""
            sim: {{timestep: 0.001, world: {world}}}
            components:
              - flex_material: {{flex: beam, young: 1.0e+5}}
                name: beam_material
            """
        ),
        encoding="utf-8",
    )

    def swept(young: str) -> float:
        cfg = load_config(
            doc, overrides=overrides_from_dotlist([f"components.beam_material.young={young}"])
        )
        engine = Engine(cfg)
        engine.ctx.seed = 0
        return _deflection(engine)

    soft, stiff = swept("1.0e+5"), swept("4.0e+5")
    assert stiff / soft == pytest.approx(1 / 4, rel=0.1), (soft, stiff)


# -- any origin, declared after it -----------------------------------------------------------------
def test_a_spawned_assets_flex_declared_after_the_material(tmp_path):
    """The component sits ABOVE the spawn that brings the flex in, and the flex is prefixed."""
    asset = _world_mjcf(tmp_path, f'<body name="holder">{_beam()}</body>', name="beam.xml")

    def spawned(material: list) -> Engine:
        return _engine(
            tmp_path,
            [
                {
                    "spawn_model": {
                        "model": asset,
                        "prefix": "p_",
                        "motion": "static",
                        "pose": {"position": {"x": 0.0, "y": 0.0, "z": 0.5}},
                    },
                    "name": "beam_prop",
                },
                *material,
            ],
        )

    nominal = _deflection(spawned([]))
    stiffer = _deflection(spawned([{"flex_material": {"flex": "p_beam", "young": 3.0e5}}]))
    assert stiffer / nominal == pytest.approx(1 / 3, rel=0.1), (nominal, stiffer)


class _BeamBuilder(Plugin):
    """A plugin that adds the flex in its own build, as a generator plugin would."""

    def build(self, spec, ctx):
        holder = spec.worldbody.add_body(name="holder", pos=[0, 0, 0.5])
        child = mujoco.MjSpec.from_string(
            f'<mujoco><worldbody><body name="b">{_beam()}</body></worldbody></mujoco>'
        )
        holder.add_frame().attach_body(child.body("b"), "built_", "")


def test_a_plugin_built_flex_declared_before_the_material(tmp_path):
    material = FlexMaterialPlugin({"flex": "built_beam", "young": 3.0e5})
    nominal = _deflection(_engine(tmp_path, [], plugins=[_BeamBuilder()]))
    stiffer = _deflection(_engine(tmp_path, [], plugins=[_BeamBuilder(), material]))
    assert stiffer / nominal == pytest.approx(1 / 3, rel=0.1), (nominal, stiffer)


def test_a_material_declared_before_its_flex_is_refused_naming_the_order(tmp_path):
    material = FlexMaterialPlugin({"flex": "built_beam", "young": 3.0e5})
    engine = _engine(tmp_path, [], plugins=[material, _BeamBuilder()])
    with pytest.raises(PluginError, match=r"no flex named 'built_beam'.*YAML order.*after the component"):
        engine.setup()


# -- refusals --------------------------------------------------------------------------------------
def test_an_unknown_flex_is_refused_naming_the_ones_there_are(tmp_path):
    world = _world_mjcf(tmp_path, f'<body name="holder" pos="0 0 .5">{_beam()}</body>')
    engine = _engine(tmp_path, [{"flex_material": {"flex": "bean", "young": 1.0e5}}], world=world)
    with pytest.raises(PluginError, match=r"no flex named 'bean'.*flexes: 'beam'"):
        engine.setup()


_SHEET = """
<flexcomp name="sheet" type="grid" count="4 4 1" spacing=".02 .02 .02" dim="2" mass=".02"
          radius=".001">
  <elasticity young="1e5" thickness=".002"/>
</flexcomp>
"""
_RIGID = """
<flexcomp name="lump" type="grid" count="3 3 3" spacing=".02 .02 .02" dim="3" mass=".05"
          rigid="true"/>
"""


@pytest.mark.parametrize(
    ("body", "material", "expected"),
    [
        (_beam(), {"thickness": 0.002}, "only a dim=2 flex"),
        (_beam(), {"elastic2d": "both"}, "only a dim=2 flex"),
        (_SHEET, {"young": 2.0e5}, "set elastic2d"),
        (_RIGID, {"young": 2.0e5}, "is rigid"),
    ],
)
def test_a_key_the_flex_would_not_read_is_refused(tmp_path, body, material, expected):
    world = _world_mjcf(tmp_path, f'<body name="holder" pos="0 0 .5">{body}</body>')
    name = "beam" if "beam" in body else "sheet" if "sheet" in body else "lump"
    engine = _engine(tmp_path, [{"flex_material": {"flex": name, **material}}], world=world)
    with pytest.raises(PluginError, match=expected):
        engine.setup()


def test_a_shell_made_elastic_in_one_block_is_accepted(tmp_path):
    """elastic2d is applied before the modulus is judged, so one block can do both."""
    world = _world_mjcf(tmp_path, f'<body name="holder" pos="0 0 .5">{_SHEET}</body>')
    engine = _engine(
        tmp_path,
        [{"flex_material": {"flex": "sheet", "young": 2.0e5, "elastic2d": "both"}}],
        world=world,
    )
    engine.setup()
    assert engine.integrator.resolved == "discrete"


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        ({"young": 1.0e5}, "'flex' is required"),
        ({"flex": "beam"}, "sets nothing"),
        ({"flex": "beam", "youngs": 1.0e5}, "'youngs' is not a flex material key"),
        ({"flex": "beam", "young": "5e5"}, "YAML 1.1"),
        ({"flex": "beam", "poisson": 0.5}, "below 0.5"),
        ({"flex": "beam", "elastic2d": "soft"}, "must be one of"),
        ({"flex": "beam", "solimp": [0.9, 0.95, 0.001, 0.5, 2, 7]}, "1 to 5 numbers"),
        ({"flex": "beam", "friction": -1.0}, "must be >= 0"),
    ],
)
def test_config_mistakes_are_refused_at_load(config, expected):
    with pytest.raises(PluginError, match=expected):
        Engine(load_config_from_dict({"sim": {}, "components": [{"flex_material": config}]}))


def test_a_short_vector_keeps_the_flexs_own_values(tmp_path):
    world = _world_mjcf(tmp_path, f'<body name="holder" pos="0 0 .5">{_beam()}</body>')
    engine = _engine(
        tmp_path,
        [{"flex_material": {"flex": "beam", "friction": 0.4, "solref": 0.03, "priority": 2}}],
        world=world,
    )
    engine.setup()
    model = engine.ctx.model
    np.testing.assert_allclose(model.flex_friction[0], [0.4, 0.005, 0.0001])
    np.testing.assert_allclose(model.flex_solref[0], [0.03, 1.0])
    assert int(model.flex_priority[0]) == 2


# -- the run-time rows of model_override -----------------------------------------------------------
def _override(field: str, to, select=("beam",)) -> dict:
    return {
        "model_override": {
            "overrides": [{"field": field, "select": list(select), "to": to}],
            "active": True,
        }
    }


def test_flex_damping_at_run_time_is_the_compiled_damping(tmp_path):
    """Written over a compiled 0.001, 0.01 runs exactly as 0.01 compiled in -- and not as 0.001."""
    world = _world_mjcf(
        tmp_path, f'<body name="holder" pos="0 0 .5">{_beam(damping="0.001")}</body>'
    )
    light = _run(_engine(tmp_path, [], world=world), 0.3)
    written = _run(_engine(tmp_path, [_override("flex_damping", 0.01)], world=world), 0.3)
    compiled = _run(
        _engine(tmp_path, [{"flex_material": {"flex": "beam", "damping": 0.01}}], world=world), 0.3
    )
    np.testing.assert_allclose(written, compiled, atol=1e-9)
    ring = lambda run: float(run[-100:, :, 2].std(axis=0).max())  # noqa: E731
    assert ring(written) < 0.6 * ring(light), (ring(written), ring(light))


def test_flex_damping_over_a_compiled_zero_is_refused(tmp_path):
    world = _world_mjcf(tmp_path, f'<body name="holder" pos="0 0 .5">{_beam(damping="0")}</body>')
    engine = _engine(tmp_path, [_override("flex_damping", 0.01)], world=world)
    with pytest.raises(PluginError, match="compiled with damping 0"):
        engine.setup()


# A soft cube resting on a floor whose own friction is negligible, so the flex's values govern.
_BLOCK = """
<geom name="floor" type="plane" size="1 1 .1" friction="0.01"/>
<flexcomp name="beam" type="grid" count="3 3 3" spacing=".02 .02 .02" pos="0 0 .0215" dim="3"
          mass=".05" radius=".001">
  <elasticity young="1e5" poisson="0.3" damping="0.002"/>
  <contact selfcollide="none" priority="1" solref="0.02 1" friction="1"/>
</flexcomp>
"""


def test_flex_friction_at_run_time_lets_a_held_block_slide(tmp_path):
    """Gravity tilted ~24 degrees: friction 1 holds the block, 0.1 lets it go."""
    world = _world_mjcf(tmp_path, _BLOCK)
    tilted = {"gravity": [4.0, 0.0, -9.0]}
    slid = lambda run: float(run[-1, :, 0].mean() - run[0, :, 0].mean())  # noqa: E731
    held = slid(_run(_engine(tmp_path, [], world=world, sim=tilted), 1.0))
    slipped = slid(
        _run(
            _engine(
                tmp_path,
                [_override("flex_friction", [0.1, 0.005, 0.0001])],
                world=world,
                sim=tilted,
            ),
            1.0,
        )
    )
    assert held < 0.05 and slipped > 0.5, (held, slipped)


@pytest.mark.parametrize(
    ("field", "to"), [("flex_solref", [0.08, 1.0]), ("flex_solimp", [0.1, 0.2, 0.01, 0.5, 2.0])]
)
def test_flex_contact_softness_at_run_time_lets_a_block_sink(tmp_path, field, to):
    world = _world_mjcf(tmp_path, _BLOCK)
    lowest = lambda run: float(run[-1, :, 2].min())  # noqa: E731
    nominal = lowest(_run(_engine(tmp_path, [], world=world), 1.0))
    softened = lowest(_run(_engine(tmp_path, [_override(field, to)], world=world), 1.0))
    assert nominal - softened > 0.005, (nominal, softened)
