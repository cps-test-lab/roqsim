"""contact_pair: the friction of ONE pair, which per-geom values cannot express.

The load-bearing test is not that pairs appear in the model -- it is that a block slides further on
the SAME floor when only its pair with that floor is changed, and that a second pair sharing the
block is unaffected. That second half is the whole reason the plugin exists: per-geom friction
combines by maximum, so one object cannot have two different pair frictions no matter how its own
value is set.
"""

from __future__ import annotations

import mujoco
import pytest

from roqsim.config import load_config_from_dict
from roqsim.engine import Engine
from roqsim.plugin import PluginError

# A block resting on the floor, plus a second block beside it. Both start with MuJoCo's default
# friction, so any difference below comes from a declared pair and nothing else.
SCENE = """
<mujoco model="pair_test">
  <worldbody>
    <geom name="floor" type="plane" size="20 20 0.05"/>
    <body name="slider" pos="0 0 0.1">
      <freejoint/>
      <geom name="slider_g" type="box" size="0.1 0.1 0.1" mass="1"/>
    </body>
    <body name="other" pos="0 2 0.1">
      <freejoint/>
      <geom name="other_g" type="box" size="0.1 0.1 0.1" mass="1"/>
    </body>
  </worldbody>
</mujoco>
"""


def _engine(tmp_path, *pairs):
    world = tmp_path / "w.xml"
    world.write_text(SCENE)
    return Engine(
        load_config_from_dict(
            {
                "sim": {"pacing": "asap", "world": str(world)},
                # A distinct `name:` per entry: a label addresses one component, so two unnamed
                # `contact_pair`s in one document are refused -- which is the substrate being
                # right, and is why a world declaring several pairs must name them.
                "components": [
                    {"contact_pair": p, "name": f"pair_{i}"} for i, p in enumerate(pairs)
                ],
            },
            base_dir=tmp_path,
        )
    )


def _slide(engine, body="slider", vx=2.0, seconds=3.0):
    """Launch a block along +x and return how far it travelled before stopping."""
    engine.setup()
    engine.reset()
    ctx = engine.ctx
    bid = mujoco.mj_name2id(ctx.model, mujoco.mjtObj.mjOBJ_BODY, body)
    adr = ctx.model.jnt_dofadr[ctx.model.body_jntadr[bid]]
    x0 = float(ctx.data.xpos[bid][0])
    ctx.data.qvel[adr] = vx
    for _ in range(int(seconds / ctx.dt)):
        engine.step()
    return float(ctx.data.xpos[bid][0]) - x0


# -- the behaviour --------------------------------------------------------------------------------
def test_a_low_friction_pair_makes_the_block_slide_further(tmp_path):
    """The point of the plugin, measured rather than asserted from the model."""
    default = _slide(_engine(tmp_path))
    slippery = _slide(
        _engine(tmp_path, {"a": {"geom": "slider_g"}, "b": {"geom": "floor"}, "friction": 0.02})
    )
    assert slippery > default * 2, (
        f"a 0.02 pair should slide much further than the default: {slippery:.3f} m vs {default:.3f} m"
    )


def test_a_pair_leaves_every_other_contact_alone(tmp_path):
    """The half per-geom friction cannot do. Lowering the slider's pair with the floor must not
    change the OTHER block, which shares that same floor geom."""
    engine = _engine(
        tmp_path, {"a": {"geom": "slider_g"}, "b": {"geom": "floor"}, "friction": 0.02}
    )
    moved_other = _slide(engine, body="other")
    baseline_other = _slide(_engine(tmp_path), body="other")
    assert moved_other == pytest.approx(baseline_other, rel=0.05), (
        "the untouched block moved differently, so the pair leaked into another contact"
    )


def test_two_pairs_sharing_one_object_get_different_frictions(tmp_path):
    """The case that has no per-geom expression at all: one object, two contacts, two values.

    MuJoCo combines per-geom friction by MAX, so the shared object's own value is a floor on both
    pairs and no assignment produces two different ones.
    """
    engine = _engine(
        tmp_path,
        {"a": {"geom": "slider_g"}, "b": {"geom": "floor"}, "friction": 0.02},
        {"a": {"geom": "slider_g"}, "b": {"geom": "other_g"}, "friction": 0.9},
    )
    engine.setup()
    m = engine.ctx.model
    got = {}
    for i in range(m.npair):
        g1 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, int(m.pair_geom1[i]))
        g2 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, int(m.pair_geom2[i]))
        got[frozenset((g1, g2))] = round(float(m.pair_friction[i][0]), 4)
    assert got[frozenset(("slider_g", "floor"))] == 0.02
    assert got[frozenset(("slider_g", "other_g"))] == 0.9


def test_entities_pair_every_geom_of_both_subtrees(tmp_path):
    """A robot base with many collision geoms must pair all of them. Asking a world to name each is
    how a pair silently misses the one that actually touches."""
    world = tmp_path / "w.xml"
    world.write_text(SCENE)
    engine = Engine(
        load_config_from_dict(
            {
                "sim": {"pacing": "asap", "world": str(world)},
                "components": [
                    {
                        "contact_pair": {
                            "a": {"body": "slider"},
                            "b": {"body": "other"},
                            "friction": 0.4,
                        }
                    },
                ],
            },
            base_dir=tmp_path,
        )
    )
    engine.setup()
    assert engine.ctx.model.npair == 1  # one geom each side here
    assert float(engine.ctx.model.pair_friction[0][0]) == pytest.approx(0.4)


def test_the_two_sides_may_be_different_kinds(tmp_path):
    """The case that forced this API: the floor is a geom while the thing sliding on it is an
    entity. Requiring both sides to be the same kind makes the commonest pair in a pushing
    experiment -- object against ground, object against pusher -- inexpressible."""
    world = tmp_path / "w.xml"
    world.write_text(SCENE)
    engine = Engine(
        load_config_from_dict(
            {
                "sim": {"pacing": "asap", "world": str(world)},
                "components": [
                    {
                        "contact_pair": {
                            "a": {"body": "slider"},
                            "b": {"geom": "floor"},
                            "friction": 0.05,
                        }
                    },
                ],
            },
            base_dir=tmp_path,
        )
    )
    engine.setup()
    assert engine.ctx.model.npair == 1
    assert float(engine.ctx.model.pair_friction[0][0]) == pytest.approx(0.05)


# -- validation -----------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "cfg, message",
    [
        ({"friction": 0.3}, "is required"),
        ({"a": {"geom": "x"}, "friction": 0.3}, "is required"),
        ({"a": "x", "b": {"geom": "y"}}, "must be a mapping"),
        ({"a": {}, "b": {"geom": "y"}}, "exactly one of"),
        ({"a": {"geom": "x", "body": "y"}, "b": {"geom": "z"}}, "exactly one of"),
        ({"a": {"geom": "x"}, "b": {"geom": "y"}, "friction": [1.0, 2.0]}, "5-element row"),
        ({"a": {"geom": "x"}, "b": {"geom": "y"}, "friction": -1.0}, ">= 0"),
        ({"a": {"geom": "x"}, "b": {"geom": "y"}, "condim": 2}, "condim"),
    ],
)
def test_a_malformed_pair_is_refused(tmp_path, cfg, message):
    with pytest.raises(PluginError, match=message):
        _engine(tmp_path, cfg)


def test_declaring_the_same_pair_twice_is_refused(tmp_path):
    """MuJoCo keeps both and uses one; which is not something a world should have to know."""
    engine = _engine(
        tmp_path,
        {"a": {"geom": "slider_g"}, "b": {"geom": "floor"}, "friction": 0.02},
        {"a": {"geom": "floor"}, "b": {"geom": "slider_g"}, "friction": 0.9},
    )
    with pytest.raises(RuntimeError, match="already an explicit pair"):
        engine.setup()


def test_an_unknown_body_fails_loudly(tmp_path):
    """A pair naming nothing would silently leave the contact at its combined per-geom value --
    which is exactly the state the plugin exists to escape."""
    engine = _engine(tmp_path, {"a": {"body": "slider"}, "b": {"body": "nope"}, "friction": 0.3})
    with pytest.raises(RuntimeError, match="not found"):
        engine.setup()
