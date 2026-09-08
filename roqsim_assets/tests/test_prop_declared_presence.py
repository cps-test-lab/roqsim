"""``present: false`` on the parametric props, and on a population of them.

The contract itself -- what the key means, what absence does, that it survives a reset -- is
:mod:`roqsim.tests.test_declared_presence`, written against the one plugin that honoured it. What
is pinned here is that it now reaches the props a campaign actually places: every plugin that
registers an entity used to ACCEPT this key and drop it, so a world declaring an obstacle absent
compiled a present one and nothing said otherwise.

The population case is the one worth reading twice. ``boxes`` registers no entity of its own -- its
instances do -- so the engine, which sees the entry and not what is under it, cannot apply their
declaration for them.
"""

from __future__ import annotations

from roqsim.config import load_config_from_dict
from roqsim.engine import Engine
from roqsim.plugin import Plugin

_SCENE = """<mujoco>
  <worldbody><geom name="floor" type="plane" size="20 20 .05"/></worldbody>
</mujoco>
"""


def _engine(tmp_path, components):
    world = tmp_path / "w.xml"
    world.write_text(_SCENE)
    return Engine(
        load_config_from_dict(
            {"sim": {"pacing": "asap", "world": str(world)}, "components": components},
            base_dir=tmp_path,
        )
    )


def _box(name, **extra):
    # A distinct MJCF prefix per entry: `box` documents that two of them in one world need one,
    # since the material each declares would otherwise collide at compile.
    return {
        "box": {
            "pose": {"position": {"x": 1.0, "y": 0.0}},
            "size": [0.5, 0.5, 1.0],
            "prefix": f"{name}_",
            **extra,
        },
        "name": name,
    }


def test_a_parametric_prop_declared_absent_starts_absent(tmp_path):
    engine = _engine(tmp_path, [_box("here"), _box("spare", present=False)])
    engine.setup()
    assert engine.ctx.entities.get("here").present is True
    assert engine.ctx.entities.get("spare").present is False
    # The control plane lists what a trial can perceive, and a spare is not that yet.
    assert "spare" not in engine.ctx.entities.names(present_only=True)


def test_a_population_declares_it_per_instance(tmp_path):
    """The instances register the entities, so the declaration is each instance's own."""
    engine = _engine(
        tmp_path,
        [
            {
                "boxes": {
                    "instances": [
                        {
                            "name": "obstacle_0",
                            "pose": {"position": {"x": 2.0, "y": 0.0}},
                            "size": [0.5, 0.5, 1.0],
                        },
                        {
                            "name": "dynamic_0",
                            "pose": {"position": {"x": 4.0, "y": 0.0}},
                            "size": [0.5, 0.5, 1.0],
                            "present": False,
                        },
                    ]
                },
                "name": "obstacles",
            }
        ],
    )
    engine.setup()
    assert engine.ctx.entities.get("obstacle_0").present is True
    assert engine.ctx.entities.get("dynamic_0").present is False


def test_a_population_instance_is_a_spare_again_after_a_reset(tmp_path):
    """Presence lives in ``model``, which ``mj_resetData`` does not restore -- and the engine
    cannot reach these instances to put them back, so their entry has to."""
    from roqsim.presence import set_present

    engine = _engine(
        tmp_path,
        [
            {
                "boxes": {
                    "instances": [
                        {
                            "name": "dynamic_0",
                            "pose": {"position": {"x": 4.0, "y": 0.0}},
                            "size": [0.5, 0.5, 1.0],
                            "present": False,
                        }
                    ]
                },
                "name": "obstacles",
            }
        ],
    )
    engine.setup()
    set_present(engine.ctx, engine.ctx.entities.get("dynamic_0"), True)  # a trial's SpawnEntity

    engine.reset()

    assert engine.ctx.entities.get("dynamic_0").present is False


def test_a_plugin_that_registers_no_entity_refuses_the_key():
    """Silence here is the same failure one step removed: a key that reads as though it did
    something. A plugin with no entity has no presence to declare."""

    class _NoEntity(Plugin):
        """A transport or a monitor: configured by a world, but nothing in the scene is it."""

    plugin = _NoEntity({"present": False}, label="n")
    assert any("registers none" in e for e in plugin.config_errors(plugin.config))
