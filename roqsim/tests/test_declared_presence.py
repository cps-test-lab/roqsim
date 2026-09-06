# Copyright (C) 2026 Frederik Pasch
#
# SPDX-License-Identifier: Apache-2.0

"""``present: false``: a world declaring what a trial may bring in, but has not yet.

roqsim never recompiles, so an obstacle that must *appear* mid-trial is compiled in up front and
started absent. This is the declaration half of that; :mod:`test_presence` covers what absence does
to a body, and the ``simulation_interfaces`` services are what move an entity either way at run time.

The distinction the key has to keep is against ``enabled: false``, which is a different question with
a similar spelling: that one never builds the body, so there is nothing left to spawn.
"""

from __future__ import annotations

import mujoco
import pytest

from roqsim.config import instantiate_plugins, load_config_from_dict
from roqsim.engine import Engine
from roqsim.presence import ABSENT_GEOM_GROUP, entity_geom_ids, set_present

CRATE = "roqsim.plugins.spawn_model:SpawnModelPlugin"


@pytest.fixture
def prop_file(tmp_path):
    """A one-body prop as a plain MJCF path, so these tests need no asset package."""
    path = tmp_path / "dummy_box.xml"
    path.write_text(
        "<mujoco><worldbody><body name='box'>"
        "<geom name='box_geom' type='box' size='.2 .2 .2'/>"
        "</body></worldbody></mujoco>"
    )
    return str(path)


@pytest.fixture
def make_engine(prop_file):
    def _make(present=None, free=True):
        config = {"model": prop_file, "pos": [1.0, 0.0, 1.0], "free": free}
        if present is not None:
            config["present"] = present
        return Engine(
            load_config_from_dict({"sim": {}, "plugins": [{CRATE: config, "name": "crate"}]})
        )

    return _make


def _entity(engine):
    return engine.ctx.entities.get("crate")


def test_a_prop_declared_absent_starts_absent(make_engine):
    engine = make_engine(present=False)
    engine.setup()
    assert _entity(engine).present is False
    groups = [int(engine.ctx.model.geom_group[g]) for g in entity_geom_ids(engine.ctx.model, "box")]
    assert groups and set(groups) == {ABSENT_GEOM_GROUP}


def test_the_default_is_present(make_engine):
    """An absent-by-accident prop is a world missing an obstacle nobody wrote down."""
    engine = make_engine()
    engine.setup()
    assert _entity(engine).present is True


def test_the_control_plane_does_not_list_it(make_engine):
    """``GetEntities`` reports what can be perceived, so an absent prop would contradict every
    sensor."""
    engine = make_engine(present=False)
    engine.setup()
    assert "crate" not in engine.ctx.entities.names(present_only=True)
    assert "crate" in engine.ctx.entities.names()


def test_it_is_still_in_the_compiled_model(make_engine):
    """The whole point: absent is spawnable, which is what separates it from ``enabled: false``."""
    engine = make_engine(present=False)
    engine.setup()
    assert mujoco.mj_name2id(engine.ctx.model, mujoco.mjtObj.mjOBJ_BODY, "box") >= 0


def test_a_disabled_entry_builds_no_body_at_all(prop_file):
    """The contrast that gives the two keys separate jobs."""
    cfg = load_config_from_dict(
        {"sim": {}, "plugins": [{CRATE: {"model": prop_file}, "name": "crate", "enabled": False}]}
    )
    assert instantiate_plugins(cfg) == []


def test_a_declared_spare_is_a_spare_again_after_a_reset(make_engine):
    """Presence is a ``model`` field and ``mj_resetData`` restores ``data``, so without the
    plugin re-applying it, every repetition after the first would begin with the obstacle already
    in the room."""
    engine = make_engine(present=False)
    engine.setup()
    entity = _entity(engine)

    set_present(engine.ctx, entity, True)  # what a scenario's SpawnEntity does
    assert entity.present is True

    engine.reset()
    assert entity.present is False


def test_an_absent_prop_does_not_fall_through_the_run(make_engine):
    """The declaration inherits the freeze: a spare waiting in a 240 s trial is still at its
    declared pose when the trial calls for it."""
    engine = make_engine(present=False)
    engine.setup()
    engine.reset()
    jid = mujoco.mj_name2id(engine.ctx.model, mujoco.mjtObj.mjOBJ_JOINT, "free")
    adr = int(engine.ctx.model.jnt_qposadr[jid])
    before = float(engine.ctx.data.qpos[adr + 2])
    for _ in range(2000):
        engine.step()
    assert float(engine.ctx.data.qpos[adr + 2]) == pytest.approx(before, abs=1e-3)


def test_present_must_be_a_bool(prop_file):
    """``present: "false"`` is truthy, so a string would silently mean the opposite."""
    cfg = load_config_from_dict(
        {"sim": {}, "plugins": [{CRATE: {"model": prop_file, "present": "false"}, "name": "c"}]}
    )
    with pytest.raises(Exception, match="'present' must be true or false"):
        instantiate_plugins(cfg)
