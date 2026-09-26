"""A contact with a flex is attributed to the flex, never to the model's last geom.

MuJoCo gives a flex's side of a contact ``geom = -1`` and the flex's id in ``contact.flex``. Every
per-geom mask indexed with that ``-1`` reads the model's LAST geom, so before the contact scope knew
about flexes, a soft body resting on a crate was a collision of whatever geom happened to be
compiled last. The scene below makes that geom a hovering robot that touches nothing, so the
misattribution and its absence are both observable.

Each test measures the consequence through the consumer that carries it -- contact_monitor's
verdict, contact_impulse's load, contact_location's position, a recording's contact rows, the SRDF
sampler, model_override's verification, contact_pair_override's refusal -- rather than reading the
masks.
"""

from __future__ import annotations

import json
import logging
import re

import mujoco
import numpy as np
import pytest

from roqsim.config import load_config_from_dict
from roqsim.contact_scope import resolve_contact_scope, side_name
from roqsim.context import Entity, SimContext
from roqsim.engine import Engine
from roqsim.export_srdf import collision_matrix
from roqsim.flex import entity_flex_ids
from roqsim.plugin import PluginError
from roqsim.plugins.contact_impulse import ContactImpulsePlugin
from roqsim.plugins.contact_location import ContactLocationPlugin
from roqsim.plugins.contact_monitor import ContactMonitorPlugin
from roqsim.plugins.model_override import ModelOverridePlugin
from roqsim.state import contact_rows

BLOB_MASS = 0.1  # kg, the whole flex
CRATE_TOP = 0.1  # m

#: How a flex side is named: the vertex that touched, or the element (a solid flex meets a box
#: through its elements and a plane through its vertices).
FLEX_SIDE = re.compile(r"flex:blob\[[ve]\d+\]")

# A soft cube dropped onto a static crate. `robot` holds the model's LAST geom and hovers two metres
# away, touching nothing: a flex side read as geom -1 lands on it.
SCENE = f"""
<mujoco model="contact_flex">
  <option timestep="0.002"/>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.05"/>
    <body name="crate" pos="0 0 0.05">
      <geom name="crate" type="box" size="0.1 0.1 0.05" priority="{{crate_priority}}"/>
    </body>
    <body name="soft" pos="0 0 0.2">
      <flexcomp name="blob" type="grid" count="3 3 3" spacing=".03 .03 .03" radius=".005"
                dim="3" mass="{BLOB_MASS}">
        <contact selfcollide="none" internal="false" friction="{{blob_friction}}"
                 priority="{{blob_priority}}"/>
        <edge equality="true"/>
      </flexcomp>
    </body>
    <body name="robot" pos="2 0 0.5">
      <geom name="chassis" type="box" size="0.1 0.1 0.1"/>
    </body>
  </worldbody>
</mujoco>
"""


def _model(crate_priority=0, blob_friction=1.0, blob_priority=0):
    model = mujoco.MjModel.from_xml_string(
        SCENE.format(
            crate_priority=crate_priority, blob_friction=blob_friction, blob_priority=blob_priority
        )
    )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def _ctx(model, data):
    ctx = SimContext(config={})
    ctx.model, ctx.data = model, data
    for name in ("robot", "soft", "crate"):
        ctx.entities.add(
            Entity(name=name, kind="object", body=name, meta={"prefix": "", "namespace": ""})
        )
    return ctx


def _settle(ctx, plugins=(), seconds=1.0):
    """Drop the blob onto the crate and let it come to rest, stepping the plugins as the engine does."""
    for _ in range(int(seconds / ctx.model.opt.timestep)):
        mujoco.mj_step(ctx.model, ctx.data)
        for plugin in plugins:
            plugin.post_step(ctx)


def _flex_contacts(data) -> np.ndarray:
    n = int(data.ncon)
    return np.flatnonzero((data.contact.geom[:n] < 0).any(axis=1))


def _configured(cls, ctx, entity, **cfg):
    plugin = cls(dict(cfg), entity=entity)
    plugin.configure(ctx)
    plugin.on_reset(ctx)
    return plugin


# -- the regression: never the last geom ----------------------------------------------------------


def test_a_flex_contact_is_not_a_contact_of_the_last_geom():
    ctx = _ctx(*_model())
    model, data = ctx.model, ctx.data
    assert mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, model.ngeom - 1) == "chassis"
    monitor = _configured(ContactMonitorPlugin, ctx, "robot", ignore=[], min_force=0.0)
    impulse = _configured(ContactImpulsePlugin, ctx, "robot", ignore=[])
    _settle(ctx, [monitor, impulse])

    flex = _flex_contacts(data)
    assert flex.size, "the blob should be resting on the crate"
    # The precondition that made this a bug: the per-geom rule, fed MuJoCo's geom ids as they are,
    # reads every one of those contacts as the robot's.
    watched = monitor._scope.watched
    geom = data.contact.geom[flex]
    assert (watched[geom[:, 0]] ^ watched[geom[:, 1]]).all()

    assert list(monitor._scope.indices(data)) == []
    assert monitor.read_state().in_contact is False
    assert impulse.read().impulse_ns == 0.0


# -- a flex is a side, and its entity's ----------------------------------------------------------


def test_an_entity_that_is_only_a_flex_owns_it():
    model, _ = _model()
    assert entity_flex_ids(model, "soft") == [0]
    assert entity_flex_ids(model, "world") == [0]
    assert entity_flex_ids(model, "robot") == []
    scope = resolve_contact_scope(
        model, Entity(name="soft", kind="object", body="soft"), plugin="t"
    )
    assert not scope.watched.any()
    assert list(scope.watched_flex) == [True]


def test_a_flex_touching_a_geom_is_its_entitys_collision():
    ctx = _ctx(*_model())
    monitor = _configured(ContactMonitorPlugin, ctx, "soft", min_force=0.0)
    impulse = _configured(ContactImpulsePlugin, ctx, "soft")
    _settle(ctx, [monitor, impulse])

    report = monitor.read_state()
    assert report.in_contact is True
    assert report.geom_a == "crate"
    assert FLEX_SIDE.fullmatch(report.geom_b), report.geom_b

    # At rest the crate carries the blob's weight, and nothing else touches it.
    load = impulse.read()
    assert load.normal_n == pytest.approx(BLOB_MASS * 9.81, rel=0.1)
    assert FLEX_SIDE.fullmatch(load.peak_geom_b), load.peak_geom_b


def test_the_other_side_of_a_flex_contact_counts_it_too():
    """The crate is touched BY the blob: the flex is the external side, so the crate collided."""
    ctx = _ctx(*_model())
    monitor = _configured(ContactMonitorPlugin, ctx, "crate", min_force=0.0)
    _settle(ctx, [monitor])
    assert monitor.read_state().in_contact is True


def test_ignore_may_name_a_flex(caplog):
    ctx = _ctx(*_model())
    with caplog.at_level(logging.WARNING, logger="roqsim.contact_scope"):
        monitor = _configured(
            ContactMonitorPlugin, ctx, "crate", ignore=["floor", "blob"], min_force=0.0
        )
    assert "no matching" not in caplog.text
    _settle(ctx, [monitor])
    assert _flex_contacts(ctx.data).size
    assert monitor.read_state().in_contact is False


def test_a_flex_touching_itself_is_not_an_external_contact():
    model, _ = _model()
    scope = resolve_contact_scope(
        model, Entity(name="soft", kind="object", body="soft"), plugin="t"
    )
    crate = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "crate")
    geom = np.array([[-1, -1], [crate, -1]])
    flex = np.array([[0, 0], [-1, 0]])
    assert list(scope.qualifying(geom, flex)) == [False, True]


def test_a_side_naming_neither_a_geom_nor_a_flex_is_refused():
    model, _ = _model()
    scope = resolve_contact_scope(
        model, Entity(name="soft", kind="object", body="soft"), plugin="t"
    )
    with pytest.raises(RuntimeError, match="neither a geom nor a flex"):
        scope.qualifying(np.array([[0, -1]]), np.array([[-1, -1]]))
    with pytest.raises(ValueError, match="neither a geom nor a flex"):
        side_name(model, -1, -1)


def test_side_names():
    model, _ = _model()
    assert side_name(model, 1, -1) == "crate"
    assert side_name(model, -1, 0, 4) == "flex:blob[v4]"
    assert side_name(model, -1, 0, -1, 7) == "flex:blob[e7]"


# -- where, on a flex -----------------------------------------------------------------------------


def test_contact_location_reports_where_a_flex_rests():
    ctx = _ctx(*_model())
    # Per-vertex forces are a fraction of the blob's 1 N weight, so no force threshold.
    location = _configured(ContactLocationPlugin, ctx, "soft", frame="world", min_force=0.0)
    _settle(ctx, [location])
    reading = location.read_state()
    assert reading.in_contact is True
    assert reading.kind == "line"
    assert reading.z == pytest.approx(CRATE_TOP, abs=0.01)
    assert abs(reading.x) < 0.02 and abs(reading.y) < 0.02
    assert reading.extent > 0.03  # the blob's footprint, not one vertex


# -- a recording's contact rows -------------------------------------------------------------------


def test_contact_rows_name_the_flex_side():
    ctx = _ctx(*_model())
    _settle(ctx)
    rows = contact_rows(ctx.model, ctx.data)
    flex_rows = [r for r in rows if r["flex2"] == "blob"]
    assert flex_rows, rows
    for row in flex_rows:
        assert row["geom1"] == "crate" and row["flex1"] is None
        assert row["geom2"] is None
        # A vertex or an element, and exactly one of them.
        (index,) = [row[k] for k in ("vert2", "elem2") if row[k] is not None]
        assert isinstance(index, int) and index >= 0
        assert row["vert1"] is None and row["elem1"] is None
    assert sum(r["force.normal"] for r in flex_rows) == pytest.approx(BLOB_MASS * 9.81, rel=0.1)
    json.dumps(rows)  # the record is printed as JSON


# -- consumers that must not read -1 as a geom ---------------------------------------------------


def test_the_srdf_sampler_does_not_read_a_flex_as_a_link(monkeypatch):
    """A flex resting on link_a while link_far holds the last geom: they must not become 'Always'.

    The sampler runs kinematics and collision only, which leaves a flex's vertices unpositioned and
    so out of every contact. Its collision pass is given the flex here (``mj_flex`` first), which is
    the case the guard is for: a flex side is skipped, not read as the last geom's link.
    """
    collide = mujoco.mj_collision

    def with_flexes(model, data):
        mujoco.mj_flex(model, data)
        collide(model, data)

    monkeypatch.setattr(mujoco, "mj_collision", with_flexes)
    model = mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <worldbody>
            <body name="link_a"><geom name="g_a" type="box" size="0.1 0.1 0.1"/></body>
            <body name="soft" pos="0 0 0.1">
              <flexcomp name="blob" type="grid" count="2 2 2" spacing=".05 .05 .05"
                        radius=".01" dim="3" mass="0.1"/>
            </body>
            <body name="link_far" pos="5 0 0"><geom name="g_far" type="box" size="0.1 0.1 0.1"/></body>
          </worldbody>
        </mujoco>
        """
    )
    data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, data)
    with_flexes(model, data)
    assert _flex_contacts(data).size, "the flex should overlap link_a"
    links = {
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n): n for n in ("link_a", "link_far")
    }
    matrix = {(a, b): r for a, b, r in collision_matrix(model, links, samples=5)}
    assert matrix[("link_a", "link_far")] == "Never"


def test_model_override_names_a_flex_that_governs_the_friction(caplog):
    """The blob outranks the crate, so zeroing the crate's friction changes no contact -- and the
    report says which side governs, as a flex rather than as the robot's chassis."""
    ctx = _ctx(*_model(blob_friction=0.9, blob_priority=1))
    _settle(ctx)
    plugin = ModelOverridePlugin(
        {"overrides": [{"field": "geom_friction", "select": ["crate"], "to": 0.0}]},
        name="slip",
    )
    plugin.configure(ctx)
    plugin.on_reset(ctx)
    with caplog.at_level(logging.WARNING):
        plugin.set_active(True)
        for _ in range(20):
            ctx.drain_commands()
            mujoco.mj_step(ctx.model, ctx.data)
            plugin.post_step(ctx)
    assert plugin.read_state().verified == "no_effect"
    assert FLEX_SIDE.search(caplog.text), caplog.text
    assert "chassis" not in caplog.text
    assert "flex's own friction" in caplog.text


# -- a <pair> cannot name a flex ------------------------------------------------------------------


def _pair_engine(tmp_path, pair):
    world = tmp_path / "w.xml"
    world.write_text(SCENE.format(crate_priority=0, blob_friction=1.0, blob_priority=0))
    return Engine(
        load_config_from_dict(
            {
                "sim": {"pacing": "asap", "world": str(world)},
                "components": [{"contact_pair_override": pair, "name": "pair"}],
            },
            base_dir=tmp_path,
        )
    )


@pytest.mark.parametrize(
    "side",
    [{"geom": "blob"}, {"body": "soft"}, {"body": "world"}],
    ids=["geom-names-a-flex", "body-owns-a-flex", "subtree-owns-a-flex"],
)
def test_a_pair_side_that_is_a_flex_is_refused(tmp_path, side):
    engine = _pair_engine(tmp_path, {"a": side, "b": {"geom": "floor"}, "friction": 0.3})
    engine.ctx.seed = 0
    with pytest.raises((RuntimeError, PluginError), match="flex's own friction and priority"):
        engine.setup()


def test_a_pair_beside_a_flex_is_still_declared(tmp_path):
    engine = _pair_engine(
        tmp_path, {"a": {"body": "crate"}, "b": {"geom": "floor"}, "friction": 0.3}
    )
    engine.ctx.seed = 0
    engine.setup()
    assert engine.ctx.model.npair == 1
