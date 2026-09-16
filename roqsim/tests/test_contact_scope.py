"""contact_scope: the one rule an entity's contact observables share.

What is load-bearing here is not any single plugin's behaviour but that there is only one rule:
``contact_monitor`` and ``contact_impulse`` resolve the SAME masks from the same config, so a
verdict and a severity can never be about different contacts. The rest is the failure the resolver
exists to make loud -- a body that does not resolve, a subtree with nothing to watch, an ignore
entry that matches no geom -- because each of them otherwise leaves a meter reporting a clean run
forever.
"""

from __future__ import annotations

import logging

import mujoco
import numpy as np
import pytest

from roqsim.contact_scope import resolve_contact_scope
from roqsim.context import Entity, SimContext
from roqsim.plugins.contact_impulse import ContactImpulsePlugin
from roqsim.plugins.contact_monitor import ContactMonitorPlugin

SCENE = """
<mujoco model="contact_scope">
  <worldbody>
    <geom name="floor" type="plane" size="10 10 0.05"/>
    <geom name="ground_strip" type="box" size="1 1 0.01" pos="3 0 0.01"/>
    <geom name="wall" type="box" size="0.1 2 0.5" pos="1.0 0 0.5"/>
    <body name="marker" pos="0 3 0"/>
    <body name="base_link" pos="0 0 0.2">
      <freejoint name="base_free"/>
      <geom name="chassis" type="box" size="0.2 0.15 0.1" mass="10"/>
      <body name="wheel" pos="0.25 0 -0.1">
        <joint name="wheel_joint" type="hinge" axis="0 1 0"/>
        <geom name="wheel_geom" type="sphere" size="0.1" mass="1"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


def _model():
    model = mujoco.MjModel.from_xml_string(SCENE)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def _entity(body="base_link"):
    return Entity(name="robot", kind="robot", body=body, meta={"prefix": "", "namespace": ""})


def _gid(model, name):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)


def _names(model, mask):
    return {
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(g)) for g in np.flatnonzero(mask)
    }


# -- the rule -------------------------------------------------------------------------------------


def test_the_watched_mask_is_the_whole_subtree():
    model, _ = _model()
    scope = resolve_contact_scope(model, _entity(), plugin="test")
    assert _names(model, scope.watched) == {"chassis", "wheel_geom"}


def test_only_one_side_watched_qualifies():
    """Neither side is another pair's business; both sides is a self-contact."""
    model, _ = _model()
    scope = resolve_contact_scope(model, _entity(), plugin="test", ignore=[])
    chassis, wheel, wall = (_gid(model, n) for n in ("chassis", "wheel_geom", "wall"))
    geom1 = np.array([chassis, chassis, wall])
    geom2 = np.array([wall, wheel, wall])
    assert list(scope.qualifying(geom1, geom2)) == [True, False, False]


def test_an_ignored_geom_takes_its_contacts_out():
    model, _ = _model()
    scope = resolve_contact_scope(model, _entity(), plugin="test", ignore=["wall"])
    chassis, wall, floor = (_gid(model, n) for n in ("chassis", "wall", "floor"))
    assert list(scope.qualifying(np.array([chassis, chassis]), np.array([wall, floor]))) == [
        False,
        True,  # `floor` is only the default, and this scope was given its own list
    ]


def test_ignore_prefixes_take_a_family_out():
    model, _ = _model()
    scope = resolve_contact_scope(model, _entity(), plugin="test", ignore_prefixes=["ground"])
    assert _names(model, scope.ignored) == {"floor", "ground_strip"}


def test_indices_are_ascending_and_empty_without_contacts():
    """A caller reading the FIRST contact of a step depends on the order MuJoCo listed them in."""
    model, data = _model()
    scope = resolve_contact_scope(model, _entity(), plugin="test")
    assert list(scope.indices(data)) == []  # nothing has touched yet at t=0

    for _ in range(300):  # settle onto the floor, which this scope ignores
        mujoco.mj_step(model, data)
    data.qvel[0] = 3.0
    hit = []
    for _ in range(600):
        mujoco.mj_step(model, data)
        idx = scope.indices(data)
        assert list(idx) == sorted(idx)
        hit.extend(int(i) for i in idx)
    assert hit, "the robot was driven into the wall and something should have qualified"


# -- the two plugins are one rule ------------------------------------------------------------------


def test_the_two_observables_resolve_the_same_scope():
    """The guarantee: a verdict and a severity are about the same contacts, by construction."""
    model, data = _model()
    ctx = SimContext(config={})
    ctx.model, ctx.data = model, data
    ctx.entities.add(_entity())
    cfg = {"ignore": ["floor"], "ignore_prefixes": ["ground"]}

    monitor = ContactMonitorPlugin(dict(cfg), entity="robot")
    impulse = ContactImpulsePlugin(dict(cfg), entity="robot")
    monitor.configure(ctx)
    impulse.configure(ctx)

    assert monitor._scope.body == impulse._scope.body
    assert np.array_equal(monitor._scope.watched, impulse._scope.watched)
    assert np.array_equal(monitor._scope.ignored, impulse._scope.ignored)


# -- what it refuses to do quietly -----------------------------------------------------------------


def test_a_body_that_does_not_resolve_fails_loudly():
    """Silently watching nothing would report a clean run forever -- and pass every trial."""
    model, _ = _model()
    with pytest.raises(RuntimeError, match="contact_monitor: base body 'nope' not found"):
        resolve_contact_scope(model, _entity(body="nope"), plugin="contact_monitor")


def test_a_subtree_without_geoms_fails_loudly():
    """A body that resolves and carries nothing is the same blindness with a valid name."""
    model, _ = _model()
    with pytest.raises(RuntimeError, match="carry no geoms to watch"):
        resolve_contact_scope(model, _entity(body="marker"), plugin="test")


def test_an_ignore_entry_matching_nothing_is_warned_about(caplog):
    """How a ground plane starts counting as a collision: a renamed floor geom, silently unmatched."""
    model, _ = _model()
    with caplog.at_level(logging.WARNING, logger="roqsim.contact_scope"):
        resolve_contact_scope(model, _entity(), plugin="test", ignore=["floor", "carpet"])
    assert "carpet" in caplog.text
    assert "floor" not in caplog.text.split("no matching geom:")[-1]
