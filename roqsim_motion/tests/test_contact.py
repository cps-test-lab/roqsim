"""Holds: welds added at build, switched while the run goes on, read for their force and slip."""

from __future__ import annotations

import mujoco
import numpy as np
from roqsim_motion.contact import Hold, add_holds, check_solref, contact_force, touching

MJCF = """<mujoco><option timestep="0.002"/>
<worldbody>
  <geom name="floor" type="plane" size="5 5 .1"/>
  <body name="hand" mocap="true" pos="0 0 -50"><site name="hand/grip"/></body>
  <body name="box" pos="0.5 0 0.1"><freejoint/><geom type="box" size="0.15 0.1 0.1" mass="3"/>
    <site name="box/grip_l" pos="0 0.13 0.08"/></body>
</worldbody></mujoco>"""


def _model():
    spec = mujoco.MjSpec.from_string(MJCF)
    names = add_holds(spec, ["hand/grip"], ["box/grip_l"])
    return spec.compile(), names


def test_a_hold_is_inactive_until_switched_and_carries_by_physics_then_lets_go():
    m, names = _model()
    assert names == ["hold:hand/grip->box/grip_l"]
    d = mujoco.MjData(m)
    hold = Hold.find(m, "hand/grip", "box/grip_l")
    assert not hold.active(d) and m.neq == 1
    for _ in range(100):
        mujoco.mj_step(m, d)
    d.mocap_pos[0] = d.site("box/grip_l").xpos
    mujoco.mj_forward(m, d)
    hold.set(d, True)
    z0 = d.body("box").xpos[2]
    for _ in range(1000):
        d.mocap_pos[0][2] += 0.0003
        mujoco.mj_step(m, d)
    assert d.body("box").xpos[2] - z0 > 0.25
    assert hold.separation(d) < 0.02
    assert 10.0 < hold.force(d) < 200.0  # carrying 3 kg, about its weight, not a spike
    hold.set(d, False)
    for _ in range(500):
        mujoco.mj_step(m, d)
    assert d.body("box").xpos[2] < 0.11
    assert hold.force(d) == 0.0


def test_a_hold_stiffer_than_the_step_allows_is_refused():
    m, _ = _model()
    assert check_solref(m.opt) == ""
    assert "floor" in check_solref(m.opt, (0.001, 1.0))


def test_contact_force_and_touching_read_a_body():
    m, _ = _model()
    d = mujoco.MjData(m)
    for _ in range(300):
        mujoco.mj_step(m, d)
    box = m.body("box").id
    assert 20.0 < contact_force(m, d, box) < 40.0  # 3 kg standing on the floor
    assert touching(m, d, box, ignore=set())
    assert not touching(m, d, box, ignore={0})  # the floor is the world body's
    assert np.isfinite(contact_force(m, d, box, exclude={0}))


def test_holds_are_added_on_sites_added_after_a_compile_made_while_building():
    """A plugin may compile the spec mid-build (the ceiling's removal measures the world so);
    ``MjSpec.site`` then misses sites added after it, and holds on them are added all the same."""
    spec = mujoco.MjSpec()
    floor = spec.worldbody.add_body(name="floor")
    floor.add_geom(size=[1.0, 1.0, 0.1])
    floor.add_site(name="floor/mark")
    spec.compile()
    hand = spec.worldbody.add_body(name="hand", mocap=True)
    hand.add_site(name="hand/grip")
    box = spec.worldbody.add_body(name="box")
    box.add_freejoint()
    box.add_geom(size=[0.1, 0.1, 0.1])
    box.add_site(name="box/grip")
    assert spec.site("box/grip") is None  # the lookup this works round
    (name,) = add_holds(spec, ["hand/grip"], ["box/grip"])
    model = spec.compile()
    eq = model.equality(name).id
    assert model.body(int(model.eq_obj1id[eq])).name == "box"
    assert model.body(int(model.eq_obj2id[eq])).name == "hand"


def test_a_humanoids_press_force_is_its_press_max_force():
    """One name for the cap a press pushes at most: ``press: {max_force}`` in the world, as a
    press move's ``max_force``; the world key ``press_force`` is refused naming it."""
    from roqsim_motion.moves import HumanoidPlugin

    assert HumanoidPlugin.validate_schema({"press": {"max_force": 25.0}}) == []
    (error,) = HumanoidPlugin.validate_schema({"press_force": 25.0})
    assert "'press_force' is not a setting" in error and "press: {max_force: <N>}" in error
