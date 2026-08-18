"""contact_monitor: the substrate's contact observable.

Written to reconstruct a local-trajectory-planning comparison whose only trial failure criterion is
a collision. The load-bearing
behaviour is the pair of negatives: a robot standing and driving on the floor must NOT report a
collision, and a wheel touching a wall must.
"""

from __future__ import annotations

import mujoco
import pytest

from roqsim.context import Entity, SimContext
from roqsim.plugins.contact_monitor import ContactMonitorPlugin

# A two-body "robot": a chassis with a free joint plus a child wheel, so the subtree walk is
# exercised. A wall sits ahead of it; the floor is underneath.
SCENE = """
<mujoco model="contact_monitor_test">
  <worldbody>
    <geom name="floor" type="plane" size="10 10 0.05"/>
    <geom name="wall" type="box" size="0.1 2 0.5" pos="{wall_x} 0 0.5"/>
    <body name="base_link" pos="0 0 0.2">
      <freejoint/>
      <geom name="chassis" type="box" size="0.2 0.15 0.1" mass="10"/>
      <body name="wheel" pos="0.25 0 -0.1">
        <joint name="wheel_joint" type="hinge" axis="0 1 0"/>
        <geom name="wheel_geom" type="sphere" size="0.1" mass="1"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


def _build(wall_x=5.0):
    model = mujoco.MjModel.from_xml_string(SCENE.format(wall_x=wall_x))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def _plugin(model, data, **cfg):
    ctx = SimContext(config={})
    ctx.model, ctx.data = model, data
    ctx.entities.add(
        Entity(name="robot", kind="robot", body="base_link", meta={"prefix": "", "namespace": ""})
    )
    plugin = ContactMonitorPlugin({"robot": "robot", **cfg})
    plugin.configure(ctx)
    plugin.on_reset(ctx)
    return ctx, plugin


def _settle(ctx, plugin, seconds, push=0.0):
    for _ in range(int(seconds / ctx.model.opt.timestep)):
        if push:
            ctx.data.qvel[0] = push
        mujoco.mj_step(ctx.model, ctx.data)
        plugin.post_step(ctx)
    endpoint = next(e for e in ctx.interface.all() if e.name == "contact")
    return endpoint.read()


def test_watches_the_whole_subtree():
    """The wheel is watched too -- a wheel clipping a box is as much a collision as the bumper."""
    model, data = _build()
    _, plugin = _plugin(model, data)
    watched = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) for g in plugin._watched}
    assert watched == {"chassis", "wheel_geom"}


def test_resting_on_the_floor_is_not_a_collision():
    """The negative that matters most: a wheeled robot is in permanent floor contact by design."""
    model, data = _build()
    ctx, plugin = _plugin(model, data)
    report = _settle(ctx, plugin, 2.0)
    assert report.in_contact is False
    assert report.first_time == -1.0
    assert data.ncon > 0, "the test is vacuous unless the robot is actually touching the floor"


def test_hitting_a_wall_is_a_collision_and_is_attributable():
    model, data = _build(wall_x=1.0)
    ctx, plugin = _plugin(model, data)
    report = _settle(ctx, plugin, 3.0, push=1.0)
    assert report.in_contact is True
    assert report.first_time > 0.0
    assert "wall" in (report.geom_a, report.geom_b)


def test_latch_holds_a_failed_trial_failed():
    """A trial that touched something stays failed even after the robot bounces off."""
    model, data = _build(wall_x=1.0)
    ctx, plugin = _plugin(model, data, latch=True)
    _settle(ctx, plugin, 3.0, push=1.0)
    report = _settle(ctx, plugin, 2.0, push=-1.0)  # reverse away from the wall
    assert report.in_contact is True
    assert report.count == 0, "no CURRENT contacts, but the trial is still failed"


def test_latch_off_reports_only_current_contacts():
    model, data = _build(wall_x=1.0)
    ctx, plugin = _plugin(model, data, latch=False)
    assert _settle(ctx, plugin, 3.0, push=1.0).in_contact is True
    assert _settle(ctx, plugin, 2.0, push=-1.0).in_contact is False


def test_on_reset_clears_the_report():
    model, data = _build(wall_x=1.0)
    ctx, plugin = _plugin(model, data)
    assert _settle(ctx, plugin, 3.0, push=1.0).in_contact is True
    plugin.on_reset(ctx)
    endpoint = next(e for e in ctx.interface.all() if e.name == "contact")
    assert endpoint.read().in_contact is False
    assert endpoint.read().first_time == -1.0


def test_missing_body_fails_loudly():
    """A monitor watching nothing would pass every trial. That must never be silent."""
    model, data = _build()
    ctx = SimContext(config={})
    ctx.model, ctx.data = model, data
    ctx.entities.add(Entity(name="robot", kind="robot", body="nope", meta={"prefix": ""}))
    with pytest.raises(RuntimeError, match="not found"):
        ContactMonitorPlugin({"robot": "robot"}).configure(ctx)


def test_min_force_filters_grazing_contacts():
    """A huge min_force must suppress even a real wall hit -- proves the filter is wired in."""
    model, data = _build(wall_x=1.0)
    ctx, plugin = _plugin(model, data, min_force=1e6)
    assert _settle(ctx, plugin, 3.0, push=1.0).in_contact is False
