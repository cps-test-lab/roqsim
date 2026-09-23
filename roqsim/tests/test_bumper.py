"""bumper: WHICH ZONE of a shell is being pushed, every step.

The third contact observable over one geometry. contact_monitor says whether, contact_location
says where as a point; this says which switch. What a bumper-consuming safety stack is graded on is
the zone NAME, so these tests drive a box into an obstacle at known bearings and check that the
right zone -- and only that zone -- reads pressed, that it releases, and that a contact outside
every declared zone presses nothing.
"""

from __future__ import annotations

import math

import mujoco
import pytest

from roqsim.context import Entity, SimContext
from roqsim.plugins.bumper import BumperPlugin, _in_sector

# A box "robot" with an obstacle that can be placed at any bearing. The chassis is 0.4 x 0.3 x 0.2 m
# and settles on the floor with its centre at z = 0.1; the roof rack on top of it is what a contact
# that is NOT on the shell lands on.
SCENE = """
<mujoco model="bumper_test">
  <worldbody>
    <geom name="floor" type="plane" size="10 10 0.05"/>
    <geom name="post" type="box" size="{sx} {sy} 0.5" pos="{px} {py} 0.5"/>
    <body name="base_link" pos="0 0 0.2">
      <freejoint/>
      <geom name="chassis" type="box" size="0.2 0.15 0.1" mass="10"/>
      <geom name="roof_rack" type="box" size="0.05 0.05 0.02" pos="0 0 0.12" mass="0.1"/>
    </body>
  </worldbody>
</mujoco>
"""

# The Create 3's five zones, as its simulator zones them: bearings counter-clockwise from +x.
ZONES = {
    "bump_right": [-math.pi / 2, -3 * math.pi / 10],
    "bump_front_right": [-3 * math.pi / 10, -math.pi / 10],
    "bump_front_center": [-math.pi / 10, math.pi / 10],
    "bump_front_left": [math.pi / 10, 3 * math.pi / 10],
    "bump_left": [3 * math.pi / 10, math.pi / 2],
}


def _build(px=5.0, py=5.0, sx=0.03, sy=0.03):
    model = mujoco.MjModel.from_xml_string(SCENE.format(px=px, py=py, sx=sx, sy=sy))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def _plugin(model, data, **cfg):
    ctx = SimContext(config={})
    ctx.model, ctx.data = model, data
    ctx.entities.add(
        Entity(name="robot", kind="robot", body="base_link", meta={"prefix": "", "namespace": ""})
    )
    cfg.setdefault("zones", ZONES)
    plugin = BumperPlugin(dict(cfg), entity="robot")
    assert plugin.validate_config(dict(cfg)) == []
    plugin.configure(ctx)
    plugin.on_reset(ctx)
    return ctx, plugin


def _drive(ctx, plugin, seconds, vx=0.0, vy=0.0):
    """Drive at a fixed velocity; the LAST reading (what a stack sees after the motion)."""
    return _drive_seen(ctx, plugin, seconds, vx, vy)[0]


def _drive_seen(ctx, plugin, seconds, vx=0.0, vy=0.0):
    """The last reading and every zone pressed at ANY step of the drive.

    A body shoved into an obstacle at a fixed velocity slides off it, so the zone a contact pressed
    is read while it lasted rather than after the body has scraped past."""
    seen: set[str] = set()
    for _ in range(int(seconds / ctx.model.opt.timestep)):
        ctx.data.qvel[0] = vx
        ctx.data.qvel[1] = vy
        # Held square: the bearing of a contact is only well-defined for a body that has not
        # been spun round by the very contact under test.
        ctx.data.qvel[3:6] = 0.0
        mujoco.mj_step(ctx.model, ctx.data)
        plugin.post_step(ctx)
        seen |= _pressed(plugin.read_state())
    return plugin.read_state(), seen


def _pressed(reading) -> set[str]:
    return {z for z, p in reading.pressed.items() if p}


def test_nothing_pressed_on_the_floor():
    """The ignored ground plane is the one surface a wheeled robot touches by design."""
    r = _drive(*_plugin(*_build()), 0.5)
    assert r.any_pressed is False
    assert _pressed(r) == set()


def test_a_head_on_contact_presses_the_centre_zone_only():
    """The load-bearing check: the ZONE, not merely the fact of contact. A head-on post lands at
    bearing 0, inside bump_front_center and no other."""
    ctx, plugin = _plugin(*_build(px=0.5, py=0.0))
    r, seen = _drive_seen(ctx, plugin, 2.0, vx=0.4)
    assert _pressed(r) == {"bump_front_center"}
    assert seen == {"bump_front_center"}


def test_an_offset_contact_presses_the_matching_side_zone():
    """A post ahead and to the left: the contact lands on the front face at y ~ 0.12, bearing
    atan2(0.12, 0.2) ~ 0.54 rad, inside bump_front_left [0.31, 0.94] -- and the body then scrapes
    past it, so the zone is read while the contact lasts."""
    ctx, plugin = _plugin(*_build(px=0.45, py=0.12))
    _, seen = _drive_seen(ctx, plugin, 2.0, vx=0.4)
    assert "bump_front_left" in seen
    assert "bump_front_right" not in seen
    assert "bump_right" not in seen


def test_a_side_contact_presses_the_side_zone():
    """Driving sideways into a post on the right: bearing atan2(-0.15, 0.05) ~ -1.25, in bump_right."""
    ctx, plugin = _plugin(*_build(px=0.05, py=-0.4))
    _, seen = _drive_seen(ctx, plugin, 2.0, vy=-0.4)
    assert "bump_right" in seen
    assert "bump_left" not in seen
    assert "bump_front_center" not in seen


def test_a_contact_behind_a_front_bumper_presses_nothing():
    """A rear collision is a collision (contact_monitor's business) but presses no front switch."""
    ctx, plugin = _plugin(*_build(px=-0.5, py=0.0))
    r = _drive(ctx, plugin, 2.0, vx=-0.4)
    assert r.any_pressed is False


def test_the_reading_releases_when_the_robot_backs_off():
    """Not latched: a bumper that stayed pressed after the robot backed away would hold the base's
    safety stack in its stopped state forever."""
    ctx, plugin = _plugin(*_build(px=0.5, py=0.0))
    assert _drive(ctx, plugin, 2.0, vx=0.4).any_pressed is True
    assert _drive(ctx, plugin, 2.0, vx=-0.4).any_pressed is False


def test_a_wrapping_rear_zone_is_pressed_from_behind():
    """A sector with from > to wraps through +/-pi: that is how a rear zone is spelled."""
    zones = {"rear": [2.6, -2.6]}
    ctx, plugin = _plugin(*_build(px=-0.5, py=0.0), zones=zones)
    r = _drive(ctx, plugin, 2.0, vx=-0.4)
    assert _pressed(r) == {"rear"}


def test_geoms_restrict_the_shell_to_the_named_geoms():
    """A real bumper is a shell. A beam at roof-rack height (above the chassis top at z = 0.2)
    presses no switch when the shell is named -- and does when it is not, because the default is
    the whole subtree."""
    scene = SCENE.format(px=5.0, py=5.0, sx=0.03, sy=0.03).replace(
        '<geom name="post" type="box" size="0.03 0.03 0.5" pos="5.0 5.0 0.5"/>',
        '<geom name="beam" type="box" size="0.05 2 0.0425" pos="0.6 0 0.2575"/>',
    )
    model = mujoco.MjModel.from_xml_string(scene)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    ctx, plugin = _plugin(model, data, geoms=["chassis"])
    _, seen = _drive_seen(ctx, plugin, 2.0, vx=0.4)
    assert seen == set(), "the beam only touches the roof rack, which is not the shell"

    model = mujoco.MjModel.from_xml_string(scene)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    ctx, plugin = _plugin(model, data)
    _, seen = _drive_seen(ctx, plugin, 2.0, vx=0.4)
    assert seen, "without a named shell the whole subtree is the bumper"


def test_a_named_shell_geom_that_does_not_exist_fails_loudly():
    model, data = _build()
    ctx = SimContext(config={})
    ctx.model, ctx.data = model, data
    ctx.entities.add(Entity(name="robot", kind="robot", body="base_link", meta={}))
    with pytest.raises(RuntimeError, match="not found"):
        BumperPlugin({"zones": ZONES, "geoms": ["shell"]}, entity="robot").configure(ctx)


def test_missing_body_fails_loudly():
    model, data = _build()
    ctx = SimContext(config={})
    ctx.model, ctx.data = model, data
    ctx.entities.add(Entity(name="robot", kind="robot", body="nope", meta={}))
    with pytest.raises(RuntimeError, match="not found"):
        BumperPlugin({"zones": ZONES}, entity="robot").configure(ctx)


def test_one_endpoint_per_zone_and_a_blackboard_handle():
    """Each zone is its own bool endpoint (a switch each), lazy, under bumper/<zone>."""
    ctx, plugin = _plugin(*_build())
    names = {e.name for e in ctx.interface.all()}
    assert names == {f"bumper/{z}" for z in ZONES}
    assert all(e.lazy for e in ctx.interface.all())
    assert all(e.backend["ros2"]["type"] == "std_msgs.msg.Bool" for e in ctx.interface.all())
    assert ctx.blackboard.get(f"bumper:{plugin.address}")() is plugin.read_state()


def test_endpoints_read_the_current_zone_state():
    ctx, plugin = _plugin(*_build(px=0.5, py=0.0))
    _drive(ctx, plugin, 2.0, vx=0.4)
    by_name = {e.name: e for e in ctx.interface.all()}
    assert by_name["bumper/bump_front_center"].read() is True
    assert by_name["bumper/bump_left"].read() is False


def test_min_force_filters_grazing_contacts():
    ctx, plugin = _plugin(*_build(px=0.5, py=0.0), min_force=1e9)
    assert _drive(ctx, plugin, 2.0, vx=0.4).any_pressed is False


def test_declared_at_the_top_of_a_document_it_is_refused():
    assert BumperPlugin.requires_owner is True


@pytest.mark.parametrize(
    "bad",
    [
        {},
        {"zones": {}},
        {"zones": {"a": [0.0]}},
        {"zones": {"a": [0.0, 0.0]}},
        {"zones": {"a": [0.0, 4.0]}},
        {"zones": {"a/b": [0.0, 1.0]}},
        {"zones": ZONES, "rate_hz": 0},
        {"zones": ZONES, "geoms": "chassis"},
    ],
)
def test_bad_config_is_reported(bad):
    assert BumperPlugin(bad, entity="robot").validate_config(bad) != []


@pytest.mark.parametrize(
    "bearing, lo, hi, inside",
    [
        (0.0, -0.3, 0.3, True),
        (0.5, -0.3, 0.3, False),
        (3.0, 2.6, -2.6, True),
        (-3.0, 2.6, -2.6, True),
        (0.0, 2.6, -2.6, False),
    ],
)
def test_sector_membership(bearing, lo, hi, inside):
    assert _in_sector(bearing, lo, hi) is inside
