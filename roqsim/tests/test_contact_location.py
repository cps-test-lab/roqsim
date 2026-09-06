"""contact_location: WHERE the robot is being touched, every step.

The sibling of test_contact_monitor. What that one checks is a pair of negatives (the floor is not a
collision, a wall is); what this one checks is a pair of POSITIONS, because a contact sensor that
fires at the right time and reports the wrong place is the failure a green run hides -- a controller
steering away from a contact that is not there.

Written for a tactile pushing strategy whose control law reads the contact location every step, so
the reading has to be current and in the robot's own frame rather than latched and in the world's.
"""

from __future__ import annotations

import math

import mujoco
import pytest

from roqsim.context import Entity, SimContext
from roqsim.plugins.contact_location import ContactLocationPlugin

# A box "robot" between two walls it can be pushed into, so a contact can be produced on a KNOWN
# side at a KNOWN height. The chassis is 0.4 x 0.3 x 0.2 m centred at z = 0.2.
SCENE = """
<mujoco model="contact_location_test">
  <worldbody>
    <geom name="floor" type="plane" size="10 10 0.05"/>
    <geom name="wall_front" type="box" size="0.1 2 0.5" pos="{front_x} 0 0.5"/>
    <geom name="post_left" type="cylinder" size="0.05 0.5" pos="0 {left_y} 0.5"/>
    <body name="base_link" pos="0 0 0.2">
      <freejoint/>
      <geom name="chassis" type="box" size="0.2 0.15 0.1" mass="10"/>
    </body>
  </worldbody>
</mujoco>
"""


def _build(front_x=5.0, left_y=5.0):
    model = mujoco.MjModel.from_xml_string(SCENE.format(front_x=front_x, left_y=left_y))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def _plugin(model, data, **cfg):
    ctx = SimContext(config={})
    ctx.model, ctx.data = model, data
    ctx.entities.add(
        Entity(name="robot", kind="robot", body="base_link", meta={"prefix": "", "namespace": ""})
    )
    plugin = ContactLocationPlugin(dict(cfg), entity="robot")
    plugin.configure(ctx)
    plugin.on_reset(ctx)
    return ctx, plugin


def _drive(ctx, plugin, seconds, vx=0.0, vy=0.0):
    for _ in range(int(seconds / ctx.model.opt.timestep)):
        ctx.data.qvel[0] = vx
        ctx.data.qvel[1] = vy
        mujoco.mj_step(ctx.model, ctx.data)
        plugin.post_step(ctx)
    return plugin.read_state()


def test_standing_on_the_floor_reports_no_contact():
    """The ignored ground plane is the one surface a wheeled robot touches by design."""
    ctx, plugin = _plugin(*_build())
    r = _drive(ctx, plugin, 0.5)
    assert r.in_contact is False
    assert r.kind == "none"
    assert r.count == 0


def test_a_frontal_contact_is_reported_in_front_of_the_robot():
    """The load-bearing check: the SIGN and rough magnitude of the reported position.

    The wall sits ahead on +x; the chassis half-length is 0.2 m, so a contact against it must land
    near x = +0.2 in the base frame, and at y ~ 0. A plugin reporting the world position instead
    would give x ~ 0.6 here, and one reporting the body origin would give 0.0 -- both are contacts
    at the right TIME and the wrong PLACE, which is what a bumper controller would steer on.
    """
    ctx, plugin = _plugin(*_build(front_x=0.6))
    r = _drive(ctx, plugin, 2.0, vx=0.4)
    assert r.in_contact is True
    assert r.x == pytest.approx(0.2, abs=0.05), f"contact should be on the front face, got {r.x}"
    assert abs(r.y) < 0.1, f"a head-on contact should be centred in y, got {r.y}"


def test_a_side_contact_is_reported_on_that_side():
    """Same check on the other axis, which is what distinguishes 'in front' from 'to my left'."""
    ctx, plugin = _plugin(*_build(left_y=0.5))
    r = _drive(ctx, plugin, 2.0, vy=0.4)
    assert r.in_contact is True
    assert r.y == pytest.approx(0.15, abs=0.06), f"contact should be on the left face, got {r.y}"
    assert abs(r.x) < 0.12, f"a square-on side contact should be centred in x, got {r.x}"


def test_the_base_frame_reading_does_not_move_with_the_robot():
    """The reason `frame: base` is the default: a controller's rule ("contact on my left") must not
    depend on where in the world the robot happens to be standing."""
    near = _drive(*_plugin(*_build(front_x=0.6)), 2.0, vx=0.4)
    far = _drive(*_plugin(*_build(front_x=3.6)), 6.0, vx=0.6)
    assert far.in_contact is True
    assert far.x == pytest.approx(near.x, abs=0.05)
    # ...and in the world frame it DOES move, which is what makes the two settings different.
    ctx, plugin = _plugin(*_build(front_x=3.6), frame="world")
    world = _drive(ctx, plugin, 6.0, vx=0.6)
    assert world.x > 3.0, f"the world-frame reading should be out at the wall, got {world.x}"


def test_a_flat_face_contact_is_one_line_not_many_points():
    """MuJoCo solves several contacts across two flat faces meeting. A 2 cm-taxel skin reports that
    as one region; counting each solver contact separately would make every flat push look like a
    dozen sensing areas. `merge_radius` is what collapses them, and `extent` is what survives."""
    ctx, plugin = _plugin(*_build(front_x=0.6), merge_radius=0.02)
    r = _drive(ctx, plugin, 2.0, vx=0.4)
    assert r.kind == "line", "a box face against a wall is a region, not a point"
    assert r.count >= 2
    # The front face is 0.3 m wide and 0.2 m tall, and `extent` is a 3-D distance, so the widest
    # separation two corner contacts on it can have is that face's DIAGONAL, not its width:
    # sqrt(0.3^2 + 0.2^2) = 0.361 m. A value above that would mean contacts on two different faces
    # were being averaged into one region.
    face_diagonal = math.hypot(0.3, 0.2)
    assert 0.0 < r.extent <= face_diagonal + 1e-3, f"extent {r.extent} exceeds the front face"


def test_merge_radius_wide_enough_collapses_the_face_to_one_area():
    """The same contact through a coarser skin: one sensing area, hence a point."""
    ctx, plugin = _plugin(*_build(front_x=0.6), merge_radius=1.0)
    r = _drive(ctx, plugin, 2.0, vx=0.4)
    assert r.count == 1
    assert r.kind == "point"
    assert r.extent == 0.0


def test_the_reading_is_current_not_latched():
    """The whole reason this is not contact_monitor: that one LATCHES a failed trial, this one must
    go back to 'none' the moment the robot stops touching anything, or a controller keeps steering
    away from a contact that ended."""
    ctx, plugin = _plugin(*_build(front_x=0.6))
    assert _drive(ctx, plugin, 2.0, vx=0.4).in_contact is True
    assert _drive(ctx, plugin, 2.0, vx=-0.4).in_contact is False


def test_missing_body_fails_loudly():
    """A sensor watching nothing reports 'no contact' forever, which a controller cannot tell from
    open space -- so it must not be reachable by a typo."""
    model, data = _build()
    ctx = SimContext(config={})
    ctx.model, ctx.data = model, data
    ctx.entities.add(Entity(name="robot", kind="robot", body="nope", meta={}))
    with pytest.raises(RuntimeError, match="not found"):
        ContactLocationPlugin({}, entity="robot").configure(ctx)


def test_declared_at_the_top_of_a_document_it_is_refused():
    """Ownership is position: at the top of a document there is no entity to read."""
    assert ContactLocationPlugin.requires_owner is True


def test_two_sensors_get_two_handles():
    """Keyed on the address, so two watched bodies do not overwrite each other's reading."""
    model, data = _build()
    ctx = SimContext(config={})
    ctx.model, ctx.data = model, data
    ctx.entities.add(
        Entity(name="robot", kind="robot", body="base_link", meta={"prefix": "", "namespace": ""})
    )
    a = ContactLocationPlugin({}, name="skin_a", entity="robot")
    b = ContactLocationPlugin({}, name="skin_b", entity="robot")
    a.configure(ctx)
    b.configure(ctx)
    assert ctx.blackboard.get(f"contact_location:{a.address}") is not None
    assert ctx.blackboard.get(f"contact_location:{b.address}") is not None
    assert a.address != b.address, "two named sensors must not share one blackboard key"


def test_min_force_filters_grazing_contacts():
    """Same guard as contact_monitor's: a numerically-touching pair is not a tactile reading."""
    ctx, plugin = _plugin(*_build(front_x=0.6), min_force=1e9)
    r = _drive(ctx, plugin, 2.0, vx=0.4)
    assert r.in_contact is False


def test_extent_is_a_distance_not_a_coordinate():
    """Guards a plausible mis-implementation: reporting the spread as a signed span would let it
    come out negative, and a consumer scaling a control gain by it would reverse."""
    ctx, plugin = _plugin(*_build(front_x=0.6))
    r = _drive(ctx, plugin, 2.0, vx=0.4)
    assert r.extent >= 0.0
    assert not math.isnan(r.extent)


# -- cost -----------------------------------------------------------------------------------------

CROWDED = """
<mujoco model="contact_location_crowd">
  <worldbody>
    <geom name="floor" type="plane" size="30 30 0.05"/>
    <geom name="wall_front" type="box" size="0.1 2 0.5" pos="0.6 0 0.5"/>
    <body name="base_link" pos="0 0 0.2">
      <freejoint/>
      <geom name="chassis" type="box" size="0.2 0.15 0.1" mass="10"/>
    </body>
    {props}
  </worldbody>
</mujoco>
"""


def _crowded(n_props=60):
    """A scene whose contacts are overwhelmingly NOT the robot's: props resting on the floor."""
    props = "".join(
        f'<body name="p{i}" pos="{-6 + (i % 10) * 0.5} {-3 + (i // 10) * 0.5} 0.06">'
        f'<freejoint/><geom name="pg{i}" type="box" size="0.2 0.2 0.05" mass="1"/></body>'
        for i in range(n_props)
    )
    model = mujoco.MjModel.from_xml_string(CROWDED.format(props=props))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def test_a_contact_force_is_queried_only_for_the_robots_own_contacts(monkeypatch):
    """The scan must cost the robot's contacts, not the world's.

    A trial's contacts are overwhelmingly pairs the robot is not in -- props on the floor, a
    crowd's feet -- and `mj_contactForce` is a C call per contact. Querying every one of them made
    this plugin cost more than the physics step that produced the contacts. The geom filter runs
    first, over the whole contact array at once, and only survivors are priced.
    """
    ctx, plugin = _plugin(*_crowded())
    _drive(ctx, plugin, 1.0)  # let the props settle so the world is full of contacts
    calls = []
    real = mujoco.mj_contactForce
    monkeypatch.setattr(
        mujoco, "mj_contactForce", lambda m, d, i, r: (calls.append(i), real(m, d, i, r))[1]
    )
    plugin.post_step(ctx)
    assert ctx.data.ncon > 50, f"the fixture should be contact-heavy, got {ctx.data.ncon}"
    # The robot touches only the floor here, which is ignored, so nothing should be priced at all.
    assert len(calls) <= 4, f"queried {len(calls)} contact forces out of {ctx.data.ncon}"


def test_the_reading_is_correct_in_a_contact_heavy_world(monkeypatch):
    """The filter must not be a shortcut that loses the robot's own contact among the crowd."""
    ctx, plugin = _plugin(*_crowded())
    r = _drive(ctx, plugin, 3.0, vx=0.4)
    assert ctx.data.ncon > 50
    assert r.in_contact is True, "the wall contact was lost among the props' contacts"
    assert r.x == pytest.approx(0.2, abs=0.05)


def test_compute_rate_hz_decimates_the_whole_computation():
    """The escape hatch for a world where even the filtered scan is too much.

    Counted by how often the reading is refreshed, not by what it says: decimation is about how
    much work is done, and a reading that happens to be the same either way would hide it. Default
    is 0 (every step), because unlike a latching monitor this plugin cannot recover a contact it
    did not look at -- one shorter than the interval is simply missed, and that trade is the
    caller's to make rather than a default imposed on them.
    """

    def refreshes(seconds, **cfg):
        ctx, plugin = _plugin(*_build(front_x=0.6), **cfg)
        stamps = set()
        for _ in range(int(seconds / ctx.model.opt.timestep)):
            ctx.data.qvel[0] = 0.4
            mujoco.mj_step(ctx.model, ctx.data)
            plugin.post_step(ctx)
            stamps.add(plugin.read_state().time)
        return len(stamps), plugin.read_state()

    every_step, r_full = refreshes(2.0)
    decimated, r_dec = refreshes(2.0, compute_rate_hz=10.0)

    assert every_step > 900, f"undecimated should refresh once per step, got {every_step}"
    assert decimated <= 25, f"10 Hz over 2 s should refresh ~20 times, got {decimated}"
    # Both still see the contact: decimating trades resolution in time, not the signal itself.
    assert r_full.in_contact is True
    assert r_dec.in_contact is True
    assert r_dec.x == pytest.approx(r_full.x, abs=0.05)
