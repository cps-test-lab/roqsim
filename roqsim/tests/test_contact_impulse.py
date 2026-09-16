"""contact_impulse: the substrate's contact-severity observable.

The load-bearing behaviour is the oracle a collision supplies for nothing: the normal impulse a
wall delivers to a body that touches nothing else IS that body's change in momentum, so the
integral is checked against ``m * dv`` rather than against another integral of the same numbers.
Beside it sit the two the plugin's shape depends on -- that it counts exactly what
``contact_monitor`` counts, and that a collision is over before a publish period is.

Two scenes, because the two questions want different bodies. The sliding block is one degree of
freedom against a wall, which is what makes the momentum oracle exact. The chassis-and-wheel robot
is the one ``contact_monitor``'s own tests use, so the subtree walk, the floor and the agreement
between the two plugins are exercised over one geometry.
"""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from roqsim.context import Entity, SimContext
from roqsim.plugins.contact_impulse import ContactImpulsePlugin
from roqsim.plugins.contact_monitor import ContactMonitorPlugin

#: One block, one axis, one wall, and no gravity: everything the block's momentum changes by came
#: from the wall, so `m * dv` is the impulse without anything having to measure it.
SLIDE_SCENE = """
<mujoco model="contact_impulse_slide">
  <option timestep="0.002" gravity="0 0 0"/>
  <worldbody>
    <geom name="floor" type="plane" size="10 10 0.05"/>
    <geom name="wall" type="box" size="0.1 2 0.5" pos="1.0 0 0.5"/>
    <body name="base_link" pos="0 0 0.5">
      <joint name="base_slide" type="slide" axis="1 0 0"/>
      <geom name="chassis" type="box" size="0.2 0.15 0.1" mass="10"/>
    </body>
  </worldbody>
</mujoco>
"""

#: A two-body "robot": a chassis with a free joint plus a child wheel, so the subtree walk is
#: exercised. A wall sits ahead of it; the floor is underneath.
ROBOT_SCENE = """
<mujoco model="contact_impulse_robot">
  <worldbody>
    <geom name="floor" type="plane" size="10 10 0.05"/>
    <geom name="wall" type="box" size="0.1 2 0.5" pos="{wall_x} 0 0.5"/>
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

#: The sliding block's mass, from the MJCF above. What the momentum oracle is written against.
BLOCK_MASS = 10.0


def _build(xml):
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def _slide():
    return _build(SLIDE_SCENE)


def _robot(wall_x=5.0):
    return _build(ROBOT_SCENE.format(wall_x=wall_x))


def _ctx(model, data):
    ctx = SimContext(config={})
    ctx.model, ctx.data = model, data
    ctx.entities.add(
        Entity(
            name="robot",
            kind="robot",
            body="base_link",
            meta={"prefix": "", "namespace": "", "base_joint": "base_free"},
        )
    )
    return ctx


def _plugin(model, data, **cfg):
    ctx = _ctx(model, data)
    plugin = ContactImpulsePlugin(dict(cfg), entity="robot")
    plugin.configure(ctx)
    plugin.on_reset(ctx)
    return ctx, plugin


def _endpoint(ctx):
    return next(e for e in ctx.interface.all() if e.name == "contact_impulse")


def _run(ctx, plugins, seconds, push=None):
    """Step for ``seconds``, driving every plugin's ``post_step`` as the engine does.

    ``push`` is applied ONCE, at the start: a body nothing holds at a velocity is free from the
    first step on, which is what makes its change in momentum the impulse it was given.
    """
    if push is not None:
        ctx.data.qvel[0] = push
    for _ in range(int(seconds / ctx.model.opt.timestep)):
        mujoco.mj_step(ctx.model, ctx.data)
        for plugin in plugins:
            plugin.post_step(ctx)


def _force_series(ctx, plugin, seconds, push):
    """The integrand, step by step: what a consumer would have to sample to integrate downstream.

    Returns the per-step normal force and the per-step qualifying-contact count side by side; the
    two do not switch on and off together, and that difference is a documented property.
    """
    ctx.data.qvel[0] = push
    force, counted = [], []
    for _ in range(int(seconds / ctx.model.opt.timestep)):
        mujoco.mj_step(ctx.model, ctx.data)
        plugin.post_step(ctx)
        force.append(plugin.read().normal_n)
        counted.append(plugin.read().count)
    return np.asarray(force), np.asarray(counted)


# -- the measurement ----------------------------------------------------------------------------


@pytest.mark.parametrize("speed", [0.5, 1.0, 2.0])
def test_the_impulse_is_the_momentum_the_wall_took_out(speed):
    """The oracle: a free body's normal impulse IS its change in momentum.

    One sliding degree of freedom and no gravity, so the wall is the only thing that can change the
    block's momentum, and nothing else can carry any away.
    """
    model, data = _slide()
    ctx, plugin = _plugin(model, data)
    _run(ctx, [plugin], 2.0, push=speed)

    report = _endpoint(ctx).read()
    momentum_change = BLOCK_MASS * (speed - float(data.qvel[0]))
    assert report.impulse_ns == pytest.approx(momentum_change, rel=1e-6)
    assert report.peak_normal_n > 0.0
    assert report.peak_time > 0.0


def test_a_harder_hit_is_a_larger_impulse():
    """Severity, which is the whole point: it orders two collisions a bit reports identically."""
    impulses = []
    for speed in (0.5, 2.0):
        model, data = _slide()
        ctx, plugin = _plugin(model, data)
        _run(ctx, [plugin], 2.0, push=speed)
        impulses.append(_endpoint(ctx).read().impulse_ns)
    slow, fast = impulses
    assert fast == pytest.approx(4.0 * slow, rel=1e-3), "four times the momentum to take out"


def test_the_peak_force_names_the_geoms_it_was_against():
    model, data = _slide()
    ctx, plugin = _plugin(model, data)
    _run(ctx, [plugin], 2.0, push=1.0)

    report = _endpoint(ctx).read()
    assert "wall" in (report.peak_geom_a, report.peak_geom_b)
    assert report.peak_normal_n >= report.normal_n


def test_the_contact_time_is_the_time_a_qualifying_contact_existed():
    """The monitor's notion of touching, not a second one: every step it counts a contact is a step
    this adds to the total. That is longer than the force was nonzero, because MuJoCo goes on
    listing a pair while the two still overlap on the way apart."""
    model, data = _slide()
    ctx, plugin = _plugin(model, data)
    force, counted = _force_series(ctx, plugin, 2.0, push=1.0)

    steps = int((counted > 0).sum())
    loaded = int((force > 0.0).sum())
    assert 0 < loaded < steps, "the pair outlives the force it transmits"
    assert plugin.read().contact_time_s == pytest.approx(steps * model.opt.timestep, rel=1e-9)


def test_nothing_touched_reads_a_measured_zero():
    """The wall is out of reach and the floor is ignored: zero impulse, and no time in contact."""
    model, data = _robot(wall_x=5.0)
    ctx, plugin = _plugin(model, data)
    _run(ctx, [plugin], 2.0)

    report = _endpoint(ctx).read()
    assert report.impulse_ns == 0.0
    assert report.peak_normal_n == 0.0
    assert report.contact_time_s == 0.0
    assert report.normal_n == 0.0
    assert report.count == 0
    assert report.peak_time == -1.0
    assert report.peak_geom_a == ""
    assert data.ncon > 0, "the test is vacuous unless the robot is actually touching the floor"


# -- a collision is shorter than a publish period ------------------------------------------------


def test_a_collision_is_shorter_than_the_period_it_would_be_published_at():
    """Why the integral is accumulated here rather than sampled and integrated downstream.

    The contact is tens of milliseconds, so at the endpoint's default rate a single sample lands
    inside it: one point on a force that rose and fell between two publications. What that one
    sample happens to catch is where in the pulse it fell, so an estimate built from the published
    series is wrong by a large factor and in either direction -- while the integral is exact.
    """
    errors = []
    for speed in (0.5, 1.0, 2.0):
        model, data = _slide()
        ctx, plugin = _plugin(model, data)
        series, _ = _force_series(ctx, plugin, 2.0, push=speed)
        dt = model.opt.timestep

        touching = int((series > 0.0).sum())
        assert touching * dt < 1.0 / 20.0, "a free collision is tens of milliseconds"

        stride = int(round((1.0 / plugin.rate_hz) / dt))
        published = series[::stride]
        assert int((published > 0.0).sum()) <= 2, "at most two samples land inside the contact"

        # What a consumer integrating the published series would get.
        estimate = float(published.sum()) * stride * dt
        true = plugin.read().impulse_ns
        errors.append(abs(estimate - true) / true)

    assert max(errors) > 0.5, "the published series does not carry the area under the pulse"


def test_the_publish_rate_does_not_change_the_integral():
    """The corollary: ``rate_hz`` decides when the total leaves, never what it is."""
    totals = []
    for rate in (1.0, 200.0):
        model, data = _slide()
        ctx, plugin = _plugin(model, data, rate_hz=rate)
        _run(ctx, [plugin], 2.0, push=1.0)
        totals.append(_endpoint(ctx).read().impulse_ns)
    assert totals[0] == pytest.approx(totals[1])


# -- it counts what contact_monitor counts -------------------------------------------------------


def test_it_counts_exactly_what_the_contact_monitor_counts():
    """Two observables over one geometry, step for step.

    The monitor runs at ``min_force: 0`` because its force filter is the one rule of its own this
    plugin deliberately does not share.
    """
    model, data = _robot(wall_x=1.0)
    ctx = _ctx(model, data)
    impulse = ContactImpulsePlugin({}, entity="robot")
    monitor = ContactMonitorPlugin({"min_force": 0.0, "latch": False}, entity="robot")
    for plugin in (impulse, monitor):
        plugin.configure(ctx)
        plugin.on_reset(ctx)

    for _ in range(int(3.0 / model.opt.timestep)):
        data.qvel[0] = 1.0  # driven, so the wall is reached over a floor that has friction
        mujoco.mj_step(model, data)
        impulse.post_step(ctx)
        monitor.post_step(ctx)
        assert impulse.read().count == monitor.read_state().count

    assert monitor.read_state().first_time > 0.0
    assert impulse.read().impulse_ns > 0.0


def test_the_floor_is_ignored_the_same_way():
    """The default ``ignore`` is the monitor's: a wheeled robot rests on the ground by design, and
    billing it for its own weight makes standing still the heaviest contact of the trial."""
    model, data = _robot(wall_x=5.0)
    ctx, plugin = _plugin(model, data)
    _run(ctx, [plugin], 2.0)
    assert plugin.read().impulse_ns == 0.0

    model, data = _robot(wall_x=5.0)
    ctx, counting_the_floor = _plugin(model, data, ignore=[])
    _run(ctx, [counting_the_floor], 2.0)
    assert counting_the_floor.read().impulse_ns > 0.0


def test_ignore_prefixes_take_a_family_of_geoms_out():
    model, data = _robot(wall_x=1.0)
    ctx, plugin = _plugin(model, data, ignore=[], ignore_prefixes=["wal", "flo"])
    for _ in range(int(3.0 / model.opt.timestep)):
        data.qvel[0] = 1.0
        mujoco.mj_step(model, data)
        plugin.post_step(ctx)
    assert plugin.read().impulse_ns == 0.0
    assert plugin.read().count == 0


def test_the_whole_subtree_is_watched():
    """A wheel clipping a box is as much a collision as the bumper, and as much of an impulse."""
    model, data = _robot()
    _, plugin = _plugin(model, data)
    watched = {
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(g))
        for g in np.flatnonzero(plugin._watched)
    }
    assert watched == {"chassis", "wheel_geom"}


# -- lifecycle -----------------------------------------------------------------------------------


def test_on_reset_clears_the_integral():
    """One process serves several trials; a leaked integral puts the first trial's collisions on
    the second one's bill."""
    model, data = _slide()
    ctx, plugin = _plugin(model, data)
    _run(ctx, [plugin], 2.0, push=1.0)
    assert _endpoint(ctx).read().impulse_ns > 0.0

    plugin.on_reset(ctx)
    report = _endpoint(ctx).read()
    assert report.impulse_ns == 0.0
    assert report.peak_normal_n == 0.0
    assert report.contact_time_s == 0.0
    assert report.peak_time == -1.0
    assert report.peak_geom_a == ""


def test_a_contact_after_a_reset_accumulates_again():
    """Clearing is not disarming, and the second collision is measured as its own."""
    model, data = _slide()
    ctx, plugin = _plugin(model, data)
    _run(ctx, [plugin], 2.0, push=1.0)
    plugin.on_reset(ctx)

    before = float(data.qvel[0])
    _run(ctx, [plugin], 2.0, push=1.0)  # driven back into the wall it bounced off
    assert _endpoint(ctx).read().impulse_ns == pytest.approx(
        BLOCK_MASS * (1.0 - float(data.qvel[0])), rel=1e-6
    )
    assert before < 0.0, "precondition: it had bounced away before being pushed back"


def test_spawning_the_watched_entity_restarts_the_integral():
    """A re-spawned entity carries no history -- ``contact_monitor``'s rule, so that neither plugin
    keeps a contact the other has forgotten."""
    model, data = _robot(wall_x=0.25)  # compiled inside the wall
    ctx, plugin = _plugin(model, data)
    _run(ctx, [plugin], 0.05)
    assert plugin.read().impulse_ns > 0.0, "precondition: it was loaded touching the wall"

    _respawn_clear_of_the_wall(ctx, plugin)

    report = plugin.read()
    assert report.impulse_ns == 0.0
    assert report.contact_time_s == 0.0
    assert report.peak_time == -1.0


def test_reset_on_spawn_can_be_turned_off():
    """For a re-spawned entity that should carry its history forward."""
    model, data = _robot(wall_x=0.25)
    ctx, plugin = _plugin(model, data, reset_on_spawn=False)
    _run(ctx, [plugin], 0.05)
    carried = plugin.read().impulse_ns
    assert carried > 0.0

    _respawn_clear_of_the_wall(ctx, plugin)
    assert plugin.read().impulse_ns == pytest.approx(carried)


def _respawn_clear_of_the_wall(ctx, plugin):
    """Absent, moved, present again -- what a trial spawning a robot into position does."""
    from roqsim.placement import place_body
    from roqsim.presence import set_present

    robot = ctx.entities.get("robot")
    for present in (False, True):
        set_present(ctx, robot, present)
        place_body(ctx, robot, (-5.0, 0.0, 0.2), (1.0, 0.0, 0.0, 0.0))
        mujoco.mj_step(ctx.model, ctx.data)
        plugin.post_step(ctx)


# -- what it refuses -----------------------------------------------------------------------------


def test_a_force_threshold_is_refused_rather_than_ignored():
    """The calibration constant an impulse metric exists to avoid. Accepted and dropped, a block
    copied from ``contact_monitor`` would read as though grazing had been filtered out."""
    errors = ContactImpulsePlugin({}).validate_config({"min_force": 1.0})
    assert any("min_force" in e for e in errors)


def test_a_publish_rate_of_zero_is_refused():
    assert any("rate_hz" in e for e in ContactImpulsePlugin({}).validate_config({"rate_hz": 0.0}))


def test_missing_body_fails_loudly():
    """A meter watching nothing reports a zero impulse forever, which reads exactly like a trial
    that touched nothing gently -- and would be averaged in as one."""
    model, data = _robot()
    ctx = SimContext(config={})
    ctx.model, ctx.data = model, data
    ctx.entities.add(Entity(name="robot", kind="robot", body="nope", meta={"prefix": ""}))
    with pytest.raises(RuntimeError, match="not found"):
        ContactImpulsePlugin({}, entity="robot").configure(ctx)


def test_declared_at_the_top_of_a_document_it_is_refused():
    """It watches an entity, so there is nothing for it to watch at the top of a document."""
    from roqsim.config import PluginError, load_config_from_dict

    doc = {
        "sim": {"world": "empty_room"},
        "components": [
            {"roqsim.plugins.contact_impulse:ContactImpulsePlugin": {"ignore": ["floor"]}}
        ],
    }
    with pytest.raises(PluginError, match="nested"):
        load_config_from_dict(doc)


def test_two_meters_get_two_handles():
    """Keyed on the address, not on ``name`` -- which defaults to the CLASS name, so two unnamed
    instances would write to one key and report one robot's impulse as another's."""
    model, data = _robot()
    ctx = _ctx(model, data)
    ctx.entities.add(
        Entity(name="other", kind="robot", body="base_link", meta={"prefix": "", "namespace": ""})
    )
    for entity in ("robot", "other"):
        ContactImpulsePlugin({"ignore": ["floor"]}, entity=entity).configure(ctx)

    assert ctx.blackboard.get("contact_impulse:robot.ContactImpulsePlugin") is not None
    assert ctx.blackboard.get("contact_impulse:other.ContactImpulsePlugin") is not None
