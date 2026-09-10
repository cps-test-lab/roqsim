"""upright_monitor: did a body a trial drives in the plane stop being in the plane?

The scene is the body the failure happens to: 1.7 m of cylinder on a free joint, standing on a
floor with friction. Drive such a thing at its centre of mass and the drive force at the middle
and the floor's friction at the base make a couple, so it tips -- correct physics about a model
that was wrong, and before this the run carried on reporting positions and distances about a
pedestrian lying on its side.

The load-bearing behaviour is the pair of negatives: a body driving along the floor, and one
turning on the spot, must NOT be reported as fallen.
"""

from __future__ import annotations

import math

import mujoco
import pytest

from roqsim.context import Entity, SimContext
from roqsim.plugins.upright_monitor import UprightMonitorPlugin

#: A pedestrian as one is easily and wrongly modelled: 1.7 m of cylinder on a free joint.
SCENE = """
<mujoco model="upright_monitor_test">
  <option timestep="0.002"/>
  <worldbody>
    <geom name="floor" type="plane" size="20 20 0.05" friction="1 0.005 0.0001"/>
    <body name="base_link" pos="0 0 0.85">
      <freejoint name="base_free"/>
      <geom name="torso" type="cylinder" size="0.2 0.85" mass="70" friction="1 0.005 0.0001"/>
    </body>
  </worldbody>
</mujoco>
"""


def _ctx():
    model = mujoco.MjModel.from_xml_string(SCENE)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    ctx = SimContext(config={})
    ctx.model, ctx.data = model, data
    ctx.entities.add(
        Entity(name="walker", kind="robot", body="base_link",
               meta={"prefix": "", "namespace": "", "base_joint": "base_free"})
    )
    return ctx


def _plugin(ctx, **cfg):
    """A configured monitor. ``settle_s`` defaults to 0 here, not to the plugin's 0.5.

    These tests put the body in the state they are about and then step briefly, so the plugin's
    grace window would swallow every one of them. The window itself is what the two settling tests
    below are for, and they ask for it explicitly.
    """
    cfg.setdefault("settle_s", 0.0)
    plugin = UprightMonitorPlugin(dict(cfg), entity="walker")
    plugin.configure(ctx)
    plugin.on_reset(ctx)
    return plugin


def _drive(ctx, plugin, seconds, *, vx=0.0, wz=0.0, lift=0.0):
    """Step the world holding the whole planar twist on the free joint: this body IS in the plane.

    The full twist and not just ``vx``: holding the linear velocity alone still leaves the base's
    friction free to tip the cylinder -- which is the very failure this plugin reports, and would
    make a "must not be flagged" test measure the opposite of what it claims. Held on every axis,
    the body is kinematically planar by construction, so a finding here would be a false positive.
    """
    for _ in range(int(seconds / ctx.model.opt.timestep)):
        ctx.data.qvel[:6] = [vx, 0.0, lift, 0.0, 0.0, wz]
        mujoco.mj_step(ctx.model, ctx.data)
        plugin.post_step(ctx)
    return plugin.read_state()


def _run(ctx, plugin, seconds):
    """Step the world and let the solver do whatever it does."""
    for _ in range(int(seconds / ctx.model.opt.timestep)):
        mujoco.mj_step(ctx.model, ctx.data)
        plugin.post_step(ctx)
    return plugin.read_state()


def test_a_body_driving_along_the_floor_is_upright():
    """The negative that matters most: an ordinary trial must produce no finding at all."""
    ctx = _ctx()
    plugin = _plugin(ctx)
    report = _drive(ctx, plugin, 3.0, vx=1.2)

    assert report.upright is True
    assert report.first_time == -1.0
    assert report.reason == ""


def test_turning_on_the_spot_is_not_leaving_the_plane():
    """Tilt is measured against world +z, so yaw is invisible to it -- which is the point.

    A pedestrian that turns to walk the other way is doing exactly what a planar trial expects,
    and a monitor built on Euler angles would have to pick a convention to avoid calling it a fall.
    """
    ctx = _ctx()
    plugin = _plugin(ctx)
    report = _drive(ctx, plugin, 3.0, wz=2.0)

    assert report.upright is True
    assert report.tilt_deg < 1.0


def test_a_toppled_body_is_reported_with_the_time_and_the_reason():
    """The finding the ticket asked for, and the reason names which mistake to look for."""
    ctx = _ctx()
    plugin = _plugin(ctx)
    # Laid over by hand rather than tipped by a couple: this file is about the MONITOR, and
    # reproducing the tip would be a test of MuJoCo's contact solver.
    ctx.data.qpos[3:7] = [math.cos(math.pi / 4), math.sin(math.pi / 4), 0.0, 0.0]  # 90 deg about x
    mujoco.mj_forward(ctx.model, ctx.data)

    report = _run(ctx, plugin, 0.1)

    assert report.upright is False
    assert report.reason == "tilt"
    assert report.first_time >= 0.0
    assert report.worst_tilt_deg > 80.0


def test_a_body_that_leaves_the_ground_is_reported_as_height():
    """Airborne is out of the plane too, and it points somewhere else than a tilt does."""
    ctx = _ctx()
    plugin = _plugin(ctx, max_rise_m=0.2)
    report = _drive(ctx, plugin, 1.0, lift=2.0)

    assert report.upright is False
    assert report.reason == "height"
    assert report.worst_rise_m > 0.2


def test_sinking_counts_as_much_as_rising():
    """A run where the ground gave way is no more usable than one where the walker flew.

    The symmetry is stated because the obvious implementation compares ``z - reference`` against a
    positive bound and silently passes every body falling through the floor.
    """
    ctx = _ctx()
    plugin = _plugin(ctx, max_rise_m=0.2)
    report = _drive(ctx, plugin, 1.0, lift=-2.0)

    assert report.upright is False
    assert report.reason == "height"
    assert report.worst_rise_m < -0.2


def test_the_reference_height_is_where_it_settled_not_where_it_was_spawned():
    """A body spawned a little above the floor drops onto it, and that is not a departure.

    Measuring against the spawn pose would flag every world whose author did not place a body to
    the millimetre -- a monitor that fires on correct worlds is one people turn off.
    """
    ctx = _ctx()
    ctx.data.qpos[2] = 1.05  # 20 cm of drop, twice the default tolerance
    mujoco.mj_forward(ctx.model, ctx.data)
    plugin = _plugin(ctx, settle_s=0.5)

    report = _run(ctx, plugin, 2.0)

    assert report.upright is True, "settling onto the floor is not leaving the plane"


def test_settle_zero_judges_from_the_first_step():
    """The escape hatch, and the proof that the grace window is what buys the test above.

    A trial that starts in contact, or one whose first instant is the measurement, opts out -- and
    then the same 20 cm drop IS a departure, because nothing said to wait for it.
    """
    ctx = _ctx()
    ctx.data.qpos[2] = 1.05
    mujoco.mj_forward(ctx.model, ctx.data)
    plugin = _plugin(ctx, settle_s=0.0)

    assert _run(ctx, plugin, 2.0).upright is False


def test_a_reset_forgets_the_verdict_and_the_reference():
    """A repetition compares against its own start, like every other per-episode state."""
    ctx = _ctx()
    plugin = _plugin(ctx)
    ctx.data.qpos[3:7] = [math.cos(math.pi / 4), math.sin(math.pi / 4), 0.0, 0.0]
    mujoco.mj_forward(ctx.model, ctx.data)
    assert _run(ctx, plugin, 0.1).upright is False

    mujoco.mj_resetData(ctx.model, ctx.data)
    mujoco.mj_forward(ctx.model, ctx.data)
    plugin.on_reset(ctx)

    assert plugin.read_state().upright is True
    assert _run(ctx, plugin, 1.0).upright is True


def test_a_fallen_body_stays_fallen():
    """A trial is failed, not un-failed -- the rule contact_monitor already follows.

    A pedestrian that topples and rolls back upright spent the run's middle on the floor, and the
    positions recorded there are no better for its having got up.
    """
    ctx = _ctx()
    plugin = _plugin(ctx)
    ctx.data.qpos[3:7] = [math.cos(math.pi / 4), math.sin(math.pi / 4), 0.0, 0.0]
    mujoco.mj_forward(ctx.model, ctx.data)
    _run(ctx, plugin, 0.05)

    ctx.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]  # stood back up
    mujoco.mj_forward(ctx.model, ctx.data)
    report = _run(ctx, plugin, 0.05)

    assert report.upright is False
    assert report.tilt_deg < 5.0, "the CURRENT tilt is small; the verdict is about the episode"


def test_latch_false_reports_the_current_state():
    """For a consumer reading the pose live rather than judging the trial afterwards."""
    ctx = _ctx()
    plugin = _plugin(ctx, latch=False)
    ctx.data.qpos[3:7] = [math.cos(math.pi / 4), math.sin(math.pi / 4), 0.0, 0.0]
    mujoco.mj_forward(ctx.model, ctx.data)
    assert _run(ctx, plugin, 0.05).upright is False

    ctx.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    mujoco.mj_forward(ctx.model, ctx.data)
    report = _run(ctx, plugin, 0.05)

    assert report.upright is True
    assert report.first_time >= 0.0, "when it first happened is still on the record"


def test_an_unknown_body_is_refused_by_name():
    ctx = _ctx()
    with pytest.raises(RuntimeError, match="was not found"):
        UprightMonitorPlugin({"body": "nope"}, entity="walker").configure(ctx)


@pytest.mark.parametrize("cfg,expected", [
    ({"max_tilt_deg": 0.0}, "max_tilt_deg"),
    ({"max_tilt_deg": 181.0}, "max_tilt_deg"),
    ({"max_rise_m": 0.0}, "max_rise_m"),
    ({"rate_hz": 0.0}, "rate_hz"),
])
def test_the_thresholds_are_checked(cfg, expected):
    errors = UprightMonitorPlugin(cfg, entity="walker").validate_config(cfg)
    assert any(expected in e for e in errors), errors


# -- a body no solver integrates -------------------------------------------------------------

MOCAP_SCENE = SCENE.replace(
    '<body name="base_link" pos="0 0 0.85">\n      <freejoint name="base_free"/>',
    '<body name="base_link" pos="0 0 0.85" mocap="true">',
)


def _mocap_ctx():
    model = mujoco.MjModel.from_xml_string(MOCAP_SCENE)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    ctx = SimContext(config={})
    ctx.model, ctx.data = model, data
    ctx.entities.add(
        Entity(name="walker", kind="pedestrian", body="base_link",
               meta={"prefix": "", "namespace": ""})
    )
    return ctx


def test_a_driven_body_nothing_mishandles_stays_quiet():
    """A mocap body has no degrees of freedom to topple with, so there is nothing to report.

    Worth pinning rather than assuming: a monitor that fired on the one embodiment that cannot
    fall would be noise on every pedestrian world in the corpus.
    """
    ctx = _mocap_ctx()
    plugin = _plugin(ctx)
    assert _run(ctx, plugin, 2.0).upright is True


def test_a_driven_body_written_into_a_bad_pose_is_still_reported():
    """The failure a mocap body DOES have: not falling, but being put somewhere wrong.

    Its pose comes from a gait, a navigation output or a scenario placing it, and any of those can
    lay it flat or sink it through the floor. This reads ``xpos``/``xmat``, so it reports that
    exactly as it reports a fall -- which is why nothing that moves a body needs to know it exists.
    """
    ctx = _mocap_ctx()
    plugin = _plugin(ctx)
    _run(ctx, plugin, 0.05)
    assert plugin.read_state().upright is True

    ctx.data.mocap_quat[0] = [math.cos(math.pi / 4), math.sin(math.pi / 4), 0.0, 0.0]
    mujoco.mj_forward(ctx.model, ctx.data)
    report = _run(ctx, plugin, 0.05)

    assert report.upright is False
    assert report.reason == "tilt"
    assert report.worst_tilt_deg > 80.0
