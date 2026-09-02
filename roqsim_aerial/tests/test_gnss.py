"""gnss: the projection, the rate, and the switch that denies the fix.

The noise terms are set to zero wherever the geometry is under test -- a receiver whose bias is
drifting cannot answer "is the projection right?", and the two questions must not be mixed.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np
import pytest

from roqsim.context import Entity, SimContext
from roqsim_aerial.plugins.gnss import R_EARTH, GnssPlugin

SCENE = """
<mujoco model="gnss_test">
  <option timestep="0.002" gravity="0 0 0"/>
  <worldbody>
    <body name="drone" pos="0 0 0">
      <freejoint name="base_free"/>
      <geom name="drone" type="box" size="0.1 0.1 0.05" mass="1"/>
    </body>
  </worldbody>
</mujoco>
"""

DATUM = {"lat": 47.397742, "lon": 8.545594, "alt": 488.0}

#: Noise off: what is under test is the tangent-plane projection, not the receiver's error model.
QUIET = {
    "datum": DATUM,
    "horizontal_noise": 0.0,
    "vertical_noise": 0.0,
    "velocity_noise": 0.0,
}


def _ctx(*, seed=7):
    model = mujoco.MjModel.from_xml_string(SCENE)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    ctx = SimContext(config={})
    ctx.model, ctx.data = model, data
    ctx.seed = seed
    # A real Entity, exactly as spawn_robot builds one: the root body is the first-class `body`
    # attribute and `meta` carries only prefix/namespace/pose. A fake whose meta was stuffed with
    # whatever key the plugin happened to read could not catch a wrong key name -- which is how a
    # dead `meta["root_body"]` lookup survived here in the first place.
    ctx.entities.add(Entity(name="drone", kind="robot", body="drone", meta={"prefix": ""}))
    return ctx


def _plugin(config, *, name="gnss"):
    plugin = GnssPlugin(config, name=name, entity="drone")
    errors = plugin.validate_config(config)
    if errors:
        raise ValueError("; ".join(errors))
    return plugin


def _place(ctx, east, north, up):
    """Put the drone at a world ENU offset and re-derive the kinematics."""
    ctx.data.qpos[0:3] = [east, north, up]
    mujoco.mj_forward(ctx.model, ctx.data)


def _fix_at(east, north, up, config=QUIET):
    ctx = _ctx()
    plugin = _plugin(config)
    plugin.configure(ctx)
    plugin.on_reset(ctx)
    _place(ctx, east, north, up)
    plugin.post_step(ctx)
    return plugin.read_fix()


# -- the datum -----------------------------------------------------------------------------------
def test_a_datum_is_required():
    """A fix against an invented origin is a wrong answer wearing a right answer's clothes."""
    errors = GnssPlugin({}).validate_config({})
    assert any("datum" in e for e in errors), errors


def test_a_partial_datum_is_refused():
    errors = GnssPlugin({}).validate_config({"datum": {"lat": 47.0, "lon": 8.0}})
    assert any("datum.alt" in e for e in errors), errors


def test_refuses_a_seed_of_its_own():
    with pytest.raises(ValueError, match="seed"):
        _plugin({**QUIET, "seed": 3})


# -- the projection ------------------------------------------------------------------------------
def test_the_origin_reports_the_datum():
    fix = _fix_at(0.0, 0.0, 0.0)
    assert fix["lat"] == pytest.approx(DATUM["lat"], abs=1e-12)
    assert fix["lon"] == pytest.approx(DATUM["lon"], abs=1e-12)
    assert fix["alt"] == pytest.approx(DATUM["alt"], abs=1e-9)
    assert fix["valid"] and fix["fix_type"] == 3


def test_a_hundred_metres_north_moves_latitude():
    fix = _fix_at(0.0, 100.0, 0.0)
    assert math.radians(fix["lat"] - DATUM["lat"]) == pytest.approx(100.0 / R_EARTH, rel=1e-9)
    assert fix["lon"] == pytest.approx(DATUM["lon"], abs=1e-12)  # north alone must not move it


def test_a_hundred_metres_east_moves_longitude_by_the_cosine_scaled_amount():
    fix = _fix_at(100.0, 0.0, 0.0)
    cos_lat = math.cos(math.radians(DATUM["lat"]))
    assert math.radians(fix["lon"] - DATUM["lon"]) == pytest.approx(
        100.0 / (R_EARTH * cos_lat), rel=1e-9
    )
    assert fix["lat"] == pytest.approx(DATUM["lat"], abs=1e-12)


def test_altitude_is_the_datum_plus_up():
    assert _fix_at(0.0, 0.0, 25.0)["alt"] == pytest.approx(DATUM["alt"] + 25.0, abs=1e-9)


# -- the receiver --------------------------------------------------------------------------------
def test_the_fix_is_held_between_updates():
    """10 Hz is 10 measurements a second and nothing in between. A consumer that saw the fix creep
    each tick would be reading interpolation the receiver never performed."""
    ctx = _ctx()
    plugin = _plugin({**QUIET, "rate": 10.0})
    plugin.configure(ctx)
    plugin.on_reset(ctx)

    plugin.post_step(ctx)  # t = 0: the first fix
    first = plugin.read_fix()

    period_ticks = int(round((1.0 / 10.0) / ctx.dt))
    seen = []
    for tick in range(1, period_ticks + 1):
        _place(ctx, 0.0, tick * 1.0, 0.0)  # move a metre north every tick
        ctx.data.time += ctx.dt
        plugin.post_step(ctx)
        seen.append(plugin.read_fix()["lat"])

    assert seen[:-1] == [first["lat"]] * (period_ticks - 1)  # held for the whole period
    assert seen[-1] != first["lat"]  # and updated exactly once the period elapsed


def test_no_fix_before_the_first_update():
    plugin = _plugin(QUIET)
    assert plugin.read_fix()["valid"] is False


def test_denial_reports_no_fix_without_removing_the_receiver():
    """GNSS denial is an experiment factor, so it must be a level of a config key -- not the
    presence or absence of a plugin, which would change the world rather than a condition in it."""
    fix = _fix_at(100.0, 100.0, 5.0, {**QUIET, "denied": True})
    assert fix["valid"] is False
    assert fix["fix_type"] == 0
    assert fix["satellites"] == 0
    assert (fix["lat"], fix["lon"], fix["alt"]) == (0.0, 0.0, 0.0)


def test_velocity_is_reported_in_ned():
    ctx = _ctx()
    plugin = _plugin(QUIET)
    plugin.configure(ctx)
    plugin.on_reset(ctx)
    ctx.data.qvel[0:3] = [3.0, 4.0, 2.0]  # ENU: east 3, north 4, up 2
    mujoco.mj_forward(ctx.model, ctx.data)
    plugin.post_step(ctx)
    fix = plugin.read_fix()
    assert fix["vel_e"] == pytest.approx(3.0)
    assert fix["vel_n"] == pytest.approx(4.0)
    assert fix["vel_d"] == pytest.approx(-2.0)
    assert fix["vel"] == pytest.approx(5.0)
    assert fix["cog"] == pytest.approx(math.degrees(math.atan2(3.0, 4.0)))


# -- the noise -----------------------------------------------------------------------------------
def _track(seed, ticks=400, config=None):
    ctx = _ctx(seed=seed)
    plugin = _plugin(config or {"datum": DATUM, "rate": 50.0})
    plugin.configure(ctx)
    plugin.on_reset(ctx)
    out = []
    for _ in range(ticks):
        plugin.post_step(ctx)
        out.append(plugin.read_fix()["lat"])
        ctx.data.time += ctx.dt
    return np.array(out)


def test_the_noise_follows_the_run_seed():
    assert np.array_equal(_track(7), _track(7))
    # Compared in metres, not degrees: a relative tolerance on a latitude near 47 is worth tens of
    # metres, and would call two entirely different tracks equal.
    apart = (_track(7) - _track(99)) * R_EARTH * math.pi / 180.0
    assert np.abs(apart).max() > 0.1, np.abs(apart).max()


def test_the_bias_is_correlated_rather_than_white():
    """A pure-white GNSS averages away and flatters an estimator; the drift is the thing a
    navigation experiment is about, so it must actually be there."""
    config = {"datum": DATUM, "rate": 50.0, "bias_time": 2.0, "horizontal_noise": 1.0}
    track = _track(7, ticks=4000, config=config)  # 400 updates at 50 Hz
    error = (track - DATUM["lat"]) * R_EARTH * math.pi / 180.0  # back to metres north
    # Successive updates of a white sequence are uncorrelated; a slow bias makes them not.
    unique = error[np.diff(error, prepend=np.nan) != 0]
    corr = float(np.corrcoef(unique[:-1], unique[1:])[0, 1])
    assert corr > 0.25, corr


# -- the entity contract -------------------------------------------------------------------------
def test_the_body_comes_from_the_entity_not_from_a_meta_key():
    """Regression: the root body is `Entity.body`, the attribute spawn_robot fills in. Reading a
    meta key instead is a lookup that always returns None, and it failed only in a real world."""
    ctx = _ctx()
    assert "root_body" not in ctx.entities.get("drone").meta
    plugin = _plugin(QUIET)
    plugin.configure(ctx)  # would raise "no body to measure" against a meta-key lookup
    assert plugin._bid == mujoco.mj_name2id(ctx.model, mujoco.mjtObj.mjOBJ_BODY, "drone")


def test_a_configured_body_takes_the_entity_prefix():
    """`Entity.body` is already prefixed (spawn_robot resolves it against the compiled model); a
    config `body:` names the model's own body and takes the prefix. Never both."""
    ctx = _ctx()
    ctx.entities.remove("drone")
    ctx.entities.add(Entity(name="drone", kind="robot", body="drone", meta={"prefix": ""}))
    plugin = _plugin({**QUIET, "body": "drone"})
    plugin.configure(ctx)
    assert plugin._bid >= 0


def test_an_entity_with_no_body_at_all_still_fails_loudly():
    ctx = _ctx()
    ctx.entities.remove("drone")
    ctx.entities.add(Entity(name="drone", kind="robot", body=None, meta={"prefix": ""}))
    with pytest.raises(RuntimeError, match="no body to measure"):
        _plugin(QUIET).configure(ctx)
