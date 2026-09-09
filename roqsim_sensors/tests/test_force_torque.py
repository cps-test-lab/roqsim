"""``force_torque`` checks: what the wrench reads, in which frame, and that its noise is reproducible.

The scene is the smallest thing that produces a real reading: one jointed link hanging from the world
with a heavy tool welded below it, damped so it settles. A MuJoCo site force sensor reports the
interaction between the site's body and its **parent**, so everything in the subtree below that joint
-- the tool AND the link's own mass -- loads the sensor. That is the un-tared behaviour the plugin
documents, and pinning the number here is what would catch a future "helpful" gravity compensation.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np
import pytest
from roqsim_sensors.plugins.force_torque import ForceTorquePlugin

from roqsim.config import load_config_from_dict
from roqsim.context import SimContext
from roqsim.plugin import Plugin

LINK_MASS = 0.5
TOOL_MASS = 2.0
GRAVITY = 9.81
#: What the sensor must read: the whole subtree below the joint, un-tared.
EXPECTED_FZ = (LINK_MASS + TOOL_MASS) * GRAVITY


class _ArmScene(Plugin):
    """A damped prismatic link carrying a tool, with an ``fts_site`` at the cut between them."""

    #: Site rotation about x, in degrees. 0 -> site frame == world frame; 90 -> they differ, which is
    #: what makes the `frame:` option observable.
    site_roll_deg = 0.0
    #: Whether the scene ships its own <force>/<torque> pair (a vendor MJCF that already has one).
    ships_own_sensors = False

    def build(self, spec: mujoco.MjSpec, ctx: SimContext) -> None:
        spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[5, 5, 0.1])
        link = spec.worldbody.add_body(name="link", pos=[0, 0, 1])
        link.add_joint(name="j", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[0, 0, 1], damping=1000.0)
        link.add_geom(
            type=mujoco.mjtGeom.mjGEOM_CAPSULE,
            fromto=[0, 0, 0, 0, 0, -0.1],
            size=[0.02, 0, 0],
            mass=LINK_MASS,
        )
        roll = math.radians(self.site_roll_deg)
        link.add_site(
            name="fts_site", pos=[0, 0, -0.1], quat=[math.cos(roll / 2), math.sin(roll / 2), 0, 0]
        )
        tool = link.add_body(name="tool", pos=[0, 0, -0.1])
        tool.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[0.05, 0.05, 0.05],
            pos=[0, 0, -0.05],
            mass=TOOL_MASS,
        )
        if self.ships_own_sensors:
            for suffix, kind in (
                ("force", mujoco.mjtSensor.mjSENS_FORCE),
                ("torque", mujoco.mjtSensor.mjSENS_TORQUE),
            ):
                s = spec.add_sensor()
                s.name = f"fts_site_{suffix}"
                s.type = kind
                s.objtype = mujoco.mjtObj.mjOBJ_SITE
                s.objname = "fts_site"


class _RolledArmScene(_ArmScene):
    """Same scene with the site rolled, so the ``sensor`` and ``world`` frames disagree."""

    site_roll_deg = 90.0


class _VendorSensorScene(_ArmScene):
    """A model that already ships its own ``<force>``/``<torque>`` pair on the site."""

    ships_own_sensors = True


def _settled(scene: str = f"{__name__}:_ArmScene", *, name=None, **ft_config):
    """An engine stepped until the link hangs static, so the wrench is the static load.

    `name` is the entry's reserved sibling -- an instance is identified by its label, not by a
    config key it carries."""
    from roqsim.engine import Engine

    entry = {"force_torque": {"site": "fts_site", **ft_config}}
    if name is not None:
        entry["name"] = name
    cfg = load_config_from_dict({"sim": {}, "components": [{scene: {}}, entry]})
    engine = Engine(cfg)
    # A test driving an Engine IS the driver, and `ctx.seed` is driver-owned: `rng_for`
    # refuses an unset one. The world's own `sim.seed` is honoured so declaring one here
    # does what it looks like it does; the fallback is fixed, not drawn, so a noisy test
    # stays reproducible.
    engine.ctx.seed = 0 if cfg.seed is None else int(cfg.seed)
    engine.setup()
    engine.reset()
    for _ in range(200):
        engine.step()
    return engine


def _plugin(engine) -> ForceTorquePlugin:
    return next(p for p in engine.plugins if isinstance(p, ForceTorquePlugin))


# -- what it reads ---------------------------------------------------------------------------


def test_reads_the_static_load_of_everything_below_the_cut():
    ft = _plugin(_settled(invert=False))
    force, torque = ft.read()
    # +z: the parent holds the subtree up. The link's own 0.5 kg is in there too -- a simulated FT
    # sensor is not tared against the tool, which is why a force-integrating metric needs either
    # zero gravity or a near-massless tool.
    assert force[2] == pytest.approx(EXPECTED_FZ, rel=1e-3)
    assert abs(force[0]) < 1e-6 and abs(force[1]) < 1e-6
    assert torque.shape == (3,)


def test_invert_is_the_default_and_flips_the_sign():
    plain = _plugin(_settled(invert=False)).read()[0]
    inverted = _plugin(_settled()).read()[0]  # invert defaults to true
    assert inverted == pytest.approx(-plain)
    # The convention a real FT sensor and its users assume: the load the environment applies.
    assert inverted[2] < 0


def test_frame_rotates_the_wrench_out_of_the_site_frame():
    rolled = f"{__name__}:_RolledArmScene"
    sensor_frame = _plugin(_settled(scene=rolled, invert=False)).read()[0]
    world_frame = _plugin(_settled(scene=rolled, frame="world", invert=False)).read()[0]
    # The site is rolled 90 deg about x, so the load that is +z in the world lies along the site's
    # own y. Reporting in the wrong frame silently splits a wrench onto the wrong axes -- and a
    # metric that separates an insertion axis from the plane orthogonal to it is exactly that split.
    assert abs(sensor_frame[1]) == pytest.approx(EXPECTED_FZ, rel=1e-3)
    assert abs(sensor_frame[2]) < 1e-3
    assert world_frame[2] == pytest.approx(EXPECTED_FZ, rel=1e-3)


# -- wiring ---------------------------------------------------------------------------------


def test_blackboard_reader_agrees_with_the_endpoint():
    engine = _settled(name="ft")
    reader = engine.ctx.blackboard.get("ft:ft")
    assert reader is not None and reader.frame == "sensor"
    endpoint = next(e for e in engine.ctx.interface.all() if e.name == "wrench")
    force, torque = reader.read()
    ep_force, ep_torque = endpoint.read()
    # Same instant, same wrench -- a controller reading the blackboard and a bag recording the topic
    # must not disagree about the force at one time.
    assert np.allclose(force, ep_force) and np.allclose(torque, ep_torque)


def test_an_existing_sensor_pair_is_reused_not_duplicated():
    engine = _settled(scene=f"{__name__}:_VendorSensorScene", invert=False)
    m = engine.ctx.model
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_SENSOR, i) for i in range(m.nsensor)]
    assert names.count("fts_site_force") == 1 and names.count("fts_site_torque") == 1
    # And it still reads: the plugin bound to the model's own pair.
    assert _plugin(engine).read()[0][2] == pytest.approx(EXPECTED_FZ, rel=1e-3)


def test_two_sensors_cannot_share_a_blackboard_key():
    """Two entries answering to one label is refused when the document LOADS, before anything is
    built -- the blackboard key it would collide on is derived from that label. The plugin keeps its
    own guard for a directly-constructed pair (below), but a world can no longer reach it."""
    from roqsim.config import PluginError

    with pytest.raises(PluginError, match="labelled 'ft'"):
        load_config_from_dict(
            {
                "sim": {},
                "components": [
                    {f"{__name__}:_ArmScene": {}},
                    {"force_torque": {"site": "fts_site"}, "name": "ft"},
                    {"force_torque": {"site": "fts_site"}, "name": "ft"},
                ],
            }
        )


def test_the_plugins_own_duplicate_guard_still_holds():
    """Reachable by constructing two directly, which an embedding driver may do."""
    engine = _settled(name="ft")
    ctx = engine.ctx
    with pytest.raises(RuntimeError, match="already registered"):
        ForceTorquePlugin({"site": "fts_site"}, label="ft").configure(ctx)


# -- config -------------------------------------------------------------------------------


def test_site_is_required_and_frame_is_checked():
    errors = ForceTorquePlugin().validate_config({})
    assert any("'site' is required" in e for e in errors)
    assert any(
        "'frame' must be one of" in e
        for e in ForceTorquePlugin().validate_config({"site": "s", "frame": "elbow"})
    )


def test_a_per_sensor_seed_is_rejected_rather_than_ignored():
    # The noise comes from the run's seed via ctx.rng_for; accepting `seed:` here would leave a world
    # believing it had pinned the stream.
    errors = ForceTorquePlugin().validate_config({"site": "s", "seed": 7})
    assert any("not a force_torque setting" in e for e in errors)


# -- noise ---------------------------------------------------------------------------------


def test_noise_is_identical_for_two_readers_in_one_step_and_changes_between_steps():
    engine = _settled(noise_force_stddev=1.0, noise_torque_stddev=0.1, invert=False)
    ft = _plugin(engine)
    first, second = ft.read()[0], ft.read()[0]
    # Two reads at one instant are the same measurement: rng_for is keyed on (seed, sim_time, sensor),
    # so it does not advance a stream between the endpoint and the blackboard consumer.
    assert np.allclose(first, second)
    assert not np.allclose(first, [0, 0, EXPECTED_FZ])  # noise was actually applied
    engine.step()
    assert not np.allclose(first, ft.read()[0])


def test_the_same_run_seed_reproduces_the_noise():
    def run(seed):
        engine = _settled(noise_force_stddev=1.0)
        engine.ctx.seed = seed
        return _plugin(engine).read()[0]

    assert np.allclose(run(4), run(4))
    assert not np.allclose(run(4), run(5))


# -- taring: zeroing the tool's own load ------------------------------------------------------


def test_nothing_is_tared_unless_asked():
    """The default reads the whole load, and the number above is what a tare must not change.

    Negative because ``invert`` is the default: the ENVIRONMENT pushes the tool up.
    """
    engine = _settled()
    force, _ = _plugin(engine).read()
    assert force[2] == pytest.approx(-EXPECTED_FZ, rel=1e-3)


def test_a_tare_zeroes_the_standing_load():
    """What the option is for: a contact task starts from zero, not from the tool's weight."""
    engine = _settled()
    plugin = _plugin(engine)
    plugin.tare()
    force, torque = plugin.read()
    assert np.allclose(force, 0.0, atol=1e-9)
    assert np.allclose(torque, 0.0, atol=1e-9)


def test_tare_at_s_fires_once_the_clock_reaches_it():
    """Captured at a stated sim time, so a world can zero after the arm has settled.

    Before that time the sensor reads the full load -- a tare armed for later must not quietly
    apply early, or a world that meant to settle first would zero a moving arm.
    """
    engine = _settled(tare_at_s=1e9)  # never reached in this run
    plugin = _plugin(engine)
    assert plugin.read()[0][2] == pytest.approx(-EXPECTED_FZ, rel=1e-3)

    plugin.tare_at_s = engine.ctx.sim_time
    assert np.allclose(plugin.read()[0], 0.0, atol=1e-9)


def test_a_tare_is_forgotten_on_reset():
    """An offset carried into the next episode is a measurement of the previous one.

    The same rule the noise follows, and for the same reason: a repetition that starts from the
    last trial's zero is not a repetition.
    """
    engine = _settled()
    plugin = _plugin(engine)
    plugin.tare()
    assert np.allclose(plugin.read()[0], 0.0, atol=1e-9)

    engine.reset()
    for _ in range(200):
        engine.step()
    assert plugin.read()[0][2] == pytest.approx(-EXPECTED_FZ, rel=1e-3)


def test_taring_twice_zeroes_against_the_load_now_not_the_residual():
    """Re-taring is the way to use this on a tool that turns, so it must not compound.

    Taken through ``read`` instead of the raw pair, the second tare would capture the residual of
    the first -- leaving the first offset standing forever and the second doing nothing.
    """
    engine = _settled()
    plugin = _plugin(engine)
    plugin.tare()
    first = plugin._offset_force.copy()
    plugin.tare()

    assert np.allclose(plugin._offset_force, first, atol=1e-9), \
        "the standing load has not changed, so neither should the offset"
    assert np.allclose(plugin.read()[0], 0.0, atol=1e-9)


def test_the_offset_is_the_mean_so_what_remains_is_the_noise_alone():
    """A tare taken after the noise would bake one draw into every later reading.

    Captured before it, the residual is a zero-mean signal rather than one shifted by whatever
    the sensor happened to read at the instant somebody pressed the button.
    """
    engine = _settled(noise_force_stddev=0.5)
    plugin = _plugin(engine)
    plugin.tare()

    samples = []
    for _ in range(200):
        engine.step()
        samples.append(plugin.read()[0][2])

    assert abs(float(np.mean(samples))) < 0.15, "the residual must be centred on zero"
    assert float(np.std(samples)) > 0.2, "and it must still carry the noise"


def test_the_reader_on_the_blackboard_can_tare():
    """The in-process half of the ticket: a controller zeroes at a moment it chooses."""
    engine = _settled(name="ft")
    reader = engine.ctx.blackboard.get("ft:ft")
    assert reader.tare is not None
    reader.tare()
    assert np.allclose(reader.read()[0], 0.0, atol=1e-9)


def test_a_negative_tare_time_is_refused_as_a_time():
    from roqsim.engine import Engine

    cfg = load_config_from_dict({
        "sim": {},
        "components": [{f"{__name__}:_ArmScene": {}},
                       {"force_torque": {"site": "fts_site", "tare_at_s": -1.0}}],
    })
    with pytest.raises(Exception, match="tare_at_s"):
        Engine(cfg)


def test_the_zero_button_is_a_service_a_scenario_can_press():
    """What a user reaches for, and what real hardware exposes: a command taking no argument.

    An FT driver's zero is a service (`zero_ftsensor`), not a configured time -- the moment to
    zero is known by whatever is running the task, not by whoever wrote the world. A service also
    has a reply, which is what lets a scenario fail instead of measuring against an offset it only
    assumed was applied.
    """
    engine = _settled(name="ft")
    endpoint = next(e for e in engine.ctx.interface.all() if e.name == "tare")

    assert endpoint.direction == "in"
    assert endpoint.backend["ros2"]["service"] == "std_srvs.srv.Trigger"

    plugin = _plugin(engine)
    assert plugin.read()[0][2] == pytest.approx(-EXPECTED_FZ, rel=1e-3)
    endpoint.write(None)
    engine.step()  # the command is posted to the physics thread, like every other write

    assert np.allclose(plugin.read()[0], 0.0, atol=1e-9)


def test_all_three_doors_zero_the_same_sensor():
    """One implementation behind the service, the reader and the configured time.

    Three surfaces onto one offset, so a world that tares at a time and a controller that re-tares
    on approach cannot end up disagreeing about what the sensor reads.
    """
    engine = _settled(name="ft")
    plugin = _plugin(engine)
    reader = engine.ctx.blackboard.get("ft:ft")
    endpoint = next(e for e in engine.ctx.interface.all() if e.name == "tare")

    for press in (plugin.tare, reader.tare, lambda: (endpoint.write(None), engine.step())):
        plugin.on_reset(engine.ctx)
        assert plugin.read()[0][2] == pytest.approx(-EXPECTED_FZ, rel=1e-3)
        press()
        assert np.allclose(plugin.read()[0], 0.0, atol=1e-9)
