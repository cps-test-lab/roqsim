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


def _settled(scene: str = f"{__name__}:_ArmScene", **ft_config):
    """An engine stepped until the link hangs static, so the wrench is the static load."""
    from roqsim.engine import Engine

    cfg = load_config_from_dict(
        {"sim": {}, "plugins": [{scene: {}}, {"force_torque": {"site": "fts_site", **ft_config}}]}
    )
    engine = Engine(cfg)
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
    from roqsim.engine import Engine

    cfg = load_config_from_dict(
        {
            "sim": {},
            "plugins": [
                {f"{__name__}:_ArmScene": {}},
                {"force_torque": {"site": "fts_site", "name": "ft"}},
                {"force_torque": {"site": "fts_site", "name": "ft"}},
            ],
        }
    )
    engine = Engine(cfg)
    with pytest.raises(RuntimeError, match="already registered"):
        engine.setup()


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
