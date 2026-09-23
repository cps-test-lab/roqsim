"""Standalone lidar checks -- a synthetic scene, no dependency on roqsim_mobile.

What a scan publishes for each class of ray (too close, measured, no return), the inclusive ray
layout, the noise model and the config validation, plus the frame and mount-TF rules.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np
import pytest
from roqsim_sensors.plugins.lidar import LidarPlugin

from roqsim.config import load_config_from_dict
from roqsim.context import Entity, SimContext
from roqsim.engine import Engine
from roqsim.plugin import Plugin

#: The wall face _OneWallScene puts in front of the lidar, along +x.
WALL_FACE = 1.95


class _OneWallScene(Plugin):
    """A single wall 2m in front of a lidar site at the origin, facing +x."""

    def build(self, spec: mujoco.MjSpec, ctx: SimContext) -> None:
        spec.worldbody.add_site(name="lidar", pos=[0, 0, 0.1])
        spec.worldbody.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX, pos=[2, 0, 0.1], size=[0.05, 5, 0.5]
        )


def _world(**lidar_config):
    cfg = {
        "sim": {},
        "plugins": [
            {f"{__name__}:_OneWallScene": {}},
            {"roqsim_sensors.plugins.lidar:LidarPlugin": lidar_config},
        ],
    }
    return load_config_from_dict(cfg)


def _scan(engine: Engine):
    ep = next(e for e in engine.ctx.interface.all() if e.name == "scan")
    return ep.read()


def _one_scan(**lidar_config):
    engine = Engine(_world(**lidar_config))
    engine.ctx.seed = 1
    engine.setup()
    engine.reset()
    engine.step()
    return engine, _scan(engine)


def _same(a: float, b: float) -> bool:
    """Equal, with NaN equal to NaN and each infinity equal only to itself."""
    if math.isnan(b):
        return math.isnan(a)
    return a == pytest.approx(b, abs=1e-6) if math.isfinite(b) else a == b


def test_validate_config_rejects_bad_values():
    errors = LidarPlugin().validate_config({"rays": 0, "max_range": -1, "dropout_percent": 150})
    assert len(errors) == 3


@pytest.mark.parametrize(
    ("config", "message"),
    [
        ({"too_close": "zero"}, "'too_close' must be a number or one of"),
        ({"too_close": -0.5}, "'too_close' must be >= 0"),
        ({"too_close": True}, "'too_close' must be a number or one of"),
        ({"no_return": "raw"}, "'no_return' cannot be 'raw'"),
        ({"no_return": "-inf"}, "'no_return' cannot be -inf"),
        ({"no_return": float("-inf")}, "'no_return' cannot be -inf"),
        ({"detection_min": 5.0, "max_range": 4.0}, "the detection limits are empty"),
        ({"range_min": 1.0, "detection_max": 0.5}, "the detection limits are empty"),
        ({"detection_min": -0.1}, "'detection_min' must be >= 0"),
        ({"detection_max": 0.0}, "'detection_max' must be > 0"),
        ({"range_resolution": -0.01}, "'range_resolution' must be >= 0"),
        ({"range_stddev_relative": -0.1}, "'range_stddev_relative' must be >= 0"),
        ({"rays": 1, "angle_min": 0.0, "angle_max": 1.0}, "a single ray has no increment"),
        ({"rays": 4, "angle_min": 1.0, "angle_max": 1.0}, "'angle_max' must be > 'angle_min'"),
    ],
)
def test_validate_config_names_each_bad_key(config, message):
    errors = LidarPlugin().validate_config(config)
    assert len(errors) == 1 and message in errors[0], errors


@pytest.mark.parametrize(
    "config",
    [
        {"too_close": "-inf", "no_return": "+inf"},
        {"too_close": "+inf", "no_return": "inf"},
        {"too_close": "nan", "no_return": "nan"},
        {"too_close": "raw"},
        {"too_close": 0.004, "no_return": 65.533},
        {"too_close": float("-inf"), "no_return": float("inf")},  # YAML's -.inf / .inf
        {"too_close": float("nan"), "no_return": 0.0},
        {"detection_min": 0.05, "detection_max": 30.0, "range_min": 0.01, "max_range": 29.0},
        {"range_stddev": 0.01, "range_stddev_relative": 0.035, "range_stddev_relative_from": 0.5},
        {"range_resolution": 0.01},
        # A threshold with the fraction off is harmless: what a fault switching relative noise off
        # leaves behind.
        {"range_stddev_relative": 0.0, "range_stddev_relative_from": 0.5},
        {"rays": 1, "angle_min": 0.3, "angle_max": 0.3},
    ],
)
def test_validate_config_accepts_the_vocabulary(config):
    assert LidarPlugin().validate_config(config) == []


def test_a_fault_is_validated_with_the_same_rules():
    errors = LidarPlugin().validate_config({"fault": {"detection_max": -1.0}})
    assert any("'detection_max' must be > 0" in e for e in errors), errors


# -- what each class of ray publishes --------------------------------------------------------------

#: Four rays at 0, pi/2, pi and 3pi/2: ray 0 meets the wall, the other three meet nothing.
FAN = {"rays": 4, "angle_min": 0.0, "angle_max": 1.5 * math.pi}


def test_rep117_defaults():
    _, scan = _one_scan(**FAN, range_min=0.1, max_range=4.0)
    assert scan.ranges[0] == pytest.approx(WALL_FACE, abs=1e-3)
    assert np.all(scan.ranges[1:] == math.inf)

    _, scan = _one_scan(**FAN, range_min=2.5, max_range=4.0)
    assert scan.ranges[0] == -math.inf, "a too-close return is -inf, not raised to range_min"
    assert np.all(scan.ranges[1:] == math.inf)

    _, scan = _one_scan(**FAN, range_min=0.1, max_range=1.0)
    assert scan.ranges[0] == math.inf, "a wall beyond max_range is no return"


@pytest.mark.parametrize(
    ("too_close", "expected"),
    [
        ("-inf", -math.inf),
        ("+inf", math.inf),
        ("nan", math.nan),
        ("raw", WALL_FACE),
        (0.004, 0.004),
    ],
)
def test_too_close_publishes_the_declared_value(too_close, expected):
    # Too close: the wall at 1.95 is inside range_min.
    _, scan = _one_scan(**FAN, range_min=2.5, max_range=4.0, too_close=too_close)
    assert _same(float(scan.ranges[0]), expected), scan.ranges
    assert np.all(scan.ranges[1:] == math.inf), "the rays that miss are still no return"
    # Measured: the same setting leaves an in-range return alone.
    _, scan = _one_scan(**FAN, range_min=0.1, max_range=4.0, too_close=too_close)
    assert scan.ranges[0] == pytest.approx(WALL_FACE, abs=1e-3)
    # None: nothing within range, so the ray is no return, not too close.
    _, scan = _one_scan(**FAN, range_min=0.1, max_range=1.0, too_close=too_close)
    assert scan.ranges[0] == math.inf


@pytest.mark.parametrize(
    ("no_return", "expected"),
    [("+inf", math.inf), ("inf", math.inf), ("nan", math.nan), (65.533, 65.533), (0.0, 0.0)],
)
def test_no_return_publishes_the_declared_value(no_return, expected):
    # Measured: ray 0 reads the wall; the rest are no return.
    _, scan = _one_scan(**FAN, range_min=0.1, max_range=4.0, no_return=no_return)
    assert scan.ranges[0] == pytest.approx(WALL_FACE, abs=1e-3)
    assert all(_same(float(r), expected) for r in scan.ranges[1:]), scan.ranges
    # None within range: the wall's ray is no return too.
    _, scan = _one_scan(**FAN, range_min=0.1, max_range=1.0, no_return=no_return)
    assert all(_same(float(r), expected) for r in scan.ranges), scan.ranges
    # Too close stays too close.
    _, scan = _one_scan(**FAN, range_min=2.5, max_range=4.0, no_return=no_return)
    assert scan.ranges[0] == -math.inf


def test_the_header_publishes_range_min_and_max_while_detection_limits_classify():
    # Header 0.01 .. 1.0, physics 2.5 .. 4.0: the wall is too close although above range_min.
    _, scan = _one_scan(**FAN, range_min=0.01, max_range=1.0, detection_min=2.5, detection_max=4.0)
    assert (scan.range_min, scan.range_max) == (0.01, 1.0)
    assert scan.ranges[0] == -math.inf
    # Physics 0.05 .. 4.0 under a 1.0 header: the wall is measured and published above range_max,
    # where a REP 117 consumer discards it.
    _, scan = _one_scan(**FAN, range_min=0.01, max_range=1.0, detection_min=0.05, detection_max=4.0)
    assert scan.ranges[0] == pytest.approx(WALL_FACE, abs=1e-3)
    assert scan.ranges[0] > scan.range_max


def test_a_fault_on_max_range_moves_the_far_limit_that_follows_it():
    engine, scan = _one_scan(**FAN, range_min=0.1, max_range=4.0)
    assert scan.ranges[0] == pytest.approx(WALL_FACE, abs=1e-3)
    lidar = next(p for p in engine.plugins if isinstance(p, LidarPlugin))
    lidar.range_max = 1.0  # what a `fault: {max_range: 1.0}` writes
    for _ in range(200):
        engine.step()
    assert _scan(engine).ranges[0] == math.inf


def test_dropout_makes_a_ray_no_return():
    _, scan = _one_scan(
        **FAN, range_min=0.1, max_range=4.0, dropout_percent=100.0, no_return=65.533
    )
    assert np.all(scan.ranges == pytest.approx(65.533))


# -- layout -------------------------------------------------------------------------------------


def test_the_last_ray_is_exactly_at_angle_max():
    engine, scan = _one_scan(rays=5, angle_min=-math.pi / 2, angle_max=math.pi / 2, max_range=4.0)
    lidar = next(p for p in engine.plugins if isinstance(p, LidarPlugin))
    assert scan.angle_increment == pytest.approx(math.pi / 4)
    assert scan.angle_min + (len(scan.ranges) - 1) * scan.angle_increment == pytest.approx(
        scan.angle_max
    )
    dirs = lidar._build_directions()
    np.testing.assert_allclose(dirs[0], [0.0, -1.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(dirs[-1], [0.0, 1.0, 0.0], atol=1e-12)
    # The middle ray looks along +x at the wall; the two edge rays run parallel to it.
    assert scan.ranges[2] == pytest.approx(WALL_FACE, abs=1e-3)


def test_the_default_fan_is_a_full_turn_with_no_bearing_published_twice():
    plugin = LidarPlugin({"rays": 360})
    assert plugin.angle_min == 0.0
    assert plugin.angle_max == pytest.approx(2 * math.pi * 359 / 360)
    assert plugin.angle_increment == pytest.approx(math.radians(1.0))
    bearings = np.degrees(np.angle(np.exp(1j * np.linspace(0, plugin.angle_max, 360))))
    assert len(np.unique(np.round(bearings, 6))) == 360


def test_a_minus_pi_to_pi_fan_publishes_the_seam_bearing_twice():
    """The layout the RPLIDAR C1's driver publishes: first and last ray on the same bearing."""
    plugin = LidarPlugin({"rays": 720, "angle_min": -math.pi, "angle_max": math.pi})
    dirs = plugin._build_directions()
    np.testing.assert_allclose(dirs[0], dirs[-1], atol=1e-12)
    assert plugin.angle_increment == pytest.approx(2 * math.pi / 719)


# -- noise ------------------------------------------------------------------------------------------

#: A narrow fan of many rays onto the wall: true ranges 1.95 / cos(bearing), all hits.
NOISE_FAN = {"rays": 4001, "angle_min": -0.3, "angle_max": 0.3, "range_min": 0.1, "max_range": 4.0}


def _residual_over_sigma(sigma_of_true, **noise):
    engine, scan = _one_scan(**NOISE_FAN, **noise)
    lidar = next(p for p in engine.plugins if isinstance(p, LidarPlugin))
    true = WALL_FACE / np.cos(np.linspace(lidar.angle_min, lidar.angle_max, lidar.num_rays))
    return (np.asarray(scan.ranges) - true) / sigma_of_true(true)


def test_a_relative_sigma_scales_with_the_distance_beyond_its_threshold():
    z = _residual_over_sigma(
        lambda t: 0.02 * t,
        range_stddev=0.001,
        range_stddev_relative=0.02,
        range_stddev_relative_from=1.0,
    )
    assert abs(z.mean()) < 0.1
    assert z.std() == pytest.approx(1.0, rel=0.06)


def test_nearer_than_its_threshold_the_sigma_is_the_constant():
    z = _residual_over_sigma(
        lambda t: np.full_like(t, 0.01),
        range_stddev=0.01,
        range_stddev_relative=0.2,
        range_stddev_relative_from=3.0,
    )
    assert abs(z.mean()) < 0.1
    assert z.std() == pytest.approx(1.0, rel=0.06)


def test_a_constant_sigma_is_unchanged():
    z = _residual_over_sigma(lambda t: np.full_like(t, 0.03), range_stddev=0.03)
    assert z.std() == pytest.approx(1.0, rel=0.06)


def test_a_constant_too_close_value_carries_no_noise():
    _, scan = _one_scan(**FAN, range_min=2.5, max_range=4.0, too_close=0.004, range_stddev=0.5)
    assert scan.ranges[0] == 0.004


# -- quantisation -----------------------------------------------------------------------------------


def test_range_resolution_quantises_every_published_distance():
    _, scan = _one_scan(**NOISE_FAN, range_resolution=0.01)
    steps = np.asarray(scan.ranges) / 0.01
    np.testing.assert_allclose(steps, np.round(steps), atol=1e-9)
    assert scan.ranges[len(scan.ranges) // 2] == pytest.approx(1.95, abs=1e-9)


def test_quantisation_applies_after_noise():
    _, scan = _one_scan(**NOISE_FAN, range_resolution=0.25, range_stddev=0.05)
    assert set(np.round(np.asarray(scan.ranges) / 0.25, 9)) <= {7.0, 8.0, 9.0}


# -- frame and mount transform ---------------------------------------------------------------------


def _scan_hints(engine: Engine) -> dict:
    ep = next(e for e in engine.ctx.interface.all() if e.name == "scan")
    return ep.backend["ros2"]


def test_frame_id_defaults_to_the_site():
    """The scan is stamped in the frame the rays are cast from, not a hardcoded robot's frame.

    A frame hardwired to "rplidar_link" for every robot would put a Husky's scan in a TurtleBot's
    frame. The static mount TF's child comes from the same hint, so the two cannot
    disagree.
    """
    engine = Engine(_world(site="lidar"))
    engine.setup()
    hints = _scan_hints(engine)
    assert hints["frame_id"] == "lidar"
    # The child of the mount TF is that same frame_id, applied by the bridge. The parent is the
    # world here because this scene has no base_link to hang it from -- see the mount-TF tests.
    assert hints["static_tf"]["parent"] == "world"


def test_frame_id_can_be_declared_by_the_model():
    """A model whose real description names the frame (TurtleBot 4's URDF: rplidar_link) says so."""
    engine = Engine(_world(site="lidar", frame_id="rplidar_link"))
    engine.setup()
    assert _scan_hints(engine)["frame_id"] == "rplidar_link"


def test_ray_along_x_hits_the_wall_at_expected_range():
    # max_range below the empty_room perimeter (walls at +-5) so the -x miss actually misses.
    engine = Engine(_world(rays=4, angle_min=0.0, angle_max=1.5 * np.pi, max_range=4.0))
    engine.setup()
    engine.reset()
    engine.step()
    scan = _scan(engine)
    # ray 0 points along +x (angle 0) straight at the wall face at x=1.95 (2.0 - half-thickness).
    assert np.isclose(scan.ranges[0], 1.95, atol=1e-3)
    # ray 2 points along -x (angle pi): nothing in range -> inf.
    assert np.isinf(scan.ranges[2])


# -- the static mount transform's parent frame ---------------------------------------------------


class _MastScene(Plugin):
    """A scanner site on a mast body, and a wall to see -- no ``base_link`` anywhere."""

    def build(self, spec: mujoco.MjSpec, ctx: SimContext) -> None:
        mast = spec.worldbody.add_body(name="mast", pos=[0, 0, 0])
        mast.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.05, 0.5, 0])
        mast.add_site(name="lidar", pos=[0, 0, 1.2])
        spec.worldbody.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX, pos=[2, 0, 0.1], size=[0.05, 5, 0.5]
        )


def _mast_world(**lidar_config):
    cfg = {
        "sim": {},
        "plugins": [
            {f"{__name__}:_MastScene": {}},
            {"roqsim_sensors.plugins.lidar:LidarPlugin": lidar_config},
        ],
    }
    return load_config_from_dict(cfg)


def test_a_world_mounted_scanner_hangs_its_frame_off_the_world():
    """No exclude_body resolves, so the transform is measured from the world -- and says so.

    Naming the parent `base_link` regardless would, in a world with no base_link, orphan the frame,
    and in a world where some other robot has one, bolt the frame onto that robot at a pose measured
    from somewhere else.
    """
    engine = Engine(_mast_world(site="lidar"))
    engine.setup()
    st = _scan_hints(engine)["static_tf"]
    assert st["parent"] == "world"
    # And the numbers are the site's world pose, which is what "measured from the world" means.
    assert st["translation"] == [0.0, 0.0, 1.2]


def test_excluding_nothing_explicitly_is_not_an_empty_parent_frame():
    """`exclude_body: ''` must not publish a transform whose frame_id is the empty string, which
    tf2 drops -- the sensor frame would never enter the tree at all."""
    engine = Engine(_mast_world(site="lidar", exclude_body=""))
    engine.setup()
    assert _scan_hints(engine)["static_tf"]["parent"] == "world"


def test_a_resolved_exclude_body_is_still_the_parent():
    """With no carrier, the transform is measured from the excluded body and named for it."""
    engine = Engine(_mast_world(site="lidar", exclude_body="mast"))
    engine.setup()
    st = _scan_hints(engine)["static_tf"]
    assert st["parent"] == "mast"
    assert st["translation"] == [0.0, 0.0, 1.2]  # the mast sits at the origin


class _Robot(Plugin):
    """A robot as far as a lidar can tell: a prefixed root body carrying the scan site, and an entity
    naming that body. The root is not called ``base_link``, so the parent can only come from the
    entity."""

    provides_entity = True

    def build(self, spec: mujoco.MjSpec, ctx: SimContext) -> None:
        chassis = spec.worldbody.add_body(name="rb_chassis", pos=[0.0, 0.0, 0.3])
        chassis.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.2, 0.2, 0.05])
        chassis.add_site(name="rb_lidar", pos=[0.1, 0.0, 0.2])

    def configure(self, ctx: SimContext) -> None:
        ctx.entities.add(
            Entity(name=self.address, kind="robot", body="rb_chassis", meta={"prefix": "rb_"})
        )


def test_a_robot_carried_scanner_without_exclude_body_hangs_its_frame_off_the_robots_base():
    """Nothing is excluded by default, and the frame's parent is the carrying robot's root body."""
    engine = Engine(
        load_config_from_dict(
            {
                "sim": {},
                "components": [
                    {
                        f"{__name__}:_Robot": {},
                        "name": "robot",
                        "components": [
                            {"roqsim_sensors.plugins.lidar:LidarPlugin": {"site": "lidar"}}
                        ],
                    }
                ],
            }
        )
    )
    engine.setup()
    (lidar,) = [p for p in engine.plugins if isinstance(p, LidarPlugin)]
    assert lidar.exclude_body == "" and lidar._bodyexclude == -1
    st = _scan_hints(engine)["static_tf"]
    assert st["parent"] == "chassis"
    assert np.allclose(st["translation"], [0.1, 0.0, 0.2])


# -- tf_parent: the frame's parent, decoupled from what the rays skip -----------------------------


class _HousedScene(Plugin):
    """A scanner in a housing body of its own, on a base: the housing is excluded, the base is the
    link the vendor hangs the frame from."""

    def build(self, spec: mujoco.MjSpec, ctx: SimContext) -> None:
        base = spec.worldbody.add_body(name="base", pos=[0, 0, 0.5])
        base.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.2, 0.2, 0.05])
        housing = base.add_body(name="housing", pos=[0, 0, 0.5])
        # The site sits INSIDE the housing geom, so only excluding the housing lets a ray out.
        housing.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.05, 0.05, 0.05])
        housing.add_site(name="lidar", pos=[0, 0, 0.0])
        spec.worldbody.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX, pos=[2, 0, 1.0], size=[0.05, 5, 0.5]
        )


def _housed_world(**lidar_config):
    return load_config_from_dict(
        {
            "sim": {},
            "plugins": [
                {f"{__name__}:_HousedScene": {}},
                {"roqsim_sensors.plugins.lidar:LidarPlugin": lidar_config},
            ],
        }
    )


def test_tf_parent_names_and_measures_the_frame_independently_of_exclude_body():
    engine = Engine(
        _housed_world(site="lidar", exclude_body="housing", tf_parent="base", rays=4, max_range=4.0)
    )
    engine.setup()
    st = _scan_hints(engine)["static_tf"]
    assert st["parent"] == "base"
    assert np.allclose(st["translation"], [0.0, 0.0, 0.5])  # housing sits 0.5 above the base
    engine.reset()
    engine.step()
    # ...while the rays still skip the housing and reach the wall.
    assert np.isclose(_scan(engine).ranges[0], 1.95, atol=1e-3)


def test_without_tf_parent_the_excluded_body_is_still_the_parent():
    engine = Engine(_housed_world(site="lidar", exclude_body="housing"))
    engine.setup()
    st = _scan_hints(engine)["static_tf"]
    assert st["parent"] == "housing" and np.allclose(st["translation"], [0.0, 0.0, 0.0])


def test_a_tf_parent_that_is_not_a_body_is_refused():
    engine = Engine(_housed_world(site="lidar", exclude_body="housing", tf_parent="chassis"))
    with pytest.raises(RuntimeError, match="tf_parent 'chassis' not found"):
        engine.setup()
