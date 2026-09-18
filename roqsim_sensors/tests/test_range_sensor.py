"""range_sensor: a ray grid published as one LaserScan -- the cliff / IR proximity / ToF model.

What a consumer of these sensors relies on: the nearest return in the grid, `+inf` where nothing
is in range (a hole under a cliff sensor), the grid's length and row-major order, and that the
sensor points where its site points. The scene puts a floor at a known standoff under a pitched
sensor and a wall at a known distance in front of a level one.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np
import pytest
from roqsim_sensors.plugins.range_sensor import RangeSensorPlugin

from roqsim.config import load_config_from_dict
from roqsim.context import SimContext
from roqsim.engine import Engine
from roqsim.plugin import Plugin

#: Wall face along +x in front of the level sensor, and the level sensor's height over the floor.
WALL_FACE = 0.10
HEIGHT = 0.05
#: The pitched sensor looks 80 deg down from a site 0.0192 m over the floor, as a Create 3 cliff
#: sensor does; its boresight meets the floor at HEIGHT_CLIFF / sin(80 deg).
HEIGHT_CLIFF = 0.0192
PITCH = math.radians(80.0)


class _Scene(Plugin):
    """A floor with a hole, a wall in front of a level sensor, a pitched sensor over the floor."""

    def build(self, spec: mujoco.MjSpec, ctx: SimContext) -> None:
        # Floor as two slabs with a gap at x in [1.0, 2.0]: the hole.
        spec.worldbody.add_geom(
            name="floor_a", type=mujoco.mjtGeom.mjGEOM_BOX, pos=[0, 0, -0.05], size=[1, 5, 0.05]
        )
        spec.worldbody.add_geom(
            name="floor_b", type=mujoco.mjtGeom.mjGEOM_BOX, pos=[3, 0, -0.05], size=[1, 5, 0.05]
        )
        # Level sensor at the origin, facing +x, a wall 0.1 m ahead.
        spec.worldbody.add_site(name="front", pos=[0, 0, HEIGHT])
        spec.worldbody.add_geom(
            name="wall",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[WALL_FACE + 0.05, 0, 0.5],
            size=[0.05, 1, 0.5],
        )
        # Pitched sensors: one over the floor, one over the hole. Pitch about y by +80 deg turns +x
        # towards -z. Quaternion (w, x, y, z) for a rotation of PITCH about y.
        q = [math.cos(PITCH / 2), 0.0, math.sin(PITCH / 2), 0.0]
        spec.worldbody.add_site(name="cliff_floor", pos=[-0.5, 0, HEIGHT_CLIFF], quat=q)
        spec.worldbody.add_site(name="cliff_hole", pos=[1.5, 0, HEIGHT_CLIFF], quat=q)


def _engine(*sensors: dict) -> Engine:
    cfg = {
        "sim": {},
        "plugins": [{f"{__name__}:_Scene": {}}]
        + [{"roqsim_sensors.plugins.range_sensor:RangeSensorPlugin": s} for s in sensors],
    }
    engine = Engine(load_config_from_dict(cfg))
    engine.ctx.seed = 1
    engine.setup()
    engine.reset()
    engine.step()
    return engine


def _scans(engine: Engine):
    return [e.read() for e in engine.ctx.interface.all() if e.name == "scan"]


CLIFF = {"h_rays": 1, "v_rays": 1, "range_min": 0.0001, "max_range": 0.15, "rate_hz": 62.0}


def test_a_single_ray_over_the_floor_reads_the_standoff():
    """The comparison a cliff detector makes: floor at the standoff, nothing above threshold."""
    (scan,) = _scans(_engine({"site": "cliff_floor", **CLIFF}))
    assert scan.ranges.shape == (1,)
    assert scan.ranges[0] == pytest.approx(HEIGHT_CLIFF / math.sin(PITCH), abs=2e-3)
    assert scan.angle_min == scan.angle_max == 0.0
    assert scan.angle_increment == 0.0


def test_a_single_ray_over_a_hole_reads_no_return():
    """REP 117: nothing within max_range is +inf, which a min-over-ranges consumer reads as a cliff."""
    (scan,) = _scans(_engine({"site": "cliff_hole", **CLIFF}))
    assert math.isinf(scan.ranges[0]) and scan.ranges[0] > 0


def test_a_grid_facing_a_wall_reads_the_wall_in_every_cell():
    """A 5x5 IR sensor: 25 finite ranges, the centre ray exactly the wall distance and every other
    ray slightly farther (an off-axis ray reaches the plane at distance / cos)."""
    (scan,) = _scans(
        _engine(
            {
                "site": "front",
                "h_rays": 5,
                "v_rays": 5,
                "h_fov": math.radians(10),
                "v_fov": math.radians(10),
                "range_min": 0.025,
                "max_range": 0.2,
            }
        )
    )
    assert scan.ranges.shape == (25,)
    assert np.all(np.isfinite(scan.ranges))
    centre = scan.ranges[12]
    assert centre == pytest.approx(WALL_FACE, abs=1e-3)
    assert scan.ranges.min() == pytest.approx(centre, abs=1e-6)
    assert scan.ranges.max() > centre
    assert scan.angle_min == pytest.approx(-math.radians(5))
    assert scan.angle_max == pytest.approx(math.radians(5))
    assert scan.angle_increment == pytest.approx(math.radians(2.5))


def test_rows_are_top_first_and_columns_sweep_left_to_right():
    """The published order is the grid's, row-major, so an index means one direction."""
    plugin = RangeSensorPlugin(
        {"h_rays": 3, "v_rays": 2, "h_fov": math.radians(20), "v_fov": math.radians(20)}
    )
    d = plugin._build_directions()
    assert d.shape == (6, 3)
    # First row is the top one (positive z), second the bottom.
    assert np.all(d[:3, 2] > 0) and np.all(d[3:, 2] < 0)
    # Within a row, y runs from -h_fov/2 (right) to +h_fov/2 (left): increasing y.
    assert d[0, 1] < d[1, 1] < d[2, 1]
    assert np.allclose(np.linalg.norm(d, axis=1), 1.0)


def test_the_grid_is_lazy_free_and_named_by_its_site():
    engine = _engine({"site": "front"})
    ep = next(e for e in engine.ctx.interface.all() if e.name == "scan")
    assert ep.backend["ros2"]["type"] == "sensor_msgs.msg.LaserScan"
    assert ep.backend["ros2"]["frame_id"] == "front"
    assert ep.backend["ros2"]["topic"] == "range"


def test_a_topic_override_names_the_scan():
    engine = _engine({"site": "front", "topics": {"scan": "_internal/cliff_front_left/scan"}})
    ep = next(e for e in engine.ctx.interface.all() if e.name == "scan")
    assert ep.backend["ros2"]["topic"] == "_internal/cliff_front_left/scan"


@pytest.mark.parametrize(
    "config, message",
    [
        ({"rays": 5}, "lidar fan"),
        ({"angle_min": 0.0}, "lidar fan"),
        ({"h_rays": 0}, ">= 1"),
        ({"h_rays": 1, "h_fov": 0.1}, "single ray"),
        ({"h_rays": 3, "h_fov": 0.0}, "h_fov"),
        ({"v_rays": 3, "v_fov": 4.0}, "v_fov"),
        ({"max_range": 0}, "max_range"),
    ],
)
def test_validate_config_names_each_bad_key(config, message):
    errors = RangeSensorPlugin(config).validate_config(config)
    assert any(message in e for e in errors), errors


def test_the_defaults_are_a_single_ray():
    assert RangeSensorPlugin({}).validate_config({}) == []
    assert RangeSensorPlugin({}).num_rays == 1
