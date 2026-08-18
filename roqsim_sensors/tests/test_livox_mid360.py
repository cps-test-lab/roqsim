"""Standalone Livox Mid-360 sanity checks -- a synthetic scene, no dependency on roqsim_mobile.

Confirms the plugin configures, casts a 3D ray grid, and exposes a PointCloud2 cloud endpoint.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np
from roqsim_sensors.plugins.livox_mid360 import LivoxMid360Plugin

from roqsim.config import load_config_from_dict
from roqsim.context import SimContext
from roqsim.engine import Engine
from roqsim.plugin import Plugin


class _OneWallScene(Plugin):
    """A single tall wall 2m in front of a lidar site at the origin, facing +x.

    Declares ``provides_world`` so the engine skips the default walled room (see world.py); the only
    geometry in the scene is this one wall, keeping the ray-hit counts below deterministic.
    """

    provides_world = True

    def build(self, spec: mujoco.MjSpec, ctx: SimContext) -> None:
        spec.worldbody.add_site(name="lidar", pos=[0, 0, 0.1])
        spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, pos=[2, 0, 0.1], size=[0.05, 5, 5])


def _world(**cfg):
    config = {
        "sim": {},
        "plugins": [
            {f"{__name__}:_OneWallScene": {}},
            {"roqsim_sensors.plugins.livox_mid360:LivoxMid360Plugin": cfg},
        ],
    }
    return load_config_from_dict(config)


def _endpoint(engine: Engine):
    return next(e for e in engine.ctx.interface.all() if e.name == "cloud")


def _cloud(engine: Engine):
    return _endpoint(engine).read()


def test_validate_config_rejects_bad_values():
    errors = LivoxMid360Plugin().validate_config(
        {"horizontal_rays": 0, "vertical_rays": -1, "max_range": -1, "dropout_percent": 150}
    )
    assert len(errors) == 4


def test_num_rays_is_the_grid_product():
    plugin = LivoxMid360Plugin({"horizontal_rays": 360, "vertical_rays": 56})
    assert plugin.num_rays == 360 * 56


def test_cloud_endpoint_declares_pointcloud2_and_frame():
    engine = Engine(_world(site="lidar", frame_id="livox_frame"))
    engine.setup()
    hints = _endpoint(engine).backend["ros2"]
    assert hints["type"] == "sensor_msgs.msg.PointCloud2"
    assert hints["frame_id"] == "livox_frame"
    assert hints["static_tf"]["parent"] == "base_link"  # child is frame_id, applied by the bridge


def test_frame_id_defaults_to_the_site():
    engine = Engine(_world(site="lidar"))
    engine.setup()
    assert _endpoint(engine).backend["ros2"]["frame_id"] == "lidar"


def test_rays_hit_the_wall_at_expected_range():
    # A horizontal ring only (v_fov = 0) so the whole grid points at the wall's mid-height.
    engine = Engine(
        _world(
            horizontal_rays=8,
            vertical_rays=1,
            v_fov_min=0.0,
            v_fov_max=0.0,
            max_range=10.0,
        )
    )
    engine.setup()
    engine.reset()
    engine.step()
    cloud = _cloud(engine)
    # The ray along +x hits the wall face at x=1.95 (2.0 - half-thickness); points come back in the
    # sensor frame, so that hit is the point with the largest x.
    assert cloud.points.shape[1] == 3
    assert np.isclose(cloud.points[:, 0].max(), 1.95, atol=1e-3)
    # Only the +x hemisphere sees the wall; a full ring of 8 rays -> 3 finite returns (+x and its two
    # diagonal neighbours), the rest miss into the void.
    assert cloud.points.shape[0] == 3


def test_vertical_fov_spreads_points_in_z():
    engine = Engine(
        _world(horizontal_rays=16, vertical_rays=16, v_fov_min=-0.2, v_fov_max=0.2, max_range=10.0)
    )
    engine.setup()
    engine.reset()
    engine.step()
    cloud = _cloud(engine)
    # The wall spans a range of elevations, so the returns are not all coplanar in z.
    assert np.ptp(cloud.points[:, 2]) > 0.1
    # Elevation stays within the configured band: |z| <= r * tan(0.2) for a hit at horizontal range r.
    horiz = np.hypot(cloud.points[:, 0], cloud.points[:, 1])
    assert np.all(np.abs(cloud.points[:, 2]) <= horiz * math.tan(0.2) + 1e-3)
