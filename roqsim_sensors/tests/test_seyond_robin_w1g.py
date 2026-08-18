"""Standalone Seyond Robin W1G sanity checks -- a synthetic scene, no dependency on roqsim_mobile.

Confirms the forward-facing solid-state lidar configures, casts a *bounded* 3D ray grid (azimuth is a
band with inclusive endpoints, not a wrapping 360deg sweep), and exposes a PointCloud2 cloud endpoint.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np
from roqsim_sensors.plugins.seyond_robin_w1g import SeyondRobinW1GPlugin

from roqsim.config import load_config_from_dict
from roqsim.context import SimContext
from roqsim.engine import Engine
from roqsim.plugin import Plugin


class _OneWallScene(Plugin):
    """A single tall wall 3 m in front of a lidar site at the origin, facing +x (the boresight)."""

    provides_world = True

    def build(self, spec: mujoco.MjSpec, ctx: SimContext) -> None:
        spec.worldbody.add_site(name="robin_w1g", pos=[0, 0, 0.1])
        spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, pos=[3, 0, 0.1], size=[0.05, 8, 8])


def _world(**cfg):
    config = {
        "sim": {},
        "plugins": [
            {f"{__name__}:_OneWallScene": {}},
            {"roqsim_sensors.plugins.seyond_robin_w1g:SeyondRobinW1GPlugin": cfg},
        ],
    }
    return load_config_from_dict(config)


def _endpoint(engine: Engine):
    return next(e for e in engine.ctx.interface.all() if e.name == "cloud")


def _cloud(engine: Engine):
    return _endpoint(engine).read()


def test_defaults_are_the_datasheet_forward_fov():
    p = SeyondRobinW1GPlugin()
    assert p.h_rays == 192 and p.v_rays == 112
    assert math.isclose(p.h_fov_min, math.radians(-60.0)) and math.isclose(
        p.h_fov_max, math.radians(60.0)
    )
    assert math.isclose(p.v_fov_min, math.radians(-35.0)) and math.isclose(
        p.v_fov_max, math.radians(35.0)
    )
    assert p.range_min == 0.1 and p.range_max == 70.0
    # A bounded forward FoV: azimuth must NOT wrap like the Mid-360 dome it subclasses.
    assert p.AZIMUTH_WRAPS is False


def test_azimuth_uses_inclusive_endpoints_not_wrapping():
    # 3 azimuth samples across +-60 deg with inclusive endpoints -> exactly -60, 0, +60 (no wrap).
    p = SeyondRobinW1GPlugin(
        {"horizontal_rays": 3, "vertical_rays": 1, "v_fov_min": 0.0, "v_fov_max": 0.0}
    )
    dirs = p._build_directions()
    az = np.degrees(np.arctan2(dirs[:, 1], dirs[:, 0]))
    assert np.allclose(sorted(az), [-60.0, 0.0, 60.0], atol=1e-6)


def test_cloud_endpoint_declares_pointcloud2_topic_and_frame():
    engine = Engine(_world(frame_id="seyond_lidar"))
    engine.setup()
    hints = _endpoint(engine).backend["ros2"]
    assert hints["type"] == "sensor_msgs.msg.PointCloud2"
    assert hints["topic"] == "seyond/points"  # Seyond driver default, not livox/lidar
    assert hints["frame_id"] == "seyond_lidar"


def test_boresight_ray_hits_wall_in_front():
    engine = Engine(
        _world(horizontal_rays=41, vertical_rays=1, v_fov_min=0.0, v_fov_max=0.0, max_range=10.0)
    )
    engine.setup()
    engine.reset()
    engine.step()
    cloud = _cloud(engine)
    # The +x boresight ray hits the wall face at x=2.95 (3.0 - half-thickness); points are in the
    # sensor frame, so that hit has the largest x.
    assert cloud.points.shape[1] == 3
    assert np.isclose(cloud.points[:, 0].max(), 2.95, atol=1e-3)


def test_returns_stay_within_the_forward_fov_bounds():
    engine = Engine(_world(horizontal_rays=120, vertical_rays=70, max_range=10.0))
    engine.setup()
    engine.reset()
    engine.step()
    pts = _cloud(engine).points
    az = np.degrees(np.arctan2(pts[:, 1], pts[:, 0]))
    el = np.degrees(np.arctan2(pts[:, 2], np.hypot(pts[:, 0], pts[:, 1])))
    assert az.min() >= -60.0 - 1e-3 and az.max() <= 60.0 + 1e-3
    assert el.min() >= -35.0 - 1e-3 and el.max() <= 35.0 + 1e-3
    # Nothing behind the sensor: a forward lidar never returns points with x < 0.
    assert np.all(pts[:, 0] >= 0.0)
