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


# -- the two things the 3D lidars used to get wrong ------------------------------------------------
#
# Both were live until the raycast seam and the shared lidar base landed: this plugin passed
# ``geomgroup=None`` (so an absent obstacle was still a cloud point) and never applied ``max_range``
# (so a return past the device's range was still a point, because ``cutoff`` is a culling hint and
# not a clamp). The 2D ``lidar`` had always done both; these keep the pair from drifting again.


class _SingleWallScene(Plugin):
    """One wall at ``WALL_X`` on +x. A class attribute rather than config, so the scene stays
    referable by module path the way the other fixture in this file is."""

    provides_world = True
    WALL_X = 2.0

    def build(self, spec: mujoco.MjSpec, ctx: SimContext) -> None:
        spec.worldbody.add_site(name="lidar", pos=[0, 0, 0.1])
        body = spec.worldbody.add_body(name="wall", pos=[self.WALL_X, 0, 0.1])
        body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.05, 1, 1])


def _single_ray_world(wall_x: float, **cfg):
    """One ray along +x, so a point count is a yes/no about that one wall."""
    _SingleWallScene.WALL_X = wall_x
    return load_config_from_dict(
        {
            "sim": {},
            "plugins": [
                {f"{__name__}:_SingleWallScene": {}},
                {
                    "roqsim_sensors.plugins.livox_mid360:LivoxMid360Plugin": {
                        "horizontal_rays": 1,
                        "vertical_rays": 1,
                        "h_fov_min": 0.0,
                        "h_fov_max": 0.0,
                        "v_fov_min": 0.0,
                        "v_fov_max": 0.0,
                        "exclude_body": "",
                        **cfg,
                    }
                },
            ],
        }
    )


def _settled_cloud(engine: Engine, steps: int = 120):
    """Step past the 10 Hz cast gate -- one step reads whatever the previous cast left behind."""
    for _ in range(steps):
        engine.step()
    return _cloud(engine)


def test_an_absent_obstacle_is_not_a_cloud_point():
    """The presence mask, which this plugin used to skip by passing ``geomgroup=None``.

    The geom is left fully OPAQUE, so the alpha-zeroing half of ``presence.set_present`` cannot be
    what hides it -- only the ``geomgroup`` mask can.
    """
    from roqsim.presence import ABSENT_GEOM_GROUP

    engine = Engine(_single_ray_world(2.0, max_range=10.0))
    engine.setup()
    engine.reset()
    assert len(_settled_cloud(engine).points) == 1

    m = engine.ctx.model
    wall = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "wall")
    for g in range(m.ngeom):
        if m.geom_bodyid[g] == wall:
            m.geom_group[g] = ABSENT_GEOM_GROUP
            m.geom_rgba[g][3] = 1.0
    assert len(_settled_cloud(engine).points) == 0


class _GroundPlaneScene(Plugin):
    """A bare ground plane, with the sensor 1 m above it.

    A *plane* is the geometry that makes ``max_range`` load-bearing. MuJoCo culls a compact geom
    whose bounding volume is past ``cutoff``, so a distant wall never reaches the clamp -- it is
    simply a miss. A plane is unbounded and always tested, so a shallow downward ray reports a hit
    far beyond ``cutoff``, which is precisely what "``cutoff`` is a culling hint, not a clamp" means.
    """

    provides_world = True

    def build(self, spec: mujoco.MjSpec, ctx: SimContext) -> None:
        spec.worldbody.add_site(name="lidar", pos=[0, 0, 1.0])
        spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05])


def _grazing_world(**cfg):
    """A single ray angled 0.02 rad below horizontal from 1 m up -> a floor hit near 50 m."""
    return load_config_from_dict(
        {
            "sim": {},
            "plugins": [
                {f"{__name__}:_GroundPlaneScene": {}},
                {
                    "roqsim_sensors.plugins.livox_mid360:LivoxMid360Plugin": {
                        "horizontal_rays": 1,
                        "vertical_rays": 1,
                        "h_fov_min": 0.0,
                        "h_fov_max": 0.0,
                        "v_fov_min": -0.02,
                        "v_fov_max": -0.02,
                        "exclude_body": "",
                        **cfg,
                    }
                },
            ],
        }
    )


def test_a_return_beyond_max_range_is_not_a_cloud_point():
    """``max_range`` is a clamp here, not just the ``cutoff`` culling hint handed to MuJoCo.

    The floor hit sits near 50 m, and the plane is never culled, so with ``max_range`` at 10 m
    MuJoCo still *reports* the hit and the plugin is what has to reject it.
    """
    far = Engine(_grazing_world(max_range=100.0))
    far.setup()
    far.reset()
    reachable = _settled_cloud(far).points
    assert len(reachable) == 1
    hit_range = float(np.linalg.norm(reachable[0]))
    assert hit_range > 40.0, f"expected a far floor hit, got {hit_range:.2f} m"

    clamped = Engine(_grazing_world(max_range=10.0))
    clamped.setup()
    clamped.reset()
    assert len(_settled_cloud(clamped).points) == 0
