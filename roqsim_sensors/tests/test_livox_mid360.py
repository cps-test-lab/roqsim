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
    # The child is that frame_id, applied by the bridge. The parent is the world: this scene has no
    # base_link, and the transform is measured from the world -- see test_lidar's mount-TF tests.
    assert hints["static_tf"]["parent"] == "world"


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


# -- too close and no return, as the device's driver publishes them -------------------------------


def test_too_close_and_no_return_are_dropped_by_default():
    # One ray at a wall 0.05 m away (inside the 0.1 m blind zone), one into the void.
    engine = Engine(_single_ray_world(0.1, max_range=10.0))
    engine.setup()
    engine.reset()
    assert len(_settled_cloud(engine).points) == 0


def test_origin_publishes_a_point_at_the_sensor_for_too_close_and_no_return():
    # The wall face is at 0.05 m: nearer than range_min 0.1, so the device cannot measure it.
    near = Engine(_single_ray_world(0.1, max_range=10.0, too_close="origin"))
    near.setup()
    near.reset()
    np.testing.assert_array_equal(_settled_cloud(near).points, [[0.0, 0.0, 0.0]])

    void = Engine(_single_ray_world(20.0, max_range=10.0, no_return="origin"))
    void.setup()
    void.reset()
    np.testing.assert_array_equal(_settled_cloud(void).points, [[0.0, 0.0, 0.0]])


def test_a_measured_return_is_unaffected_by_the_output_settings():
    engine = Engine(_single_ray_world(2.0, max_range=10.0, too_close="origin", no_return="origin"))
    engine.setup()
    engine.reset()
    (point,) = _settled_cloud(engine).points
    assert np.isclose(point[0], 1.95, atol=1e-3)


def test_validate_config_refuses_an_unknown_output():
    errors = LivoxMid360Plugin().validate_config({"too_close": "-inf", "no_return": "nan"})
    assert any("too_close" in e for e in errors) and any("no_return" in e for e in errors)


# -- the mid360 device model -----------------------------------------------------------------------

MANUAL_ORIGIN_ABOVE_BOTTOM = 0.047  # Livox Mid-360 User Manual v1.2, Appendix, p. 19
MANUAL_HEIGHT = 0.060
MANUAL_FOOTPRINT = 0.065


def _device_world(**spawn):
    return load_config_from_dict(
        {
            "sim": {},
            "plugins": [{"spawn_sensor": {"model": "mid360", **spawn}, "name": "lidar"}],
        }
    )


def test_the_device_compiles_from_committed_files_only():
    from roqsim_sensors.models import MODELS_DIR

    xml = (MODELS_DIR / "mid360" / "mid360.xml").read_text()
    assert "<mesh" not in xml, "a mesh reference would make the device depend on a generated asset"
    mujoco.MjModel.from_xml_path(str(MODELS_DIR / "mid360" / "mid360.xml"))


def test_the_scan_site_is_the_manual_origin_above_the_housing():
    from roqsim_sensors.models import MODELS_DIR

    m = mujoco.MjModel.from_xml_path(str(MODELS_DIR / "mid360" / "mid360.xml"))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    site = d.site_xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "mid360")]
    body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "mount")
    lo = np.full(3, np.inf)
    hi = np.full(3, -np.inf)
    for g in range(m.ngeom):
        if m.geom_bodyid[g] != body or g == mujoco.mj_name2id(
            m, mujoco.mjtObj.mjOBJ_GEOM, "mid360_connector"
        ):
            continue
        lo = np.minimum(lo, d.geom_xpos[g] - m.geom_aabb[g][3:])
        hi = np.maximum(hi, d.geom_xpos[g] + m.geom_aabb[g][3:])
    np.testing.assert_allclose(site[2] - lo[2], MANUAL_ORIGIN_ABOVE_BOTTOM, atol=1e-4)
    np.testing.assert_allclose(hi[2] - lo[2], MANUAL_HEIGHT, atol=2e-4)
    np.testing.assert_allclose(hi[:2] - lo[:2], [MANUAL_FOOTPRINT] * 2, atol=1e-4)
    np.testing.assert_allclose(m.body_mass[body], 0.265)


def test_the_mount_publishes_livox_frame_at_the_scan_site_and_the_driver_conventions():
    engine = Engine(_device_world(pos=[0.0, 0.0, 1.0]))
    engine.setup()
    (plugin,) = [p for p in engine.plugins if isinstance(p, LivoxMid360Plugin)]
    assert (plugin.frame_id, plugin.too_close, plugin.no_return) == (
        "livox_frame",
        "origin",
        "origin",
    )
    assert (plugin.range_min, plugin.range_max, plugin.exclude_body) == (0.1, 40.0, "mount")
    cloud = _endpoint(engine).backend["ros2"]
    assert cloud["frame_id"] == "livox_frame" and cloud["topic"] == "livox/lidar"
    assert "static_tf" not in cloud  # the mount publishes the frames: chain instead
    frames = next(e for e in engine.ctx.interface.all() if e.name == "frames")
    (tf,) = frames.backend["ros2"]["static_tf"]
    assert (tf["parent"], tf["child"]) == ("world", "livox_frame")
    np.testing.assert_allclose(tf["translation"], [0.0, 0.0, 1.0], atol=1e-9)
    m, d = engine.ctx.model, engine.ctx.data
    mujoco.mj_forward(m, d)
    np.testing.assert_allclose(d.site_xpos[plugin._site_id], [0.0, 0.0, 1.0], atol=1e-9)


def test_the_device_publishes_one_point_per_ray_with_nothing_in_range():
    # The default world is a walled room, so point the dome at the open sky: every ray is no return.
    engine = Engine(_device_world(pos=[0.0, 0.0, 50.0]))
    engine.ctx.seed = (
        1  # the manifest's range noise draws, and a test driving an Engine owns the seed
    )
    engine.setup()
    engine.reset()
    (plugin,) = [p for p in engine.plugins if isinstance(p, LivoxMid360Plugin)]
    engine.step()
    points = plugin.latest.points
    assert len(points) == plugin.num_rays
    assert not points.any()
