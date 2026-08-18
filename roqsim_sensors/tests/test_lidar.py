"""Standalone lidar sanity checks -- a synthetic scene, no dependency on roqsim_mobile.

The richer behavioral coverage (walls, range noise, dropout) already lives in
roqsim_mobile/tests/test_turtlebot_scene.py against the real TurtleBot 4 scene; this file just
confirms the moved package configures and ray-casts correctly on its own.
"""

from __future__ import annotations

import mujoco
import numpy as np
from roqsim_sensors.plugins.lidar import LidarPlugin

from roqsim.config import load_config_from_dict
from roqsim.context import SimContext
from roqsim.engine import Engine
from roqsim.plugin import Plugin


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


def test_validate_config_rejects_bad_values():
    errors = LidarPlugin().validate_config({"rays": 0, "max_range": -1, "dropout_percent": 150})
    assert len(errors) == 3


def _scan_hints(engine: Engine) -> dict:
    ep = next(e for e in engine.ctx.interface.all() if e.name == "scan")
    return ep.backend["ros2"]


def test_frame_id_defaults_to_the_site():
    """The scan is stamped in the frame the rays are cast from, not a hardcoded robot's frame.

    This used to be hardwired to "rplidar_link" for every robot, so a Husky published its scan in a
    TurtleBot's frame. The static mount TF's child comes from the same hint, so the two cannot
    disagree.
    """
    engine = Engine(_world(site="lidar"))
    engine.setup()
    hints = _scan_hints(engine)
    assert hints["frame_id"] == "lidar"
    assert hints["static_tf"]["parent"] == "base_link"  # child is frame_id, applied by the bridge


def test_frame_id_can_be_declared_by_the_model():
    """A model whose real description names the frame (TurtleBot 4's URDF: rplidar_link) says so."""
    engine = Engine(_world(site="lidar", frame_id="rplidar_link"))
    engine.setup()
    assert _scan_hints(engine)["frame_id"] == "rplidar_link"


def test_ray_along_x_hits_the_wall_at_expected_range():
    # max_range below the empty_room perimeter (walls at +-5) so the -x miss actually misses.
    engine = Engine(_world(rays=4, angle_min=0.0, angle_max=2 * np.pi, max_range=4.0))
    engine.setup()
    engine.reset()
    engine.step()
    scan = _scan(engine)
    # ray 0 points along +x (angle 0) straight at the wall face at x=1.95 (2.0 - half-thickness).
    assert np.isclose(scan.ranges[0], 1.95, atol=1e-3)
    # ray pointing -x (angle pi, index len//2 for 4 evenly spaced rays): nothing in range -> inf.
    assert np.isinf(scan.ranges[2])
