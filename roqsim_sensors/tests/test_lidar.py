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
    engine = Engine(_world(rays=4, angle_min=0.0, angle_max=2 * np.pi, max_range=4.0))
    engine.setup()
    engine.reset()
    engine.step()
    scan = _scan(engine)
    # ray 0 points along +x (angle 0) straight at the wall face at x=1.95 (2.0 - half-thickness).
    assert np.isclose(scan.ranges[0], 1.95, atol=1e-3)
    # ray pointing -x (angle pi, index len//2 for 4 evenly spaced rays): nothing in range -> inf.
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

    It used to name the parent `base_link` regardless, which in a world with no base_link is an
    orphaned frame and in a world where some other robot has one is a frame bolted onto that robot
    at a pose measured from somewhere else.
    """
    engine = Engine(_mast_world(site="lidar"))
    engine.setup()
    st = _scan_hints(engine)["static_tf"]
    assert st["parent"] == "world"
    # And the numbers are the site's world pose, which is what "measured from the world" means.
    assert st["translation"] == [0.0, 0.0, 1.2]


def test_excluding_nothing_explicitly_is_not_an_empty_parent_frame():
    """`exclude_body: ''` used to publish a transform whose frame_id was the empty string, which
    tf2 drops -- so the sensor frame never entered the tree at all."""
    engine = Engine(_mast_world(site="lidar", exclude_body=""))
    engine.setup()
    assert _scan_hints(engine)["static_tf"]["parent"] == "world"


def test_a_resolved_exclude_body_is_still_the_parent():
    """The ordinary case is untouched: the transform is measured from that body and named for it."""
    engine = Engine(_mast_world(site="lidar", exclude_body="mast"))
    engine.setup()
    st = _scan_hints(engine)["static_tf"]
    assert st["parent"] == "mast"
    assert st["translation"] == [0.0, 0.0, 1.2]  # the mast sits at the origin
