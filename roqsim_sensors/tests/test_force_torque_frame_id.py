"""The wrench's header names a frame TF can resolve, in each of the three reporting frames.

TF knows a robot's frames without the entity's MJCF prefix, and a site is a frame only once its
mount transform is published.
"""

from __future__ import annotations

from roqsim.config import load_config_from_dict
from roqsim.engine import Engine


def _hints(frame):
    """The wrench endpoint's ros2 hints, and the arm entity's root body as compiled."""
    cfg = load_config_from_dict(
        {
            "sim": {},
            "components": [
                {
                    "spawn_arm": {"model": "ur5e", "prefix": "ur5e_"},
                    "name": "ur5e",
                    "components": [{"force_torque": {"site": "fts_site", "frame": frame}}],
                }
            ],
        }
    )
    engine = Engine(cfg)
    engine.ctx.seed = 0
    engine.setup()
    try:
        ep = next(e for e in engine.ctx.interface.all() if e.name == "wrench")
        return ep.backend["ros2"], engine.ctx.entities.get("ur5e").body
    finally:
        engine.shutdown()


def test_the_sensor_frame_is_the_bare_site_with_its_mount_transform():
    hints, _ = _hints("sensor")
    assert hints["frame_id"] == "fts_site"
    assert not hints["static_tf"]["parent"].startswith("ur5e_")
    assert len(hints["static_tf"]["translation"]) == 3 and len(hints["static_tf"]["rotation"]) == 4


def test_the_base_frame_is_the_entitys_root_body_unprefixed():
    hints, root = _hints("base")
    assert hints["frame_id"] == root.removeprefix("ur5e_")


def test_the_world_frame_is_world():
    hints, _ = _hints("world")
    assert hints["frame_id"] == "world"
