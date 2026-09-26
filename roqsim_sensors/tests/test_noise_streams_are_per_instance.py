"""Two unnamed sensors of one kind on two robots draw two noise streams, not one stream twice.

A manifest-injected component has no ``name:`` sibling, so ``Plugin.name`` falls back to the class
name and every robot's ``lidar`` is called ``LidarPlugin``. Keying a draw on that name gives every
such sensor the same Philox counter at the same ``sim_time``: bit-identical noise, dropout on the same
ray indices, and a fleet whose "independent" statistics are one sample. The key is the instance's
address, which is unique in a document.
"""

from __future__ import annotations

import roqsim  # noqa: F401, I001  (selects MuJoCo's GL backend before anything imports mujoco)

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from roqsim.config import load_config_from_dict  # noqa: E402
from roqsim.context import Entity  # noqa: E402
from roqsim.engine import Engine  # noqa: E402
from roqsim.plugin import Plugin  # noqa: E402
from roqsim_sensors.plugins.lidar import LidarPlugin  # noqa: E402


class _Robot(Plugin):
    """A box with a lidar site, one per entity, so two robots see the same room."""

    provides_entity = True

    def build(self, spec, ctx):
        prefix = self.config["prefix"]
        body = spec.worldbody.add_body(name=prefix + "chassis", pos=[0.0, 0.0, 0.3])
        body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.2, 0.2, 0.1])
        body.add_site(name=prefix + "lidar", pos=[0.1, 0.0, 0.2])

    def configure(self, ctx):
        prefix = self.config["prefix"]
        ctx.entities.add(
            Entity(
                name=self.address, kind="robot", body=prefix + "chassis", meta={"prefix": prefix}
            )
        )


def _robot(label: str, prefix: str) -> dict:
    return {
        f"{__name__}:_Robot": {"prefix": prefix},
        "name": label,
        # No `name:` on the lidar, as a manifest injects it: both fall back to the class name.
        "components": [{"lidar": {"site": "lidar", "range_stddev": 0.05}}],
    }


def test_two_unnamed_lidars_on_two_robots_draw_different_noise():
    cfg = load_config_from_dict(
        {"sim": {}, "components": [_robot("robot_a", "a_"), _robot("robot_b", "b_")]}
    )
    engine = Engine(cfg)
    engine.ctx.seed = 7
    engine.setup()
    engine.reset()
    try:
        for _ in range(60):
            engine.step()
        lidars = [p for p in engine.plugins if isinstance(p, LidarPlugin)]
        assert [p.name for p in lidars] == ["LidarPlugin", "LidarPlugin"]
        a, b = (np.asarray(p.latest.ranges) for p in lidars)
        assert np.isfinite(a).any(), "no returns at all: the room is not what the test assumes"
        assert not np.array_equal(a, b), "both sensors drew one noise stream"
    finally:
        engine.shutdown()
