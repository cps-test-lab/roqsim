"""Every humanoid whose manifest declares its ``lidar`` directly: what its own scan hits of itself.

Two properties are pinned per robot, from the plugin's own ray pattern and its resolved exclusion:

* **No ray starts inside robot geometry.** A first surface met from inside (normal . ray > 0) means
  the scan origin lies within a geom that is not excluded, and the published scan reads that geom's
  inner face on every such ray.
* **The robot bodies hit from outside, with their ray counts.** These are real returns of a sensor
  that sees part of its own robot; a change to the model, the stance, the mount or the exclusion
  shows up here.

Each robot is spawned as a world spawns it -- ``spawn_robot`` with a prefix, its manifest's controller
setting the stance at reset (a humanoid's legs, the G2's hanging arms) -- into the default world
(``empty_room``: a floor and perimeter walls), so every non-world body is the robot's.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import mujoco
import numpy as np
import pytest

from roqsim import raycast
from roqsim.config import load_config_from_dict
from roqsim.engine import Engine

PREFIX = "r_"

#: model -> (sim timestep, excluded body, {robot body hit from outside: ray count}). No manifest here
#: sets exclude_body, so the lidar's default (base_link) is what is excluded.
CASES = {
    "unitree_g1": (0.002, "base_link", {}),
    # The waist is a jointed link, not the excluded pelvis, so its surface 0.062 m out around the whole
    # fan is a real return by the rule a lidar follows. Whether the site belongs at this height is an
    # open question about the mount pose, not about the exclusion.
    "oli": (0.001, "base_link", {"waist_pitch_link": 360}),
    # The torso column stands behind the chassis-front site and fills the rear of the fan.
    "agibot_g2": (0.002, "base_link", {"body_link1": 89}),
}


def _spawn(model: str, timestep: float) -> Engine:
    world = {
        "sim": {"timestep": timestep},
        "components": [{"spawn_robot": {"model": model, "prefix": PREFIX}, "name": "robot"}],
    }
    engine = Engine(load_config_from_dict(world, base_dir=Path(".")))
    engine.ctx.seed = 1  # a test driving an Engine is the driver, and the seed is driver-owned
    engine.setup()
    engine.reset()
    engine.step()  # the rate gate starts open, so the first step casts
    return engine


def _body(model, bid: int) -> str:
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(bid))


@pytest.mark.parametrize("model", list(CASES))
def test_the_scan_meets_no_robot_geometry_from_inside(model):
    timestep, excluded, expected_outside = CASES[model]
    engine = _spawn(model, timestep)
    try:
        m, d = engine.ctx.model, engine.ctx.data
        lidars = [p for p in engine.plugins if type(p).__name__ == "LidarPlugin"]
        assert [p.address for p in lidars] == ["robot.lidar"]
        (lidar,) = lidars
        foreign = [_body(m, b) for b in range(1, m.nbody) if not _body(m, b).startswith(PREFIX)]
        assert not foreign, f"bodies not of the spawned robot: {foreign}"
        assert lidar._bodyexclude >= 0 and _body(m, lidar._bodyexclude) == PREFIX + excluded

        # The plugin's own rays from its own site: `_local_dirs @ rot.T`, as post_step casts them.
        origin = d.site_xpos[lidar._site_id].copy()
        dirs = lidar._build_directions() @ d.site_xmat[lidar._site_id].reshape(3, 3).T
        hits = raycast.cast(
            m,
            d,
            origin,
            dirs,
            cutoff=lidar.range_max,
            bodyexclude=lidar._bodyexclude,
            out=raycast.buffers(len(dirs), normals=True),
        )
        np.testing.assert_array_equal(hits.geomid, lidar._hits.geomid)

        on_robot = (hits.geomid >= 0) & (m.geom_bodyid[np.maximum(hits.geomid, 0)] != 0)
        from_inside = on_robot & (np.einsum("ij,ij->i", hits.normal, dirs) > 0)
        inside = Counter(_body(m, m.geom_bodyid[g]) for g in hits.geomid[from_inside])
        assert not inside, f"rays start inside robot geometry: {dict(inside)}"

        outside = Counter(_body(m, m.geom_bodyid[g]) for g in hits.geomid[on_robot])
        assert dict(outside) == {PREFIX + b: n for b, n in expected_outside.items()}
    finally:
        engine.shutdown()
