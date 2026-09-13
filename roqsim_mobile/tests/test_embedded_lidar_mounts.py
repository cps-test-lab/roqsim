"""Every mobile base whose manifest declares its ``lidar`` directly: what its own scan hits of itself.

A robot that mounts a scanner device model (``spawn_sensor``) is covered by its own mount test. This
module covers the rest: bases whose lidar is a site in the robot's MJCF, cast with an ``exclude_body``
the manifest names. Two properties are pinned per robot, from the plugin's own ray pattern and its
resolved exclusion:

* **No ray starts inside robot geometry.** A first surface met from inside (normal . ray > 0) means
  the scan origin lies within a collision or visual geom that is not excluded, and the published scan
  reads that geom's inner face on every such ray.
* **The robot bodies hit from outside, with their ray counts.** These are real returns of a sensor
  that sees part of its own robot; a change to the model, the mount or the exclusion shows up here.

The robot is spawned through ``spawn_robot`` with a prefix into the default world (``empty_room``: a
floor and perimeter walls), so every non-world body is the robot's.

``makerspet_mini`` and ``oomwoo_one`` also declare their lidar directly; their scene tests already
recast the scan with normals and pin the same two properties, so they are not repeated here.
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

#: model -> (excluded body, {robot body hit from outside: ray count}).
CASES = {
    "husky_a200": ("base_link", {}),
    "clearpath_jackal": ("base_link", {}),
    # The chassis body's side panels stand in the scan plane ~0.29 m out, over a narrow sector.
    "panther": ("base_link", {"body_link": 12}),
    "warthog": ("base_link", {}),
    "lgdxrobot2": ("base_link", {}),
    "ridgeback": ("base_link", {}),
}

#: Pinned failures, each a property of the model as it stands rather than of the test.
XFAIL = {
    "ridgeback": "the lidar site lies inside chassis_link's deck geometry, which exclude_body "
    "(base_link) does not cover, so every ray starts inside the chassis; the mount is pending a "
    "decision on the Clearpath scanner model",
}


def _spawn(model: str) -> Engine:
    world = {
        "sim": {},
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


@pytest.mark.parametrize(
    "model",
    [
        pytest.param(name, marks=pytest.mark.xfail(strict=True, reason=XFAIL[name]))
        if name in XFAIL
        else name
        for name in CASES
    ],
)
def test_the_scan_meets_no_robot_geometry_from_inside(model):
    excluded, expected_outside = CASES[model]
    engine = _spawn(model)
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
