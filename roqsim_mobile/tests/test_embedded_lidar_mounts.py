"""Every mobile base whose manifest declares its ``lidar`` directly: what its own scan hits of itself.

A robot that mounts a scanner device model (``spawn_sensor``) is covered by its own mount test. This
module covers the rest: bases whose lidar is a site in the robot's MJCF. The rule a lidar follows is
that it never excludes robot geometry, so the scan is cast here with **no exclusion at all**, from the
plugin's own ray pattern and site, and two properties are pinned per robot:

* **No ray starts inside robot geometry.** A first surface met from inside (normal . ray > 0) means
  the scan origin lies within a collision or visual geom, and a published scan would read that geom's
  inner face on every such ray -- unless something robot-sized is excluded, which is what this test
  refuses to do.
* **The robot bodies hit from outside, with their ray counts.** These are real returns of a sensor
  that sees part of its own robot; a change to the model or the mount shows up here.

A manifest that still excludes robot geometry is pinned in ``STILL_EXCLUDES``, so the exclusion is a
visible exception rather than a default nobody chose. Where the manifest excludes nothing, the cast
here must also equal the plugin's own.

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

#: model -> {robot body hit from outside: ray count}, cast with no exclusion.
CASES = {
    # The chassis body's side panels stand in the scan plane ~0.29 m out, over a narrow sector.
    "panther": {"body_link": 12},
    "warthog": {},
}

#: model -> the robot body its manifest still excludes from the published scan.
STILL_EXCLUDES = {"panther": "base_link"}

#: Pinned failures, each a property of the model as it stands rather than of the test.
XFAIL: dict[str, str] = {}


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
    expected_outside = CASES[model]
    engine = _spawn(model)
    try:
        m, d = engine.ctx.model, engine.ctx.data
        lidars = [p for p in engine.plugins if type(p).__name__ == "LidarPlugin"]
        assert [p.address for p in lidars] == ["robot.lidar"]
        (lidar,) = lidars
        foreign = [_body(m, b) for b in range(1, m.nbody) if not _body(m, b).startswith(PREFIX)]
        assert not foreign, f"bodies not of the spawned robot: {foreign}"
        bid = lidar._bodyexclude
        excluded = _body(m, bid).removeprefix(PREFIX) if bid >= 0 else None
        assert excluded == STILL_EXCLUDES.get(model), f"the manifest excludes {excluded!r}"

        # The plugin's own rays from its own site: `_local_dirs @ rot.T`, as post_step casts them.
        origin = d.site_xpos[lidar._site_id].copy()
        dirs = lidar._build_directions() @ d.site_xmat[lidar._site_id].reshape(3, 3).T
        hits = raycast.cast(
            m,
            d,
            origin,
            dirs,
            cutoff=lidar.range_max,
            bodyexclude=-1,
            out=raycast.buffers(len(dirs), normals=True),
        )
        if excluded is None:
            np.testing.assert_array_equal(hits.geomid, lidar._hits.geomid)

        on_robot = (hits.geomid >= 0) & (m.geom_bodyid[np.maximum(hits.geomid, 0)] != 0)
        from_inside = on_robot & (np.einsum("ij,ij->i", hits.normal, dirs) > 0)
        inside = Counter(_body(m, m.geom_bodyid[g]) for g in hits.geomid[from_inside])
        assert not inside, f"rays start inside robot geometry: {dict(inside)}"

        outside = Counter(_body(m, m.geom_bodyid[g]) for g in hits.geomid[on_robot])
        assert dict(outside) == {PREFIX + b: n for b, n in expected_outside.items()}
    finally:
        engine.shutdown()
