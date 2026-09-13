"""Every quadruped whose manifest declares its ``lidar`` directly: what its own scan hits of itself.

The rule a lidar follows is that it never excludes robot geometry, so the scan is cast here with **no
exclusion at all**, from the plugin's own ray pattern and site, and two properties are pinned per
robot:

* **No ray starts inside robot geometry.** A first surface met from inside (normal . ray > 0) means
  the scan origin lies within a geom, and a published scan would read that geom's inner face on every
  such ray -- unless something robot-sized is excluded, which is what this test refuses to do.
* **The robot bodies hit from outside, with their ray counts.** These are real returns of a sensor
  that sees part of its own robot; a change to the model, the stance or the mount shows up here.

A manifest that still excludes robot geometry would be pinned in ``STILL_EXCLUDES``; where it excludes
nothing, the cast here must also equal the plugin's own.

The robot is spawned as a world spawns it -- ``spawn_robot`` by its registered plugin name, with a
prefix, its manifest's locomotion controller setting the standing stance at reset -- into the default
world (``empty_room``: a floor and perimeter walls), so every non-world body is the robot's. The
controller loads the fetched policy (``python -m roqsim_quadruped.policy.fetch_policy``).
"""

from __future__ import annotations

import os
from collections import Counter
from pathlib import Path

import mujoco
import numpy as np
import pytest
from roqsim_quadruped.policy import POLICY_DIR

from roqsim import raycast
from roqsim.config import load_config_from_dict
from roqsim.engine import Engine

PREFIX = "r_"

#: The policy the controller loads at spawn. It is a fetched external asset, not part of the tree, so
#: a checkout that has not fetched it has no stance to test -- a skip that says how to get it, not a
#: failure.
_POLICY = Path(os.environ.get("SPOT_POLICY_PATH") or POLICY_DIR / "spot_policy.pt")
pytestmark = pytest.mark.skipif(
    not _POLICY.is_file(),
    reason=f"no Spot policy at {_POLICY}: python -m roqsim_quadruped.policy.fetch_policy",
)

#: model -> (sim timestep, {robot body hit from outside: ray count}), cast with no exclusion.
CASES = {
    "spot": (0.002, {}),
}

#: model -> the robot body its manifest still excludes from the published scan.
STILL_EXCLUDES: dict[str, str] = {}


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
    timestep, expected_outside = CASES[model]
    engine = _spawn(model, timestep)
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

        # The TF parent is named now that nothing is excluded, so the frame still hangs off the base.
        tf = next(e for e in engine.ctx.interface.all() if e.name == "scan").backend["ros2"]
        assert tf["static_tf"]["parent"] == "base_link"
    finally:
        engine.shutdown()
