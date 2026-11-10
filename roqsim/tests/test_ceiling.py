"""The ceiling's measuring compile leaves the world's spec as it was for the plugins built after it."""

from __future__ import annotations

import mujoco

from roqsim.context import SimContext
from roqsim.plugins.ceiling import CeilingPlugin


def test_names_added_after_the_ceiling_is_built_are_found():
    spec = mujoco.MjSpec()
    room = spec.worldbody.add_body(name="room")
    room.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_BOX, size=[2.0, 2.0, 0.05])
    room.add_geom(
        name="roof", type=mujoco.mjtGeom.mjGEOM_BOX, size=[2.0, 2.0, 0.05], pos=[0.0, 0.0, 3.0]
    )
    CeilingPlugin({"above_z": 2.5, "keep": False}).build(spec, SimContext(config={}))
    box = spec.worldbody.add_body(name="box")
    box.add_freejoint()
    box.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.1, 0.1, 0.1])
    box.add_site(name="box/grip")
    assert spec.body("box") is not None
    assert spec.site("box/grip") is not None and spec.site("box/grip").parent.name == "box"
    assert {g.name for g in spec.geoms} >= {"floor"} and "roof" not in {g.name for g in spec.geoms}
