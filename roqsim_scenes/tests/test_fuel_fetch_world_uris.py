"""``fuel-fetch --world`` reads a world's Fuel URIs past the comments every real world carries."""

from __future__ import annotations

from roqsim_scenes.cli.fuel_fetch import _world_uris

_WORLD = """<?xml version="1.0"?>
<sdf version="1.8">
  <world name="depot">
    <!-- <include><uri>https://fuel.gazebosim.org/1.0/OpenRobotics/models/Retired</uri></include> -->
    <include>
      <uri>https://fuel.gazebosim.org/1.0/OpenRobotics/models/Depot</uri>
    </include>
    <?ignition a-processing-instruction?>
  </world>
</sdf>
"""


def test_a_world_with_comments_yields_its_fuel_uris(tmp_path):
    world = tmp_path / "depot.sdf"
    world.write_text(_WORLD)
    assert _world_uris(world) == ["https://fuel.gazebosim.org/1.0/OpenRobotics/models/Depot"]
