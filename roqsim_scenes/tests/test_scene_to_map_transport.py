"""A world that publishes over ROS 2 is still mappable where ROS is not.

``scene-to-map --world`` wants the scene, not a running simulation: a transport plugin publishes what
the others built and adds no geometry. So the loader drops it the way ``roqsim render`` does, which is
what lets a ``*_ros`` world be mapped in a pip-only environment -- or in a container whose ROS
overlay nobody sourced, which is how a campaign's input generator runs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from roqsim_scenes.cli import scene_to_map

# The default room (a floor and four walls) plus the two transport plugins a ROS world declares.
WORLD = """
sim:
  world: empty_room
components:
  - ros2_bridge: {}
  - sim_interfaces: {}
"""


@pytest.fixture
def ros_world(tmp_path: Path) -> Path:
    path = tmp_path / "ros_world.yaml"
    path.write_text(WORLD)
    return path


def test_a_world_declaring_the_bridge_maps_without_it(ros_world: Path, capsys):
    tris, lo, hi, _ = scene_to_map._load_world(str(ros_world))
    assert tris, "the room's geometry survived the drop"
    assert np.all(hi > lo)
    out = capsys.readouterr()
    assert "skipping" in out.out + out.err


def test_the_command_writes_a_map_for_it(ros_world: Path, tmp_path: Path):
    out = tmp_path / "map"
    rc = scene_to_map.main(
        [
            "--world",
            str(ros_world),
            "--out",
            str(out),
            "--scan-height",
            "0.5",
            "--resolution",
            "0.1",
            "--free-from",
            "0",
            "0",
        ]
    )
    assert rc == 0
    assert out.with_suffix(".pgm").is_file() and out.with_suffix(".yaml").is_file()
