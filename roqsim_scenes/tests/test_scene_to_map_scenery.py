"""A scene object that does not collide is still in the map built from the scene directory.

``sdf-to-scene --no-collide`` imports a building shell as visual-only geometry and promises it
"still generates the occupancy grid"; the scene path, like the world path, counts every object
because the lidar hits every one.
"""

from __future__ import annotations

import json

from roqsim_scenes.cli.scene_to_map import _load_scene


def test_a_visual_only_wall_is_mapped(tmp_path):
    (tmp_path / "meshes").mkdir()
    (tmp_path / "meshes" / "shell.obj").write_text(
        "v 0 0 0\nv 4 0 0\nv 4 0 3\nv 0 0 3\nf 1 2 3\nf 1 3 4\n"
    )
    (tmp_path / "scene.json").write_text(
        json.dumps({"objects": [{"name": "shell", "mesh": "meshes/shell.obj", "collide": False}]})
    )
    assert len(_load_scene(tmp_path)) == 1
