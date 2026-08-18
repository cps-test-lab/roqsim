"""The ground is two geoms, and each has one job it must not lose.

A baked scene's floor answers to two constraints that pull apart. The **collider** must sit exactly at
the ground height, because that is what the robot stands on and what ``contact_monitor: {ignore:
[floor]}`` names. The **visual** must never hide a floor the scene brought of its own -- one geom doing
both put a drawn plane at the same z as a scene's own floor mesh, and they z-fought across the room,
which is why the drawn floor was removed the first time.

The cases below are mostly ones no scene in the tree exercises, so they are pinned here rather than
argued: a scene whose floor lies *below* its stated ground height, a scene that states no ground height
at all, and a scene with geometry far under its floor.
"""

from __future__ import annotations

import json

import mujoco
import numpy as np
import pytest

from roqsim_scenes import scene_mesh_io as mio
from roqsim_scenes.cli import scene_to_mjcf


def _quad(path, z, half=4.0):
    """A flat square at height *z* -- stands in for whatever floor a scene brings of its own."""
    verts = np.array(
        [[-half, -half, z], [half, -half, z], [half, half, z], [-half, half, z]], float
    )
    mio.write_obj(path, verts, np.array([[0, 1, 2], [0, 2, 3]]))


def _scene(tmp_path, *, ground_z, objects, name="s"):
    """Write a minimal scene dir; *objects* is a list of (obj_name, z, render)."""
    meshes = tmp_path / "meshes"
    for obj_name, z, _ in objects:
        _quad(meshes / f"{obj_name}.obj", z)
    manifest = {
        "name": name,
        "unit_scale": 1.0,
        "bounds_min": [-4.0, -4.0, min([z for _, z, _ in objects], default=0.0)],
        "bounds_max": [4.0, 4.0, 2.0],
        "objects": [
            {
                "name": obj_name,
                "mesh": f"meshes/{obj_name}.obj",
                "rgba": [0.7, 0.7, 0.7, 1.0],
                "collide": False,
                "render": render,
            }
            for obj_name, _, render in objects
        ],
    }
    if ground_z is not None:
        manifest["ground_z"] = ground_z
    (tmp_path / "scene.json").write_text(json.dumps(manifest))
    return tmp_path / "scene.json"


def _bake(scene_json, out):
    scene_to_mjcf.main(["--scene", str(scene_json), "--out", str(out)])
    m = mujoco.MjModel.from_xml_path(str(out))
    ids = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i): i for i in range(m.ngeom)}
    return m, ids


def test_the_collider_keeps_its_height_name_and_contacts(tmp_path):
    """The half nothing may disturb: every existing world's physics runs through this geom."""
    sj = _scene(tmp_path, ground_z=0.0, objects=[("Floor", 0.0, True)])
    m, ids = _bake(sj, tmp_path / "w.xml")

    f = ids["floor"]
    assert m.geom_pos[f][2] == pytest.approx(0.0)
    assert m.geom_contype[f] != 0, "the robot stands on this"
    assert m.geom_group[f] == 3 and m.geom_rgba[f][3] == 0.0, "and nobody draws it"


def test_the_visual_goes_under_a_floor_modelled_below_the_ground_height(tmp_path):
    """The case a fixed offset below ``ground_z`` would get wrong.

    A scene may model its floor with thickness downward, or recess it. A plane at ``ground_z - 2 mm``
    would then be *above* that floor and cover it up -- the scene's own texture replaced by our
    checker. Placing the visual under the lowest renderable vertex instead cannot do that, whatever
    shape the scene has.
    """
    sj = _scene(tmp_path, ground_z=0.0, objects=[("Floor", -0.05, True)])
    m, ids = _bake(sj, tmp_path / "w.xml")

    assert m.geom_pos[ids["floor"]][2] == pytest.approx(0.0), "the collider still sits at ground_z"
    assert m.geom_pos[ids["floor_visual"]][2] < -0.05, "the visual is under the scene's own floor"


def test_the_visual_never_collides_and_is_actually_drawn(tmp_path):
    sj = _scene(tmp_path, ground_z=0.0, objects=[("Wall", 1.0, True)])
    m, ids = _bake(sj, tmp_path / "w.xml")

    v = ids["floor_visual"]
    assert m.geom_contype[v] == 0 and m.geom_conaffinity[v] == 0
    assert m.geom_group[v] < 3, "group 3 is the never-drawn convention, and export_web skips it"
    assert m.geom_matid[v] >= 0, "a material, or there is nothing to see"


def test_a_scene_that_states_no_ground_height_gets_no_drawn_floor_but_says_so(tmp_path, capsys):
    """Silence here is the bug this whole feature came from.

    The height would have to be guessed from the scene's lowest point, and a floor drawn at a guessed
    height makes everything standing on it hover or sink. So it is skipped -- but an unexplained void
    under the robot in the run view is exactly the report that started this, so the bake must name the
    cause and the fix.
    """
    sj = _scene(tmp_path, ground_z=None, objects=[("Floor", 0.0, True)])
    m, ids = _bake(sj, tmp_path / "w.xml")

    assert "floor" in ids, "it still collides"
    assert "floor_visual" not in ids
    out = capsys.readouterr().out
    assert "no floor is DRAWN" in out
    assert "ground_z" in out, "the message has to name the fix, not just the symptom"


def test_a_scene_with_geometry_far_below_its_floor_reports_the_drop(tmp_path, capsys):
    """The visual still goes under everything -- but a metre down reads as a step at the edge."""
    sj = _scene(tmp_path, ground_z=0.0, objects=[("Floor", 0.0, True), ("Pit", -1.0, True)])
    m, ids = _bake(sj, tmp_path / "w.xml")

    assert m.geom_pos[ids["floor_visual"]][2] < -1.0
    assert "below the ground height" in capsys.readouterr().out


def test_a_non_renderable_object_does_not_drag_the_visual_down(tmp_path):
    """Only what a viewer draws can be covered up, so only that decides the height.

    A collision-only part often reaches below the floor (a wall's footing). Letting it pull the
    backdrop down would put a visible step at the scene's edge for no reason.
    """
    sj = _scene(tmp_path, ground_z=0.0, objects=[("Floor", 0.0, True), ("Footing", -0.5, False)])
    m, ids = _bake(sj, tmp_path / "w.xml")

    assert m.geom_pos[ids["floor_visual"]][2] == pytest.approx(-0.002, abs=1e-6)


def test_ground_plane_false_still_suppresses_both(tmp_path):
    sj = _scene(tmp_path, ground_z=0.0, objects=[("Floor", 0.0, True)])
    (tmp_path / "scene.yaml").write_text("ground_plane: false\n")
    m, ids = _bake(sj, tmp_path / "w.xml")

    assert "floor" not in ids and "floor_visual" not in ids
