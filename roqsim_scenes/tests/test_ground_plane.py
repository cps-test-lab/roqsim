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


def _prop(path, z0, z1, half=0.5):
    """An upright block from *z0* to *z1* -- a prop, as opposed to a floor. Only its z range matters."""
    xy = [(-half, -half), (half, -half), (half, half), (-half, half)]
    verts = np.array([[x, y, z] for x, y in xy for z in (z0, z1)], float)
    faces = []
    for k in range(4):
        a, b = 2 * k, 2 * ((k + 1) % 4)
        faces += [[a, b, b + 1], [a, b + 1, a + 1]]
    faces += [[0, 2, 4], [0, 4, 6], [1, 5, 3], [1, 7, 5]]
    mio.write_obj(path, verts, np.array(faces))


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


def test_a_prop_sunk_into_the_floor_does_not_drag_the_drawn_floor_down_with_it(tmp_path, capsys):
    """A source world may bury part of a prop; the floor is meant to hide exactly that part.

    ``turtlebot3_world`` sinks the largest of its ornaments 0.5 m into the ground. Reading that as
    "there is scene floor 0.5 m down" put the drawn plane below it, which exposed the ornament's buried
    underside -- drawing MORE than the source world does -- and left a half-metre step around the room
    for everything else. A prop that rises above the stated ground cannot be hidden by a floor at the
    ground, so it has no say in where that floor goes.
    """
    sj = _scene(tmp_path, ground_z=0.0, objects=[("Wall", 1.0, True)])
    _prop(tmp_path / "meshes" / "Ornament.obj", -0.5, 1.5)
    manifest = json.loads(sj.read_text())
    manifest["bounds_min"][2] = -0.5
    manifest["objects"].append(
        {
            "name": "Ornament",
            "mesh": "meshes/Ornament.obj",
            "rgba": [0.1, 0.8, 0.1, 1.0],
            "collide": True,
            "render": True,
        }
    )
    sj.write_text(json.dumps(manifest))
    m, ids = _bake(sj, tmp_path / "w.xml")

    assert m.geom_pos[ids["floor_visual"]][2] == pytest.approx(-0.002), (
        "drawn at the ground, not 0.5 m under it"
    )
    assert "note: the drawn floor sits" not in capsys.readouterr().out


def test_a_slab_entirely_below_the_ground_still_lowers_the_drawn_floor(tmp_path):
    """The distinction the rule turns on: floor down there, not a prop reaching down.

    A recessed slab or an outdoor apron IS the floor where it lies, so covering it with our own checker
    is the failure the drawn plane's depth exists to prevent. It stays covered.
    """
    sj = _scene(tmp_path, ground_z=0.0, objects=[("Floor", 0.0, True), ("Apron", -0.4, True)])
    m, ids = _bake(sj, tmp_path / "w.xml")

    assert m.geom_pos[ids["floor_visual"]][2] < -0.4
