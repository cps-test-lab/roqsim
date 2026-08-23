"""The TurtleBot 4 as one merged mesh -- what `roqsim export mesh` produces for a real robot.

The assertions are against the platform's DOCUMENTED dimensions (`nav2_minimal_tb4_description`, the
same reference `test_turtlebot4_scene.py` uses) rather than against numbers copied out of a previous
export: body radius 0.164 m, wheel r=0.03575 m, ``wheel_separation`` 0.233 m, caster r=0.01 m. An
export that agrees with the datasheet is right for a reason; one that agrees with yesterday's file only
proves nothing changed.

Two facts about the shipped visual meshes are pinned here on purpose. ``tower_standoff`` closes only
because vertices that are merely duplicated get welded, and ``shell``/``rplidar`` do not close at all.
Both are properties of the source OBJs, so if either changes this test is where it shows up.
"""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from roqsim.export_mesh import MeshExporter, _edge_stats, _signed_volume
from roqsim.models import apply_assets, resolve_model

BODY_RADIUS = 0.164
WHEEL_RADIUS = 0.03575
WHEEL_SEPARATION = 0.233
CASTER_RADIUS = 0.01


@pytest.fixture(scope="module")
def exported():
    asset = resolve_model("roqsim_mobile:turtlebot4")
    spec = mujoco.MjSpec.from_file(str(asset.path))
    apply_assets(spec, asset)
    exporter = MeshExporter(spec.compile())
    exporter.collect()
    return exporter, exporter.merge(1.0)


def geom_of(exporter, name):
    return [g for g in exporter.geoms if g["name"] == name]


def test_the_frame_is_base_link_without_being_told(exported):
    """base_link is the wheels' and the chassis' common ancestor, so it is found, not configured."""
    exporter, _ = exported
    assert exporter.frame == "base_link"


def test_the_whole_visible_robot_travels(exported):
    exporter, _ = exported
    names = [g["name"] for g in exporter.geoms]
    assert names.count("tower_standoff") == 4
    assert {
        "shell",
        "body_visual",
        "bumper_visual",
        "tower_sensor_plate",
        "rplidar",
        "camera_bracket",
    } <= set(names)
    # The wheels and the caster are group 0, not the visual group: a "visual geoms only" selection
    # drops them and narrows the base by their width.
    assert [g["body"] for g in exporter.geoms if g["type"] == "cylinder"] == [
        "left_wheel",
        "right_wheel",
    ]
    assert "caster" in names
    # ... while the collision cylinder, which swallows the whole chassis, does not travel.
    assert not geom_of(exporter, "body_collision")
    assert all(g["group"] != 3 for g in exporter.geoms)


def test_the_camera_is_carried_by_its_bracket(exported):
    """Regression: the bracket mesh was missing from the port, so the camera body hung in mid-air.

    Asserted as a distance rather than as "the mesh exists", because a bracket present but misplaced
    looks the same in a part list and still leaves the camera floating. The chain the numbers come from
    is base_link -> shell_link (0 0 0.0942) -> bracket (-0.118 0 0.05257) -> oakd (0.0584 0 0.09676).
    """
    exporter, _ = exported
    parts = {g["name"]: g["verts"] for g in exporter.geoms}
    assert "camera_bracket" in parts, "the OAK-D bracket mesh is not in the model"

    def closest(a, b):
        return float(np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)).min())

    # It stands on the shell, and it reaches the camera: the box spans x -0.0818..-0.0593 and the
    # bracket ends at x -0.0820, so they abut in x while overlapping in z.
    assert closest(parts["camera_bracket"], parts["shell"]) < 1e-3
    bracket_hi, camera_lo = parts["camera_bracket"].max(axis=0), parts["base_link_box"].min(axis=0)
    assert camera_lo[0] - bracket_hi[0] == pytest.approx(0.0, abs=5e-4)
    assert camera_lo[2] < bracket_hi[2]  # the camera's underside is below the bracket's top


def test_the_extent_matches_the_platform(exported):
    _exporter, mesh = exported
    lo, hi = mesh["verts"].min(axis=0), mesh["verts"].max(axis=0)
    assert lo[2] == pytest.approx(0.0, abs=5e-3)  # sits on the floor in its own frame
    assert hi[2] == pytest.approx(0.35, abs=0.01)  # 351 mm to the top of the sensor plate
    # The bumper reaches slightly past the chassis radius; nothing reaches past it by much.
    assert BODY_RADIUS <= hi[0] <= BODY_RADIUS + 0.02
    assert hi[1] == pytest.approx(WHEEL_SEPARATION / 2 + 0.054, abs=0.01)


def test_the_wheels_are_where_the_datasheet_puts_them(exported):
    exporter, _ = exported
    for geom, side in zip(
        geom_of(exporter, "left_wheel_cylinder") + geom_of(exporter, "right_wheel_cylinder"),
        (1, -1),
        strict=True,
    ):
        centre = geom["verts"].mean(axis=0)
        assert centre[1] == pytest.approx(side * WHEEL_SEPARATION / 2, abs=1e-6)
        # The wheel axis is +y in the base frame, so the radius shows in x and z.
        extent = geom["verts"].max(axis=0) - geom["verts"].min(axis=0)
        assert extent[0] == pytest.approx(2 * WHEEL_RADIUS, rel=0.02)
        assert extent[2] == pytest.approx(2 * WHEEL_RADIUS, rel=0.02)


def test_the_caster_is_a_ten_millimetre_ball(exported):
    exporter, _ = exported
    caster = geom_of(exporter, "caster")[0]
    extent = caster["verts"].max(axis=0) - caster["verts"].min(axis=0)
    assert extent == pytest.approx(np.full(3, 2 * CASTER_RADIUS), rel=0.02)


def test_the_materials_travel(exported):
    exporter, _ = exported
    assert {name for name, _rgba in exporter.materials} == {"tb4_black", "tb4_dark", "tb4_grey"}


def test_the_standoffs_close_only_because_of_the_weld(exported):
    """Their seam vertices are duplicated in the source OBJ: 20 of them, 80 phantom boundary edges."""
    exporter, _ = exported
    welded = geom_of(exporter, "tower_standoff")[0]
    assert _edge_stats(welded["faces"])["closed"]

    unwelded = MeshExporter(exporter.model, weld=0.0)
    unwelded.collect()
    raw = geom_of(unwelded, "tower_standoff")[0]
    assert _edge_stats(raw["faces"])["boundary"] == 80
    assert len(raw["verts"]) - len(welded["verts"]) == 20


def test_which_visual_meshes_are_not_watertight(exported):
    """A property of the shipped OBJs, not of the export -- and the reason CAD may need a repair pass."""
    exporter, _ = exported
    open_meshes = {g["name"] for g in exporter.geoms if not _edge_stats(g["faces"])["closed"]}
    assert open_meshes == {"shell", "rplidar"}


def test_every_closed_part_winds_outward(exported):
    """An inverted solid renders as a hole and reads as a void; nothing about the file looks wrong."""
    exporter, _ = exported
    for geom in exporter.geoms:
        if _edge_stats(geom["faces"])["closed"]:
            assert _signed_volume(geom["verts"], geom["faces"]) > 0, geom["name"]
