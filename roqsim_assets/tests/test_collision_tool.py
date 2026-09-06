"""``roqsim assets collision``: what a prop collides as, measured against what it looks like.

The defects here are the ones a picture and a drive-test both miss. A collision model can be plausible
in a render at any size a reviewer will look at, and can pass a robot's pass/stop trial while a whole
part of it is absent -- the trial only proves the lanes it drives. So these fixtures are deliberately
wrong in ways that leave both of those green, and assert that the measurement is not fooled.
"""

from __future__ import annotations

import numpy as np
import pytest

from roqsim_assets.cli import collision

# A table: a top on two legs, with a span underneath meant to stay open. The visual mesh is a box
# per part (group 2, no contacts); the collider is stated separately so a test can get it wrong.
TABLE = """
<mujoco model="fixture_table">
  <worldbody>
    <body name="table">
      <geom name="v_top"   type="box" size="0.60 0.30 0.02" pos="0 0 0.70" contype="0" conaffinity="0" group="2"/>
      <geom name="v_leg_l" type="box" size="0.03 0.03 0.35" pos="-0.55 0 0.35" contype="0" conaffinity="0" group="2"/>
      <geom name="v_leg_r" type="box" size="0.03 0.03 0.35" pos="0.55 0 0.35" contype="0" conaffinity="0" group="2"/>
      {collider}
    </body>
  </worldbody>
</mujoco>
"""

FAITHFUL = """
      <geom name="c_top"   type="box" size="0.60 0.30 0.02" pos="0 0 0.70" group="3"/>
      <geom name="c_leg_l" type="box" size="0.03 0.03 0.35" pos="-0.55 0 0.35" group="3"/>
      <geom name="c_leg_r" type="box" size="0.03 0.03 0.35" pos="0.55 0 0.35" group="3"/>
"""
# One leg never got written. Nothing drives through a leg in a pass/stop trial, so the trial stays
# green; a robot arm reaching past it meets nothing.
MISSING_LEG = """
      <geom name="c_top"   type="box" size="0.60 0.30 0.02" pos="0 0 0.70" group="3"/>
      <geom name="c_leg_l" type="box" size="0.03 0.03 0.35" pos="-0.55 0 0.35" group="3"/>
"""
# A leg in the wrong place: still a leg, still stops something, 200 mm from the one it stands for.
DISPLACED_LEG = """
      <geom name="c_top"   type="box" size="0.60 0.30 0.02" pos="0 0 0.70" group="3"/>
      <geom name="c_leg_l" type="box" size="0.03 0.03 0.35" pos="-0.55 0 0.35" group="3"/>
      <geom name="c_leg_r" type="box" size="0.03 0.03 0.35" pos="0.55 0.20 0.35" group="3"/>
"""


def _model(tmp_path, collider, name="table"):
    path = tmp_path / f"{name}.xml"
    path.write_text(TABLE.format(collider=collider))
    return str(path)


def _diff(tmp_path, collider, **kw):
    # Fewer samples than the CLI default: these fixtures are boxes, and the assertions are about
    # hundreds of millimetres, so the run stays fast without changing any verdict.
    return collision.diff(_model(tmp_path, collider), samples=6000, **kw)


def test_a_faithful_collider_agrees_with_the_mesh(tmp_path):
    report = _diff(tmp_path, FAITHFUL)
    assert report["verdict"] == "ok"
    assert report["coverage"]["max_mm"] < 10
    assert report["overreach"]["max_mm"] < 10


def test_a_missing_part_is_a_coverage_hole_and_is_located(tmp_path):
    report = _diff(tmp_path, MISSING_LEG)
    assert report["verdict"] == "WARN"
    # The absent leg is 60 mm across, so its far face is that far from anything that collides.
    assert report["coverage"]["max_mm"] > 50
    region = report["coverage"]["region"]
    assert region["min"][0] > 0.4, (
        "the hole should be located at the leg that is gone, not the prop"
    )
    assert region["max"][2] < 0.71
    # Nothing was ADDED, so the collider never stands where the prop does not.
    assert report["overreach"]["beyond_tol"] == pytest.approx(0.0, abs=0.01)


def test_a_displaced_part_is_overreach_and_is_named(tmp_path):
    report = _diff(tmp_path, DISPLACED_LEG)
    assert report["verdict"] == "WARN"
    assert report["overreach"]["max_mm"] > 100
    blamed = report["overreach"]["worst_geoms"]
    assert blamed, "overreach must name the geom responsible, or the fix has no address"
    assert blamed[0]["geom"] == "c_leg_r"


def test_a_colliding_mesh_fails_because_physics_sees_its_hull(tmp_path):
    # The shape the import pipeline emits: one mesh geom, collidable. Its hull fills the span.
    path = tmp_path / "hull.xml"
    path.write_text(
        """
        <mujoco model="hull">
          <asset><mesh name="wedge" vertex="0 0 0  1 0 0  0 1 0  0 0 1"/></asset>
          <worldbody><body name="p"><geom type="mesh" mesh="wedge"/></body></worldbody>
        </mujoco>
        """
    )
    report = collision.diff(str(path), samples=2000)
    assert report["verdict"] == "FAIL"
    assert "convex hull" in report["reason"]
    # An unnamed geom is named as such: the contact report a robot produces would say no more.
    assert report["hull_colliders"] == ["<unnamed geom 0>"]


def test_the_same_question_twice_gives_the_same_answer(tmp_path):
    # The numbers exist to be compared across an edit, so they must not move on their own.
    first = _diff(tmp_path, DISPLACED_LEG)
    second = _diff(tmp_path, DISPLACED_LEG)
    assert first["overreach"]["max_mm"] == second["overreach"]["max_mm"]
    assert first["coverage"]["p95_mm"] == second["coverage"]["p95_mm"]
    # ... and a different seed is a different point set, so it must be asked for explicitly.
    assert _diff(tmp_path, DISPLACED_LEG, seed=7)["overreach"]["max_mm"] != pytest.approx(
        first["overreach"]["max_mm"], abs=1e-12
    )


def test_a_geom_is_visual_by_its_contact_masks_not_its_group(tmp_path):
    # A model that draws a collider in group 2, or a decoration in group 3, still measures correctly:
    # what pairs with anything is MuJoCo's contype/conaffinity, and that is what the split reads.
    path = tmp_path / "groups.xml"
    path.write_text(
        """
        <mujoco model="groups">
          <worldbody><body name="p">
            <geom name="drawn_in_2_but_solid" type="box" size=".1 .1 .1" group="2"/>
            <geom name="drawn_in_3_but_ghost" type="box" size=".2 .2 .2" pos="1 0 0"
                  contype="0" conaffinity="0" group="3"/>
          </body></worldbody>
        </mujoco>
        """
    )
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(path))
    visual, solid = collision.split_geoms(model)
    assert [collision._name(model, g) for g in solid] == ["drawn_in_2_but_solid"]
    assert [collision._name(model, g) for g in visual] == ["drawn_in_3_but_ghost"]


@pytest.mark.parametrize("argv", [["--tol", "500", "diff"], ["diff", "--tol", "500"]])
def test_a_shared_flag_is_accepted_on_either_side_of_the_subcommand(tmp_path, capsys, argv):
    # argparse re-applies a parent's default inside the subparser, so a value given before the
    # subcommand is silently replaced by the default given after it. At 500 mm every fixture passes;
    # if the tolerance were dropped, DISPLACED_LEG would not.
    assert collision.main([*argv, _model(tmp_path, DISPLACED_LEG)]) == 0
    assert "tolerance 500 mm" in capsys.readouterr().out


def test_audit_separates_a_hull_collider_from_a_primitive_one(tmp_path, capsys):
    faithful = _model(tmp_path, FAITHFUL, name="faithful")
    rows = [collision.audit_one(faithful, tol=0.01)]
    assert rows[0]["verdict"] == "ok"
    assert rows[0]["collider"] == "primitive"
    collision.print_audit(rows, tol=0.01)
    assert "0 of 1 collide as a convex hull" in capsys.readouterr().out


def test_a_model_that_will_not_compile_is_a_row_not_a_crash(tmp_path):
    # An audit runs over a whole provider; one unreadable model must not end the sweep.
    bad = tmp_path / "bad.xml"
    bad.write_text("<mujoco><worldbody><geom type='box'/></worldbody></mujoco>")
    row = collision.audit_one(str(bad), tol=0.01)
    assert row["verdict"] == "ERROR"
    assert row["reason"]


def test_the_exit_code_is_the_verdict(tmp_path, capsys):
    # Same two levels `inspect-prop` uses: a FAIL stops a CI step or an agent, a WARN is an accepted
    # judgement that still has to be read. Everything past `ok` is in the report either way.
    assert collision.main(["diff", _model(tmp_path, FAITHFUL, name="good")]) == 0
    assert collision.main(["diff", _model(tmp_path, DISPLACED_LEG, name="displaced")]) == 0
    hull = tmp_path / "hull.xml"
    hull.write_text(
        "<mujoco><asset><mesh name='w' vertex='0 0 0  1 0 0  0 1 0  0 0 1'/></asset>"
        "<worldbody><body name='p'><geom type='mesh' mesh='w'/></body></worldbody></mujoco>"
    )
    assert collision.main(["diff", str(hull)]) == 1
    capsys.readouterr()


def test_json_is_the_whole_report(tmp_path, capsys):
    import json

    assert collision.main(["diff", "--json", _model(tmp_path, DISPLACED_LEG)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["verdict"] == "WARN"
    assert report["overreach"]["worst_geoms"][0]["geom"] == "c_leg_r"
    assert np.isfinite(report["resolution_mm"])


# -- effort: what a skeleton will cost, before anyone writes one -----------------------------------


def test_effort_separates_how_many_from_what_kind(tmp_path):
    # Two props of similar complexity, one built from slabs and one from tilted members. The count
    # does not tell them apart; `boxy` does, and it is the one that says whether plain axis-aligned
    # boxes will land on the shape or every part costs a rotation.
    import trimesh

    slabs = [
        trimesh.creation.box(extents=(1.0, 0.5, 0.04)),
        trimesh.creation.box(extents=(0.05, 0.05, 0.7)),
    ]
    tilted = trimesh.creation.box(extents=(1.0, 0.5, 0.04))
    tilted.apply_transform(trimesh.transformations.rotation_matrix(0.6, [1, 0, 0]))
    assert collision.effort(trimesh, slabs, 0.05)["axis_frac"] == pytest.approx(1.0, abs=0.01)
    assert collision.effort(trimesh, [tilted], 0.05)["axis_frac"] < 0.1


def test_effort_counts_more_boxes_for_a_more_broken_up_shape(tmp_path):
    # Eight shelves cost more primitives than one, which is the ordering the column exists to give.
    import trimesh

    def shelves(n):
        return [
            trimesh.creation.box(extents=(1.0, 0.4, 0.03)).apply_translation([0, 0, 0.2 * i])
            for i in range(n)
        ]

    assert (
        collision.effort(trimesh, shelves(1), 0.01)["boxes_upper"]
        < collision.effort(trimesh, shelves(8), 0.01)["boxes_upper"]
    )


def test_effort_is_an_upper_bound_not_the_answer(tmp_path):
    # The number exists to be ORDERED, not believed: a greedy voxel pass needs many more primitives
    # than someone reading the shape, so it must never be quoted as the size of the job.
    import trimesh

    ring = [
        trimesh.creation.box(extents=(0.05, 0.05, 1.0)).apply_translation([x, y, 0])
        for x in (-0.5, 0.5)
        for y in (-0.3, 0.3)
    ]
    measured = collision.effort(trimesh, ring, 0.05)
    assert measured["boxes_upper"] >= len(ring)


# -- what the two surfaces must NOT be scored on --------------------------------------------------

# A prop modelled the way artwork usually is: an open shell with no underside, standing on the floor.
OPEN_BOTTOM = """
<mujoco model="open_bottom">
  <asset>
    <!-- four walls and a lid, no floor face: the box is open where it stands -->
    <mesh name="shell" vertex="-0.2 -0.2 0  0.2 -0.2 0  0.2 0.2 0  -0.2 0.2 0
                               -0.2 -0.2 0.4  0.2 -0.2 0.4  0.2 0.2 0.4  -0.2 0.2 0.4"
          face="0 1 5  0 5 4  1 2 6  1 6 5  2 3 7  2 7 6  3 0 4  3 4 7  4 5 6  4 6 7"/>
  </asset>
  <worldbody>
    <body name="p">
      <geom name="shell" type="mesh" mesh="shell" contype="0" conaffinity="0" group="2"/>
      <geom name="solid" type="box" size="0.2 0.2 0.2" pos="0 0 0.2" group="3"/>
    </body>
  </worldbody>
</mujoco>
"""


def test_the_patch_a_prop_stands_on_is_not_overreach(tmp_path):
    # The collider has a bottom face; the artwork has none, because nobody models the underside of a
    # cabinet. Scoring that difference would charge every floor-standing prop for its own footprint.
    path = tmp_path / "open_bottom.xml"
    path.write_text(OPEN_BOTTOM)
    report = collision.diff(str(path), samples=8000)
    assert report["overreach"]["max_mm"] < 5
    assert report["verdict"] == "ok"


def test_enclosed_geometry_counts_as_covered(tmp_path):
    # A solid primitive standing in for a hollow shell encloses the shell's inner faces. Measured to
    # the collider's outer SURFACE those read as uncovered, and a prop can report metres of holes
    # while being completely enclosed -- so containment is the test and distance the fallback.
    path = tmp_path / "hollow.xml"
    path.write_text(
        """
        <mujoco model="hollow">
          <worldbody><body name="p">
            <geom name="inner" type="box" size="0.10 0.10 0.10" pos="0 0 0.10"
                  contype="0" conaffinity="0" group="2"/>
            <geom name="outer" type="box" size="0.20 0.20 0.20" pos="0 0 0.20" group="3"/>
          </body></worldbody>
        </mujoco>
        """
    )
    report = collision.diff(str(path), samples=8000)
    assert report["coverage"]["max_mm"] == pytest.approx(0.0, abs=1e-9)
    assert report["coverage"]["beyond_tol"] == 0.0


def test_containment_is_analytic_per_shape(tmp_path):
    import mujoco as mj

    path = tmp_path / "shapes.xml"
    path.write_text(
        """
        <mujoco><worldbody><body name="p">
          <geom name="b" type="box" size=".1 .1 .1" pos="0 0 0"/>
          <geom name="s" type="sphere" size=".1" pos="1 0 0"/>
          <geom name="c" type="cylinder" size=".1 .1" pos="2 0 0"/>
          <geom name="k" type="capsule" size=".1 .1" pos="3 0 0"/>
        </body></worldbody></mujoco>
        """
    )
    model = mj.MjModel.from_xml_path(str(path))
    data = mj.MjData(model)
    mj.mj_forward(model, data)
    ids = list(range(model.ngeom))
    centres = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]], float)
    outside = centres + np.array([0, 0.5, 0])
    assert collision.inside_any(model, data, ids, centres).all()
    assert not collision.inside_any(model, data, ids, outside).any()
