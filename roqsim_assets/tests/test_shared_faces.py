"""Drawn parts showing a face in one place: what roqsim_assets.shared_faces finds, what it leaves
alone, what it refuses to pass, and how inspect-prop reports it."""

import json
import math
from pathlib import Path

import mujoco
import pytest

from roqsim_assets.cli import inspect_prop
from roqsim_assets.shared_faces import UncomparableGeoms, shared_faces


def _model(body: str, asset: str = "") -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_string(
        f"<mujoco><asset>{asset}</asset><worldbody>{body}</worldbody></mujoco>"
    )


#: A 0.2 m cube on the floor, and a slab ``b`` overlapping its upper half from the side: their tops
#: lie in one plane at z = 0.2, and so do their front and back faces (y = +-0.1).
SLAB = '<geom name="b" type="box" size="0.1 0.1 0.05" pos="0.15 0 0.15" {extra}/>'
CUBE = '<geom name="a" type="box" size="0.1 0.1 0.1" pos="0 0 0.1"/>'


def _boxes(second: str = "") -> mujoco.MjModel:
    return _model(CUBE + SLAB.format(extra=second))


def test_two_boxes_showing_a_top_in_one_plane_are_found_with_where_they_touch():
    (pair,) = shared_faces(_boxes())
    assert (pair.first, pair.second, pair.kind) == ("a", "b", "plane")
    # The tops share x in [0.05, 0.1] by the full 0.2 m depth, the larger of the planes they share.
    assert pair.direction == pytest.approx((0, 0, 1))
    assert pair.at == pytest.approx((0.075, 0, 0.2), abs=1e-9)
    assert pair.area == pytest.approx(0.05 * 0.2)


def test_a_part_inset_by_a_millimetre_shows_no_face_in_the_others_plane():
    # b moved 1 mm in from every face it shared with a: the top, the front and the back.
    inset = '<geom name="b" type="box" size="0.1 0.099 0.0495" pos="0.15 0 0.1495"/>'
    assert shared_faces(_model(CUBE + inset)) == []


def test_boxes_pressed_back_to_back_are_hidden_and_pass():
    stacked = _model(
        '<geom name="a" type="box" size="0.1 0.1 0.1" pos="0 0 0.1"/>'
        '<geom name="b" type="box" size="0.05 0.05 0.05" pos="0 0 0.25"/>'
    )
    assert shared_faces(stacked) == []


@pytest.mark.parametrize(
    "hidden",
    ['group="3"', 'group="4"', 'rgba="1 0 0 0"'],
    ids=["collision group", "hidden group", "alpha 0"],
)
def test_a_part_no_render_shows_may_share_a_face(hidden):
    assert shared_faces(_boxes(second=hidden)) == []


def test_a_material_s_alpha_counts_where_the_geom_keeps_the_default_colour():
    asset = '<material name="clear" rgba="1 1 1 0"/>'
    body = (
        '<geom name="a" type="box" size="0.1 0.1 0.1" pos="0 0 0.1"/>'
        '<geom name="b" type="box" size="0.1 0.1 0.1" pos="0.15 0 0.1" material="clear"/>'
    )
    assert shared_faces(_model(body, asset)) == []
    tinted = body.replace('material="clear"', 'material="clear" rgba="1 0 0 1"')
    assert len(shared_faces(_model(tinted, asset))) == 1


def test_a_cylinder_end_flush_with_a_box_top_is_found_and_one_set_into_it_is_not():
    flush = _model(
        '<geom name="box" type="box" size="0.2 0.2 0.1" pos="0 0 0.1"/>'
        '<geom name="post" type="cylinder" size="0.05 0.1" pos="0 0 0.1"/>'
    )
    (pair,) = shared_faces(flush)
    assert pair.kind == "plane" and pair.direction == pytest.approx((0, 0, 1))
    assert pair.at == pytest.approx((0, 0, 0.2), abs=1e-9)
    sunk = _model(
        '<geom name="box" type="box" size="0.2 0.2 0.1" pos="0 0 0.1"/>'
        '<geom name="post" type="cylinder" size="0.05 0.0995" pos="0 0 0.0995"/>'
    )
    assert shared_faces(sunk) == []


def test_coaxial_cylinder_and_capsule_of_one_radius_share_their_side():
    (pair,) = shared_faces(
        _model(
            '<geom name="tube" type="cylinder" size="0.03 0.2" pos="0 0 0.5" euler="0 90 0"/>'
            '<geom name="grip" type="capsule" size="0.03 0.05" pos="0.1 0 0.5" euler="0 90 0"/>'
        )
    )
    assert pair.kind == "cylinder" and abs(pair.direction[0]) == pytest.approx(1)
    thinner = '<geom type="capsule" size="0.029 0.05" pos="0.1 0 0.5" euler="0 90 0"/>'
    tube = '<geom type="cylinder" size="0.03 0.2" pos="0 0 0.5" euler="0 90 0"/>'
    assert shared_faces(_model(tube + thinner)) == []


#: A unit cube as an OBJ, its faces wound counter-clockwise seen from outside.
CUBE_OBJ = """v 0 0 0
v 1 0 0
v 1 1 0
v 0 1 0
v 0 0 1
v 1 0 1
v 1 1 1
v 0 1 1
f 1 3 2
f 1 4 3
f 5 6 7
f 5 7 8
f 1 2 6
f 1 6 5
f 2 3 7
f 2 7 6
f 3 4 8
f 3 8 7
f 4 1 5
f 4 5 8
"""


def test_a_mesh_face_in_a_box_face_s_plane_is_found(tmp_path):
    (tmp_path / "cube.obj").write_text(CUBE_OBJ)
    xml = tmp_path / "m.xml"
    xml.write_text(
        '<mujoco><compiler meshdir="."/><asset><mesh name="cube" file="cube.obj" scale="0.2 0.2 0.2"/>'
        '</asset><worldbody><geom name="mesh" type="mesh" mesh="cube"/>'
        '<geom name="lid" type="box" size="0.05 0.05 0.05" pos="0.1 0.1 0.15"/>'
        "</worldbody></mujoco>"
    )
    (pair,) = shared_faces(mujoco.MjModel.from_xml_path(str(xml)))
    # The lid's top is at 0.2 like the mesh's, both facing up -- outward by the mesh's winding.
    assert (pair.first, pair.second) == ("mesh", "lid")
    assert pair.direction == pytest.approx((0, 0, 1))
    assert pair.area == pytest.approx(0.1 * 0.1)


def test_a_sphere_on_a_capsule_end_of_its_radius_shares_that_end_and_one_off_it_passes():
    joint = '<geom name="joint" type="sphere" size="0.02" pos="0 0 {z}"/>'
    limb = '<geom name="limb" type="capsule" size="0.02 0.1" pos="0 0 1.1"/>'  # ends at z 1 and 1.2
    (pair,) = shared_faces(_model(joint.format(z=1) + limb))
    assert (pair.first, pair.second, pair.kind, pair.direction) == ("joint", "limb", "sphere", None)
    assert pair.at == pytest.approx((0, 0, 1))
    assert pair.area == pytest.approx(2 * math.pi * 0.02**2)  # the end's half of the sphere
    # A millimetre off the end, along the limb or away from it: two spheres of one radius on
    # different centres meet in a circle, which shows no area.
    assert shared_faces(_model(joint.format(z=1.001) + limb)) == []
    assert shared_faces(_model(joint.format(z=0.999) + limb)) == []
    # A sphere of the capsule's radius inside its straight part touches the side in a circle only.
    assert shared_faces(_model(joint.format(z=1.1) + limb)) == []


def test_two_capsules_of_one_radius_meeting_end_on_end_at_an_angle_share_the_elbow():
    elbow = (
        '<geom name="run" type="capsule" size="0.05" fromto="0 0 1 1 0 1"/>'
        '<geom name="rise" type="capsule" size="0.05" fromto="1 0 1 1 1 1"/>'
    )
    (pair,) = shared_faces(_model(elbow))
    assert (pair.first, pair.second, pair.kind) == ("run", "rise", "sphere")
    assert pair.at == pytest.approx((1, 0, 1))
    # Outward axes +x and -y, 90 degrees apart: the ends share a lune of 90 degrees, a quarter sphere.
    assert pair.area == pytest.approx(math.pi * 0.05**2)
    apart = elbow.replace('fromto="1 0 1 1 1 1"', 'fromto="1 0.001 1 1 1 1"')
    assert shared_faces(_model(apart)) == []
    thinner = elbow.replace('size="0.05" fromto="1 0 1', 'size="0.04" fromto="1 0 1')
    assert shared_faces(_model(thinner)) == []
    # End to end on one axis, the two ends look away from each other: one circle, no area.
    straight = elbow.replace('fromto="1 0 1 1 1 1"', 'fromto="1 0 1 2 0 1"')
    assert shared_faces(_model(straight)) == []


def test_a_straight_capsule_chain_off_the_axes_shares_nothing():
    # Axes written to six decimals leave the angle between two links' ends a hair short of 180
    # degrees: a lune narrower than the tolerance, which shows nothing.
    d = (0.267261, 0.534522, 0.801784)  # one direction, rounded
    joints = [tuple(round(0.3 * k * c, 6) for c in d) for k in range(4)]
    chain = "".join(
        f'<geom name="link{k}" type="capsule" size="0.04" '
        f'fromto="{" ".join(map(str, a))} {" ".join(map(str, b))}"/>'
        for k, (a, b) in enumerate(zip(joints, joints[1:], strict=False))
    )
    assert shared_faces(_model(chain)) == []


def _rotated_wall_and_sill(ox: float, oy: float, inset: float = 0.0) -> mujoco.MjModel:
    """A 3 m wall mesh at (ox, oy) turned 13 degrees and a sill box set 0.1 m into its end, sides
    flush (or ``inset`` inside them), every number written to six decimals as an exporter would."""
    a = math.radians(13)
    c, s = math.cos(a), math.sin(a)

    def at(x, y):
        return ox + c * x - s * y, oy + s * x + c * y

    corners = [(*at(x, y), z) for x in (-1.5, 1.5) for y in (-0.1, 0.1) for z in (0, 2)]
    centre = [sum(v[i] for v in corners) / 8 for i in range(3)]
    faces = []
    for f in [[0, 2, 3], [0, 3, 1], [4, 5, 7], [4, 7, 6], [0, 1, 5], [0, 5, 4],
              [2, 6, 7], [2, 7, 3], [0, 4, 6], [0, 6, 2], [1, 3, 7], [1, 7, 5]]:  # fmt: skip
        p, q, r = (corners[i] for i in f)
        u = [q[i] - p[i] for i in range(3)]
        w = [r[i] - p[i] for i in range(3)]
        n = [u[1] * w[2] - u[2] * w[1], u[2] * w[0] - u[0] * w[2], u[0] * w[1] - u[1] * w[0]]
        outward = sum(n[i] * (p[i] - centre[i]) for i in range(3)) > 0
        faces.append(f if outward else f[::-1])
    vertex = " ".join(f"{v:.6f}" for corner in corners for v in corner)
    face = " ".join(str(i) for f in faces for i in f)
    x, y = at(1.7, 0)
    return _model(
        '<geom name="wall" type="mesh" mesh="wall"/>'
        f'<geom name="sill" type="box" size="0.3 {0.1 - inset} 0.02" pos="{x:.6f} {y:.6f} 0.5" '
        'euler="0 0 13"/>',
        f'<mesh name="wall" vertex="{vertex}" face="{face}"/>',
    )


@pytest.mark.parametrize("origin", [(3, 2), (100, 100)])
def test_a_rotated_mesh_written_to_six_decimals_is_compared_near_and_far_out(origin):
    # Rounding and single-precision vertices leave the faces micrometres out of one plane; the
    # plane is measured across the faces, so a model far out is compared as one near the origin
    # (the module's TOL says how far that holds for a mesh written in world coordinates).
    (pair,) = shared_faces(_rotated_wall_and_sill(*origin))
    assert (pair.first, pair.second, pair.kind) == ("wall", "sill", "plane")
    assert pair.area == pytest.approx(0.1 * 0.04, rel=1e-3)
    assert shared_faces(_rotated_wall_and_sill(*origin, inset=1e-4)) == []


def test_the_pose_given_is_the_one_compared():
    model = _model(
        CUBE + '<body name="lid" pos="1 0 0"><freejoint/>'
        '<geom name="b" type="box" size="0.1 0.1 0.05" pos="0 0 0.15"/></body>'
    )
    assert shared_faces(model) == []  # qpos0: the lid stands a metre off
    data = mujoco.MjData(model)
    data.qpos[:3] = (0.15, 0, 0)
    mujoco.mj_kinematics(model, data)
    assert [(p.first, p.second) for p in shared_faces(model, data)] == [("a", "b")]


def test_an_ellipsoid_touching_another_round_part_is_not_passed_and_one_apart_is_ignored():
    touching = '<geom type="ellipsoid" size="0.1 0.1 0.2"/><geom type="sphere" size="0.1"/>'
    with pytest.raises(UncomparableGeoms, match="ellipsoid"):
        shared_faces(_model(touching))
    apart = (
        '<geom type="ellipsoid" size="0.1 0.1 0.2"/><geom type="sphere" size="0.1" pos="1 0 0"/>'
    )
    assert shared_faces(_model(apart)) == []


def _prop(tmp_path: Path, body: str) -> Path:
    """A finalized, credited prop folder whose MJCF is ``body``."""
    prop = tmp_path / "panel"
    prop.mkdir()
    (prop / "panel.obj").write_text(CUBE_OBJ)
    (prop / "CREDITS.txt").write_text("panel\nlicensed under CC0 (cc0)\n")
    (prop / "panel.xml").write_text(
        '<mujoco model="panel"><compiler meshdir="."/><asset>'
        '<mesh name="panel" file="panel.obj" scale="0.2 0.2 0.2"/></asset>'
        f'<worldbody><body name="panel"><geom name="shell" type="mesh" mesh="panel"/>{body}'
        "</body></worldbody></mujoco>"
    )
    return prop


def test_inspect_prop_fails_a_prop_with_a_shared_face_and_says_where(tmp_path, capsys):
    prop = _prop(
        tmp_path, '<geom name="badge" type="box" size="0.02 0.05 0.05" pos="0.1 0.1 0.15"/>'
    )
    with pytest.raises(SystemExit) as code:
        inspect_prop.main([str(prop)])
    assert code.value.code == 5
    line = next(ln for ln in capsys.readouterr().out.splitlines() if "shared_faces" in ln)
    assert "[FAIL]" in line and "shell / badge" in line and "inset the touching part" in line

    with pytest.raises(SystemExit):
        inspect_prop.main([str(prop), "--json"])
    (check,) = [
        c for c in json.loads(capsys.readouterr().out)["checks"] if c["check"] == "shared_faces"
    ]
    assert check["status"] == "FAIL" and check["not_compared"] == []
    (pair,) = check["pairs"]
    assert (pair["first"], pair["second"], pair["kind"]) == ("shell", "badge", "plane")
    assert pair["at"][2] == pytest.approx(0.2)


def test_inspect_prop_passes_the_part_once_inset(tmp_path, capsys):
    prop = _prop(
        tmp_path, '<geom name="badge" type="box" size="0.02 0.05 0.0495" pos="0.1 0.1 0.15"/>'
    )
    with pytest.raises(SystemExit) as code:
        inspect_prop.main([str(prop), "--json"])
    (check,) = [
        c for c in json.loads(capsys.readouterr().out)["checks"] if c["check"] == "shared_faces"
    ]
    assert check["status"] == "PASS" and code.value.code == 0


def test_inspect_prop_fails_a_pair_it_cannot_compare(tmp_path, capsys):
    prop = _prop(
        tmp_path,
        '<geom name="knob" type="ellipsoid" size="0.01 0.01 0.02" pos="0.1 0.1 0.5"/>'
        '<geom name="lever" type="capsule" size="0.01 0.05" pos="0.1 0.1 0.55"/>',
    )
    with pytest.raises(SystemExit) as code:
        inspect_prop.main([str(prop), "--json"])
    (check,) = [
        c for c in json.loads(capsys.readouterr().out)["checks"] if c["check"] == "shared_faces"
    ]
    assert code.value.code == 5 and check["status"] == "FAIL" and check["pairs"] == []
    assert [(p["first"], p["second"]) for p in check["not_compared"]] == [("knob", "lever")]
