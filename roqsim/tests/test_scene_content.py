"""scene_content.walk: what every scene export writes is consistent and FK-correct.

A format with joints reproduces MuJoCo's body world transforms from the walk alone: each body's rest
transform composed with its joints' motion about ``axis`` anchored at ``pos``. The round-trip test
recomputes that FK in numpy and holds it against ``data.xpos``/``data.xquat``; the flex tests hold a
viewer's skinning of the walk's skins against MuJoCo's vertices.
"""

from __future__ import annotations

import logging

import mujoco
import numpy as np
import pytest

from roqsim.scene_content import walk

LOG = logging.getLogger("t")

# A compact world exercising every FK path: a hinge, a slide, a free body, a textured plane, a box,
# and two meshes (single tetras, one UV-mapped) so mesh buffer offsets and the texcoord path are
# exercised too.
_MJCF = """
<mujoco>
  <asset>
    <texture name="grid" type="2d" builtin="checker" width="32" height="32"
             rgb1="0.2 0.3 0.4" rgb2="0.1 0.15 0.2"/>
    <material name="floor_mat" texture="grid" texrepeat="4 4" texuniform="true"/>
    <material name="uv_mat" texture="grid" texrepeat="3 3"/>
    <mesh name="tetra" vertex="0 0 0  1 0 0  0 1 0  0 0 1"/>
    <mesh name="uv_tetra" vertex="0 0 0  1 0 0  0 1 0  0 0 1"
          face="0 2 1  0 1 3  1 2 3  2 0 3" texcoord="0 0  1 0  1 1  0 1"/>
  </asset>
  <worldbody>
    <geom name="floor" type="plane" size="3 3 0.05" material="floor_mat"/>
    <geom name="g_uv_mesh" type="mesh" mesh="uv_tetra" material="uv_mat" pos="1 1 0"/>
    <body name="link1" pos="0.1 0.2 0.3" euler="0 0 15">
      <joint name="j_hinge" type="hinge" axis="0 0 1" pos="0.05 0 0"/>
      <geom name="g_box" type="box" size="0.1 0.1 0.1" rgba="1 0 0 1"/>
      <body name="link2" pos="0.3 0 0">
        <joint name="j_slide" type="slide" axis="1 0 0"/>
        <geom name="g_mesh" type="mesh" mesh="tetra" rgba="0 1 0 1"/>
      </body>
    </body>
    <body name="freebody" pos="0 0 0">
      <freejoint name="j_free"/>
      <geom name="g_sphere" type="sphere" size="0.1"/>
    </body>
  </worldbody>
</mujoco>
"""


def _quat_to_mat(q):
    """MuJoCo (w,x,y,z) quaternion -> 3x3 rotation matrix."""
    m = np.zeros(9)
    mujoco.mju_quat2Mat(m, np.asarray(q, dtype=float))
    return m.reshape(3, 3)


def _rest_mat(pos, quat):
    T = np.eye(4)
    T[:3, :3] = _quat_to_mat(quat)
    T[:3, 3] = pos
    return T


def _joint_motion(joint, q):
    """A joint's local motion matrix at value ``q``: rotation about, or translation along, its axis.

    ``joint`` is a :class:`roqsim.scene_content.Joint` or a mapping with the same keys, as a glTF
    file's ``extras`` carry it.
    """
    field = joint.__getitem__ if isinstance(joint, dict) else lambda key: getattr(joint, key)
    axis = np.asarray(field("axis"), dtype=float)
    axis = axis / np.linalg.norm(axis)
    T = np.eye(4)
    if field("type") == "hinge":
        anchor = np.asarray(field("pos"), dtype=float)
        rot = np.eye(4)
        rot[:3, :3] = _quat_to_mat(_axisangle_quat(axis, q))
        trans = np.eye(4)
        trans[:3, 3] = anchor
        untrans = np.eye(4)
        untrans[:3, 3] = -anchor
        T = trans @ rot @ untrans
    elif field("type") == "slide":
        T[:3, 3] = axis * q
    return T


def _axisangle_quat(axis, angle):
    q = np.zeros(4)
    mujoco.mju_axisAngle2Quat(q, axis, angle)
    return q


def _compile():
    model = mujoco.MjSpec.from_string(_MJCF).compile()
    return model, mujoco.MjData(model)


def test_content_structure():
    model, data = _compile()
    content = walk(model, data, LOG)

    # Named joints present with correct types.
    joints = {j.name: j for j in content.joints}
    assert joints["j_hinge"].type == "hinge"
    assert joints["j_slide"].type == "slide"
    assert joints["j_free"].type == "free"

    # Every geom here is drawn, and a mesh geom references a mesh.
    assert {g.name for g in content.geoms} == {model.geom(g).name for g in range(model.ngeom)}
    mesh_geoms = [g for g in content.geoms if g.type == "mesh"]
    assert mesh_geoms and all(g.mesh is not None for g in mesh_geoms)

    # The floor material carries the checker texture in its RGB role.
    floor_geom = next(g for g in content.geoms if g.type == "plane")
    floor_mat = content.materials[floor_geom.matid]
    assert floor_mat.texture >= 0
    assert content.textures[floor_mat.texture].id == model.texture("grid").id

    # Material mapping fields are carried: the floor tiles per metre, uv_mat does not.
    assert floor_mat.texuniform is True
    assert floor_mat.texrepeat == [4.0, 4.0]
    assert content.materials[model.material("uv_mat").id].texuniform is False


# A tetra whose texcoords are indexed separately from its vertices, OBJ-style: 5 vt for 4 v, which
# MuJoCo keeps split (it does not always re-index -- the TurtleBot 4 body is 3995 vt over 2002 v).
# The walk re-indexes this to one shared index, which is what a GPU buffer needs.
_SPLIT_UV_OBJ = """\
v 0 0 0
v 1 0 0
v 0 1 0
v 0 0 1
vt 0 0
vt 1 0
vt 1 1
vt 0 1
vt 0.5 0.5
f 1/1 3/3 2/2
f 1/1 2/2 4/4
f 2/2 3/3 4/5
f 3/3 1/1 4/4
"""


def _compile_split_uv(tmp_path):
    """Compile a one-mesh world from _SPLIT_UV_OBJ, asserting MuJoCo really did keep it split."""
    obj = tmp_path / "split_uv.obj"
    obj.write_text(_SPLIT_UV_OBJ)
    mjcf = f"""
    <mujoco>
      <asset>
        <texture name="grid" type="2d" builtin="checker" width="8" height="8"
                 rgb1="1 0 0" rgb2="0 0 1"/>
        <material name="split_mat" texture="grid"/>
        <mesh name="split" file="{obj}"/>
      </asset>
      <worldbody><geom name="g_split" type="mesh" mesh="split" material="split_mat"/></worldbody>
    </mujoco>
    """
    model = mujoco.MjSpec.from_string(mjcf).compile()
    assert int(model.mesh_texcoordnum[0]) != int(model.mesh_vertnum[0]), (
        "this MuJoCo re-indexed the mesh itself, so the test no longer covers the split path"
    )
    return model, mujoco.MjData(model)


def test_split_texcoords_are_reindexed(tmp_path):
    """A mesh whose texcoords are indexed separately still yields usable per-vertex UVs.

    The re-index is the part that can silently corrupt a mesh, so check it the way a loader will
    read it: for every triangle corner, the (position, uv) pair must be the pair MuJoCo names
    through its two separate indices.
    """
    model, data = _compile_split_uv(tmp_path)
    (mesh,) = walk(model, data, LOG).meshes

    # One index for both attributes -- the whole point of the re-index.
    assert len(mesh.uv) == len(mesh.vert)
    assert mesh.index.shape == (int(model.mesh_facenum[0]), 3)

    want_vert = model.mesh_vert[model.mesh_face[: len(mesh.index)]]
    want_uv = model.mesh_texcoord[model.mesh_facetexcoord[: len(mesh.index)]]
    np.testing.assert_allclose(mesh.vert[mesh.index], want_vert, rtol=1e-6)
    np.testing.assert_allclose(mesh.uv[mesh.index], want_uv, rtol=1e-6)


def test_uv_mapped_mesh_carries_its_texcoords():
    """A mesh whose UVs MuJoCo already aligned per-vertex carries them verbatim; a UV-less one does not.

    Dropping them is not a cosmetic loss: a loader then projects the texture from geometry, which
    cannot reconstruct an atlas -- every surface samples an arbitrary region of it.
    """
    model, data = _compile()
    content = walk(model, data, LOG)
    by_name = {g.name: g for g in content.geoms}

    mesh = content.meshes[by_name["g_uv_mesh"].mesh]
    mid = int(model.geom("g_uv_mesh").dataid.item())
    vn = int(model.mesh_vertnum[mid])
    tca = int(model.mesh_texcoordadr[mid])
    assert mesh.uv is not None and mesh.uv.shape == (vn, 2)
    np.testing.assert_allclose(mesh.uv.ravel(), model.mesh_texcoord[tca : tca + vn].ravel())

    # The plain tetra has no texcoords, so it gains none.
    assert content.meshes[by_name["g_mesh"].mesh].uv is None


@pytest.mark.parametrize("qh, qs", [(0.0, 0.0), (0.7, 0.2), (-1.2, 0.35)])
def test_fk_round_trip(qh, qs):
    """Rest pose composed with joint motion reproduces MuJoCo's data.xpos/xquat for the arm bodies."""
    model, data = _compile()
    content = walk(model, data, LOG)

    adr_h = model.jnt_qposadr[model.joint("j_hinge").id]
    adr_s = model.jnt_qposadr[model.joint("j_slide").id]
    data.qpos[adr_h] = qh
    data.qpos[adr_s] = qs
    mujoco.mj_forward(model, data)

    joints_by_body: dict[int, list] = {}
    for j in content.joints:
        joints_by_body.setdefault(j.body, []).append(j)
    qval = {"j_hinge": qh, "j_slide": qs}

    # Each body's world transform from the walk, parent-relative as a loader composes it.
    world = [np.eye(4)] * len(content.bodies)
    for i, b in enumerate(content.bodies):
        local = _rest_mat(b.pos, b.quat)
        for j in joints_by_body.get(i, []):
            if j.type in ("hinge", "slide"):
                local = local @ _joint_motion(j, qval[j.name])
        world[i] = world[b.parent] @ local if i != 0 else local

    for i, b in enumerate(content.bodies):
        # Free bodies are placed at qpos0 (static); their live xpos moves, so skip them here.
        if any(j.type == "free" for j in joints_by_body.get(i, [])):
            continue
        got_pos = world[i][:3, 3]
        exp_pos = data.xpos[i]
        assert np.allclose(got_pos, exp_pos, atol=1e-6), f"{b.name} pos {got_pos} != {exp_pos}"
        got_rot = world[i][:3, :3]
        exp_rot = _quat_to_mat(data.xquat[i])
        assert np.allclose(got_rot, exp_rot, atol=1e-6), f"{b.name} rot mismatch"


# -- flexes, as skins ------------------------------------------------------------------------------
#
# A flex has no mesh: MuJoCo computes each vertex from bodies (``flexvert_xpos``). The walk draws it
# as a skin over those bodies, so the check is the viewer's own skinning formula --
# ``v = sum_k w_k * T_k * inv(B_k) * v_bind`` -- evaluated here in numpy against MuJoCo's vertices.


def _flex_world(flexcomp: str, holder_joint: str = '<freejoint name="holder_free"/>') -> str:
    """One flex declared in a body named ``holder``, which is free unless told otherwise."""
    return f"""
    <mujoco>
      <worldbody>
        <geom name="floor" type="plane" size="1 1 0.05"/>
        <body name="holder" pos="0.1 -0.2 0.5" euler="10 20 30">
          {holder_joint}
          <geom type="box" size="0.01 0.01 0.01"/>
          {flexcomp}
        </body>
      </worldbody>
    </mujoco>
    """


_GRID3 = (
    '<flexcomp name="blk" type="grid" count="3 3 3" spacing=".05 .05 .05" dim="3" radius=".001" '
    'rgba="0.2 0.6 0.9 1"><edge equality="true"/></flexcomp>'
)
_PINNED3 = (
    '<flexcomp name="blk" type="grid" count="3 3 3" spacing=".05 .05 .05" dim="3" radius=".001">'
    '<edge equality="true"/><pin id="0 1 2 9 10 11"/></flexcomp>'
)
_SHEET = (
    '<flexcomp name="cloth" type="grid" count="4 3 1" spacing=".05 .05 .05" dim="2" radius=".001">'
    '<edge equality="true"/></flexcomp>'
)
_CABLE = (
    '<flexcomp name="cable" type="grid" count="5 1 1" spacing=".05 .05 .05" dim="1" radius=".004">'
    '<edge equality="true"/></flexcomp>'
)
_TRILINEAR = (
    '<flexcomp name="blk" type="grid" count="4 4 4" spacing=".05 .05 .05" dim="3" radius=".001" '
    'dof="trilinear"><edge equality="true"/><contact selfcollide="none"/></flexcomp>'
)
_QUADRATIC = _TRILINEAR.replace('dof="trilinear"', 'dof="quadratic"')


def _flex_model(flexcomp, **kw):
    model = mujoco.MjSpec.from_string(_flex_world(flexcomp, **kw)).compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def _flex_skin(content, name):
    """(vertices, faces, skin, skin_index, skin_weight) of the skin drawing flex ``name``."""
    skin = next(s for s in content.skins if s.flex == name)
    return (
        np.asarray(skin.mesh.vert, dtype=float),
        np.asarray(skin.mesh.index),
        skin,
        np.asarray(skin.skin_index),
        np.asarray(skin.skin_weight, dtype=float),
    )


def _skinned(verts, skin, index, weight, pose):
    """The viewer's skinning: each vertex blended over its bones' motion since the bind pose.

    ``pose(name) -> (pos, wxyz)`` is a bone's current world pose.
    """
    motion = []
    for name, bpos, bquat in zip(skin.bones, skin.bindpos, skin.bindquat, strict=True):
        pos, quat = pose(name)
        motion.append(_rest_mat(pos, quat) @ np.linalg.inv(_rest_mat(bpos, bquat)))
    motion = np.asarray(motion)
    homo = np.hstack([verts, np.ones((len(verts), 1))])
    out = np.zeros_like(verts)
    for k in range(4):
        moved = np.einsum("vij,vj->vi", motion[index[:, k]], homo)[:, :3]
        out += weight[:, k : k + 1] * moved
    return out


def _deform(model, data, rng, amplitude=0.01):
    """Displace every vertex/node slide joint and turn the holder: an arbitrary deformed state."""
    for j in range(model.njnt):
        adr = model.jnt_qposadr[j]
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_SLIDE:
            data.qpos[adr] = rng.uniform(-amplitude, amplitude)
        elif model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            data.qpos[adr : adr + 3] += rng.uniform(-0.1, 0.1, 3)
            quat = rng.normal(size=4)
            data.qpos[adr + 3 : adr + 7] = quat / np.linalg.norm(quat)
    mujoco.mj_forward(model, data)


def _flex_verts(model, data, flex=0):
    adr, num = model.flex_vertadr[flex], model.flex_vertnum[flex]
    return data.flexvert_xpos[adr : adr + num]


def _body_pose(model, data):
    def pose(name):
        bid = model.body(name).id
        return data.xpos[bid], data.xquat[bid]

    return pose


@pytest.mark.parametrize(
    ("flexcomp", "name"),
    [(_GRID3, "blk"), (_PINNED3, "blk"), (_TRILINEAR, "blk")],
    ids=["full", "pinned", "trilinear"],
)
def test_a_flex_skin_binds_at_its_vertices_and_follows_its_bodies(caplog, flexcomp, name):
    """At rest the skin IS the flex; deformed, blending the bones reproduces MuJoCo's vertices."""
    model, data = _flex_model(flexcomp)
    with caplog.at_level("WARNING"):
        content = walk(model, data, LOG)
    assert "flex" not in caplog.text, "every drawn vertex here follows at most four bodies"
    verts, faces, skin, index, weight = _flex_skin(content, name)

    assert len(verts) == model.flex_vertnum[0]
    assert np.allclose(verts, _flex_verts(model, data), atol=1e-6)
    assert np.allclose(weight.sum(axis=1), 1.0, atol=1e-6)
    # Bones are named bodies the walk carries: a viewer binds them by name.
    assert set(skin.bones) <= {b.name for b in content.bodies}

    # Drawn vertices only: a trilinear block's interior vertices follow all eight nodes and are
    # reduced to four, which no one sees.
    drawn = np.unique(faces)
    rng = np.random.default_rng(0)
    for _ in range(3):
        _deform(model, data, rng)
        got = _skinned(verts, skin, index, weight, _body_pose(model, data))
        assert np.abs(got - _flex_verts(model, data))[drawn].max() < 1e-5


def test_a_pinned_vertex_binds_to_the_body_it_is_pinned_to():
    model, data = _flex_model(_PINNED3)
    content = walk(model, data, LOG)
    _verts, _faces, skin, index, weight = _flex_skin(content, "blk")
    holder = skin.bones.index("holder")
    pinned = [0, 1, 2, 9, 10, 11]
    assert (index[pinned, 0] == holder).all() and np.allclose(weight[pinned, 0], 1.0)
    assert "holder" not in [skin.bones[i] for i in index[12:, 0]]


def test_a_solid_flex_draws_its_boundary_facing_outward():
    model, data = _flex_model(_GRID3)
    content = walk(model, data, LOG)
    verts, faces, *_ = _flex_skin(content, "blk")
    # 3x3x3 vertices: six faces of 2x2 squares, two triangles each.
    assert len(faces) == 6 * 4 * 2
    tri = verts[faces]
    normal = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    outward = np.einsum("ij,ij->i", normal, tri.mean(axis=1) - verts.mean(axis=0))
    assert (outward > 0).all(), "a viewer draws front faces only: the shell must wind outward"
    # The skin's material is the flex's own.
    (skin,) = content.skins
    assert skin.rgba == pytest.approx([0.2, 0.6, 0.9, 1.0])


def test_a_sheet_flex_is_drawn_from_both_sides():
    model, data = _flex_model(_SHEET)
    content = walk(model, data, LOG)
    verts, faces, skin, index, weight = _flex_skin(content, "cloth")
    n = model.flex_vertnum[0]
    elems = model.flex_elem[: model.flex_elemnum[0] * 3].reshape(-1, 3)
    assert len(verts) == 2 * n and np.allclose(verts[:n], _flex_verts(model, data), atol=1e-6)
    assert np.allclose(verts[n:], verts[:n])
    front, back = faces[: len(elems)], faces[len(elems) :]
    assert (front == elems).all() and (back == elems[:, ::-1] + n).all()

    _deform(model, data, np.random.default_rng(1))
    got = _skinned(verts, skin, index, weight, _body_pose(model, data))
    assert np.allclose(got[:n], _flex_verts(model, data), atol=1e-5)
    assert np.allclose(got[n:], got[:n])


def test_a_line_flex_is_drawn_as_a_tube_of_its_radius():
    model, data = _flex_model(_CABLE)
    content = walk(model, data, LOG)
    verts, faces, skin, index, weight = _flex_skin(content, "cable")
    rest = _flex_verts(model, data)
    edges = model.flex_vertnum[0] - 1
    assert len(verts) == edges * 2 * 8 and len(faces) == edges * 2 * 8
    # Every ring point is one radius from the vertex whose bone it follows, and the ring is centred
    # on it.
    follows = np.array([skin.bones[i] for i in index[:, 0]])
    centre = np.array([rest[int(name.rsplit("_", 1)[1])] for name in follows])
    assert np.allclose(np.linalg.norm(verts - centre, axis=1), 0.004, atol=1e-6)
    tri = verts[faces]
    normal = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    axis_point = centre[faces].mean(axis=1)
    assert (np.einsum("ij,ij->i", normal, tri.mean(axis=1) - axis_point) > 0).all()


def test_a_quadratic_flex_is_capped_at_four_bones_and_says_so(caplog):
    """Nine nodes move a face vertex; the viewer takes four. Rest and affine motion stay exact."""
    model, data = _flex_model(_QUADRATIC)
    with caplog.at_level("WARNING"):
        content = walk(model, data, LOG)
    assert "follow more than 4 node bodies" in caplog.text
    verts, _faces, skin, index, weight = _flex_skin(content, "blk")
    assert np.allclose(verts, _flex_verts(model, data), atol=1e-6)
    assert np.allclose(weight.sum(axis=1), 1.0, atol=1e-5)

    # An affine deformation of the nodes about the block's centre: a stretch and a shear.
    strain = np.array([[0.05, 0.02, 0.0], [0.0, -0.03, 0.01], [0.01, 0.0, 0.04]])
    holder = model.body("holder").id
    rot = data.xmat[holder].reshape(3, 3)
    for j in range(model.njnt):
        if model.jnt_type[j] != mujoco.mjtJoint.mjJNT_SLIDE:
            continue
        body = model.jnt_bodyid[j]
        local = rot.T @ (data.xpos[body] - data.xpos[holder])  # the node's rest offset
        axis_local = model.jnt_axis[j]
        data.qpos[model.jnt_qposadr[j]] = axis_local @ (strain @ local)
    mujoco.mj_forward(model, data)
    got = _skinned(verts, skin, index, weight, _body_pose(model, data))
    assert np.allclose(got, _flex_verts(model, data), atol=1e-5)


def test_a_settled_flex_binds_at_rest_where_the_viewer_seats_its_bodies():
    """Unnamed joints have no exported value, so a viewer seats them at 0 -- bind there."""
    model, data = _flex_model(_GRID3)
    _deform(model, data, np.random.default_rng(2))
    content = walk(model, data, LOG)
    assert "" not in content.initial_joints
    verts, *_ = _flex_skin(content, "blk")
    rest = mujoco.MjData(model)
    rest.qpos[:] = data.qpos
    for j in range(model.njnt):
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_SLIDE:
            rest.qpos[model.jnt_qposadr[j]] = 0.0
    mujoco.mj_forward(model, rest)
    assert np.allclose(verts, _flex_verts(model, rest), atol=1e-6)


def test_a_flex_in_the_collision_group_is_not_drawn():
    model, data = _flex_model(_GRID3.replace('name="blk"', 'name="blk" group="3"'))
    content = walk(model, data, LOG)
    assert not any(s.flex for s in content.skins)
