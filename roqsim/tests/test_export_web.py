"""export_web: the browser scene descriptor is structurally consistent and FK-correct.

The exporter's contract is that the web loader can reproduce MuJoCo's body world transforms from the
descriptor alone (rest transform composed with per-joint motion about ``axis`` anchored at ``pos``).
The round-trip test recomputes that FK in numpy exactly the way ``mujocoSceneLoader.ts`` does and
asserts it matches ``data.xpos``/``data.xquat`` -- proving the JS side will animate correctly before
we ever run a browser.
"""

from __future__ import annotations

import json

import mujoco
import numpy as np
import pytest

from roqsim.export_web import export_scene

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
    """Reproduce the loader's per-joint local motion matrix for a given joint value."""
    axis = np.asarray(joint["axis"], dtype=float)
    axis = axis / np.linalg.norm(axis)
    T = np.eye(4)
    if joint["type"] == "hinge":
        anchor = np.asarray(joint["pos"], dtype=float)
        rot = np.eye(4)
        rot[:3, :3] = _quat_to_mat(_axisangle_quat(axis, q))
        trans = np.eye(4)
        trans[:3, 3] = anchor
        untrans = np.eye(4)
        untrans[:3, 3] = -anchor
        T = trans @ rot @ untrans
    elif joint["type"] == "slide":
        T[:3, 3] = axis * q
    return T


def _axisangle_quat(axis, angle):
    q = np.zeros(4)
    mujoco.mju_axisAngle2Quat(q, axis, angle)
    return q


def _compile():
    model = mujoco.MjSpec.from_string(_MJCF).compile()
    return model, mujoco.MjData(model)


def test_descriptor_structure(tmp_path):
    model, data = _compile()
    scene = export_scene(model, data, tmp_path, __import__("logging").getLogger("t"), max_tex_dim=0)

    # Files exist and scene.bin length matches the largest buffer reference.
    assert (tmp_path / "scene.json").is_file()
    assert (tmp_path / "scene.bin").is_file()
    disk = json.loads((tmp_path / "scene.json").read_text())
    assert disk == scene
    bin_len = (tmp_path / "scene.bin").stat().st_size

    # Named joints present with correct types/metadata.
    joints = {j["name"]: j for j in scene["joints"]}
    assert joints["j_hinge"]["type"] == "hinge"
    assert joints["j_slide"]["type"] == "slide"
    assert joints["j_free"]["type"] == "free"

    # Collision-only geoms would be group 3; here all are visual, and the mesh geom references a mesh.
    mesh_geoms = [g for g in scene["geoms"] if g["type"] == "mesh"]
    assert mesh_geoms and mesh_geoms[0]["mesh"] is not None

    # Every mesh buffer slice lies within scene.bin.
    for m in scene["meshes"]:
        for ref in (m["vert"], m["index"]):
            elem = 4  # float32 / uint32
            assert ref["off"] + ref["count"] * elem <= bin_len

    # The checker texture is procedural (no source file) -> packed raw into the bin, referenced by the
    # floor material's RGB role.
    floor_geom = next(g for g in scene["geoms"] if g["type"] == "plane")
    floor_mat = scene["materials"][floor_geom["matid"]]
    assert floor_mat["texture"] >= 0
    assert "raw" in scene["textures"][floor_mat["texture"]]

    # Material mapping fields survive the round trip: the floor tiles per metre, uv_mat does not.
    assert floor_mat["texuniform"] is True
    assert floor_mat["texrepeat"] == [4.0, 4.0]
    assert scene["materials"][model.material("uv_mat").id]["texuniform"] is False


# A tetra whose texcoords are indexed separately from its vertices, OBJ-style: 5 vt for 4 v, which
# MuJoCo keeps split (it does not always re-index -- the TurtleBot 4 body is 3995 vt over 2002 v).
# The exporter has to re-index this to one shared index before it can become a GPU buffer.
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
    """A mesh whose texcoords are indexed separately still exports usable per-vertex UVs.

    The re-index is the part that can silently corrupt a mesh, so check it the way the loader will
    read it: for every triangle corner, the exported (position, uv) pair must be the pair MuJoCo
    names through its two separate indices.
    """
    model, data = _compile_split_uv(tmp_path)
    out = tmp_path / "scene"
    scene = export_scene(model, data, out, __import__("logging").getLogger("t"), max_tex_dim=0)
    raw = (out / "scene.bin").read_bytes()

    mesh = scene["meshes"][0]
    verts = np.frombuffer(raw, np.float32, mesh["vert"]["count"], mesh["vert"]["off"]).reshape(
        -1, 3
    )
    index = np.frombuffer(raw, np.uint32, mesh["index"]["count"], mesh["index"]["off"]).reshape(
        -1, 3
    )
    uv = np.frombuffer(raw, np.float32, mesh["uv"]["count"], mesh["uv"]["off"]).reshape(-1, 2)

    # One index for both attributes -- the whole point of the re-index.
    assert len(uv) == len(verts)
    assert index.shape == (int(model.mesh_facenum[0]), 3)

    want_vert = model.mesh_vert[model.mesh_face[: len(index)]]
    want_uv = model.mesh_texcoord[model.mesh_facetexcoord[: len(index)]]
    np.testing.assert_allclose(verts[index], want_vert)
    np.testing.assert_allclose(uv[index], want_uv)


def test_uv_mapped_mesh_carries_its_texcoords(tmp_path):
    """A mesh whose UVs MuJoCo already aligned per-vertex exports them verbatim; a UV-less one does not.

    Dropping them is not a cosmetic loss: the loader falls back to projecting the texture from
    geometry, which cannot reconstruct an atlas -- every surface then samples an arbitrary region of
    it.
    """
    model, data = _compile()
    scene = export_scene(model, data, tmp_path, __import__("logging").getLogger("t"), max_tex_dim=0)
    raw = (tmp_path / "scene.bin").read_bytes()

    # Geoms carry no name in the descriptor; uv_mat is only on the UV-mapped mesh, so its matid names it.
    uv_geom = next(g for g in scene["geoms"] if g["matid"] == model.material("uv_mat").id)
    mesh = scene["meshes"][uv_geom["mesh"]]
    mid = int(model.geom("g_uv_mesh").dataid.item())
    vn = int(model.mesh_vertnum[mid])
    tca = int(model.mesh_texcoordadr[mid])

    assert "uv" in mesh
    assert mesh["uv"]["count"] == 2 * vn
    got = np.frombuffer(raw, np.float32, count=2 * vn, offset=mesh["uv"]["off"])
    np.testing.assert_allclose(got, model.mesh_texcoord[tca : tca + vn].ravel())

    # The plain tetra has no texcoords (and no material), so it must not gain a uv buffer.
    plain = next(g for g in scene["geoms"] if g["type"] == "mesh" and g["matid"] < 0)
    assert "uv" not in scene["meshes"][plain["mesh"]]


@pytest.mark.parametrize("qh, qs", [(0.0, 0.0), (0.7, 0.2), (-1.2, 0.35)])
def test_fk_round_trip(tmp_path, qh, qs):
    """Descriptor FK (rest ∘ joint motion) reproduces MuJoCo's data.xpos/xquat for the arm bodies."""
    model, data = _compile()
    # Place the free body somewhere non-trivial, then set the articulated joints.
    scene = export_scene(model, data, tmp_path, __import__("logging").getLogger("t"), max_tex_dim=0)

    adr_h = model.jnt_qposadr[model.joint("j_hinge").id]
    adr_s = model.jnt_qposadr[model.joint("j_slide").id]
    data.qpos[adr_h] = qh
    data.qpos[adr_s] = qs
    mujoco.mj_forward(model, data)

    bodies = scene["bodies"]
    joints_by_body: dict[int, list] = {}
    for j in scene["joints"]:
        joints_by_body.setdefault(j["body"], []).append(j)
    qval = {"j_hinge": qh, "j_slide": qs}

    # Compute each body's world transform from the descriptor, parent-relative like the loader.
    world = [np.eye(4)] * len(bodies)
    for i, b in enumerate(bodies):
        local = _rest_mat(b["pos"], b["quat"])
        for j in joints_by_body.get(i, []):
            if j["type"] in ("hinge", "slide"):
                local = local @ _joint_motion(j, qval[j["name"]])
        world[i] = world[b["parent"]] @ local if i != 0 else local

    for i, b in enumerate(bodies):
        # Free bodies are exported at qpos0 (static); their live xpos moves, so skip them here.
        if any(j["type"] == "free" for j in joints_by_body.get(i, [])):
            continue
        got_pos = world[i][:3, 3]
        exp_pos = data.xpos[i]
        assert np.allclose(got_pos, exp_pos, atol=1e-6), f"{b['name']} pos {got_pos} != {exp_pos}"
        got_rot = world[i][:3, :3]
        exp_rot = _quat_to_mat(data.xquat[i])
        assert np.allclose(got_rot, exp_rot, atol=1e-6), f"{b['name']} rot mismatch"
