"""export_gltf: a world as one binary glTF file, with the body tree, the frame and the content of
``export web``.

The file is read back here by a small GLB reader (header, JSON chunk, BIN chunk, accessors into
numpy), so every check is on what a glTF loader would see, not on the exporter's own structures.
"""

from __future__ import annotations

import io
import json
import logging
import os
import shutil
import struct
import subprocess
from collections import Counter
from pathlib import Path

import mujoco
import numpy as np
import pytest
from test_export_web import (
    _CABLE,
    _GRID3,
    _MJCF,
    _PINNED3,
    _QUADRATIC,
    _SHEET,
    _TRILINEAR,
    _body_pose,
    _compile_split_uv,
    _deform,
    _flex_skin,
    _flex_verts,
    _flex_world,
    _joint_motion,
    _quat_to_mat,
    _rest_mat,
    _skinned,
)

from roqsim import exit_status, export_gltf
from roqsim.export_gltf import WORLD_ROTATION
from roqsim.export_gltf import export_gltf as write_glb
from roqsim.export_web import export_scene

LOG = logging.getLogger("t")

# -- reading a GLB ----------------------------------------------------------------------------------

_COMPONENTS = {5120: "<i1", 5121: "<u1", 5122: "<i2", 5123: "<u2", 5125: "<u4", 5126: "<f4"}
_WIDTH = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}


class Glb:
    """A GLB file as a loader sees it."""

    def __init__(self, path):
        blob = path.read_bytes()
        magic, version, length = struct.unpack_from("<III", blob, 0)
        assert (magic, version, length) == (0x46546C67, 2, len(blob))
        jlen, jtype = struct.unpack_from("<II", blob, 12)
        assert jtype == 0x4E4F534A and jlen % 4 == 0
        self.json = json.loads(blob[20 : 20 + jlen])
        blen, btype = struct.unpack_from("<II", blob, 20 + jlen)
        assert btype == 0x004E4942
        self.bin = blob[28 + jlen : 28 + jlen + blen]
        self.nodes = self.json["nodes"]

    def accessor(self, i: int) -> np.ndarray:
        acc = self.json["accessors"][i]
        view = self.json["bufferViews"][acc["bufferView"]]
        width = _WIDTH[acc["type"]]
        flat = np.frombuffer(
            self.bin,
            dtype=_COMPONENTS[acc["componentType"]],
            count=acc["count"] * width,
            offset=view.get("byteOffset", 0) + acc.get("byteOffset", 0),
        )
        return flat.reshape(acc["count"], width) if width > 1 else flat

    def local(self, n: int) -> np.ndarray:
        node = self.nodes[n]
        T = np.eye(4)
        x, y, z, w = node.get("rotation", [0.0, 0.0, 0.0, 1.0])
        T[:3, :3] = _quat_to_mat([w, x, y, z])
        T[:3, 3] = node.get("translation", [0.0, 0.0, 0.0])
        T[:3, :3] *= np.asarray(node.get("scale", [1.0, 1.0, 1.0]))
        return T

    def globals(self) -> dict[int, np.ndarray]:
        """Every node's matrix in the scene's (Y-up) frame."""
        out: dict[int, np.ndarray] = {}

        def visit(n, parent):
            out[n] = parent @ self.local(n)
            for c in self.nodes[n].get("children", []):
                visit(c, out[n])

        for root in self.json["scenes"][self.json.get("scene", 0)]["nodes"]:
            visit(root, np.eye(4))
        return out

    def parents(self) -> dict[int, int]:
        return {c: n for n, node in enumerate(self.nodes) for c in node.get("children", [])}

    def body_nodes(self) -> dict[int, int]:
        """body id -> node index, for the nodes that are bodies."""
        return {
            node["extras"]["body_id"]: n
            for n, node in enumerate(self.nodes)
            if "body_id" in node.get("extras", {})
        }

    def geom_nodes(self) -> list[int]:
        return [n for n, node in enumerate(self.nodes) if "geom_id" in node.get("extras", {})]

    def primitive(self, n: int) -> dict:
        (prim,) = self.json["meshes"][self.nodes[n]["mesh"]]["primitives"]
        return prim


def _export(model, data, tmp_path, name="scene.glb", **kw) -> Glb:
    out = tmp_path / name
    write_glb(model, data, out, log=LOG, **kw)
    return Glb(out)


def _compile(xml):
    model = mujoco.MjSpec.from_string(xml).compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def _pose(T):
    return T[:3, 3], T[:3, :3]


#: The turn the ``world`` node carries: world point (x, y, z) -> scene point (x, z, -y).
_W = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]], dtype=float)


# -- bodies are nodes, named as the bodies are -----------------------------------------------------

_NAMES = """
<mujoco>
  <worldbody>
    <geom name="floor" type="plane" size="2 2 0.1"/>
    <body name="robot/base.link" pos="0 0 0.2">
      <joint name="j" type="hinge" axis="0 0 1"/>
      <geom name="chassis" type="box" size="0.2 0.1 0.05" rgba="0.2 0.4 0.6 1"/>
      <body pos="0.3 0 0">
        <geom type="sphere" size="0.05"/>
        <body name="tip: [1]" pos="0.1 0 0"><geom type="capsule" size="0.02 0.05"/></body>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


def test_every_body_is_a_node_named_as_the_body_in_the_body_tree(tmp_path):
    model, data = _compile(_NAMES)
    glb = _export(model, data, tmp_path)
    bodies = glb.body_nodes()
    parents = glb.parents()

    assert sorted(bodies) == list(range(model.nbody)), "one node per body, no more"
    world = bodies[0]
    assert glb.nodes[world]["name"] == "world"
    assert glb.json["scenes"][0]["nodes"][0] == world
    for b, n in bodies.items():
        name = model.body(b).name
        node = glb.nodes[n]
        assert node.get("name", "") == name, "verbatim, '/', '.', ':' and brackets included"
        assert node["extras"]["body"] == name
        if b:
            assert parents[n] == bodies[int(model.body_parentid[b])]
    unnamed = next(b for b in range(1, model.nbody) if not model.body(b).name)
    assert "name" not in glb.nodes[bodies[unnamed]]

    # A geom is an unnamed node under its body's node; its own name and its body's are in extras.
    geoms = {glb.nodes[n]["extras"]["geom_id"]: n for n in glb.geom_nodes()}
    assert sorted(geoms) == list(range(model.ngeom))
    for g, n in geoms.items():
        node = glb.nodes[n]
        body = int(model.geom_bodyid[g])
        assert "name" not in node, "a geom name would compete with body names in a lookup"
        assert parents[n] == bodies[body]
        assert node["extras"]["body"] == model.body(body).name
        assert node["extras"].get("geom", "") == model.geom(g).name


# -- a point in the world node's frame is a world coordinate ---------------------------------------


def test_the_world_node_turns_z_up_onto_y_up(tmp_path):
    model, data = _compile(_NAMES)
    glb = _export(model, data, tmp_path)
    world = glb.body_nodes()[0]
    assert glb.nodes[world]["rotation"] == pytest.approx(WORLD_ROTATION)
    np.testing.assert_allclose(glb.local(world), _W, atol=1e-12)
    np.testing.assert_allclose(glb.local(world)[:3, :3] @ [1.0, 2.0, 3.0], [1.0, 3.0, -2.0])


@pytest.mark.parametrize("qh, qs", [(0.0, 0.0), (0.7, 0.2), (-1.2, 0.35)])
def test_a_point_in_the_world_nodes_frame_is_a_world_coordinate(tmp_path, qh, qs):
    """Every body sits, in the frame of node ``world``, where MuJoCo puts it at the exported state --
    jointed links at their configured values, a free body where it was placed."""
    model = mujoco.MjSpec.from_string(_MJCF).compile()
    data = mujoco.MjData(model)
    data.qpos[model.jnt_qposadr[model.joint("j_hinge").id]] = qh
    data.qpos[model.jnt_qposadr[model.joint("j_slide").id]] = qs
    free = model.jnt_qposadr[model.joint("j_free").id]
    data.qpos[free : free + 7] = [0.4, -0.3, 0.25, np.cos(0.3), 0.0, 0.0, np.sin(0.3)]
    mujoco.mj_forward(model, data)
    glb = _export(model, data, tmp_path)

    G = glb.globals()
    world = glb.body_nodes()[0]
    to_world = np.linalg.inv(G[world])
    for b, n in glb.body_nodes().items():
        pos, rot = _pose(to_world @ G[n])
        np.testing.assert_allclose(pos, data.xpos[b], atol=1e-6, err_msg=model.body(b).name)
        np.testing.assert_allclose(rot, data.xmat[b].reshape(3, 3), atol=1e-6)

    # The converse, as a viewer meets it: a world point given in node `world`'s frame lands in the
    # scene at (x, z, -y), and a geom's vertices land where MuJoCo draws them.
    p = np.array([1.5, -0.5, 2.0, 1.0])
    np.testing.assert_allclose((G[world] @ p)[:3], [1.5, 2.0, 0.5])
    box = model.geom("g_box").id
    node = next(n for n in glb.geom_nodes() if glb.nodes[n]["extras"]["geom_id"] == box)
    verts = glb.accessor(glb.primitive(node)["attributes"]["POSITION"])
    scene = (G[node] @ np.c_[verts, np.ones(len(verts))].T).T[:, :3]
    expect = data.geom_xpos[box] + verts @ data.geom_xmat[box].reshape(3, 3).T
    np.testing.assert_allclose(scene, expect @ _W[:3, :3].T, atol=1e-6)


# -- the same content as export web ----------------------------------------------------------------


def _web_world_poses(scene: dict) -> list[np.ndarray]:
    """Body world matrices as a ``roqsim.web_scene`` viewer computes them: rest pose, then joints."""
    joints: dict[int, list] = {}
    for j in scene["joints"]:
        joints.setdefault(j["body"], []).append(j)
    world = []
    for i, b in enumerate(scene["bodies"]):
        local = _rest_mat(b["pos"], b["quat"])
        for j in joints.get(i, []):
            if j["type"] in ("hinge", "slide"):
                local = local @ _joint_motion(j, scene["initialJoints"].get(j["name"], 0.0))
        world.append(world[b["parent"]] @ local if i else local)
    return world


def _colour(rgba):
    return tuple(round(min(max(float(c), 0.0), 1.0), 5) for c in rgba)


def _web_geoms(scene: dict) -> Counter:
    out = Counter()
    for g in scene["geoms"]:
        if g.get("skin") is not None:
            continue
        own = [round(c, 6) for c in g["rgba"]]
        mat = g["matid"] >= 0 and own == [0.5, 0.5, 0.5, 1.0]
        rgba = scene["materials"][g["matid"]]["rgba"] if mat else g["rgba"]  # as MuJoCo draws it
        pose = tuple(np.round(np.r_[g["pos"], g["quat"]], 6))
        out[(scene["bodies"][g["body"]]["name"], g["type"], pose, _colour(rgba))] += 1
    return out


def _glb_geoms(glb: Glb, model) -> Counter:
    out = Counter()
    for n in glb.geom_nodes():
        node = glb.nodes[n]
        g = node["extras"]["geom_id"]
        x, y, z, w = node["rotation"]
        pose = tuple(np.round(np.r_[node["translation"], [w, x, y, z]], 6))
        material = glb.json["materials"][glb.primitive(n)["material"]]
        rgba = material["pbrMetallicRoughness"]["baseColorFactor"]
        kind = mujoco.mjtGeom(int(model.geom_type[g])).name.removeprefix("mjGEOM_").lower()
        out[(node["extras"]["body"], kind, pose, _colour(rgba))] += 1
    return out


_FIXTURES = {"mixed": _MJCF, "names": _NAMES}


@pytest.mark.parametrize("fixture", sorted(_FIXTURES))
def test_gltf_and_web_exports_hold_the_same_scene(tmp_path, fixture):
    """The two exports do not drift: same bodies where the same viewer state puts them, same geoms
    with the same pose and colour."""
    model, data = _compile(_FIXTURES[fixture])
    for j in range(model.njnt):  # a non-trivial state: every named scalar joint away from rest
        if model.jnt_type[j] in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
            data.qpos[model.jnt_qposadr[j]] = 0.3 + 0.1 * j
    mujoco.mj_forward(model, data)
    web = export_scene(model, data, tmp_path / "web", LOG, max_tex_dim=0)
    glb = _export(model, data, tmp_path)

    G = glb.globals()
    to_world = np.linalg.inv(G[glb.body_nodes()[0]])
    web_world = _web_world_poses(web)
    nodes = glb.body_nodes()
    assert [b["name"] for b in web["bodies"]] == [
        glb.nodes[nodes[b]]["extras"]["body"] for b in range(len(web["bodies"]))
    ]
    for b, T in enumerate(web_world):
        np.testing.assert_allclose(to_world @ G[nodes[b]], T, atol=1e-6)
    assert _glb_geoms(glb, model) == _web_geoms(web)


# -- tessellation and normals ----------------------------------------------------------------------

_PRIMITIVES = """
<mujoco>
  <worldbody>
    <geom name="box" type="box" size="0.3 0.2 0.1"/>
    <geom name="sphere" type="sphere" size="0.2" pos="1 0 0"/>
    <geom name="capsule" type="capsule" size="0.1 0.2" pos="2 0 0"/>
    <geom name="cylinder" type="cylinder" size="0.15 0.25" pos="3 0 0"/>
    <geom name="ellipsoid" type="ellipsoid" size="0.3 0.2 0.1" pos="4 0 0"/>
    <geom name="ground" type="plane" size="0 0 1"/>
  </worldbody>
</mujoco>
"""


def _triangles(glb, n):
    prim = glb.primitive(n)
    verts = glb.accessor(prim["attributes"]["POSITION"]).astype(float)
    normals = glb.accessor(prim["attributes"]["NORMAL"]).astype(float)
    faces = glb.accessor(prim["indices"]).astype(int).reshape(-1, 3)
    return verts, normals, faces


def test_primitives_wind_outward_and_their_normals_agree(tmp_path):
    model, data = _compile(_PRIMITIVES)
    glb = _export(model, data, tmp_path, segments=16)
    for n in glb.geom_nodes():
        name = glb.nodes[n]["extras"]["geom"]
        verts, normals, faces = _triangles(glb, n)
        tri = verts[faces]
        face_n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        assert (np.linalg.norm(face_n, axis=1) > 0).all(), f"{name}: a degenerate triangle"
        np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1.0, atol=1e-5)
        # Each corner's normal leans the way its face does.
        for k in range(3):
            assert (np.einsum("ij,ij->i", normals[faces[:, k]], face_n) > 0).all(), name
        if name == "ground":
            continue
        # Outward: away from the solid's centre (each is convex and centred on its origin).
        assert (np.einsum("ij,ij->i", face_n, tri.mean(axis=1)) > 0).all(), name


def test_a_box_is_flat_and_a_sphere_is_smooth(tmp_path):
    model, data = _compile(_PRIMITIVES)
    glb = _export(model, data, tmp_path, segments=16)
    by_name = {glb.nodes[n]["extras"]["geom"]: n for n in glb.geom_nodes()}
    _verts, normals, _faces = _triangles(glb, by_name["box"])
    axes = np.abs(normals).max(axis=1)
    np.testing.assert_allclose(axes, 1.0, atol=1e-6)  # every box normal is a face normal
    verts, normals, _faces = _triangles(glb, by_name["sphere"])
    np.testing.assert_allclose(normals, verts / np.linalg.norm(verts, axis=1)[:, None], atol=1e-6)
    # The cylinder keeps its rim: cap normals are +-z, side normals radial.
    verts, normals, _faces = _triangles(glb, by_name["cylinder"])
    cap = np.abs(normals[:, 2]) > 0.5
    np.testing.assert_allclose(np.abs(normals[cap, 2]), 1.0, atol=1e-6)
    np.testing.assert_allclose(normals[~cap, 2], 0.0, atol=1e-6)


def test_an_infinite_plane_is_a_square_of_the_models_extent(tmp_path):
    model, data = _compile(_PRIMITIVES)
    glb = _export(model, data, tmp_path)
    by_name = {glb.nodes[n]["extras"]["geom"]: n for n in glb.geom_nodes()}
    verts, _normals, _faces = _triangles(glb, by_name["ground"])
    np.testing.assert_allclose(np.abs(verts[:, :2]), model.stat.extent, rtol=1e-6)


def test_a_mesh_carries_mujocos_normals_per_corner(tmp_path):
    """MuJoCo indexes a mesh's normals apart from its vertices; one index has to keep both."""
    model, data = _compile(_MJCF)
    glb = _export(model, data, tmp_path)
    for name in ("g_mesh", "g_uv_mesh"):
        g = model.geom(name).id
        n = next(n for n in glb.geom_nodes() if glb.nodes[n]["extras"]["geom_id"] == g)
        verts, normals, faces = _triangles(glb, n)
        mid = int(model.geom_dataid[g])
        va, na = model.mesh_vertadr[mid], model.mesh_normaladr[mid]
        fa, fn = model.mesh_faceadr[mid], model.mesh_facenum[mid]
        np.testing.assert_allclose(
            verts[faces], model.mesh_vert[va + model.mesh_face[fa : fa + fn]]
        )
        np.testing.assert_allclose(
            normals[faces], model.mesh_normal[na + model.mesh_facenormal[fa : fa + fn]], atol=1e-6
        )


def test_split_texcoords_keep_their_position_pairs(tmp_path):
    model, data = _compile_split_uv(tmp_path)
    glb = _export(model, data, tmp_path)
    (n,) = glb.geom_nodes()
    prim = glb.primitive(n)
    verts = glb.accessor(prim["attributes"]["POSITION"])
    uv = glb.accessor(prim["attributes"]["TEXCOORD_0"])
    faces = glb.accessor(prim["indices"]).astype(int).reshape(-1, 3)
    np.testing.assert_allclose(verts[faces], model.mesh_vert[model.mesh_face])
    np.testing.assert_allclose(uv[faces], model.mesh_texcoord[model.mesh_facetexcoord])


def test_geoms_that_look_the_same_share_one_mesh(tmp_path):
    xml = _PRIMITIVES.replace(
        '<geom name="ground" type="plane" size="0 0 1"/>',
        '<geom name="box2" type="box" size="0.3 0.2 0.1" pos="0 2 0"/>'
        '<geom name="box3" type="box" size="0.3 0.2 0.1" pos="0 3 0" rgba="1 0 0 1"/>',
    )
    model, data = _compile(xml)
    glb = _export(model, data, tmp_path)
    by_name = {glb.nodes[n]["extras"]["geom"]: glb.nodes[n]["mesh"] for n in glb.geom_nodes()}
    assert by_name["box"] == by_name["box2"]
    assert by_name["box3"] != by_name["box"], "another colour is another material"


def test_the_authored_view_is_a_camera_where_render_puts_its_own(tmp_path):
    model, data = _compile(_NAMES)
    view = {"lookat": [0.5, -0.2, 0.3], "distance": 4.0, "azimuth": 120.0, "elevation": -30.0}
    glb = _export(model, data, tmp_path, view=view)
    (n,) = [n for n, node in enumerate(glb.nodes) if node.get("extras") == {"roqsim": "view"}]
    assert n in glb.json["scenes"][0]["nodes"], "a root of its own, outside `world`"
    cam = mujoco.MjvCamera()
    cam.lookat[:], cam.distance, cam.azimuth, cam.elevation = (
        view["lookat"],
        view["distance"],
        view["azimuth"],
        view["elevation"],
    )
    from roqsim.rendering import eye_position

    T = glb.local(n)
    np.testing.assert_allclose(T[:3, 3], _W[:3, :3] @ eye_position(cam), atol=1e-9)
    # It looks down its -z at the lookat, with the scene's up above.
    look = -T[:3, 2]
    target = _W[:3, :3] @ np.asarray(view["lookat"]) - T[:3, 3]
    np.testing.assert_allclose(look, target / np.linalg.norm(target), atol=1e-9)
    assert T[1, 1] > 0
    yfov = glb.json["cameras"][glb.nodes[n]["camera"]]["perspective"]["yfov"]
    assert yfov == pytest.approx(np.radians(model.vis.global_.fovy))


# -- skins and flexes ------------------------------------------------------------------------------
#
# A glTF viewer skins a vertex as ``sum_k w_k * J_k * IBM_k * v``, ``J_k`` the joint node's matrix in
# the scene. Here the joint nodes are posed the way a viewer replaying a run poses them -- each body
# node at the body's world pose, seen through the ``world`` node -- and the result is held against
# MuJoCo, or against ``export web``'s skin of the same flex, whose skinning its own tests hold
# against MuJoCo.

_SKIN = """
<mujoco>
  <worldbody>
    <body name="b1" pos="0 0 1"><freejoint/><geom type="sphere" size=".01"/></body>
    <body name="b2" pos="1 0 1"><freejoint/><geom type="sphere" size=".01"/></body>
  </worldbody>
  <deformable>
    <skin name="s" rgba="0.8 0.2 0.2 1" vertex="0 0 1  1 0 1  0.5 0.5 1  0.5 -0.5 1"
          face="0 1 2  0 3 1">
      <bone body="b1" bindpos="0 0 1" bindquat="1 0 0 0" vertid="0 2 3" vertweight="1 0.5 0.5"/>
      <bone body="b2" bindpos="1 0 1" bindquat="1 0 0 0" vertid="1 2 3" vertweight="1 0.5 0.5"/>
    </skin>
  </deformable>
</mujoco>
"""


def _skin_nodes(glb):
    return [n for n, node in enumerate(glb.nodes) if "skin" in node]


def _glb_skinned(glb, n, joint_matrix):
    """The scene positions of skin node ``n``'s vertices, its joints at ``joint_matrix(node)``."""
    prim = glb.primitive(n)
    skin = glb.json["skins"][glb.nodes[n]["skin"]]
    verts = glb.accessor(prim["attributes"]["POSITION"]).astype(float)
    joints = glb.accessor(prim["attributes"]["JOINTS_0"]).astype(int)
    weights = glb.accessor(prim["attributes"]["WEIGHTS_0"]).astype(float)
    ibm = glb.accessor(skin["inverseBindMatrices"]).reshape(-1, 4, 4).transpose(0, 2, 1)
    motion = np.array([joint_matrix(j) @ ibm[k] for k, j in enumerate(skin["joints"])])
    homo = np.c_[verts, np.ones(len(verts))]
    out = np.zeros_like(verts)
    for k in range(4):
        out += weights[:, k : k + 1] * np.einsum("vij,vj->vi", motion[joints[:, k]], homo)[:, :3]
    return out


def _posed(glb, model, data):
    """Joint matrices of a viewer that seats every body node at the body's pose in ``data``."""
    node_body = {n: b for b, n in glb.body_nodes().items()}

    def matrix(node):
        b = node_body[node]
        T = np.eye(4)
        T[:3, :3] = data.xmat[b].reshape(3, 3)
        T[:3, 3] = data.xpos[b]
        return _W @ T

    return matrix


def _yup(points):
    return np.asarray(points) @ _W[:3, :3].T


def test_a_skin_is_a_skinned_mesh_over_its_bone_bodies(tmp_path):
    model, data = _compile(_SKIN)
    glb = _export(model, data, tmp_path)
    (n,) = _skin_nodes(glb)
    assert n in glb.json["scenes"][0]["nodes"], "a skinned mesh's node is a root"
    assert "translation" not in glb.nodes[n] and "rotation" not in glb.nodes[n]
    skin = glb.json["skins"][glb.nodes[n]["skin"]]
    bodies = glb.body_nodes()
    assert skin["joints"] == [bodies[model.body("b1").id], bodies[model.body("b2").id]]
    G = glb.globals()
    rest = _glb_skinned(glb, n, G.__getitem__)
    np.testing.assert_allclose(rest, _yup(model.skin_vert.reshape(-1, 3)), atol=1e-6)

    # Move one bone: the vertices it carries follow it, those half on it go half way.
    free = model.jnt_qposadr[model.body("b2").jntadr[0]]
    data.qpos[free : free + 3] += [0.0, 0.0, 0.4]
    mujoco.mj_forward(model, data)
    moved = _glb_skinned(glb, n, _posed(glb, model, data))
    lift = (moved - rest) @ _W[:3, :3]  # back to world axes
    np.testing.assert_allclose(lift[:, 2], [0.0, 0.4, 0.2, 0.2], atol=1e-6)


@pytest.mark.parametrize(
    ("flexcomp", "name"),
    [(_GRID3, "blk"), (_PINNED3, "blk"), (_TRILINEAR, "blk"), (_SHEET, "cloth"), (_CABLE, "cable")],
    ids=["solid", "pinned", "trilinear", "sheet", "cable"],
)
def test_a_flex_deforms_as_export_webs_skin_of_it_does(tmp_path, flexcomp, name):
    model, data = _compile(_flex_world(flexcomp))
    web = export_scene(model, data, tmp_path / "web", LOG)
    web_verts, faces, web_skin, index, weight = _flex_skin(web, tmp_path / "web", name)
    glb = _export(model, data, tmp_path)
    (n,) = [n for n in _skin_nodes(glb) if glb.nodes[n]["extras"].get("flex") == name]
    prim = glb.primitive(n)
    np.testing.assert_allclose(
        glb.accessor(prim["indices"]).reshape(-1, 3), faces, err_msg="same triangles"
    )
    weights = glb.accessor(prim["attributes"]["WEIGHTS_0"]).astype(float)
    assert (weights >= 0).all()
    np.testing.assert_allclose(weights.sum(axis=1), 1.0, atol=1e-6)

    # At the exported state the file draws the flex where MuJoCo has it.
    G = glb.globals()
    np.testing.assert_allclose(_glb_skinned(glb, n, G.__getitem__), _yup(web_verts), atol=1e-6)
    if name == "blk":
        np.testing.assert_allclose(
            _glb_skinned(glb, n, G.__getitem__), _yup(_flex_verts(model, data)), atol=1e-6
        )
    rng = np.random.default_rng(0)
    drawn = np.unique(faces)
    for _ in range(3):
        _deform(model, data, rng)
        got = _glb_skinned(glb, n, _posed(glb, model, data))
        want = _skinned(web_verts, web_skin, index, weight, _body_pose(model, data))
        assert np.abs(got - _yup(want))[drawn].max() < 1e-5
        if name == "blk":
            assert np.abs(got - _yup(_flex_verts(model, data)))[drawn].max() < 1e-5


def test_a_quadratic_flex_drops_its_negative_weights_and_says_so(tmp_path, caplog):
    model, data = _compile(_flex_world(_QUADRATIC))
    with caplog.at_level("WARNING"):
        glb = _export(model, data, tmp_path)
    assert "negative skin weights set to 0" in caplog.text
    (n,) = _skin_nodes(glb)
    weights = glb.accessor(glb.primitive(n)["attributes"]["WEIGHTS_0"]).astype(float)
    assert (weights >= 0).all()
    np.testing.assert_allclose(weights.sum(axis=1), 1.0, atol=1e-6)
    # Exact at rest, whatever the weights.
    rest = _glb_skinned(glb, n, glb.globals().__getitem__)
    np.testing.assert_allclose(rest, _yup(_flex_verts(model, data)), atol=1e-6)


def test_a_flex_in_the_collision_group_is_not_drawn(tmp_path):
    model, data = _compile(_flex_world(_GRID3.replace('name="blk"', 'name="blk" group="3"')))
    glb = _export(model, data, tmp_path)
    assert not _skin_nodes(glb)


# -- materials and textures ------------------------------------------------------------------------


def _image(glb, texture_index):
    from PIL import Image

    tex = glb.json["textures"][texture_index]
    image = glb.json["images"][tex["source"]]
    view = glb.json["bufferViews"][image["bufferView"]]
    data = glb.bin[view["byteOffset"] : view["byteOffset"] + view["byteLength"]]
    return image["mimeType"], Image.open(io.BytesIO(data))


def _write_png(path, pixels):
    from PIL import Image

    Image.fromarray(np.asarray(pixels, np.uint8)).save(path)
    return path


def _textured_world(tmp_path, pixels, *, material='rgba="1 1 1 1"', geom="", name="tex"):
    # A file name per image: MuJoCo caches a texture by its path.
    png = _write_png(tmp_path / f"{name}.png", pixels)
    channels = np.asarray(pixels).shape[2]
    return f"""
    <mujoco>
      <asset>
        <texture name="t" type="2d" file="{png}" nchannel="{channels}"/>
        <material name="m" texture="t" {material}/>
      </asset>
      <worldbody><geom name="g" type="box" size="0.2 0.3 0.1" material="m" {geom}/></worldbody>
    </mujoco>
    """


def _quadrants(h=64, w=96, alpha=None):
    px = np.zeros((h, w, 3 if alpha is None else 4), np.uint8)
    px[: h // 2, : w // 2, :3] = (220, 30, 30)
    px[: h // 2, w // 2 :, :3] = (30, 200, 40)
    px[h // 2 :, : w // 2, :3] = (40, 60, 210)
    px[h // 2 :, w // 2 :, :3] = (230, 210, 40)
    if alpha is not None:
        px[..., 3] = alpha
    return px


def _material_of(glb, geom_name):
    n = next(n for n in glb.geom_nodes() if glb.nodes[n]["extras"].get("geom") == geom_name)
    return glb.json["materials"][glb.primitive(n)["material"]], glb.primitive(n)


def test_an_embedded_image_is_the_compiled_texture(tmp_path):
    model, data = _compile(_textured_world(tmp_path, _quadrants()))
    glb = _export(model, data, tmp_path, max_tex_dim=0)
    material, prim = _material_of(glb, "g")
    assert "TEXCOORD_0" in prim["attributes"]
    mime, img = _image(glb, material["pbrMetallicRoughness"]["baseColorTexture"]["index"])
    assert mime == "image/png"
    tid = model.texture("t").id
    want = model.tex_data[model.tex_adr[tid] : model.tex_adr[tid] + 64 * 96 * 3].reshape(64, 96, 3)
    np.testing.assert_array_equal(np.asarray(img.convert("RGB")), want)
    # The compiled pixels are the file's, top row first: glTF's image origin is MuJoCo's.
    np.testing.assert_array_equal(want, _quadrants())


def test_a_large_texture_is_downscaled(tmp_path):
    model, data = _compile(_textured_world(tmp_path, _quadrants(64, 96)))
    glb = _export(model, data, tmp_path, max_tex_dim=24)
    material, _ = _material_of(glb, "g")
    _mime, img = _image(glb, material["pbrMetallicRoughness"]["baseColorTexture"]["index"])
    assert img.size == (24, 16)


def test_jpeg_is_for_opaque_images_only(tmp_path):
    for alpha, want in ((255, "image/jpeg"), (128, "image/png")):
        model, data = _compile(_textured_world(tmp_path, _quadrants(alpha=alpha), name=f"a{alpha}"))
        glb = _export(model, data, tmp_path, name=f"{alpha}.glb", texture_format="jpeg")
        material, _ = _material_of(glb, "g")
        mime, _img = _image(glb, material["pbrMetallicRoughness"]["baseColorTexture"]["index"])
        assert mime == want


def test_a_tint_above_one_is_folded_into_the_image(tmp_path):
    model, data = _compile(_textured_world(tmp_path, _quadrants(), material='rgba="2 1 0.5 1"'))
    glb = _export(model, data, tmp_path, max_tex_dim=0)
    material, _ = _material_of(glb, "g")
    pbr = material["pbrMetallicRoughness"]
    assert pbr["baseColorFactor"] == [1.0, 1.0, 1.0, 1.0]
    _mime, img = _image(glb, pbr["baseColorTexture"]["index"])
    want = np.clip(np.rint(_quadrants() * np.array([2.0, 1.0, 0.5])), 0, 255)
    np.testing.assert_array_equal(np.asarray(img.convert("RGB")), want)


def test_an_untextured_colour_above_one_is_clamped_and_named(tmp_path, caplog):
    xml = """<mujoco><asset><material name="hot" rgba="1.5 0.5 0.2 1"/></asset>
    <worldbody><geom type="sphere" size=".1" material="hot"/></worldbody></mujoco>"""
    model, data = _compile(xml)
    with caplog.at_level("WARNING"):
        glb = _export(model, data, tmp_path)
    assert "material 'hot'" in caplog.text
    (material,) = glb.json["materials"]
    assert material["pbrMetallicRoughness"]["baseColorFactor"] == pytest.approx(
        [1.0, 0.5, 0.2, 1.0]
    )


def test_a_geoms_own_colour_wins_over_its_materials_as_mujoco_draws_it(tmp_path):
    xml = """<mujoco><asset><material name="m" rgba="0 0 1 1"/></asset><worldbody>
    <geom name="plain" type="sphere" size=".1" material="m"/>
    <geom name="tinted" type="sphere" size=".1" material="m" rgba="1 0 0 1" pos="1 0 0"/>
    </worldbody></mujoco>"""
    model, data = _compile(xml)
    glb = _export(model, data, tmp_path)
    plain, _ = _material_of(glb, "plain")
    tinted, _ = _material_of(glb, "tinted")
    assert plain["pbrMetallicRoughness"]["baseColorFactor"] == [0.0, 0.0, 1.0, 1.0]
    assert tinted["pbrMetallicRoughness"]["baseColorFactor"] == [1.0, 0.0, 0.0, 1.0]


def test_a_cube_texture_is_refused_by_name(tmp_path, capsys):
    xml = """<mujoco><asset>
    <texture name="cube" type="cube" builtin="checker" width="8" height="8" rgb1="1 0 0" rgb2="0 0 1"/>
    <material name="dice" texture="cube"/></asset>
    <worldbody><geom type="box" size=".1 .1 .1" material="dice"/></worldbody></mujoco>"""
    scene = tmp_path / "cube.xml"
    scene.write_text(xml)
    model, data = _compile(xml)
    with pytest.raises(
        export_gltf.GltfExportError, match="material 'dice'.*'cube', a cube texture"
    ):
        _export(model, data, tmp_path)
    status = export_gltf.main(["--mjcf", str(scene), "--out", str(tmp_path / "x.glb")])
    assert status == exit_status.BAD_INPUT
    assert "cube texture" in capsys.readouterr().err


# -- the texture coordinates MuJoCo projects -------------------------------------------------------
#
# MuJoCo renders the scene with ambient light only, so a pixel is its texel. The same pixel's ray is
# cast with mj_ray; where it meets a geom, the exported mesh of that geom is found under the hit
# point, the exported texture coordinate interpolated there, and the embedded image sampled. Pixels
# whose coordinate falls near a cell edge of the test texture are skipped: filtering blends there.

_CELLS = 4  # the test texture is a 4 x 4 grid of distinct colours, asymmetric in both directions
_PALETTE = np.array(
    [
        [230, 25, 75], [60, 180, 75], [255, 225, 25], [0, 130, 200],
        [245, 130, 48], [145, 30, 180], [70, 240, 240], [240, 50, 230],
        [210, 245, 60], [250, 190, 212], [0, 128, 128], [220, 190, 255],
        [170, 110, 40], [255, 250, 200], [128, 0, 0], [0, 0, 128],
    ],
    np.uint8,
)  # fmt: skip


def _grid_texture(cell=32):
    idx = np.arange(_CELLS * _CELLS).reshape(_CELLS, _CELLS)
    return np.kron(_PALETTE[idx], np.ones((cell, cell, 1), np.uint8)).astype(np.uint8)


_PROJECTED = """
<mujoco>
  <visual>
    <headlight ambient="1 1 1" diffuse="0 0 0" specular="0 0 0"/>
    <quality shadowsize="0" numslices="64" numstacks="32"/>
  </visual>
  <asset>
    <texture name="grid" type="2d" file="{png}"/>
    <material name="plain" texture="grid" specular="0" shininess="0" reflectance="0"/>
    <material name="rep" texture="grid" texrepeat="2 1.5" specular="0" shininess="0" reflectance="0"/>
    <material name="uni" texture="grid" texrepeat="0.5 0.5" texuniform="true" specular="0"
              shininess="0" reflectance="0"/>
    <material name="dense" texture="grid" texrepeat="4 4" texuniform="true" specular="0"
              shininess="0" reflectance="0"/>
    <mesh name="block" vertex="-0.3 -0.2 0  0.3 -0.2 0  0.3 0.2 0  -0.3 0.2 0
                               -0.3 -0.2 0.25  0.3 -0.2 0.25  0.3 0.2 0.25  -0.3 0.2 0.25"/>
  </asset>
  <worldbody>
    <camera name="look" pos="0 -3.2 3.0" xyaxes="1 0 0 0 0.68 0.73" fovy="50"/>
    <geom name="floor" type="plane" size="2 1.5 0.1" material="uni"/>
    <geom name="box" type="box" size="0.35 0.3 0.2" pos="-1 0.4 0.2" euler="0 0 20"
          material="plain"/>
    <geom name="cylinder" type="cylinder" size="0.2 0.2" pos="0 0.6 0.2" material="plain"/>
    <geom name="sphere" type="sphere" size="0.25" pos="1 0.3 0.25" material="plain"/>
    <geom name="capsule" type="capsule" size="0.18 0.3" pos="-0.5 -0.6 0.18" euler="0 90 30"
          material="rep"/>
    <geom name="mesh" type="mesh" mesh="block" pos="0.7 -0.6 0" material="dense"/>
  </worldbody>
</mujoco>
"""


def _surface_uv(glb, n, local):
    """The exported texture coordinate of mesh node ``n`` at the point ``local`` on it, or None."""
    verts, _normals, faces = _triangles(glb, n)
    uv = glb.accessor(glb.primitive(n)["attributes"]["TEXCOORD_0"]).astype(float)
    a, b, c = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    normal = np.cross(b - a, c - a)
    area2 = np.einsum("ij,ij->i", normal, normal)
    ok = area2 > 0
    unit = normal / np.sqrt(np.where(ok, area2, 1.0))[:, None]
    dist = np.abs(np.einsum("ij,ij->i", local - a, unit))
    proj = local - dist[:, None] * unit * np.sign(np.einsum("ij,ij->i", local - a, unit))[:, None]
    w_a = np.einsum("ij,ij->i", np.cross(c - b, proj - b), normal) / np.where(ok, area2, 1.0)
    w_b = np.einsum("ij,ij->i", np.cross(a - c, proj - c), normal) / np.where(ok, area2, 1.0)
    w_c = 1.0 - w_a - w_b
    inside = ok & (w_a >= -1e-6) & (w_b >= -1e-6) & (w_c >= -1e-6)
    if not inside.any():
        return None
    k = np.flatnonzero(inside)[np.argmin(dist[inside])]
    if dist[k] > 0.01:
        return None
    f = faces[k]
    return w_a[k] * uv[f[0]] + w_b[k] * uv[f[1]] + w_c[k] * uv[f[2]]


def test_projected_texture_coordinates_draw_what_mujoco_draws(tmp_path):
    from roqsim.rendering import FrameRenderer, GLBackendError

    png = _write_png(tmp_path / "grid.png", _grid_texture())
    model, data = _compile(_PROJECTED.format(png=png))
    width, height = 320, 240
    try:
        frame = FrameRenderer(model, width, height, camera="look").render(data)
    except (GLBackendError, mujoco.FatalError) as err:
        pytest.skip(f"no usable offscreen GL here: {err}")
    glb = _export(model, data, tmp_path, max_tex_dim=0)  # segments: the model's own 64
    texture = np.asarray(_image(glb, 0)[1].convert("RGB"))
    th, tw = texture.shape[:2]
    node_of = {glb.nodes[n]["extras"]["geom_id"]: n for n in glb.geom_nodes()}

    cam = model.camera("look").id
    origin = data.cam_xpos[cam]
    axes = data.cam_xmat[cam].reshape(3, 3)
    tan = np.tan(np.radians(model.cam_fovy[cam]) / 2)
    geomid = np.zeros(1, np.int32)
    hits = {}
    which = np.full((height, width), -1)
    for i in range(height):
        for j in range(width):
            x = ((j + 0.5) / width * 2 - 1) * tan * width / height
            y = (1 - (i + 0.5) / height * 2) * tan
            ray = axes @ np.array([x, y, -1.0])
            ray /= np.linalg.norm(ray)
            dist = mujoco.mj_ray(model, data, origin, ray, None, 1, -1, geomid)
            if dist >= 0:
                which[i, j] = int(geomid[0])
                hits[i, j] = origin + dist * ray
    uvs = {}
    for (i, j), hit in hits.items():
        g = int(which[i, j])
        if g in node_of:
            local = data.geom_xmat[g].reshape(3, 3).T @ (hit - data.geom_xpos[g])
            uvs[i, j] = _surface_uv(glb, node_of[g], local)
    checked, wrong = Counter(), Counter()
    for (i, j), uv in uvs.items():
        g = int(which[i, j])
        around = which[max(i - 1, 0) : i + 2, max(j - 1, 0) : j + 2]
        neighbours = [uvs.get((i, j + 1)), uvs.get((i + 1, j))]
        if uv is None or (around != g).any() or any(n is None for n in neighbours):
            continue  # a silhouette pixel is blended with what lies behind it
        # MuJoCo samples a mipmap: over a footprint this many cells wide, the colour is a blend.
        footprint = max(np.abs(n - uv).max() for n in neighbours) * _CELLS
        cell = (uv % 1.0) * _CELLS
        if footprint > 0.25 or (np.abs(cell - np.rint(cell)) < 0.15 + footprint).any():
            continue  # near a cell edge, or a footprint too wide: filtering decides the colour
        texel = texture[int((uv[1] % 1.0) * th), int((uv[0] % 1.0) * tw)].astype(int)
        name = model.geom(g).name
        checked[name] += 1
        if np.abs(frame[i, j].astype(int) - texel).max() > 40:
            wrong[name] += 1
    for name in ("floor", "box", "cylinder", "sphere", "capsule", "mesh"):
        assert checked[name] >= 15, f"{name}: only {checked[name]} pixels to compare"
        assert wrong[name] <= 0.03 * checked[name], f"{name}: {wrong[name]}/{checked[name]} wrong"


# -- the Khronos glTF Validator --------------------------------------------------------------------
#
# The validator is a Node.js program, pinned in tools/gltf_validator (`make test-gltf` installs it).
# Under `make test` this runs where it is installed and says it skipped where it is not; under
# `make test-gltf` (ROQSIM_GLTF_VALIDATOR=required) a missing validator fails.

_VALIDATOR = Path(__file__).resolve().parents[2] / "tools" / "gltf_validator"


def khronos(paths) -> dict:
    """The Khronos glTF Validator's report on each of ``paths``."""
    missing = shutil.which("node") is None or not (_VALIDATOR / "node_modules").is_dir()
    if missing:
        why = (
            "the Khronos glTF Validator is not installed: `make test-gltf` installs it "
            f"({_VALIDATOR / 'package.json'}), and needs Node.js"
        )
        if os.environ.get("ROQSIM_GLTF_VALIDATOR") == "required":
            pytest.fail(why)
        pytest.skip(why)
    run = subprocess.run(
        ["node", str(_VALIDATOR / "validate.mjs"), *map(str, paths)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert run.stdout, run.stderr
    return json.loads(run.stdout)


def test_khronos_validator_finds_nothing_wrong_with_a_small_world(tmp_path):
    """Bodies, joints, every primitive, meshes, a texture, a skin and the flexes, through the CLI
    and from a bare MJCF: no error and no warning."""
    paths = []
    textured = _PROJECTED.format(png=_write_png(tmp_path / "grid.png", _grid_texture()))
    for name, xml in {
        "mixed": _MJCF,
        "names": _NAMES,
        "primitives": _PRIMITIVES,
        "textured": textured,
        "skin": _SKIN,
        "solid": _flex_world(_GRID3),
        "sheet": _flex_world(_SHEET),
        "cable": _flex_world(_CABLE),
    }.items():
        scene = tmp_path / f"{name}.xml"
        scene.write_text(xml, encoding="utf-8")
        out = tmp_path / f"{name}.glb"
        assert export_gltf.main(["--mjcf", str(scene), "--out", str(out)]) == exit_status.OK
        paths.append(out)
    for path, report in khronos(paths).items():
        assert report["errors"] == 0 and report["warnings"] == 0, (path, report["messages"])


# -- the command line ------------------------------------------------------------------------------

_BARE = "<mujoco><worldbody><body name='box'><geom type='box' size='.1 .1 .1'/></body></worldbody></mujoco>"


@pytest.mark.parametrize(
    "flags",
    [
        ["--set", "sim.timestep=0.001"],
        ["--override", "run.overrides.yaml"],
        ["--skip-plugins", "floorplan"],
        ["--settle-steps", "500"],
    ],
)
def test_a_world_option_with_a_bare_mjcf_is_bad_input(tmp_path, capsys, flags):
    scene = tmp_path / "scene.xml"
    scene.write_text(_BARE, encoding="utf-8")
    out = tmp_path / "out.glb"
    with pytest.raises(SystemExit) as exit_info:
        export_gltf.main(["--mjcf", str(scene), "--out", str(out), *flags])
    assert exit_info.value.code == exit_status.BAD_INPUT
    assert "--mjcf compiles a bare MJCF" in capsys.readouterr().err
    assert not out.exists()


def test_a_bare_mjcf_exports(tmp_path):
    scene = tmp_path / "scene.xml"
    scene.write_text(_BARE, encoding="utf-8")
    out = tmp_path / "out" / "box.glb"
    assert export_gltf.main(["--mjcf", str(scene), "--out", str(out)]) == exit_status.OK
    glb = Glb(out)
    assert {node.get("name") for node in glb.nodes} >= {"world", "box"}


def test_out_must_be_a_glb(tmp_path, capsys):
    scene = tmp_path / "scene.xml"
    scene.write_text(_BARE, encoding="utf-8")
    with pytest.raises(SystemExit) as exit_info:
        export_gltf.main(["--mjcf", str(scene), "--out", str(tmp_path / "scene.gltf")])
    assert exit_info.value.code == exit_status.BAD_INPUT
    assert ".glb" in capsys.readouterr().err


@pytest.mark.parametrize("segments", ["0", "2", "-8"])
def test_too_few_segments_is_bad_input(tmp_path, capsys, segments):
    scene = tmp_path / "scene.xml"
    scene.write_text(_BARE, encoding="utf-8")
    out = tmp_path / "out.glb"
    with pytest.raises(SystemExit) as exit_info:
        export_gltf.main(["--mjcf", str(scene), "--out", str(out), "--segments", segments])
    assert exit_info.value.code == exit_status.BAD_INPUT
    assert "--segments takes 3 or more" in capsys.readouterr().err
    assert not out.exists()


def test_help_states_the_frame(capsys):
    with pytest.raises(SystemExit):
        export_gltf.main(["--help"])
    text = " ".join(capsys.readouterr().out.split())
    assert "a point in the frame of node `world` is a point in roqsim world coordinates" in text
