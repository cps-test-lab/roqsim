"""export_gltf: a world as one binary glTF file, with the body tree, the frame and the content of
``export web``.

The file is read back here by a small GLB reader (header, JSON chunk, BIN chunk, accessors into
numpy), so every check is on what a glTF loader would see, not on the exporter's own structures.
"""

from __future__ import annotations

import json
import logging
import struct
from collections import Counter

import mujoco
import numpy as np
import pytest
from test_export_web import (
    _MJCF,
    _compile_split_uv,
    _joint_motion,
    _quat_to_mat,
    _rest_mat,
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


# -- T1: bodies are nodes, named as the bodies are --------------------------------------------------

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


# -- T2: a point in the world node's frame is a world coordinate ------------------------------------


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


# -- T3: the same content as export web -------------------------------------------------------------


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
        rgba = scene["materials"][g["matid"]]["rgba"] if g["matid"] >= 0 else g["rgba"]
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


# -- T4: tessellation and normals -------------------------------------------------------------------

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


# -- T7: the command line ---------------------------------------------------------------------------

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


def test_help_states_the_frame(capsys):
    with pytest.raises(SystemExit):
        export_gltf.main(["--help"])
    text = " ".join(capsys.readouterr().out.split())
    assert "a point in the frame of node `world` is a point in roqsim world coordinates" in text
