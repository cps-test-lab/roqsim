"""The ``fr3`` looks as Franka's description draws it, and collides where it is drawn.

The visual meshes are held to ``upstream/franka/franka_description/fr3_visual.json``, the digest of
franka_description's COLLADA visuals that ``roqsim arm franka-upstream`` reads at the pin: each
primitive (one material) is one OBJ beside the MJCF, with the same triangles, the same open edges
(a decimated piece tears and pulls its outline away from its neighbour's), the same winding, and
the DAE's diffuse, emission and reflectivity as its material.

The collision geometry is held to the visual one along rays from outside the arm, which is how a
contact meets it: MuJoCo collides each mesh as its convex hull, so the hulls are what is measured.
"""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np
import pytest
from _franka_upstream import DESCRIPTION
from scipy.spatial import ConvexHull

from roqsim.models import resolve_model

MJCF = Path(resolve_model("fr3").path)
MESHES = MJCF.parent / "meshes"
DIGEST = json.loads((DESCRIPTION / "fr3_visual.json").read_text(encoding="utf-8"))
HOME = [0.0, -np.pi / 4, 0.0, -3 * np.pi / 4, 0.0, np.pi / 2, np.pi / 4]


@pytest.fixture(scope="module")
def model():
    return mujoco.MjModel.from_xml_path(str(MJCF))


@pytest.fixture(scope="module")
def spec():
    return mujoco.MjSpec.from_file(str(MJCF))


def _obj(path: Path) -> tuple[np.ndarray, np.ndarray]:
    vertices, faces = [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("v "):
            vertices.append([float(c) for c in line.split()[1:4]])
        elif line.startswith("f "):
            faces.append([int(c.split("/")[0]) - 1 for c in line.split()[1:4]])
    return np.array(vertices), np.array(faces)


def _visual_geoms(model, link: int) -> list[int]:
    body = model.body(f"fr3_link{link}").id
    return [
        g for g in range(model.ngeom) if model.geom_bodyid[g] == body and model.geom_group[g] == 2
    ]


@pytest.mark.parametrize("link", range(8))
def test_each_dae_primitive_is_one_unchanged_visual_mesh(model, link):
    pieces = DIGEST[f"link{link}"]
    geoms = _visual_geoms(model, link)
    assert [model.geom(g).name for g in geoms] == [
        f"fr3_link{link}_{j}_visual" for j in range(len(pieces))
    ]
    for j, piece in enumerate(pieces):
        vertices, faces = _obj(MESHES / f"link{link}_{j}.obj")
        corners = vertices[faces]
        edges = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), 1)
        _, uses = np.unique(edges, axis=0, return_counts=True)
        normal = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]).sum(0) / 2
        where = f"link{link}_{j} ({piece['material']})"
        assert len(faces) == piece["triangles"], where
        assert int((uses == 1).sum()) == piece["open_edges"], where
        assert corners.reshape(-1, 3).min(0) == pytest.approx(piece["bounds"][0], abs=2e-6), where
        assert corners.reshape(-1, 3).max(0) == pytest.approx(piece["bounds"][1], abs=2e-6), where
        assert normal == pytest.approx(piece["area_normal"], abs=1e-6), where  # winding kept


@pytest.mark.parametrize("link", range(8))
def test_each_visual_mesh_wears_its_daes_material(model, spec, link):
    for j, (g, piece) in enumerate(
        zip(_visual_geoms(model, link), DIGEST[f"link{link}"], strict=True)
    ):
        where = f"link{link}_{j} ({piece['material']})"
        # The mesh sits in the link frame, as the URDF's visual origin (none) puts it.
        placed = spec.geom(model.geom(g).name)
        assert np.allclose(placed.pos, 0, atol=1e-9), where
        assert np.allclose(np.abs(placed.quat), [1, 0, 0, 0], atol=1e-9), where
        mat = model.geom_matid[g]
        assert mat >= 0, where
        assert model.geom_rgba[g] == pytest.approx([0.5, 0.5, 0.5, 1])  # MuJoCo's default: unused
        assert model.mat_rgba[mat] == pytest.approx(piece["diffuse"], abs=1e-6), where
        lit = np.mean(piece["diffuse"][:3])
        assert model.mat_emission[mat] * lit == pytest.approx(
            np.mean(piece["emission"][:3]), abs=1e-4
        ), where
        if piece["reflectivity"] is not None:
            assert model.mat_specular[mat] == pytest.approx(piece["reflectivity"]), where


def test_every_collider_is_convex_and_only_colliders_collide(model):
    """What group 3 draws is what a contact meets: MuJoCo collides a mesh as its convex hull."""
    for g in range(model.ngeom):
        collides = model.geom_contype[g] or model.geom_conaffinity[g]
        assert bool(collides) == (model.geom_group[g] == 3), model.geom(g).name
        if collides:  # the mesh encloses its own convex hull's volume
            mesh = model.geom_dataid[g]
            adr = model.mesh_vertadr[mesh]
            vertices = model.mesh_vert[adr : adr + model.mesh_vertnum[mesh]].astype(float)
            fadr = model.mesh_faceadr[mesh]
            a, b, c = vertices[model.mesh_face[fadr : fadr + model.mesh_facenum[mesh]]].transpose(
                1, 0, 2
            )
            volume = abs(np.einsum("fk,fk->f", a, np.cross(b, c)).sum()) / 6
            assert volume == pytest.approx(ConvexHull(vertices).volume, rel=1e-3), model.geom(
                g
            ).name


def _first_hits(model, data, rng, rays: int) -> tuple[np.ndarray, np.ndarray]:
    """Distance along rays from outside the arm to the first visual surface, and to the first
    collider's convex hull (inf for a miss)."""
    groups = model.geom_group
    visual = [g for g in range(model.ngeom) if groups[g] == 2]
    colliders = [g for g in range(model.ngeom) if groups[g] == 3]

    def world(g, vertices):
        return vertices @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g]

    def verts(g):
        mesh = model.geom_dataid[g]
        adr = model.mesh_vertadr[mesh]
        return model.mesh_vert[adr : adr + model.mesh_vertnum[mesh]]

    planes = [ConvexHull(world(g, verts(g))).equations for g in colliders]
    points = np.concatenate([world(g, verts(g)) for g in visual])
    reach = np.linalg.norm(points.max(0) - points.min(0))
    directions = rng.normal(size=(rays, 3))
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    targets = points[rng.integers(len(points), size=rays)] + rng.normal(scale=0.01, size=(rays, 3))
    starts = targets - directions * reach
    only_visual = np.array([0, 0, 1, 0, 0, 0], np.uint8)
    hit = np.zeros(1, np.int32)
    to_visual, to_collider = np.full(rays, np.inf), np.full(rays, np.inf)
    for r, (p, u) in enumerate(zip(starts, directions, strict=True)):
        t = mujoco.mj_ray(model, data, p, u, only_visual, 1, -1, hit)
        if t >= 0:
            to_visual[r] = t
        for eq in planes:  # a ray enters a convex hull at its last entering plane
            depth, along = -(eq[:, :3] @ p + eq[:, 3]), eq[:, :3] @ u
            if np.any((along == 0) & (depth < 0)):
                continue
            with np.errstate(divide="ignore"):
                t_plane = depth / along
            enter = t_plane[along < 0].max(initial=-np.inf)
            leave = t_plane[along > 0].min(initial=np.inf)
            if enter <= leave and leave >= 0:
                to_collider[r] = min(to_collider[r], max(enter, 0.0))
    return to_visual, to_collider


def test_the_colliders_stand_where_the_arm_is_drawn(model):
    """Along rays from outside, a contact meets a collider within 20 mm of the drawn surface.

    A link's whole STL collides as one hull, which bridges the elbow's and the wrist's recesses: a
    third of such rays met a collider more than 20 mm proud of the arm. Where franka_description's
    STL leaves a drawn part out (the base's rear strip, a recess on links 3 and 4), the ray meets
    the drawing first; that stays a small share.
    """
    data = mujoco.MjData(model)
    data.qpos[:] = HOME
    mujoco.mj_forward(model, data)
    to_visual, to_collider = _first_hits(model, data, np.random.default_rng(0), 2000)
    seen = np.isfinite(to_visual) | np.isfinite(to_collider)
    proud = to_visual[seen] - to_collider[seen]  # +inf: a collider where nothing is drawn
    assert np.mean(proud > 0.02) < 0.05
    assert np.mean(proud < -0.02) < 0.05


def test_the_arm_touches_nothing_of_its_own_at_home_or_in_reach(model):
    """Neighbouring links' pieces meet at their joint; MuJoCo filters a body's contacts with its
    parent, but not once the parent is welded to the world, as link0 is."""
    data = mujoco.MjData(model)
    lo, hi = model.jnt_range[:, 0], model.jnt_range[:, 1]
    rng = np.random.default_rng(1)
    for q in [HOME] + [HOME + rng.uniform(-0.3, 0.3, 7) for _ in range(50)]:
        data.qpos[:] = np.clip(q, lo, hi)
        mujoco.mj_forward(model, data)
        touching = {
            (
                model.body(model.geom_bodyid[c.geom1]).name,
                model.body(model.geom_bodyid[c.geom2]).name,
            )
            for c in data.contact[: data.ncon]
        }
        assert not touching, (data.qpos, touching)
