"""Unit tests for the walker: skeleton FK, clips, planner, blueprint, and the plugin end-to-end."""

from __future__ import annotations

import math
import pathlib
import shutil

import mujoco
import numpy as np
import pytest

from roqsim_walker.blueprint import available_walkers, resolve_walker
from roqsim_walker.humanoid import (
    DEFAULT_SKELETON,
    JOINT_NAMES,
    build_humanoid,
    forward_kinematics,
    quat_yaw,
    to_skeleton,
)
from roqsim_walker.motion import Clip, procedural_idle, procedural_walk, smoothstep
from roqsim_walker.nav.occupancy import OccupancyGrid
from roqsim_walker.nav.planner import GridPlanner

IDENT = {n: np.array([1.0, 0.0, 0.0, 0.0]) for n in JOINT_NAMES}


# -- humanoid / FK -----------------------------------------------------------------------------
def test_forward_kinematics_rest_pose_puts_feet_near_floor():
    poses = forward_kinematics([0.0, 0.0, DEFAULT_SKELETON.root_height], 0.0, IDENT)
    assert set(poses) == set(JOINT_NAMES)
    for pos, quat in poses.values():
        assert pos.shape == (3,) and quat.shape == (4,)
    # ankles sit a little above the floor; the sole reaches it.
    assert 0.0 < poses["ankle_l"][0][2] < 0.15
    assert poses["head"][0][2] > poses["pelvis"][0][2] > poses["knee_l"][0][2]


def test_forward_kinematics_yaw_rotates_the_body():
    root = [1.0, 2.0, DEFAULT_SKELETON.root_height]
    straight = forward_kinematics(root, 0.0, IDENT)
    turned = forward_kinematics(root, math.pi / 2, IDENT)
    # pelvis is the root: position pinned, orientation follows yaw.
    np.testing.assert_allclose(turned["pelvis"][0], root)
    np.testing.assert_allclose(turned["pelvis"][1], quat_yaw(math.pi / 2), atol=1e-9)
    # the left hip, offset along +Y at rest, swings to -X after a +90 deg yaw.
    dx = straight["hip_l"][0] - np.array(root)
    dy = turned["hip_l"][0] - np.array(root)
    assert dx[1] > 0.05 and dy[0] < -0.05


def test_to_skeleton_reads_a_walker_json_block():
    skel = to_skeleton({"root_height": 1.05, "offsets": {"spine": [0.0, 0.0, 0.3]}})
    assert skel.root_height == pytest.approx(1.05)
    assert skel.offset("spine") == (0.0, 0.0, 0.3)
    assert skel.offset("head") == DEFAULT_SKELETON.offset("head")  # falls back to the default table


# -- motion clips ------------------------------------------------------------------------------
def test_procedural_clip_shape_and_sampling():
    clip = procedural_walk(frames=30)
    assert clip.joint_rot.shape == (30, len(JOINT_NAMES), 4)
    q, z = clip.sample_array(0.0)
    assert q.shape == (len(JOINT_NAMES), 4)
    np.testing.assert_allclose(np.linalg.norm(q, axis=1), 1.0, atol=1e-9)  # unit quats
    assert isinstance(z, float)


def test_clip_sampling_is_continuous_and_wraps():
    clip = procedural_walk(frames=30)
    a, _ = clip.sample_array(0.999)
    b, _ = clip.sample_array(1.001)  # phase wraps to ~0.001
    assert np.abs(a - b).max() < 0.2


def test_clip_roundtrip_reorders_joints(tmp_path):
    clip = procedural_idle(frames=8)
    path = tmp_path / "idle.npz"
    clip.save(path)
    loaded = Clip.load(path)
    assert loaded.joint_names == list(JOINT_NAMES)
    np.testing.assert_allclose(loaded.joint_rot, clip.joint_rot)


def test_smoothstep_edges():
    assert smoothstep(0.0, 1.0, 2.0) == 0.0
    assert smoothstep(3.0, 1.0, 2.0) == 1.0
    assert smoothstep(1.5, 1.0, 2.0) == pytest.approx(0.5)


# -- planner -----------------------------------------------------------------------------------
_BOUNDS = (-2.0, -2.0, 2.0, 2.0)  # OccupancyGrid.from_polygons pads this by 1 m on every side


def _wall_grid(top: float = 1.0):
    """A room bisected by a wall at x=0 running from below the padded floor up to ``top``, so the
    only way across is over the gap at the top."""
    wall = [(-0.1, -3.5), (0.1, -3.5), (0.1, top), (-0.1, top)]
    return OccupancyGrid.from_polygons([wall], resolution=0.05, bounds=_BOUNDS)


def test_planner_routes_around_a_wall():
    planner = GridPlanner(_wall_grid(), inflation_radius=0.2)
    path = planner.plan((-1.0, -1.0), (1.0, -1.0))
    assert path, "goal should be reachable via the gap above the wall"
    assert path[-1] == pytest.approx((1.0, -1.0))
    # The straight line would cross x=0 at y=-1 (solid); the plan must detour over the wall's top.
    assert max(y for _, y in path) > 1.0


def test_planner_returns_none_when_the_goal_is_walled_off():
    # A wall spanning the whole padded grid seals the two halves apart.
    sealed = _wall_grid(top=3.5)
    planner = GridPlanner(sealed, inflation_radius=0.0)
    assert planner.plan((-1.0, 0.0), (1.0, 0.0)) is None


def test_planner_simplifies_a_clear_run_to_a_straight_shot():
    grid = OccupancyGrid.from_polygons([[(9.0, 9.0), (9.1, 9.0), (9.1, 9.1)]], bounds=_BOUNDS)
    planner = GridPlanner(grid, inflation_radius=0.0)
    path = planner.plan((-1.5, -1.5), (1.5, 1.5))
    assert path is not None
    assert len(path) <= 2, f"line-of-sight string-pulling should collapse the path, got {path}"


def test_occupancy_inflation_grows_obstacles():
    grid = _wall_grid()
    base = grid.inflate(0.0).sum()
    grown = grid.inflate(0.3).sum()
    assert grown > base


# -- blueprint ---------------------------------------------------------------------------------
def test_obj_groups_key_by_material_name():
    # Regression: trimesh only keys the OBJ groups by their material names when the blueprint's .mtl
    # sits next to the OBJ. Without it every group falls back to a generic name, materials.get()
    # misses, and the character renders as flat colour instead of textured.
    from roqsim_walker import skin

    spec = resolve_walker("MaleVisitorWalk")
    part_names = {name for name, *_ in skin._load_parts(spec["mesh"])}
    assert part_names == set(spec["materials"]), (
        "OBJ material groups must match walker.json material keys (is the blueprint's .mtl present?)"
    )


def test_bundled_blueprint_resolves():
    assert "MaleVisitorWalk" in available_walkers()
    spec = resolve_walker("MaleVisitorWalk")
    assert spec["mesh"].endswith("maleVisitorWalk.obj")
    assert spec["skeleton"]["root_height"] > 0.5  # a standing adult, read from the bind pose
    assert set(spec["sole"]) == {"l", "r"}
    for kind in ("idle", "walk", "run", "short", "turn_l", "turn_r"):
        assert kind in spec["motion"], kind
    # Its own <animation> tracks are ignored; locomotion comes from the shared clip set.
    assert "/anims/" in spec["motion"]["walk"]


@pytest.mark.parametrize(
    "name,anim,flip,n_skins",
    [
        ("FemaleVisitorWalk", "female", False, 1),  # handbag group excluded at import
        ("MaleVisitorWalk", "adult", False, 1),
    ],
)
def test_imported_openrmf_actor_resolves_and_compiles(name, anim, flip, n_skins):
    """The Open-RMF actors imported by `roqsim walker import-actor`: rigged + textured, T-posed but +X
    facing (flip false, unlike CARLA), and driven by our own locomotion clips."""
    if name not in available_walkers():
        pytest.skip(f"{name} not imported (run `roqsim walker import-actor`)")
    spec = resolve_walker(name)
    assert spec["tpose"] is True and spec["flip"] is flip
    assert set(spec["sole"]) == {"l", "r"}
    assert f"/anims/{anim}/" in spec["motion"]["walk"]
    # OBJ material groups must line up with walker.json keys, or the skin binds flat.
    from roqsim_walker import skin

    assert {n for n, *_ in skin._load_parts(spec["mesh"])} == set(spec["materials"])

    mspec = mujoco.MjSpec()
    build_humanoid(
        mspec,
        name="ped",
        mesh=spec["mesh"],
        materials=spec["materials"],
        tpose=spec["tpose"],
        flip=spec["flip"],
        skeleton=spec["skeleton"],
        collision=spec["collision"],
    )
    model = mspec.compile()
    assert model.nmocap == len(JOINT_NAMES)
    assert model.nskin == n_skins
    assert model.ntex >= 1  # the character texture actually loaded


def test_imported_actor_weights_sidecar_matches_the_obj():
    """The authored weights.npz must be row-matched to the OBJ verts (skin._load_weights KD-tree
    queries by position) and normalised across our 17 joints."""
    if "FemaleVisitorWalk" not in available_walkers():
        pytest.skip("FemaleVisitorWalk not imported")
    import os

    from roqsim_walker import skin

    spec = resolve_walker("FemaleVisitorWalk")
    npz = os.path.splitext(spec["mesh"])[0] + ".weights.npz"
    assert os.path.isfile(npz), "importer must write an authored weights sidecar"
    data = np.load(npz)
    assert data["weights"].shape[1] == len(JOINT_NAMES)
    np.testing.assert_allclose(data["weights"].sum(axis=1), 1.0, atol=1e-5)
    lookup = skin._load_weights(spec["mesh"])
    assert lookup is not None  # the sidecar is picked up, not the auto-rig fallback


def test_unknown_blueprint_raises():
    from roqsim_walker.blueprint import BlueprintError

    with pytest.raises(BlueprintError, match="Nope"):
        resolve_walker("Nope")


# -- build_humanoid ----------------------------------------------------------------------------
def test_build_humanoid_capsules_compiles_with_one_mocap_body_per_joint():
    spec = mujoco.MjSpec()
    names = build_humanoid(spec, name="ped", mesh=None)
    model = spec.compile()
    assert names == list(JOINT_NAMES)
    assert model.nmocap == len(JOINT_NAMES)
    for part in JOINT_NAMES:
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"ped/{part}") >= 0


def test_build_humanoid_with_a_skinned_blueprint_compiles():
    blueprint = resolve_walker("MaleVisitorWalk")
    spec = mujoco.MjSpec()
    build_humanoid(
        spec,
        name="ped",
        mesh=blueprint["mesh"],
        materials=blueprint["materials"],
        tpose=blueprint["tpose"],
        skeleton=blueprint["skeleton"],
        collision=blueprint["collision"],
    )
    model = spec.compile()
    assert model.nmocap == len(JOINT_NAMES)
    # One <skin> per OBJ material group, so the count is the blueprint's, not a fixed number: a
    # single-material Fuel actor gives 1, a multi-material character one per group.
    assert model.nskin == len(blueprint["materials"])
    # Every material that declares a texture must bind one -- the "textured, not flat colour" check.
    # Asserting against the sidecar rather than a literal keeps this true for any blueprint.
    want = sum(1 for m in blueprint["materials"].values() if m.get("texture"))
    textured = sum(model.skin_matid[i] >= 0 for i in range(model.nskin))
    assert textured >= want, f"expected >={want} textured skins, got {textured}"
    assert model.ntex >= want, "no textures loaded for the character"


# -- blueprints resolve across packages ------------------------------------------------------------
def test_a_foreign_package_can_ship_a_blueprint(tmp_path, monkeypatch):
    """A character does not have to live in this package.

    Blueprints resolve across every registered ``roqsim.models`` provider, so a downstream package
    ships its own people without this one being edited -- and a world still names only the walker.
    Without this, a blueprint outside ``roqsim_walker/models`` is unreachable no matter how it is
    registered, because the lookup used to be one hardcoded directory.
    """
    from roqsim_walker import blueprint

    foreign = tmp_path / "foreign_models"
    actor = foreign / "people" / "ForeignActor"
    actor.mkdir(parents=True)
    # Reuse a bundled blueprint's sidecar + mesh: this test is about *where* it is found.
    bundled = pathlib.Path(blueprint.models_dir()) / "people" / "MaleVisitorWalk"
    for f in bundled.iterdir():
        if f.is_file():
            shutil.copy(f, actor / f.name)

    monkeypatch.setattr(
        blueprint, "_providers", lambda: [("foreign", foreign)], raising=False
    )
    monkeypatch.setattr(
        "roqsim.models.providers",
        lambda: [("foreign", foreign, foreign / "meshes", foreign)],
    )

    assert "ForeignActor" in blueprint.available_walkers()
    spec = blueprint.resolve_walker("ForeignActor")
    assert pathlib.Path(spec["mesh"]).is_file()
    assert str(foreign) in spec["mesh"], "the foreign package's own mesh must win"
    # It ships no anims/, so clips fall back to this package's shared set rather than failing.
    assert spec["motion"], "clips must still resolve from the bundled set"
    assert all(pathlib.Path(p).is_file() for p in spec["motion"].values())


def test_a_bundled_blueprint_is_not_shadowed_by_a_foreign_one(tmp_path, monkeypatch):
    """This package is searched first, so a same-named folder elsewhere cannot hijack a name."""
    from roqsim_walker import blueprint

    foreign = tmp_path / "foreign_models"
    (foreign / "people" / "MaleVisitorWalk").mkdir(parents=True)
    # A sidecar that would resolve, if it were ever reached.
    bundled = pathlib.Path(blueprint.models_dir()) / "people" / "MaleVisitorWalk"
    for f in bundled.iterdir():
        if f.is_file():
            shutil.copy(f, foreign / "people" / "MaleVisitorWalk" / f.name)

    monkeypatch.setattr(
        "roqsim.models.providers",
        lambda: [("foreign", foreign, foreign / "meshes", foreign)],
    )
    spec = blueprint.resolve_walker("MaleVisitorWalk")
    assert str(bundled) in spec["mesh"], "the bundled blueprint must win its own name"
