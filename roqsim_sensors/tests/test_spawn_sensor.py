"""spawn_sensor: mounts the d435 model at a pose and auto-brings its capture plugin via manifest."""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from roqsim.config import load_config_from_dict
from roqsim.engine import Engine

_MODELS = pathlib.Path(__file__).resolve().parent.parent / "src" / "roqsim_sensors" / "models"

#: The mid360, zivid and robin_w1g meshes are DERIVED from vendor CAD whose redistribution terms are
#: unclear, so they are generated locally by `make external-resources` and git-ignored -- and two of
#: the three sources sit behind a product page that cannot be fetched at all. A clean checkout (CI
#: included) therefore has the models' MJCF but not their meshes, and the engine fails at compile with
#: "Error opening file 'meshes/mid360_body.obj'". Skipping is the honest outcome: the asset is absent
#: by design, not broken. See external/external_assets.yaml and the package's *_MESH_LICENSE files.
def _needs_external_mesh(model: str, mesh: str):
    return pytest.mark.skipif(
        not (_MODELS / model / "meshes" / mesh).is_file(),
        reason=(
            f"{model}: {mesh} is a generated external asset and is not committed "
            f"(run `make external-resources RESOURCE=...` -- some sources are manual downloads)"
        ),
    )


needs_mid360 = _needs_external_mesh("mid360", "mid360_body.obj")
needs_zivid = _needs_external_mesh("zivid", "zivid_body.obj")
needs_robin = _needs_external_mesh("robin_w1g", "robin_w1g_body.obj")


def _world(**spawn_config):
    cfg = {
        "sim": {},
        "plugins": [
            {
                "spawn_sensor": {"model": "d435", "name": "d435", **spawn_config},
            },
        ],
    }
    return load_config_from_dict(cfg)


def _endpoint(engine: Engine, name: str):
    return next((e for e in engine.ctx.interface.all() if e.name == name), None)


def test_manifest_auto_brings_realsense_d435_capture():
    engine = Engine(_world())
    engine.setup()
    engine.reset()
    engine.step()
    assert "d435" in engine.ctx.entities.names()
    rgb = _endpoint(engine, "image").read()
    assert rgb.shape == (480, 640, 3) and rgb.dtype == np.uint8
    assert _endpoint(engine, "camera_info") is not None
    # The d435 manifest asks for colour only: depth/points are opt-in per world, so a plain
    # `spawn_sensor: {model: d435}` must not start paying for a 307k-point cloud.
    assert _endpoint(engine, "depth") is None
    assert _endpoint(engine, "points") is None


def test_default_plugins_false_skips_the_capture_plugin():
    engine = Engine(_world(default_plugins=False))
    engine.setup()
    engine.reset()
    engine.step()
    assert _endpoint(engine, "image") is None


def _mid360_world(**spawn_config):
    cfg = {
        "sim": {},
        "plugins": [{"spawn_sensor": {"model": "mid360", "name": "mid360", **spawn_config}}],
    }
    return load_config_from_dict(cfg)


def _fov_alpha(engine: Engine):
    import mujoco

    m = engine.ctx.model
    gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "mid360_fov")
    assert gid >= 0
    return m.geom_rgba[gid][3]


def _robin_world(**spawn_config):
    cfg = {
        "sim": {},
        "plugins": [{"spawn_sensor": {"model": "robin_w1g", "name": "robin_w1g", **spawn_config}}],
    }
    return load_config_from_dict(cfg)


@needs_mid360
def test_no_fov_geom_without_show_fov():
    # The Mid-360 ships no baked _fov mesh; its sector is synthesised at build time, and only when
    # show_fov is set -- so a plain mount has no FOV geom at all (mirrors the camera path).
    import mujoco

    engine = Engine(_mid360_world())
    engine.setup()
    assert mujoco.mj_name2id(engine.ctx.model, mujoco.mjtObj.mjOBJ_GEOM, "mid360_fov") < 0


@needs_mid360
def test_show_fov_synthesises_the_lidar_sector():
    # show_fov synthesises the Mid-360's dome sector as a mesh geom named "<site>_fov" = "mid360_fov".
    engine = Engine(_mid360_world(show_fov=True, fov_alpha=0.2))
    engine.setup()
    assert np.isclose(_fov_alpha(engine), 0.2)


@needs_mid360
def test_show_fov_default_alpha_maximises_overlap_contrast():
    # Default fov_alpha is ~0.25: the darkness step between single- and double-coverage under alpha
    # blending is largest near this value and vanishes at very low alpha.
    engine = Engine(_mid360_world(show_fov=True))
    engine.setup()
    assert np.isclose(_fov_alpha(engine), 0.25)


@needs_mid360
def test_lidar_sector_geom_is_non_colliding_mesh_in_group_2():
    import mujoco

    engine = Engine(_mid360_world(show_fov=True))
    engine.setup()
    m = engine.ctx.model
    gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "mid360_fov")
    assert gid >= 0
    assert m.geom_type[gid] == mujoco.mjtGeom.mjGEOM_MESH
    assert m.geom_contype[gid] == 0 and m.geom_conaffinity[gid] == 0  # visual only
    assert m.geom_group[gid] == 2  # FOV_GEOM_GROUP -- normally rendered (not the dropped 4/5)


@needs_mid360
def test_lidar_sector_vertex_count_matches_grid():
    # A closed sector shell is an inner + outer sheet over the (azimuth x elevation) grid: 2*na*ne.
    import math

    from roqsim_sensors.plugins.spawn_sensor import _sector_grid

    na, ne = _sector_grid(0.0, 2 * math.pi, math.radians(-7.0), math.radians(52.0))
    engine = Engine(_mid360_world(show_fov=True))
    engine.setup()
    assert _mesh_vertnum(engine, "mid360_fov") == 2 * na * ne


@needs_mid360
def test_lidar_sector_reaches_the_datasheet_range():
    # The outer shell sits at the manifest far (Mid-360: 40 m detection range), so the farthest
    # vertex from the mount is ~40 m -- the user asked for the true datasheet range, not a stub.
    engine = Engine(_mid360_world(show_fov=True))
    engine.setup()
    engine.reset()
    verts = _world_fov_verts(engine, "mid360_fov")
    assert np.isclose(np.linalg.norm(verts, axis=1).max(), 40.0, atol=0.5)


@needs_robin
def test_show_fov_synthesises_robin_w1g_forward_sector():
    # The Robin W1G is the other camera-less lidar: a bounded forward 120x70 sector, drawn to its 70 m
    # spec'd range. Same synthesis path, different manifest angles.
    import mujoco

    engine = Engine(_robin_world(show_fov=True, fov_alpha=0.2))
    engine.setup()
    engine.reset()
    m = engine.ctx.model
    gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "robin_w1g_fov")
    assert gid >= 0 and m.geom_type[gid] == mujoco.mjtGeom.mjGEOM_MESH
    assert np.isclose(m.geom_rgba[gid][3], 0.2)
    verts = _world_fov_verts(engine, "robin_w1g_fov")
    assert np.isclose(np.linalg.norm(verts, axis=1).max(), 70.0, atol=1.0)


def test_manifest_fov_angles_match_capture_plugin_defaults():
    # The drawn sector must coincide with where the plugin casts; both encode the same datasheet FoV,
    # so guard the manifest angles against silently drifting from the capture plugin's defaults.
    import math

    import yaml
    from roqsim_sensors.plugins.livox_mid360 import LivoxMid360Plugin
    from roqsim_sensors.plugins.seyond_robin_w1g import SeyondRobinW1GPlugin

    from roqsim.models import resolve_model

    for plugin_cls, model in ((LivoxMid360Plugin, "mid360"), (SeyondRobinW1GPlugin, "robin_w1g")):
        p = plugin_cls()
        asset = resolve_model(model)
        fov = yaml.safe_load((asset.path.parent / f"{asset.path.stem}.manifest.yaml").read_text())["fov"]
        assert math.isclose(fov["h_min"], p.h_fov_min, abs_tol=1e-6), model
        assert math.isclose(fov["h_max"], p.h_fov_max, abs_tol=1e-6), model
        assert math.isclose(fov["v_min"], p.v_fov_min, abs_tol=1e-6), model
        assert math.isclose(fov["v_max"], p.v_fov_max, abs_tol=1e-6), model


def test_show_fov_on_a_camera_model_synthesises_a_frustum():
    # The d435 ships no _fov mesh, but it has a camera, so show_fov synthesises a translucent frustum.
    import mujoco

    engine = Engine(_world(show_fov=True, fov_alpha=0.2, fov_range=1.5))
    engine.setup()
    m = engine.ctx.model
    gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "d435_color_fov")
    assert gid >= 0  # a frustum geom was added for the camera
    assert m.geom_type[gid] == mujoco.mjtGeom.mjGEOM_MESH
    assert m.geom_contype[gid] == 0 and m.geom_conaffinity[gid] == 0  # non-colliding
    assert np.isclose(m.geom_rgba[gid][3], 0.2)


def test_fov_alpha_out_of_range_is_rejected():
    from roqsim_sensors.plugins.spawn_sensor import SpawnSensorPlugin

    errors = SpawnSensorPlugin().validate_config({"model": "mid360", "fov_alpha": 1.5})
    assert any("fov_alpha" in e for e in errors)


def _mesh_vertnum(engine: Engine, geom_name: str) -> int:
    import mujoco

    m = engine.ctx.model
    gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
    assert gid >= 0
    return int(m.mesh_vertnum[m.geom_dataid[gid]])


def _mesh_facenum(engine: Engine, geom_name: str) -> int:
    import mujoco

    m = engine.ctx.model
    gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
    assert gid >= 0
    return int(m.mesh_facenum[m.geom_dataid[gid]])


def test_synthesised_frustum_is_double_sided():
    # A single-sided shell back-face culls to nothing when you stand inside it, so you can't tell you
    # walked through a camera's FOV. The occlusion visibility mesh carries each triangle plus its
    # reverse-wound twin, so its face count is even and positive.
    engine = Engine(_world(show_fov=True, fov_range=1.5))
    engine.setup()
    n = _mesh_facenum(engine, "d435_color_fov")
    assert n > 0 and n % 2 == 0


@needs_mid360
def test_lidar_sector_is_double_sided():
    # Same guard for the lidar dome: doubled faces so the coverage volume is visible from inside too.
    engine = Engine(_mid360_world(show_fov=True))
    engine.setup()
    n = _mesh_facenum(engine, "mid360_fov")
    assert n > 0 and n % 2 == 0


@needs_zivid
def test_camera_model_always_synthesises_occluded_frustum_not_envelope():
    # The zivid ships a bundled _fov envelope but also has a camera. Occlusion is unconditional, so the
    # plugin always synthesises an occlusion-clipped frustum from the camera and leaves the baked
    # envelope hidden -- for every fov_near, including 0 (there is no reveal-the-envelope path for a
    # model that has a camera).
    import mujoco

    for extra in ({}, {"fov_near": 0.0}, {"fov_near": 1.3, "fov_range": 2.5}):
        cfg = {
            "sim": {},
            "plugins": [
                {"spawn_sensor": {"model": "zivid", "name": "zivid", "show_fov": True, "fov_alpha": 0.2, **extra}}
            ],
        }
        engine = Engine(load_config_from_dict(cfg))
        engine.setup()
        m = engine.ctx.model
        synth = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "zivid_color_fov")
        assert synth >= 0 and np.isclose(m.geom_rgba[synth][3], 0.2), extra  # synthesised frustum drawn
        baked = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "zivid_fov")
        assert baked >= 0 and m.geom_rgba[baked][3] == 0.0, extra  # bundled envelope left hidden


def test_fov_near_beyond_range_is_rejected():
    from roqsim_sensors.plugins.spawn_sensor import SpawnSensorPlugin

    plugin = SpawnSensorPlugin()
    assert any("fov_near" in e for e in plugin.validate_config({"model": "d435", "fov_near": 2.0, "fov_range": 1.5}))
    assert any("fov_near" in e for e in plugin.validate_config({"model": "d435", "fov_near": -0.1}))


def test_fov_near_without_show_fov_is_inert():
    # Setting fov_near on a world that does not enable show_fov must not touch the compiled model
    # (no geom/mesh added) -- guards "no effect on a normal run".
    base = Engine(_world())
    base.setup()
    withn = Engine(_world(fov_near=0.5))
    withn.setup()
    assert (base.ctx.model.ngeom, base.ctx.model.nmesh) == (withn.ctx.model.ngeom, withn.ctx.model.nmesh)


def test_mount_pose_places_the_camera_in_the_world():
    import mujoco

    engine = Engine(_world(pos=[1.0, 2.0, 1.5], rpy=[np.pi, 0.0, 0.0]))
    engine.setup()
    engine.reset()
    cam_id = mujoco.mj_name2id(engine.ctx.model, mujoco.mjtObj.mjOBJ_CAMERA, "d435_color")
    assert cam_id >= 0
    forward = -engine.ctx.data.cam_xmat[cam_id].reshape(3, 3)[:, 2]
    # The standalone mounts look along +y by default (horizontal, toward a wall); roll 180 deg about
    # x flips that view direction to world -y (and the up vector to -z).
    assert np.allclose(forward, [0.0, -1.0, 0.0], atol=1e-6)


@needs_zivid
def test_standalone_camera_mounts_share_the_horizontal_look_convention():
    """All standalone camera mounts look +y (horizontal, toward a wall) with +z up at rpy [0,0,0].

    Locks the shared convention so a model can't silently drift back to looking straight up (+z),
    which is what a bare optical-frame mount does in a Z-up world (see the model header comments)."""
    import mujoco

    for model, camera in (
        ("d435", "d435_color"),
        ("d415", "d415_color"),
        ("d455", "d455_color"),
        ("zivid", "zivid_color"),
    ):
        cfg = {"sim": {}, "plugins": [{"spawn_sensor": {"model": model, "name": model}}]}
        engine = Engine(load_config_from_dict(cfg))
        engine.setup()
        engine.reset()
        m = engine.ctx.model
        cid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, camera)
        rot = engine.ctx.data.cam_xmat[cid].reshape(3, 3)
        forward, up = -rot[:, 2], rot[:, 1]
        assert np.allclose(forward, [0.0, 1.0, 0.0], atol=1e-6), f"{model} view dir {forward}"
        assert np.allclose(up, [0.0, 0.0, 1.0], atol=1e-6), f"{model} up {up}"


# -- FOV occlusion (always on): camera cone clipped against world geometry ----------------------------

_OCCLUSION_WORLD = """<mujoco>
  <worldbody>
    <light directional="true" pos="0 0 3"/>
    <geom name="floor" type="plane" size="10 10 0.1"/>
    {wall}
  </worldbody>
</mujoco>"""
_WALL = '<geom name="wall" type="box" pos="0 2.0 1.0" size="3 0.05 1.5"/>'  # near face at y=1.95


def _occlusion_engine(tmp_path, wall=True, **spawn):
    p = tmp_path / "world.xml"
    p.write_text(_OCCLUSION_WORLD.format(wall=_WALL if wall else ""))
    cfg = {
        "sim": {"world": str(p)},
        "plugins": [
            {
                "spawn_sensor": {
                    "model": "d435",
                    "name": "d435",
                    "pos": [0.0, 0.0, 1.0],  # mount looks along +y (toward the wall) at rpy 0
                    "show_fov": True,
                    "fov_near": 0.2,
                    "fov_range": 3.0,
                    "fov_rays": [16, 12],
                    **spawn,
                }
            }
        ],
    }
    return Engine(load_config_from_dict(cfg))


def _world_fov_verts(engine, geom_name):
    """FOV mesh vertices in the world frame.

    MuJoCo recentres a mesh to its CoM at compile and folds that offset into the geom frame, so the
    stored ``mesh_vert`` are already in the geom frame: world = geom_xpos + geom_xmat @ vert."""
    import mujoco

    m, d = engine.ctx.model, engine.ctx.data
    gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
    assert gid >= 0
    mid = int(m.geom_dataid[gid])
    v = m.mesh_vert[m.mesh_vertadr[mid] : m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
    return d.geom_xpos[gid] + v @ d.geom_xmat[gid].reshape(3, 3).T


def _cam_world_y(engine, camera="d435_color"):
    import mujoco

    cid = mujoco.mj_name2id(engine.ctx.model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
    return float(engine.ctx.data.cam_xpos[cid][1])


def test_fov_occlusion_stops_at_a_wall(tmp_path):
    engine = _occlusion_engine(tmp_path, wall=True)
    engine.setup()
    engine.reset()
    verts = _world_fov_verts(engine, "d435_color_fov")
    cam_y = _cam_world_y(engine)
    assert np.isclose(verts[:, 1].max(), 1.95, atol=0.03)  # far sheet clamped at the wall
    assert verts[:, 1].max() < 2.0  # not the unoccluded 3 m reach
    assert np.isclose(verts[:, 1].min(), cam_y + 0.2, atol=0.02)  # near cap at fov_near


def test_fov_occlusion_open_space_reaches_far(tmp_path):
    engine = _occlusion_engine(tmp_path, wall=False)
    engine.setup()
    engine.reset()
    verts = _world_fov_verts(engine, "d435_color_fov")
    cam_y = _cam_world_y(engine)
    assert np.isclose(verts[:, 1].max(), cam_y + 3.0, atol=0.03)  # no occluder -> full fov_range


def test_fov_occlusion_mesh_is_a_grid_not_a_hull(tmp_path):
    engine = _occlusion_engine(tmp_path, wall=True)
    engine.setup()
    assert _mesh_vertnum(engine, "d435_color_fov") == 2 * 16 * 12  # near + far sheets, not an 8-vert hull


@needs_mid360
def test_camera_less_model_is_unaffected_by_unconditional_occlusion():
    # Occlusion is unconditional for cameras, but a camera-less model (mid360) has no pinhole to
    # raycast from -- it must fall through to its synthesised sector, not error.
    engine = Engine(_mid360_world(show_fov=True))
    engine.setup()  # no RuntimeError
    assert _mesh_vertnum(engine, "mid360_fov") > 0


def test_fov_occlusion_near_zero_compiles(tmp_path):
    engine = _occlusion_engine(tmp_path, wall=True, fov_near=0.0)
    engine.setup()
    engine.reset()
    verts = _world_fov_verts(engine, "d435_color_fov")
    assert np.isclose(verts[:, 1].min(), _cam_world_y(engine), atol=5e-3)  # epsilon apex at the camera


def test_fov_rays_config_validation():
    from roqsim_sensors.plugins.spawn_sensor import SpawnSensorPlugin

    plugin = SpawnSensorPlugin()
    assert any("fov_rays" in e for e in plugin.validate_config({"model": "d435", "fov_rays": [1, 4]}))
    assert any("fov_rays" in e for e in plugin.validate_config({"model": "d435", "fov_rays": [8]}))
