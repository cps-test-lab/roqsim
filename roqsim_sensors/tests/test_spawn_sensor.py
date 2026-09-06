"""spawn_sensor: mounts the d435 model at a pose and auto-brings its capture plugin via manifest."""

from __future__ import annotations

import mujoco
import numpy as np
import pytest
from external_meshes import needs_mid360, needs_robin, needs_zivid

from roqsim.config import load_config_from_dict
from roqsim.engine import Engine
from roqsim.plugin import Plugin
from roqsim.context import SimContext


def _world(**spawn_config):
    cfg = {
        "sim": {},
        "plugins": [
            {
                "spawn_sensor": {"model": "d435", **spawn_config},
                "name": "d435",
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
        "plugins": [{"spawn_sensor": {"model": "mid360", **spawn_config}, "name": "mid360"}],
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
        "plugins": [{"spawn_sensor": {"model": "robin_w1g", **spawn_config}, "name": "robin_w1g"}],
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
        fov = yaml.safe_load((asset.path.parent / f"{asset.path.stem}.manifest.yaml").read_text())[
            "fov"
        ]
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
                {
                    "spawn_sensor": {
                        "model": "zivid",
                        "show_fov": True,
                        "fov_alpha": 0.2,
                        **extra,
                    },
                    "name": "zivid",
                }
            ],
        }
        engine = Engine(load_config_from_dict(cfg))
        engine.setup()
        m = engine.ctx.model
        synth = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "zivid_color_fov")
        assert synth >= 0 and np.isclose(m.geom_rgba[synth][3], 0.2), (
            extra
        )  # synthesised frustum drawn
        baked = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "zivid_fov")
        assert baked >= 0 and m.geom_rgba[baked][3] == 0.0, extra  # bundled envelope left hidden


def test_fov_near_beyond_range_is_rejected():
    from roqsim_sensors.plugins.spawn_sensor import SpawnSensorPlugin

    plugin = SpawnSensorPlugin()
    assert any(
        "fov_near" in e
        for e in plugin.validate_config({"model": "d435", "fov_near": 2.0, "fov_range": 1.5})
    )
    assert any("fov_near" in e for e in plugin.validate_config({"model": "d435", "fov_near": -0.1}))


def test_fov_near_without_show_fov_is_inert():
    # Setting fov_near on a world that does not enable show_fov must not touch the compiled model
    # (no geom/mesh added) -- guards "no effect on a normal run".
    base = Engine(_world())
    base.setup()
    withn = Engine(_world(fov_near=0.5))
    withn.setup()
    assert (base.ctx.model.ngeom, base.ctx.model.nmesh) == (
        withn.ctx.model.ngeom,
        withn.ctx.model.nmesh,
    )


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
        cfg = {"sim": {}, "plugins": [{"spawn_sensor": {"model": model}, "name": model}]}
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
                    "pos": [0.0, 0.0, 1.0],  # mount looks along +y (toward the wall) at rpy 0
                    "show_fov": True,
                    "fov_near": 0.2,
                    "fov_range": 3.0,
                    "fov_rays": [16, 12],
                    **spawn,
                },
                "name": "d435",
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
    assert (
        _mesh_vertnum(engine, "d435_color_fov") == 2 * 16 * 12
    )  # near + far sheets, not an 8-vert hull


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
    assert np.isclose(
        verts[:, 1].min(), _cam_world_y(engine), atol=5e-3
    )  # epsilon apex at the camera


def test_fov_rays_config_validation():
    from roqsim_sensors.plugins.spawn_sensor import SpawnSensorPlugin

    plugin = SpawnSensorPlugin()
    assert any(
        "fov_rays" in e for e in plugin.validate_config({"model": "d435", "fov_rays": [1, 4]})
    )
    assert any("fov_rays" in e for e in plugin.validate_config({"model": "d435", "fov_rays": [8]}))


@needs_mid360
def test_lidar_sector_is_clipped_by_the_walls():
    """A synthesised lidar sector must stop at world geometry, like a camera frustum already did.

    It used to draw its full physical reach straight through the building. That is wrong on its own
    terms -- the volume claims to show what the sensor sees -- and it wrecked every render of such a
    world: the Mid-360's 40 m dome and the Robin W1G's 200 m cone bounded an otherwise 10 m room, so
    MuJoCo's model-derived default camera framed a ~160 m box and the room came out as a few dark
    pixels.

    Asserted below the wall tops, where the default empty_room actually encloses the sensor. The room
    is open above them, so the upper dome legitimately escapes to the full range -- checked here too,
    because clipping everything to the walls would be the opposite bug.
    """
    import mujoco

    far = 30.0  # >> the 10 m room, so an unclipped sector is unmistakable
    engine = Engine(_mid360_world(show_fov=True, fov_range=far))
    engine.setup()
    engine.reset()
    m, d = engine.ctx.model, engine.ctx.data

    gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "mid360_fov")
    assert gid >= 0
    mesh = m.geom_dataid[gid]
    adr, num = m.mesh_vertadr[mesh], m.mesh_vertnum[mesh]
    # World-space vertices: MuJoCo recentres a mesh's local verts, and geom_xpos carries that offset,
    # so the two must be combined -- the local coordinates alone are not radii from the sensor.
    verts = d.geom_xpos[gid] + m.mesh_vert[adr : adr + num] @ d.geom_xmat[gid].reshape(3, 3).T

    wall_top, wall_inner = 1.9, 5.05
    enclosed = verts[verts[:, 2] < wall_top]
    assert len(enclosed) > 0
    assert np.abs(enclosed[:, :2]).max() <= wall_inner, "the sector passes through the walls"
    assert np.abs(verts[:, :2]).max() > 2 * wall_inner, (
        "the sector should still escape over the walls"
    )


# --- a measured lens, per placement -------------------------------------------------------------
#
# The rig these came from: three D435s, one calibration each. They are kept as literals rather than
# derived from a fovy because that is the point -- fx != fy and the principal point is off centre,
# neither of which one angle can say.
_CAM1 = {"fx": 1330.2327487213051, "fy": 1329.374480088922,
         "cx": 974.2501995447395, "cy": 538.9934564521541, "width": 1920, "height": 1080}
_CAM4 = {"fx": 1412.8497845, "fy": 1415.3488644597485,
         "cx": 964.1221796, "cy": 531.7413608, "width": 1920, "height": 1080}


class _LitBox(Plugin):
    """A red box straight ahead of the origin, so a render has something whose position can move."""

    def build(self, spec: mujoco.MjSpec, ctx: SimContext) -> None:
        spec.worldbody.add_geom(
            type=mujoco.mjtGeom.mjGEOM_PLANE, size=[5, 5, 0.1], rgba=[0.3, 0.3, 0.3, 1]
        )
        spec.worldbody.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX, pos=[2.0, 0, 0.5], size=[0.2, 0.2, 0.2],
            rgba=[0.9, 0.05, 0.05, 1],
        )


def _lens_world(intrinsics, *, width, height, prefix="cam_", name="cam", scene=False, **spawn):
    entries = [{f"{__name__}:_LitBox": {}}] if scene else []
    entries.append({
        "spawn_sensor": {"model": "d435", "prefix": prefix, **spawn,
                         **({"intrinsics": intrinsics} if intrinsics else {})},
        "name": name,
        "components": [{"realsense_d435": {"width": width, "height": height}}],
    })
    return load_config_from_dict({"sim": {}, "components": entries})


def test_a_placement_publishes_the_lens_it_was_given():
    """The four measured numbers survive the round trip through the MJCF and back out again."""
    import mujoco

    from roqsim_sensors.plugins.camera_common import intrinsics_from_model

    engine = Engine(_lens_world(_CAM1, width=1920, height=1080))
    engine.setup()
    engine.reset()
    m = engine.ctx.model
    cid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "cam_d435_color")
    intr = intrinsics_from_model(m, cid, width=1920, height=1080)
    # float32 in mjModel: 2e-4 px, i.e. four orders below the 14 px offset being asserted.
    assert intr.fx == pytest.approx(_CAM1["fx"], abs=1e-3)
    assert intr.fy == pytest.approx(_CAM1["fy"], abs=1e-3)
    assert intr.cx == pytest.approx(_CAM1["cx"], abs=1e-3)
    assert intr.cy == pytest.approx(_CAM1["cy"], abs=1e-3)
    assert intr.fx != intr.fy, "a real lens does not have square-pixel-perfect focal lengths"


def test_the_lens_scales_to_whatever_resolution_the_plugin_renders():
    """The calibration is a measurement at ITS resolution; rendering smaller must not move the lens."""
    import mujoco

    from roqsim_sensors.plugins.camera_common import intrinsics_from_model

    engine = Engine(_lens_world(_CAM1, width=960, height=540))
    engine.setup()
    engine.reset()
    m = engine.ctx.model
    cid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "cam_d435_color")
    intr = intrinsics_from_model(m, cid, width=960, height=540)
    assert intr.fx == pytest.approx(_CAM1["fx"] / 2, abs=1e-3)
    assert intr.cx == pytest.approx(_CAM1["cx"] / 2, abs=1e-3)
    assert intr.cy == pytest.approx(_CAM1["cy"] / 2, abs=1e-3)


def test_two_mounts_of_one_model_carry_two_different_lenses():
    """The reason this lives on the placement: one d435.xml, two units, two calibrations."""
    import mujoco

    from roqsim_sensors.plugins.camera_common import intrinsics_from_model

    cfg = load_config_from_dict({
        "sim": {},
        "components": [
            {"spawn_sensor": {"model": "d435", "prefix": "one_", "intrinsics": _CAM1},
             "name": "one",
             "components": [{"realsense_d435": {"width": 1920, "height": 1080}}]},
            {"spawn_sensor": {"model": "d435", "prefix": "two_", "pos": [1, 0, 0],
                              "intrinsics": _CAM4},
             "name": "two",
             "components": [{"realsense_d435": {"width": 1920, "height": 1080}}]},
        ],
    })
    engine = Engine(cfg)
    engine.setup()
    engine.reset()
    m = engine.ctx.model
    lenses = {}
    for prefix, want in (("one_", _CAM1), ("two_", _CAM4)):
        cid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, f"{prefix}d435_color")
        lenses[prefix] = intrinsics_from_model(m, cid, width=1920, height=1080)
        assert lenses[prefix].fx == pytest.approx(want["fx"], abs=1e-3)
        assert lenses[prefix].cy == pytest.approx(want["cy"], abs=1e-3)
    assert lenses["one_"].fx != lenses["two_"].fx


def test_the_principal_point_moves_the_pixels_and_not_only_the_numbers():
    """The claim the whole feature rests on: MuJoCo RENDERS the stated lens.

    A pure ``camera_info`` change would leave the image identical and tell every consumer the
    principal point is somewhere it is not -- which is the failure this replaces, so it is the one
    thing a test must not take on trust. Shifting the principal point right by N px must move the
    image content left by N px, which a column correlation reads off directly.
    """
    import numpy as np

    width, height = 640, 480
    fovy = 42.5  # the d435 model's own, so ONLY the principal point differs between the two renders
    f = height / (2 * np.tan(np.radians(fovy) / 2))
    offset = 40.0

    def column_profile(cx):
        lens = {"fx": f, "fy": f, "cx": cx, "cy": height / 2, "width": width, "height": height}
        # The yaw is what puts the red box in frame, and the measurement is only the box's: with the
        # camera facing anywhere else the profile is the room, whose shading carries no feature to
        # correlate, and the lag it reports means nothing.
        engine = Engine(_lens_world(lens, width=width, height=height, scene=True,
                                    pos=[0, 0, 0.5], rpy=[0, 0, -1.5707963]))
        engine.setup()
        engine.reset()
        engine.step()
        rgb = _endpoint(engine, "image").read().astype(np.float64)
        return rgb[..., 0].mean(axis=0) - rgb[..., 1:].mean(axis=(0, 2))  # redness per column

    centred = column_profile(width / 2)
    shifted = column_profile(width / 2 + offset)
    assert np.argmax(centred) != np.argmax(shifted), "the render ignored the principal point"
    lag = np.argmax(np.correlate(shifted - shifted.mean(), centred - centred.mean(), "full"))
    assert lag - (len(centred) - 1) == pytest.approx(-offset, abs=2.0)


def test_the_drawn_frustum_follows_the_measured_lens():
    """A cone drawn from the model's nominal fovy over a render that used another lens is a lie."""
    import mujoco

    def half_width(fy):
        lens = {"fx": fy, "fy": fy, "cx": 320, "cy": 240, "width": 640, "height": 480}
        engine = Engine(_lens_world(lens, width=640, height=480, show_fov=True, fov_range=2.0))
        engine.setup()
        engine.reset()
        m, d = engine.ctx.model, engine.ctx.data
        gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "cam_d435_color_fov")
        assert gid >= 0
        mesh = m.geom_dataid[gid]
        adr, num = m.mesh_vertadr[mesh], m.mesh_vertnum[mesh]
        verts = m.mesh_vert[adr : adr + num] @ d.geom_xmat[gid].reshape(3, 3).T
        return np.abs(verts[:, 2]).max()

    narrow, wide = half_width(1200.0), half_width(400.0)
    assert wide > narrow * 1.5, "the frustum ignored the stated focal length"


@pytest.mark.parametrize(
    "block, expected",
    [
        ({"fx": 1330.0, "fy": 1329.0, "width": 1920, "height": 1080}, "missing"),
        ({"fx": 1330.0, "fy": 1329.0, "cx": 974.0, "cy": 539.0, "width": 1920, "height": 1080,
          "fovy": 42.5}, "unknown key"),
        ({"fx": 0, "fy": 1329.0, "cx": 974.0, "cy": 539.0, "width": 1920, "height": 1080},
         "must be > 0"),
        ({"fx": 1330.0, "fy": 1329.0, "cx": 974.0, "cy": 539.0, "width": 1, "height": 1080},
         "pixel count"),
    ],
)
def test_a_half_stated_lens_is_refused(block, expected):
    """Partial or misspelled is refused, never quietly completed from the model's own defaults."""
    from roqsim_sensors.plugins.spawn_sensor import SpawnSensorPlugin

    errors = SpawnSensorPlugin(None).validate_config({"model": "d435", "intrinsics": block})
    assert any(expected in e for e in errors), errors


def test_a_lens_without_its_resolution_is_refused():
    """``fx`` in pixels is meaningless without the frame it was measured in, and there are two
    plausible wrong answers to hand (the model's own resolution, and the render size), so the block
    has to carry its own rather than inherit either."""
    from roqsim_sensors.plugins.spawn_sensor import SpawnSensorPlugin

    lens = {k: v for k, v in _CAM1.items() if k not in ("width", "height")}
    errors = SpawnSensorPlugin(None).validate_config({"model": "d435", "intrinsics": lens})
    assert any("width" in e and "height" in e for e in errors), errors
