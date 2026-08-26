"""Tests for the sensor-coverage subpackage: FOV membership, adapters, and the coverage engine."""

from __future__ import annotations

import json
import math

import mujoco
import numpy as np
import pytest
import yaml
from external_meshes import needs_mid360, needs_robin, needs_zivid
from roqsim_sensors.coverage import adapters, optimize
from roqsim_sensors.coverage.adapters import PlacedSensor, build_fov
from roqsim_sensors.coverage.catalog import CATALOG, catalog_as_dict
from roqsim_sensors.coverage.engine import coverage
from roqsim_sensors.coverage.fov import FovKind, SensorFov, in_fov
from roqsim_sensors.models import MODELS_DIR

from roqsim.manifest import manifest_fov
from roqsim.models import resolve_model
from roqsim.registry import resolve_plugin

MODEL_NAMES = sorted(
    p.name for p in MODELS_DIR.iterdir() if p.is_dir() and (p / f"{p.name}.xml").is_file()
)

# -- FOV membership ----------------------------------------------------------------------------------


def test_cone_band_elevation_and_wrap():
    fov = SensorFov(
        kind=FovKind.CONE_BAND,
        origin=[0, 0, 0],
        rot=np.eye(3),
        h_fov=(0.0, 2 * math.pi),  # full 360 -> azimuth always in range
        v_fov=(math.radians(-7), math.radians(52)),
        range_min=0.1,
        range_max=40.0,
        range_is_physical=True,
        sensor_type="livox_mid360",
    )
    pts = np.array(
        [
            [5, 0, 0.0],  # in plane, el=0 -> in band
            [0, 5, 0.0],  # azimuth 90deg -> full 360 covers it
            [0, 0, 5.0],  # straight up, el=90 -> above band
            [5, 0, -5.0],  # el=-45 -> below band
        ]
    )
    assert in_fov(fov, pts).tolist() == [True, True, False, False]


def test_narrow_azimuth_fan_excludes_behind():
    # A +/-45deg horizontal fan around +x.
    fov = SensorFov(
        kind=FovKind.CONE_BAND,
        origin=[0, 0, 0],
        rot=np.eye(3),
        h_fov=(-math.pi / 4, math.pi / 4),
        v_fov=(-0.01, 0.01),
        range_min=0.0,
        range_max=10.0,
        range_is_physical=True,
        sensor_type="lidar",
    )
    pts = np.array([[5, 0, 0.0], [0, 5, 0.0], [-5, 0, 0.0]])
    assert in_fov(fov, pts).tolist() == [True, False, False]


def test_frustum_projection_front_and_back():
    cam = build_fov(
        None,
        None,
        PlacedSensor(
            "camera",
            pos=[0, 0, 0],
            rpy=[0, 0, 0],
            config={"fovy": 60, "width": 640, "height": 480, "far": 10},
        ),
    )
    # rpy=0 points the optical axis along +x (world).
    assert np.allclose(cam.rot @ np.array([0, 0, -1.0]), [1, 0, 0], atol=1e-9)
    local = cam.to_local(np.array([[5, 0, 0.0], [-5, 0, 0.0], [0, 0, 5.0]]))
    assert in_fov(cam, local).tolist() == [True, False, False]


# -- adapters reuse plugin defaults ------------------------------------------------------------------


def test_lidar_adapter_uses_plugin_defaults():
    fov = build_fov(None, None, PlacedSensor("lidar", pos=[0, 0, 0], rpy=[0, 0, 0], config={}))
    assert fov.kind is FovKind.CONE_BAND
    assert fov.range_is_physical is True
    assert fov.h_fov == (0.0, 2 * math.pi)
    assert fov.range_min == pytest.approx(0.164)
    assert fov.range_max == pytest.approx(20.0)


def test_livox_adapter_uses_plugin_defaults():
    fov = build_fov(
        None, None, PlacedSensor("livox_mid360", pos=[0, 0, 0], rpy=[0, 0, 0], config={})
    )
    assert math.degrees(fov.v_fov[0]) == pytest.approx(-7.0, abs=1e-3)
    assert math.degrees(fov.v_fov[1]) == pytest.approx(52.0, abs=1e-3)
    assert fov.range_max == pytest.approx(40.0)


def test_camera_far_is_not_physical():
    fov = build_fov(
        None, None, PlacedSensor("oakd_camera", pos=[0, 0, 0], rpy=[0, 0, 0], config={})
    )
    assert fov.range_is_physical is False


def test_unknown_sensor_type_raises():
    with pytest.raises(KeyError):
        build_fov(None, None, PlacedSensor("mystery_sensor", pos=[0, 0, 0], rpy=[0, 0, 0]))


def test_registered_types_cover_bundled_sensors():
    types = set(adapters.registered_types())
    assert {"oakd_camera", "realsense_d435", "realsense_d415", "lidar", "livox_mid360"} <= types


# -- coverage engine: line of sight + group-4 mask ---------------------------------------------------

_OCCLUSION_XML = """
<mujoco>
  <worldbody>
    <geom name="floor" type="plane" size="10 10 0.1" group="0"/>
    <geom name="wall" type="box" pos="1 0 0.5" size="0.05 1 1" group="0"/>
    <geom name="fov_env" type="box" pos="0.5 0 0.5" size="0.1 0.4 0.4"
          group="4" contype="0" conaffinity="0"/>
  </worldbody>
</mujoco>
"""


def _occlusion_world():
    m = mujoco.MjModel.from_xml_string(_OCCLUSION_XML)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    cam = build_fov(
        m,
        d,
        PlacedSensor(
            "camera",
            pos=[0, 0, 0.5],
            rpy=[0, 0, 0],
            config={"fovy": 90, "width": 640, "height": 480, "far": 10, "near": 0.05},
        ),
    )
    return m, d, cam


def test_wall_occludes_but_group4_does_not():
    m, d, cam = _occlusion_world()
    pts = np.array([[0.9, 0, 0.5], [2.0, 0, 0.5]])  # before wall (grp4 in the way) / behind wall
    res = coverage(m, d, [cam], pts)
    assert res.counts.tolist() == [1, 0]


def test_including_group4_causes_occlusion():
    m, d, cam = _occlusion_world()
    pts = np.array([[0.9, 0, 0.5], [2.0, 0, 0.5]])
    res = coverage(m, d, [cam], pts, include_groups=(0, 1, 2, 3, 4))
    assert res.counts.tolist() == [0, 0]  # the group-4 box now blocks the near point too


def test_coverage_fraction_and_by_type():
    m, d, cam = _occlusion_world()
    pts = np.array([[0.9, 0, 0.5], [2.0, 0, 0.5]])
    res = coverage(m, d, [cam], pts)
    assert res.coverage_fraction(1) == pytest.approx(0.5)
    by_type = res.counts_by_type()
    assert "camera" in by_type
    assert by_type["camera"].tolist() == [True, False]


# -- sampling, report, and greedy on a small enclosed room -------------------------------------------

_ROOM_XML = """
<mujoco>
  <worldbody>
    <geom name="floor" type="plane" size="10 10 0.1" group="0"/>
    <geom name="wall_n" type="box" pos="0 3 1" size="3 0.1 1" group="0"/>
    <geom name="wall_s" type="box" pos="0 -3 1" size="3 0.1 1" group="0"/>
    <geom name="wall_e" type="box" pos="3 0 1" size="0.1 3 1" group="0"/>
    <geom name="wall_w" type="box" pos="-3 0 1" size="0.1 3 1" group="0"/>
    <geom name="table" type="box" pos="0 0 0.4" size="0.5 0.5 0.4" group="0"/>
  </worldbody>
</mujoco>
"""


def _room():
    m = mujoco.MjModel.from_xml_string(_ROOM_XML)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    return m, d


def test_room_volume_points_are_interior():
    from roqsim_sensors.coverage import sampling

    m, d = _room()
    pts = sampling.room_volume_points(m, d, resolution=0.5, heights=(1.0,))
    assert len(pts) > 0
    # every point is inside the +/-3 walls
    assert np.all(np.abs(pts[:, 0]) < 3.0) and np.all(np.abs(pts[:, 1]) < 3.0)


def test_object_surface_points_label_the_table_not_walls():
    from roqsim_sensors.coverage import sampling

    m, d = _room()
    pts, labels, names = sampling.object_surface_points(m, d, per_object=32)
    assert "table" in names
    assert not any("wall" in n or "floor" in n for n in names)  # structure excluded by name


def test_build_report_schema_and_gaps():
    from roqsim_sensors.coverage.report import build_report

    m, d = _room()
    cam = build_fov(
        m,
        d,
        PlacedSensor(
            "camera", pos=[0, 0, 2.5], rpy=[0, math.pi / 2, 0], config={"fovy": 90, "far": 6}
        ),
    )
    from roqsim_sensors.coverage import sampling

    pts = sampling.room_volume_points(m, d, resolution=0.5, heights=(1.0,))
    res = coverage(m, d, [cam], pts)
    rep = build_report(
        res, world="room", target={"metric": "fraction_covered", "k": 1, "value": 0.9}
    )
    assert set(rep) >= {"achieved", "target_met", "per_sensor_contribution", "uncovered_regions"}
    assert 0.0 <= rep["achieved"]["fraction_covered_k1"] <= 1.0
    assert rep["per_sensor_contribution"][0]["type"] == "camera"


def test_greedy_baseline_improves_coverage():
    from roqsim_sensors.coverage import sampling
    from roqsim_sensors.coverage.optimize import greedy_baseline

    m, d = _room()
    pts = sampling.room_volume_points(m, d, resolution=0.5, heights=(1.0,))
    candidates = [
        {
            "type": "oakd_camera",
            "pos": [0, 0, 2.5],
            "rpy": [0, math.pi / 2, 0],
            "config": {"fovy": 110, "far": 6},
        },
        {
            "type": "oakd_camera",
            "pos": [1.5, 1.5, 2.5],
            "rpy": [0, math.pi / 2, 0],
            "config": {"fovy": 110, "far": 6},
        },
        {
            "type": "oakd_camera",
            "pos": [-1.5, -1.5, 2.5],
            "rpy": [0, math.pi / 2, 0],
            "config": {"fovy": 110, "far": 6},
        },
    ]
    chosen, result = greedy_baseline(m, d, candidates, pts, target_frac=0.6, max_sensors=3, k=1)
    assert 1 <= len(chosen) <= 3
    assert result.coverage_fraction(1) > 0.0


# -- regions (per-area coverage breakdown + restriction) ---------------------------------------------


def test_region_contains_polygon_and_zband():
    from roqsim_sensors.coverage.regions import Region

    r = Region("box", polygon=[[0, 0], [2, 0], [2, 2], [0, 2]], z_min=0.5, z_max=1.5)
    pts = np.array([[1, 1, 1.0], [1, 1, 3.0], [3, 1, 1.0], [1, 1, 0.4]])
    assert list(r.contains(pts)) == [
        True,
        False,
        False,
        False,
    ]  # inside xy+z, above band, outside xy, below band


def test_load_regions_from_bbox_and_polygon():
    from roqsim_sensors.coverage.regions import load_regions

    regs = load_regions(
        {
            "regions": [
                {"name": "a", "bbox": [0, 0, 2, 2]},
                {"name": "b", "polygon": [[3, 3], [5, 3], [4, 5]]},
            ]
        }
    )
    assert [r.name for r in regs] == ["a", "b"]
    assert regs[0].contains(np.array([[1, 1, 0.0]]))[0]
    assert not regs[0].contains(np.array([[4, 4, 0.0]]))[0]


def test_regions_from_sketch_reconstructs_rooms():
    # A minimal sketch: one square room from four wall segments (unordered) -> a closed 4-gon.
    from roqsim_sensors.coverage.regions import regions_from_sketch

    sketch = {
        "rooms": [{"id": "r0", "name": "hall", "line_ids": [1, 2, 3, 4]}],
        "lines": [
            {"id": 1, "x0_m": 0, "y0_m": 0, "x1_m": 4, "y1_m": 0},
            {"id": 3, "x0_m": 4, "y0_m": 4, "x1_m": 0, "y1_m": 4},
            {"id": 2, "x0_m": 4, "y0_m": 0, "x1_m": 4, "y1_m": 4},
            {"id": 4, "x0_m": 0, "y0_m": 4, "x1_m": 0, "y1_m": 0},
        ],
    }
    regs = regions_from_sketch(sketch)
    assert len(regs) == 1 and regs[0].name == "hall"
    assert regs[0].contains(np.array([[2, 2, 0.0]]))[0]  # centre inside
    assert not regs[0].contains(np.array([[5, 2, 0.0]]))[0]  # outside


def test_select_unknown_region_raises():
    from roqsim_sensors.coverage.regions import load_regions, select

    regs = load_regions([{"name": "a", "bbox": [0, 0, 1, 1]}])
    with pytest.raises(ValueError, match="not found"):
        select(regs, "nope")


def test_footprint_mask_ignores_zband():
    # A mount at ceiling height is "above" a floor-banded region -> union_mask (z) drops it, but
    # footprint_mask (xy only) keeps it, so candidate mounts over a room are not filtered out.
    from roqsim_sensors.coverage.regions import Region, footprint_mask, union_mask

    r = Region("floor", polygon=[[0, 0], [4, 0], [4, 4], [0, 4]], z_min=0.0, z_max=1.8)
    mount = np.array([[2, 2, 3.3]])
    assert not union_mask(mount, [r])[0]
    assert footprint_mask(mount, [r])[0]


def test_per_region_coverage_matches_manual():
    from roqsim_sensors.coverage.regions import Region, per_region_coverage

    pts = np.array([[1, 1, 1.0], [1, 1, 1.0], [3, 3, 1.0]])
    counts = np.array([2, 0, 1])
    a = Region("a", polygon=[[0, 0], [2, 0], [2, 2], [0, 2]])
    rows = per_region_coverage(pts, counts, [a])
    assert rows[0]["n_points"] == 2  # two points inside region a
    assert rows[0]["fraction_covered_k1"] == 0.5  # one of two seen by >=1
    assert rows[0]["fraction_covered_k2"] == 0.5


# -- density palette (visualisation encoding) --------------------------------------------------------


def _luminance(rgb: np.ndarray) -> float:
    return float(0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2])


def test_density_ramp_gets_darker_with_more_sensors():
    # The density palette must encode "more overlapping sensors = darker": luminance strictly decreases
    # as the coverage count rises. This is the property the user asked to see.
    from roqsim_sensors.coverage.viz import _DENSITY_RAMP

    lums = [_luminance(row) for row in _DENSITY_RAMP]
    assert all(a > b for a, b in zip(lums, lums[1:], strict=False)), lums


def test_unknown_palette_raises():
    from roqsim_sensors.coverage.viz import _resolve_palette

    with pytest.raises(ValueError, match="unknown palette"):
        _resolve_palette("rainbow")


def _synthetic_result():
    from roqsim_sensors.coverage.engine import CoverageResult

    pts = np.array([[0.0, 0.0, 0.5], [1.0, 0.0, 0.5], [0.0, 1.0, 0.5], [1.0, 1.0, 0.5]])
    counts = np.array([0, 1, 2, 4])  # spans the ramp so both palettes exercise every band
    return CoverageResult(points=pts, counts=counts, by_sensor=np.zeros((4, 1), bool), fovs=[])


def test_render_heatmap_both_palettes_write_png(tmp_path):
    from roqsim_sensors.coverage import viz

    result = _synthetic_result()
    viz.render_heatmap_2d(result, str(tmp_path / "cov.png"), palette="coverage")
    viz.render_heatmap_2d(result, str(tmp_path / "dens.png"), palette="density")
    viz.render_heatmap_2d(result, str(tmp_path / "default.png"))  # default palette -> no error
    assert (tmp_path / "cov.png").stat().st_size > 0
    assert (tmp_path / "dens.png").stat().st_size > 0
    assert (tmp_path / "default.png").stat().st_size > 0


def test_render_heatmap_unknown_palette_raises(tmp_path):
    from roqsim_sensors.coverage import viz

    with pytest.raises(ValueError, match="unknown palette"):
        viz.render_heatmap_2d(_synthetic_result(), str(tmp_path / "x.png"), palette="rainbow")


# -- the CLI's two entry-point contracts -------------------------------------------------------------


@needs_mid360
@needs_zivid
@needs_robin
def test_load_world_accepts_a_world_yaml():
    """`--world` must take a world YAML, not only a bare MJCF.

    That branch imported `rst.config` / `rst.engine` / `rst.world` -- the package's name before it was
    renamed to roqsim -- so every world YAML and package ref died with
    `cannot load world ...: No module named 'rst'`, and only a raw .xml worked. The ImportError is
    caught and re-raised as SystemExit, so the dead import read as "this world is unloadable" rather
    than as a broken rename.
    """
    from pathlib import Path

    from roqsim_sensors.coverage.cli import load_world

    # all_sensors_demo mounts the mid360, the zivid and the robin_w1g, so loading it compiles all
    # three vendor meshes -- hence three markers for one assertion.
    world = Path(__file__).parents[1] / "src/roqsim_sensors/worlds/all_sensors_demo.yaml"
    model, data = load_world(str(world))
    assert model.ngeom > 0 and data is not None


def test_coverage_cli_selects_the_gl_backend_before_importing_mujoco():
    """`roqsim-coverage` is its own entry point and never reaches mujoco through roqsim.

    Its submodules `import mujoco` at module scope, and MUJOCO_GL is read once while that runs -- so
    the coverage package has to select the backend itself. When it did not, the CLI bound glfw and
    `--render 3d` had no offscreen renderer: dead on a headless node, and silently off the GPU
    everywhere else. Checked in a subprocess with MUJOCO_GL unset, since the variable binds at first
    import and the test session has already bound it.
    """
    import os
    import subprocess
    import sys

    env = {k: v for k, v in os.environ.items() if k != "MUJOCO_GL"}
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import roqsim_sensors.coverage.cli\n"
            "from roqsim.rendering import bound_gl_backend\n"
            "print(bound_gl_backend())",
        ],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )
    assert proc.stdout.strip().splitlines()[-1] != "glfw"


# -- the sampler's free-space classification -------------------------------------------------------
#
# `_classify_points` used to cast six axis rays per point and classify that point inside the loop.
# It now casts through `raycast.cast_many` and classifies every point at once with numpy. The
# vectorised form (an einsum over (P, 6, 3) normals) is where a transcription error would hide, so
# this pins it against the per-point logic it replaced.

_CLASSIFY_XML = """
<mujoco><worldbody>
  <light pos="0 0 5"/>
  <geom name="floor" type="plane" size="12 12 .1"/>
  <geom name="wall_px" type="box" pos="8 0 1.5" size=".1 8 1.5"/>
  <geom name="wall_nx" type="box" pos="-8 0 1.5" size=".1 8 1.5"/>
  <geom name="wall_py" type="box" pos="0 8 1.5" size="8 .1 1.5"/>
  <geom name="wall_ny" type="box" pos="0 -8 1.5" size="8 .1 1.5"/>
  <geom name="pillar" type="box" pos="2 2 1" size=".6 .6 1"/>
  <geom name="crate" type="box" pos="-3 1 .5" size=".8 .8 .5"/>
</worldbody></mujoco>
"""


def _classify_reference(model, data, points, *, max_dist, geomgroup):
    """The pre-change implementation: one ``mj_multiRay`` per point, classified per point."""
    from roqsim_sensors.coverage.sampling import _AXES6, _HORIZONTAL

    keep = np.zeros(len(points), dtype=bool)
    geomid = np.full(6, -1, dtype=np.int32)
    dist = np.full(6, -1.0, dtype=np.float64)
    normal = np.zeros(6 * 3, dtype=np.float64)
    dirs = np.ascontiguousarray(_AXES6.reshape(-1))
    for i, p in enumerate(points):
        mujoco.mj_multiRay(
            model,
            data,
            np.ascontiguousarray(p),
            dirs,
            geomgroup,
            1,
            -1,
            geomid,
            dist,
            normal,
            6,
            max_dist,
        )
        hit = dist >= 0.0
        if not bool(hit[_HORIZONTAL].all()):
            continue
        backface = hit & (np.einsum("ij,ij->i", normal.reshape(6, 3), _AXES6) > 0.0)
        keep[i] = int(backface.sum()) < 4
    return keep


def test_batched_classification_matches_the_per_point_loop():
    from roqsim_sensors.coverage.sampling import _classify_points, world_bounds

    model = mujoco.MjModel.from_xml_string(_CLASSIFY_XML)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    lo, hi = world_bounds(model, data)
    geomgroup = np.array([1, 1, 1, 1, 0, 1], dtype=np.uint8)
    max_dist = float(np.linalg.norm(hi - lo)) + 1.0
    xs = np.arange(lo[0], hi[0] + 0.5, 0.5)
    ys = np.arange(lo[1], hi[1] + 0.5, 0.5)
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    points = np.column_stack([gx.ravel(), gy.ravel(), np.full(gx.size, 1.0)])

    got = _classify_points(model, data, points, max_dist=max_dist, geomgroup=geomgroup)
    want = _classify_reference(model, data, points, max_dist=max_dist, geomgroup=geomgroup)
    assert np.array_equal(got, want)
    # Guard against the pair agreeing because both classified nothing.
    assert 0 < int(want.sum()) < len(points)


# -- the catalog derives its optics -------------------------------------------------------------------
#
# The catalog used to restate `fovy`/resolution/`near`/`far` that the models already declared, and the
# copies had drifted (its zivid entry claimed 704x704 against the model's 480x480). These pin the
# derivation itself, because a bug there is silent: `_model_optics` returning nothing would fall
# through to `camera_adapter`'s last-resort constants and report coverage for a lens no device has.

_DERIVED = sorted(name for name, spec in CATALOG.items() if spec.model)


@pytest.mark.parametrize("name", _DERIVED)
def test_catalog_optics_come_from_the_model(name):
    """Every derived entry's optics equal what its MJCF and manifest independently say.

    Deliberately NOT marked `needs_zivid`: the derivation parses the MJCF instead of compiling it, so
    it does not need the meshes on disk -- which is what lets the three generated-mesh sensors report
    correct optics on a fresh clone, before `make external-convert` has ever run. If that regressed,
    this test is where it would show up.
    """
    spec = CATALOG[name]
    asset = resolve_model(spec.model)
    cam = {c.name: c for c in mujoco.MjSpec.from_file(str(asset.path)).cameras}
    manifest = yaml.safe_load((asset.path.parent / f"{asset.path.stem}.manifest.yaml").read_text())
    # The manifest names the imaging camera when wiring its capture plugin; that is the one the
    # catalog must read, not merely "some camera in the file".
    wanted = next(
        cfg["camera"]
        for entry in manifest["components"]
        for cfg in [next(iter(entry.values()))]
        if cfg and "camera" in cfg
    )

    got = spec.fov_template
    assert (got["fovy"], got["width"], got["height"]) == (
        float(cam[wanted].fovy),
        *(int(v) for v in cam[wanted].resolution),
    )
    assert (got["near"], got["far"]) == (
        float(manifest["fov"]["near"]),
        float(manifest["fov"]["far"]),
    )


def test_search_cannot_propose_a_type_the_catalog_lacks():
    """`greedy --types X` must not die in `placed_from_proposal`.

    `optimize._DOWN_RPY` is what `generate_candidates` will propose and `--types` is user-facing, so a
    type listed there without a CATALOG entry is a KeyError reachable from the CLI. That is exactly
    what `realsense_d415` was.
    """
    assert set(optimize._DOWN_RPY) <= set(CATALOG)


def test_every_catalog_type_has_an_adapter():
    # The other direction is not asserted: `camera` is a generic adapter alias with no catalog entry,
    # which is correct. This one catches a typo'd `type=` that would otherwise surface as a KeyError
    # deep inside build_fov(), mid-search.
    assert set(CATALOG) <= set(adapters.registered_types())


def test_catalog_as_dict_is_json_serialisable():
    # fov_template is a property doing file IO now, so the CLI's `catalog` command (and the planner
    # reading its output) would break on a non-serialisable leak rather than at import.
    assert (
        json.loads(json.dumps(catalog_as_dict()))["realsense_d435"]["fov_template"]["near"] == 0.28
    )


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_manifest_fov_near_matches_the_capture_plugin(name):
    """A sensor model's `fov.near` is its device's min range, so it must equal the plugin's clip_near.

    d435 stated 0.2 while `realsense_d435` clipped at 0.28, so the drawn cone began 8 cm nearer than
    any depth was returned. Only `near` is pinned: both this manifest and the catalog document `far`
    as an analysis/display range deliberately independent of `clip_far`.
    """
    asset = resolve_model(f"roqsim_sensors:{name}")
    fov = manifest_fov(asset.path)
    if "near" not in fov:
        pytest.skip(f"{name}: no fov: block")
    manifest = yaml.safe_load((asset.path.parent / f"{name}.manifest.yaml").read_text())
    ref, cfg = next(iter(manifest["components"][0].items()))
    # Instantiating is how the lidar adapters read plugin defaults too: __init__ only resolves config
    # into attributes, so this reads the same clip the plugin would apply at run time.
    clip_near = getattr(resolve_plugin(ref)(cfg or {}), "clip_near", None)
    if clip_near is None:
        pytest.skip(f"{name}: {ref} has no depth clip")
    assert fov["near"] == pytest.approx(clip_near)
