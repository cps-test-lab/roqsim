# Copyright (C) 2026 Frederik Pasch
#
# SPDX-License-Identifier: Apache-2.0

"""``roqsim sensors coverage swept``: what a sensor carried through a recorded run ever covered.

One short run is recorded once for the module: a walled corridor whose own MJCF holds a gantry on a
slide joint carrying a camera down the corridor, a panel on a second slide joint that closes across
the corridor early in the run, a fixed camera on the world body, and a pillar in front of it. Both
slides are driven by a constant-force actuator bias, so the motion is in the model and needs no
controller. No entity owns any of it, so every mount is named by its MuJoCo name.

The properties asserted are the ones a broken sweep cannot satisfy: a sensor that moves covers more
than one that does not; a cell behind the panel is seen before the panel arrives and never after,
which only a per-sample restored world can say; the world's walls occlude a camera on the world body;
and every refused input is refused by name with exit status 2.
"""

from __future__ import annotations

import json
import re

import mujoco
import numpy as np
import pytest
import yaml
from roqsim_sensors.coverage import cli, sampling
from roqsim_sensors.coverage.swept import SweptError, swept_coverage

from roqsim import exit_status
from roqsim.recording import open_recording

pytest.importorskip("mcap", reason="a recording is an mcap file")

_WORLD = """
<mujoco>
  <option timestep="0.01"/>
  <worldbody>
    <light pos="0 0 4"/>
    <geom name="floor" type="plane" size="20 20 .1"/>
    <geom name="wall_w" type="box" pos="-1 0 1" size=".1 1.6 1"/>
    <geom name="wall_e" type="box" pos="11 0 1" size=".1 1.6 1"/>
    <geom name="wall_s" type="box" pos="5 -1.5 1" size="6.1 .1 1"/>
    <geom name="wall_n" type="box" pos="5 1.5 1" size="6.1 .1 1"/>
    <!-- In front of the fixed camera: what is behind it must stay unseen. -->
    <geom name="pillar" type="box" pos="1.2 -.8 1" size=".1 .2 1"/>
    <!-- Looks along +x with world-up as up: MuJoCo's camera looks along its -z. -->
    <camera name="fixed_cam" pos="0 -.8 .5" xyaxes="0 -1 0  0 0 1" fovy="70"/>
    <body name="gantry" pos="0 0 .5">
      <joint name="gantry_x" type="slide" axis="1 0 0" range="0 8" limited="true"/>
      <geom name="carriage" type="box" size=".15 .15 .15" mass="1" contype="0" conaffinity="0"/>
      <camera name="gantry_cam" pos="0 0 0" xyaxes="0 -1 0  0 0 1" fovy="70"/>
    </body>
    <!-- Starts outside the north wall and slides across the corridor at x = 4. -->
    <body name="panel" pos="4 3.2 1">
      <joint name="panel_y" type="slide" axis="0 -1 0" range="0 3.2" limited="true"/>
      <geom name="panel_geom" type="box" size=".05 1.5 1" mass="1" contype="0" conaffinity="0"/>
    </body>
  </worldbody>
  <actuator>
    <!-- force = b0 + b2 * qvel: a constant push with damping, so each slide settles at b0/-b2. -->
    <general name="gantry_drive" joint="gantry_x" gainprm="0" biastype="affine" biasprm="4 0 -2"/>
    <general name="panel_drive" joint="panel_y" gainprm="0" biastype="affine" biasprm="8 0 -2"/>
  </actuator>
</mujoco>
"""

SECONDS = 5.0
FPS = 25
#: Samples in the recorded run: one per 1/FPS of sim time, the first after the first step.
N_SAMPLES = int(SECONDS * FPS)
SAMPLE = ["--sample", "volume", "--resolution", "0.25", "--heights", "0.5"]


@pytest.fixture(scope="module")
def recording(tmp_path_factory):
    from roqsim.runner import run

    root = tmp_path_factory.mktemp("swept")
    (root / "gantry.xml").write_text(_WORLD)
    world = root / "gantry.yaml"
    world.write_text(yaml.safe_dump({"sim": {"world": "gantry.xml"}}))
    out = root / "run.mcap"
    run(str(world), headless=True, pacing="asap", seconds=SECONDS, record=str(out), capture_fps=FPS)
    return out


def _sweep(recording, frame="gantry_cam", sensor_type="oakd_camera", **kwargs):
    kwargs.setdefault("config", {"far": 4.0})

    def sample(model, data):
        return sampling.sample_set(
            model, data, volume=True, objects=False, resolution=0.25, heights=(0.5,)
        )

    with open_recording(recording) as rec:
        return swept_coverage(rec, frame=frame, sensor_type=sensor_type, sample=sample, **kwargs)


def _in(points, x=(-np.inf, np.inf), y=(-np.inf, np.inf)):
    px, py = points[:, 0], points[:, 1]
    return (px > x[0]) & (px < x[1]) & (py > y[0]) & (py < y[1])


def _slide(recording, joint: str) -> tuple[np.ndarray, np.ndarray]:
    """``(times, qpos)`` of one slide joint, read back from the recording itself."""
    with open_recording(recording) as rec:
        model, _ = rec.build()
        adr = int(model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint)])
        return np.asarray(rec.times), np.asarray(rec.qpos[:, adr])


# -- the fixture moves as the tests assume ---------------------------------------------------------


def test_the_recorded_run_moves_the_gantry_and_closes_the_panel(recording):
    times, gantry = _slide(recording, "gantry_x")
    assert len(times) == N_SAMPLES
    assert gantry[0] == pytest.approx(0.0, abs=1e-2)
    assert gantry[-1] > 7.5, "the gantry travels the corridor"
    times, panel = _slide(recording, "panel_y")
    assert panel[times <= 0.4].max() < 1.6, "the panel is still outside the corridor early on"
    assert panel[times >= 2.0].min() > 3.15, "and has closed across it by t = 2 s"


# -- what the union says -----------------------------------------------------------------------------


def test_a_moving_sensor_covers_more_than_a_still_one(recording):
    """The same camera optics, one on the gantry and one fixed near its start, over the same
    samples: only motion separates them."""
    moving = _sweep(recording, "gantry_cam")
    still = _sweep(recording, "fixed_cam")
    assert len(moving.times) == len(still.times) == N_SAMPLES
    assert still.covered.sum() > 0
    assert moving.covered.sum() >= 2 * still.covered.sum()
    # The far end of the corridor is reached only by the camera that went there.
    far = _in(moving.points, x=(8.5, 11.0))
    assert moving.covered[far].any() and not still.covered[far].any()


def test_an_occluder_blocks_only_the_samples_it_stands_in(recording):
    """Behind the panel's line is seen while the panel is outside the corridor and never once it
    has closed: the occluder is where the recorded state put it at each sample."""
    before = _sweep(recording, "fixed_cam", stop=0.4, config={"far": 8.0})
    after = _sweep(recording, "fixed_cam", start=2.0, config={"far": 8.0})
    # Each sweep builds its sample set at its own first sample, with the panel where it then was.
    behind_before = _in(before.points, x=(4.2, 5.0), y=(-1.0, 1.0))
    behind_after = _in(after.points, x=(4.2, 5.0), y=(-1.0, 1.0))
    assert behind_before.any() and behind_after.any()
    assert before.covered[behind_before].any(), "seen before the panel closes"
    assert not after.covered[behind_after].any(), "and never after"
    ahead = _in(after.points, x=(2.5, 3.8), y=(0.0, 1.0))
    assert after.covered[ahead].any(), "the panel hides only what is behind it"


def test_a_world_body_camera_is_occluded_by_the_walls(recording):
    """A camera on the world body excludes no body from its raycasts, so the pillar in front of it
    -- world geometry, like every wall -- still hides what is behind it."""
    swept = _sweep(recording, "fixed_cam", stop=0.4)
    assert swept.fov.body_exclude == -1
    shadow = _in(swept.points, x=(1.5, 2.5), y=(-1.0, -0.6))
    beside = _in(swept.points, x=(1.5, 2.5), y=(0.3, 0.9))
    assert shadow.any() and beside.any()
    assert not swept.covered[shadow].any()
    assert swept.covered[beside].any()


def test_a_camera_frame_brings_its_intrinsics_and_its_axes(recording):
    """On a camera frame the field of view is that camera's -- its fovy, not the catalog's -- and
    ``pose`` is read in its axes: ``z: -0.3`` is 0.3 m along the view, world +x here."""
    from roqsim_sensors.coverage.catalog import CATALOG
    from roqsim_sensors.plugins.camera_common import intrinsics_from_model

    assert CATALOG["oakd_camera"].fov_template["fovy"] != pytest.approx(70.0)
    swept = _sweep(recording, "gantry_cam", pose={"position": {"z": -0.3}})
    with open_recording(recording) as rec:
        model, _ = rec.build()
        cam = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "gantry_cam")
        data = rec.at(float(swept.times[-1])).data
        assert swept.fov.intrinsics == intrinsics_from_model(model, cam)
        assert swept.fov.rot == pytest.approx(data.cam_xmat[cam].reshape(3, 3), abs=1e-6)
        assert swept.fov.origin == pytest.approx(data.cam_xpos[cam] + [0.3, 0, 0], abs=1e-5)
        gantry = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "gantry")
    assert swept.fov.body_exclude == gantry
    overridden = _sweep(recording, "gantry_cam", config={"far": 4.0, "fovy": 30.0}, stop=0.01)
    assert overridden.fov.intrinsics.fy > swept.fov.intrinsics.fy, "config overrides the camera"


def test_a_body_frame_takes_the_catalog_field_of_view_along_its_x(recording):
    swept = _sweep(recording, "gantry", pose={"position": {"x": 0.2}}, stop=0.01)
    _, gantry = _slide(recording, "gantry_x")
    assert swept.fov.origin == pytest.approx([gantry[0] + 0.2, 0.0, 0.5], abs=1e-5)
    assert swept.fov.rot @ np.array([0.0, 0.0, -1.0]) == pytest.approx([1.0, 0.0, 0.0], abs=1e-6)


def test_a_rate_thins_the_samples_and_a_window_bounds_them(recording):
    thinned = _sweep(recording, "gantry_cam", rate=5.0)
    assert len(thinned.times) == N_SAMPLES // 5
    assert np.diff(thinned.times) == pytest.approx(0.2)
    window = _sweep(recording, "gantry_cam", start=1.0, stop=2.0)
    assert 1.0 <= window.times[0] < 1.0 + 1 / FPS
    assert 2.0 - 1 / FPS < window.times[-1] <= 2.0
    assert np.diff(window.times) == pytest.approx(1 / FPS)


def test_an_unknown_frame_is_a_swept_error(recording):
    with pytest.raises(SweptError, match="Did you mean 'gantry'"):
        _sweep(recording, "gantri")


# -- the command -------------------------------------------------------------------------------------


def _cli(recording, tmp_path, *extra):
    return cli.main(
        [
            "swept",
            "--recording",
            str(recording),
            "--frame",
            "gantry_cam",
            "--type",
            "oakd_camera",
            "--config",
            '{"far": 4.0}',
            "--out",
            str(tmp_path / "out"),
            "--render",
            "none",
            *SAMPLE,
            *extra,
        ]
    )


def test_the_command_writes_the_static_report_with_a_swept_block(recording, tmp_path, capsys):
    assert _cli(recording, tmp_path, "--target", "k=1,frac=0.5") == 0
    assert "SWEPT_OK" in capsys.readouterr().out
    report = json.loads((tmp_path / "out" / "report.json").read_text())
    swept = report["swept"]
    assert swept["frame"] == "gantry_cam"
    assert swept["n_evaluations"] == N_SAMPLES
    assert report["achieved"]["fraction_covered_k1"] == pytest.approx(swept["fraction"])
    assert sum(swept["visits_histogram"]) == swept["n_points"] == report["achieved"]["n_points"]
    assert 0 < swept["covered_area_m2"] <= swept["sampled_area_m2"]
    assert swept["cell_area_m2"] == pytest.approx(0.0625)
    assert report["uncovered_regions"], "the sweep leaves gaps, and the report clusters them"
    assert report["placements"] == [
        {"frame": "gantry_cam", "type": "oakd_camera", "config": {"far": 4.0}, "pose": {}}
    ]


def test_the_command_samples_exactly_what_sample_set_samples(recording, tmp_path):
    """The sweep's point set is ``sample_set``'s, taken at the first evaluated sample."""
    assert _cli(recording, tmp_path, "--from", "1.0") == 0
    report = json.loads((tmp_path / "out" / "report.json").read_text())
    with open_recording(recording) as rec:
        model, _ = rec.build()
        first = float(rec.times[rec.times >= 1.0][0])
        data = rec.at(first).data
        points, _, _ = sampling.sample_set(
            model, data, volume=True, objects=False, resolution=0.25, heights=(0.5,)
        )
    assert report["swept"]["n_points"] == len(points)
    assert report["swept"]["from"] == pytest.approx(first)


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (["--frame", "gantry_camm"], "--frame 'gantry_camm': no frame .*Did you mean 'gantry_cam'"),
        (["--from", "100"], "from 100 s to the end holds no sample: the recording spans 0.010"),
        (["--heights", "40"], "the sample set is empty"),
        (["--rate", "1000"], "above the recording's 25 samples per second"),
        (["--type", "no_such_sensor"], "unknown sensor type 'no_such_sensor'"),
        (["--pose", '{"pos": [1, 0, 0]}'], "--pose: .*has no key"),
        (["--config", "far=4"], "--config is not JSON"),
    ],
)
def test_a_wrong_input_is_refused_on_one_line(recording, tmp_path, capsys, extra, message):
    assert _cli(recording, tmp_path, *extra) == exit_status.BAD_INPUT
    err = capsys.readouterr().err
    assert err.startswith("roqsim sensors coverage: ")
    assert len(err.strip().splitlines()) == 1
    assert re.search(message, err), err


@pytest.mark.parametrize(
    ("content", "message"), [(None, "no such recording"), (b"hello", "not an mcap file")]
)
def test_a_recording_that_cannot_be_read_is_refused(tmp_path, capsys, content, message):
    path = tmp_path / "run.mcap"
    if content is not None:
        path.write_bytes(content)
    assert _cli(path, tmp_path) == exit_status.BAD_INPUT
    assert message in capsys.readouterr().err
