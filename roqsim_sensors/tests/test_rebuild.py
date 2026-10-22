"""Rebuilding a camera's and a lidar's streams from a recording (``roqsim.rebuild``).

The claim under test is that a rebuilt frame IS the sensor's frame: the same pixels, depth and scan
the sensor produces at that state with that seed, noise included -- not a look-alike from a generic
renderer -- and that every way of asking for something the recording cannot give is refused.
"""

from __future__ import annotations

import json

import mujoco
import numpy as np
import pytest
import yaml

from roqsim.capture import StateRecorder, snap_fps
from roqsim.check import check_world
from roqsim.config import load_config
from roqsim.engine import Engine
from roqsim.mcap_format import RecordingError
from roqsim.rebuild import META_REBUILD, RebuildError, main, open_streams, write_mcap
from roqsim.types import Image

_SCENE = """
<mujoco>
  <option timestep="0.002"/>
  <worldbody>
    <geom type="plane" size="5 5 .1" rgba=".3 .3 .3 1"/>
    <body name="box" pos="2 0 1">
      <freejoint/>
      <geom type="box" size=".3 .3 .3" rgba=".8 .1 .1 1"/>
    </body>
    <body name="base" pos="0 0 .3">
      <camera name="cam" pos="0 0 .3" xyaxes="0 -1 0 0 0 1" fovy="45" resolution="64 48"/>
      <site name="scan" pos="0 0 .3"/>
    </body>
  </worldbody>
</mujoco>
"""

OAKD = "roqsim_sensors.plugins.oakd_camera:OakDCameraPlugin"
D435 = "roqsim_sensors.plugins.realsense_d435:RealsenseD435Plugin"


def _world(tmp_path, *, moving=True, **camera):
    scene = tmp_path / "s.xml"
    scene.write_text(_SCENE if moving else _SCENE.replace("<freejoint/>", ""))
    doc = {
        "sim": {"world": str(scene)},
        "components": [
            {
                OAKD: {
                    "camera": "cam",
                    "depth_camera": "cam",
                    "rate_hz": 25,
                    "noise": {"gaussian_sigma": 4},
                    **camera,
                },
                "name": "oakd",
            },
            {"lidar": {"site": "scan", "num_rays": 64, "range_stddev": 0.05}, "name": "front"},
        ],
    }
    path = tmp_path / "w.yaml"
    path.write_text(yaml.safe_dump(doc))
    return path


def _record(world, path, *, steps=300, seed=7, live=None):
    """Run *world* with a recorder; with *live*, store each sample's camera and lidar capture.

    The live capture follows an ``mj_forward``: inside ``post_step`` MuJoCo's body poses are still
    the ones from before the step's integration, while the state and the stamp are after it.
    """
    engine = Engine(load_config(str(world)))
    engine.ctx.seed = seed
    engine.setup()
    engine.reset()
    rec = StateRecorder(
        engine.ctx, path, snap_fps(25, 0.002), world=str(world), config=engine.config
    )
    plugins = {p.address: p for p in engine.plugins}
    eps = {(e.producer, e.name): e for e in engine.ctx.interface.all()}
    try:
        for _ in range(steps):
            engine.step()
            if rec.sample(engine.ctx) and live is not None:
                mujoco.mj_forward(engine.ctx.model, engine.ctx.data)
                plugins["oakd"].capture(engine.ctx)
                plugins["front"].capture(engine.ctx)
                t = round(engine.ctx.sim_time, 9)
                live[t] = {
                    "image": eps["oakd", "image"].read().data.copy(),
                    "depth": eps["oakd", "depth"].read().data.copy(),
                    "scan": np.array(eps["front", "scan"].read().ranges, copy=True),
                }
        rec.close()
    finally:
        engine.shutdown()
    return path


@pytest.fixture
def recorded(tmp_path):
    live: dict = {}
    path = _record(_world(tmp_path), tmp_path / "run.mcap", live=live)
    return path, live


def test_a_rebuilt_frame_is_the_sensors_own_at_that_state_and_seed(recorded):
    path, live = recorded
    seen = 0
    with open_streams(path) as streams:
        chosen = streams.select(["oakd/image", "oakd/depth", "front/scan"])
        for frame in streams.frames(chosen, rate=25):
            want = live[round(frame.sim_time, 9)]
            if frame.stream.name == "scan":
                np.testing.assert_array_equal(frame.payload.ranges, want["scan"])
            else:
                np.testing.assert_array_equal(frame.payload.data, want[frame.stream.name])
            seen += 1
    assert seen == 3 * len(live) > 30


def test_on_a_static_scene_it_equals_what_the_live_sensor_published(tmp_path):
    """No mj_forward here: the frames are the ones the camera's own post_step produced live."""
    world = _world(tmp_path, moving=False)
    engine = Engine(load_config(str(world)))
    engine.ctx.seed = 3
    engine.setup()
    engine.reset()
    rec = StateRecorder(
        engine.ctx, tmp_path / "r.mcap", snap_fps(25, 0.002), world=str(world), config=engine.config
    )
    image = next(e for e in engine.ctx.interface.all() if e.name == "image")
    published = {}
    for _ in range(200):
        engine.step()
        rec.sample(engine.ctx)
        frame = image.read()
        published[round(frame.stamp, 9)] = frame.data.copy()
    rec.close()
    engine.shutdown()
    with open_streams(tmp_path / "r.mcap") as streams:
        frames = list(streams.frames(streams.select(["oakd/image"]), rate=25))
    matched = [f for f in frames if round(f.sim_time, 9) in published]
    assert len(matched) == len(frames) > 5
    for f in matched:
        np.testing.assert_array_equal(f.payload.data, published[round(f.sim_time, 9)])


def test_a_moment_is_restored_directly_and_matches_the_range(recorded):
    path, live = recorded
    with open_streams(path) as streams:
        chosen = streams.select(["oakd/image"])
        later = sorted(live)[-3]
        (one,) = streams.frames(chosen, at=[later + 0.001])
        assert one.sim_time == pytest.approx(later)
        np.testing.assert_array_equal(one.payload.data, live[round(one.sim_time, 9)]["image"])


def test_the_default_streams_and_their_names_are_the_bridges(recorded):
    path, _ = recorded
    with open_streams(path) as streams:
        listed = {s.path: s for s in streams.streams}
        default = [s.path for s in streams.select()]
    assert listed["oakd/image"].topic == "/rgbd_camera/image_raw"
    assert listed["oakd/image"].frame_id == "oakd_rgb_camera_optical_frame"
    assert listed["oakd/depth"].type == "sensor_msgs/msg/Image"
    assert listed["front/scan"].type == "sensor_msgs/msg/LaserScan"
    assert listed["oakd/camera_info"].cost is None and listed["oakd/image"].cost == "render"
    # JPEG colour is lossy, so the raw stream is the default and its compressed twin is not.
    assert "oakd/image" in default and "oakd/image_compressed" not in default
    assert {"oakd/camera_info", "oakd/depth", "oakd/depth_camera_info", "front/scan"} <= set(
        default
    )


def test_a_lossless_compressed_depth_replaces_its_raw_twin(tmp_path):
    world = _world(tmp_path, depth_encoding="16UC1", clip_far=10.0)
    path = _record(world, tmp_path / "run.mcap", steps=60)
    with open_streams(path) as streams:
        default = {s.path: s for s in streams.select()}
        assert "oakd/depth_compressed" in default and "oakd/depth" not in default
        frame = next(iter(streams.frames([default["oakd/depth_compressed"]], at=[0.05])))
        assert b"16UC1; compressedDepth png" in frame.message


def test_refusals(recorded, tmp_path):
    path, _ = recorded
    with open_streams(path) as streams:
        with pytest.raises(RebuildError, match="names no rebuildable sensor"):
            streams.select(["nope"])
        chosen = streams.select(["oakd"])
        with pytest.raises(RebuildError, match="above this recording's 25 fps"):
            list(streams.frames(chosen, rate=50))
        with pytest.raises(RebuildError, match="does not divide.*8.33333 and 12.5"):
            list(streams.frames(chosen, rate=10))
        with pytest.raises(RecordingError, match="outside this recording"):
            list(streams.frames(chosen, at=[99.0]))
    other = tmp_path / "other.xml"
    other.write_text(_SCENE.replace("<freejoint/>", ""))
    with pytest.raises(RecordingError, match="does not match this recording"):
        open_streams(path, str(other))


def test_a_recording_without_a_seed_is_refused(tmp_path):
    path = _record(_world(tmp_path), tmp_path / "run.mcap", steps=30)
    import roqsim.rebuild as rebuild

    rec = rebuild.open_recording(path)
    rec.meta["seed"] = None
    with pytest.raises(RebuildError, match="records no seed"):
        rebuild.Streams(rec)


def test_the_default_rate_is_the_sensors_own_on_the_recordings_grid(tmp_path):
    path = _record(_world(tmp_path, rate_hz=10), tmp_path / "run.mcap", steps=200)
    with open_streams(path) as streams:
        manifest = streams.manifest(streams.select(["oakd/image"]))
    (entry,) = manifest["streams"]
    assert entry["rate_hz"] == 10 and entry["rebuilt_rate_hz"] == pytest.approx(25 / 3)


def test_the_mcap_sink_is_indexed_and_carries_its_manifest(recorded, tmp_path):
    from mcap.reader import make_reader

    path, live = recorded
    out = tmp_path / "sensors.mcap"
    with open_streams(path) as streams:
        chosen = streams.select()
        manifest = streams.manifest(chosen, rate=25)
        counts = write_mcap(streams.frames(chosen, rate=25), out, manifest)
    assert manifest["frames"] == sum(counts.values()) == 5 * len(live)
    assert manifest["source"]["seed"] == 7 and len(manifest["source"]["sha256"]) == 64
    with open(out, "rb") as fh:
        reader = make_reader(fh)
        summary = reader.get_summary()
        assert summary.statistics.message_count == manifest["frames"]
        assert summary.chunk_indexes and all(c.compression == "zstd" for c in summary.chunk_indexes)
        topics = {c.topic: summary.schemas[c.schema_id].name for c in summary.channels.values()}
        assert topics["/rgbd_camera/image_raw"] == "sensor_msgs/msg/Image"
        assert topics["/scan"] == "sensor_msgs/msg/LaserScan"
        (meta,) = [m for m in reader.iter_metadata() if m.name == META_REBUILD]
        assert json.loads(meta.metadata["json"])["frames"] == manifest["frames"]
        t0 = min(live)
        messages = list(
            reader.iter_messages(topics=["/rgbd_camera/image_raw"], start_time=int(t0 * 1e9))
        )
    assert len(messages) == len(live)
    # The image's pixels, after the CDR header, are the sensor's.
    first = messages[0][2].data
    assert first.endswith(live[min(live)]["image"].tobytes())


def test_the_command_lists_and_writes(recorded, tmp_path, capsys):
    path, _ = recorded
    assert main(["--state", str(path), "--list"]) == 0
    listed = json.loads(capsys.readouterr().out)
    assert "oakd/image_compressed" in {s["path"] for s in listed["streams"]}
    assert "oakd/image_compressed" not in listed["default"]
    out = tmp_path / "o.mcap"
    assert main(["--state", str(path), "--sensor", "front", "--output", str(out)]) == 0
    record = json.loads(capsys.readouterr().out)
    assert record["written"] == {"front/scan": record["frames"]} and out.exists()
    assert record["timing"]["first_frame_s"] is not None
    assert main(["--state", str(path), "--sensor", "nope"]) == 2


def test_check_says_which_endpoints_a_rebuild_produces(tmp_path):
    report = check_world(str(_world(tmp_path)))
    image = next(e for e in report["world"]["endpoints"] if e["name"] == "image")
    info = next(e for e in report["world"]["endpoints"] if e["name"] == "camera_info")
    scan = next(e for e in report["world"]["endpoints"] if e["name"] == "scan")
    assert (image["lazy"], image["cost"], image["rebuildable"]) == (True, "render", True)
    assert (info["lazy"], info["cost"], info["rebuildable"]) == (False, None, True)
    assert (scan["cost"], scan["rebuildable"]) == ("raycast", True)


def test_a_rebuilt_image_message_holds_the_payload(recorded):
    path, live = recorded
    with open_streams(path) as streams:
        (frame,) = streams.frames(streams.select(["oakd/image"]), at=[min(live)])
        assert isinstance(frame.payload, Image)
        assert frame.message.endswith(frame.payload.data.tobytes())
