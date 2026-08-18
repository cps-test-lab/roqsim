"""export_capture: the run capture is layout-correct and addresses the geometry by name.

The format's whole contract is that a viewer can animate the *scene descriptor* from the *capture*
without either artifact knowing about MuJoCo: every track name must resolve to a joint or body the
descriptor exports, the binary must be readable by a browser's typed-array views, and a track must
mean exactly one thing (a body driven by joint tracks must not also carry a pose track, or two
writers fight over one transform).

The redundancy test is the one that earns its keep: an earlier cut marked only the body that *owns* a
joint as explained, which emitted a pose track for every link welded to a moving parent -- 36 tracks
instead of 2 on a real recording, silently.
"""

from __future__ import annotations

import json
import logging
import pathlib

import mujoco
import numpy as np
import pytest

from roqsim.export_capture import FORMAT, CaptureExportError, write_capture
from roqsim.export_web import export_scene

# Exercises every selection path: scalar joints (hinge/slide), a body welded to a moving parent (must
# NOT get a pose track), a free body (must), a mocap body (must -- its rest pose is a placeholder), and
# a body that never moves (must not).
_MJCF = """
<mujoco>
  <worldbody>
    <geom name="floor" type="plane" size="3 3 0.05"/>
    <body name="arm" pos="0 0 1">
      <joint name="j_hinge" type="hinge" axis="0 0 1"/>
      <geom name="g_arm" type="box" size="0.1 0.1 0.1"/>
      <body name="forearm" pos="0.3 0 0">
        <joint name="j_slide" type="slide" axis="1 0 0"/>
        <geom name="g_fore" type="box" size="0.05 0.05 0.05"/>
        <body name="welded_pad" pos="0.1 0 0">
          <geom name="g_pad" type="sphere" size="0.02"/>
        </body>
      </body>
    </body>
    <body name="cargo" pos="1 0 0.5">
      <freejoint name="j_free"/>
      <geom name="g_cargo" type="sphere" size="0.1"/>
    </body>
    <body name="ghost" pos="2 0 0.5" mocap="true">
      <geom name="g_ghost" type="capsule" size="0.05 0.1"/>
    </body>
    <body name="pillar" pos="-2 0 0.5">
      <geom name="g_pillar" type="cylinder" size="0.1 0.5"/>
    </body>
  </worldbody>
</mujoco>
"""


def _compile():
    model = mujoco.MjModel.from_xml_string(_MJCF)
    return model, mujoco.MjData(model)


def _samples(model, data, n=5):
    """Move the hinge, the slide, the free body and the mocap body; leave the pillar alone."""
    out = []
    for k in range(n):
        data.qpos[model.joint("j_hinge").qposadr[0]] = 0.1 * k
        data.qpos[model.joint("j_slide").qposadr[0]] = 0.02 * k
        free = model.joint("j_free").qposadr[0]
        data.qpos[free : free + 3] = [1.0, 0.0, 0.5 + 0.01 * k]
        data.mocap_pos[0] = [2.0, 0.1 * k, 0.5]
        mujoco.mj_forward(model, data)
        out.append((0.04 * k, data))
        yield out[-1]


def _read(path, track):
    buf = path.read_bytes()
    dtype = "<f8" if track["dtype"] == "f8" else "<f4"
    n = track["samples"] * track["width"]
    return np.frombuffer(buf, dtype=dtype, count=n, offset=track["off"])


def test_manifest_and_track_selection(tmp_path):
    model, data = _compile()
    manifest = write_capture(model, _samples(model, data), tmp_path, world="w.yaml", seed=7)

    assert manifest["format"] == FORMAT
    assert manifest["complete"] is True
    assert manifest["frame"] == "world"
    assert manifest["time"]["base"] == "sim"
    assert manifest["producer"] == "roqsim"
    assert manifest["world"] == "w.yaml" and manifest["seed"] == 7

    joints = {t["name"] for t in manifest["tracks"] if t["kind"] == "joint"}
    poses = {t["name"] for t in manifest["tracks"] if t["kind"] == "pose"}
    assert joints == {"j_hinge", "j_slide"}
    # `cargo` is free-jointed and `ghost` is mocap, so neither is reachable through joint tracks.
    assert poses == {"cargo", "ghost"}
    # The redundancy the transitive rule exists to prevent: these are determined by the joint tracks
    # (welded_pad through its moving ancestors) or by the geometry's rest pose (pillar).
    assert "welded_pad" not in poses and "arm" not in poses and "forearm" not in poses
    assert "pillar" not in poses


def test_units_and_binary_layout(tmp_path):
    model, data = _compile()
    manifest = write_capture(model, _samples(model, data), tmp_path)
    bin_path = tmp_path / "capture.bin"

    by_name = {t["name"]: t for t in manifest["tracks"]}
    assert by_name["j_hinge"]["unit"] == "rad"
    assert by_name["j_slide"]["unit"] == "m"

    # Every byte is accounted for: the f8 time track plus each f4 value track, no padding beyond
    # what alignment needs.
    total = manifest["time"]["samples"] * 8 + sum(
        t["samples"] * t["width"] * 4 for t in manifest["tracks"]
    )
    assert bin_path.stat().st_size == total

    # A browser's Float64Array/Float32Array view throws on a misaligned offset, so alignment is part
    # of the format being readable at all -- not a nicety.
    assert manifest["time"]["off"] % 8 == 0
    for t in manifest["tracks"]:
        assert t["off"] % 4 == 0

    t = _read(bin_path, manifest["time"])
    assert np.all(np.diff(t) > 0), "samples must be in strictly increasing time order"
    assert t[0] == pytest.approx(manifest["time"]["t0"])
    assert t[-1] == pytest.approx(manifest["time"]["t1"])

    hinge = _read(bin_path, by_name["j_hinge"])
    assert hinge == pytest.approx([0.0, 0.1, 0.2, 0.3, 0.4], abs=1e-6)

    # Pose tracks are sample-major (pos xyz then quat wxyz per sample) -- the property that makes a
    # time window a contiguous byte range.
    cargo = _read(bin_path, by_name["cargo"]).reshape(-1, 7)
    assert cargo.shape == (5, 7)
    assert cargo[:, 2] == pytest.approx([0.5, 0.51, 0.52, 0.53, 0.54], abs=1e-6)
    assert np.linalg.norm(cargo[0, 3:]) == pytest.approx(1.0, abs=1e-6)


def test_pose_tracks_are_world_frame_and_ordered_parents_first(tmp_path):
    """A consumer turns a world pose into a local transform, so a parent must arrive first."""
    model, data = _compile()
    manifest = write_capture(model, _samples(model, data), tmp_path)
    order = [t["name"] for t in manifest["tracks"] if t["kind"] == "pose"]
    ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n) for n in order]
    assert ids == sorted(ids), "pose tracks must follow model order (parents before children)"

    # The value is the body's world pose, matching what the geometry exports -- not a map-frame or
    # otherwise offset pose, which a viewer could not distinguish from the numbers.
    track = next(t for t in manifest["tracks"] if t["name"] == "cargo")
    cargo = _read(tmp_path / "capture.bin", track).reshape(-1, 7)
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cargo")
    assert cargo[-1, :3] == pytest.approx(data.xpos[bid], abs=1e-5)


def test_every_track_name_resolves_in_the_scene_descriptor(tmp_path):
    """The contract between the two artifacts, checked the way a viewer checks it: by name."""
    model, data = _compile()
    manifest = write_capture(model, _samples(model, data), tmp_path / "cap")
    export_scene(model, data, tmp_path / "scene", logging.getLogger(__name__))
    scene = json.loads((tmp_path / "scene" / "scene.json").read_text())

    bodies = {b["name"] for b in scene["bodies"]}
    joints = {j["name"] for j in scene["joints"]}
    for track in manifest["tracks"]:
        pool = joints if track["kind"] == "joint" else bodies
        assert track["name"] in pool, f"{track['kind']} track {track['name']!r} names nothing in scene.json"


def test_unnamed_mover_is_reported_not_dropped_silently(tmp_path, caplog):
    """A mover the format cannot address must be named in a warning, not left silently at rest."""
    model = mujoco.MjModel.from_xml_string(
        """
        <mujoco><worldbody>
          <body pos="0 0 1"><freejoint/><geom type="sphere" size="0.1"/></body>
        </worldbody></mujoco>
        """
    )
    data = mujoco.MjData(model)

    def samples():
        for k in range(3):
            data.qpos[2] = 1.0 + 0.1 * k
            mujoco.mj_forward(model, data)
            yield 0.04 * k, data

    with caplog.at_level("WARNING"):
        manifest = write_capture(model, samples(), tmp_path)
    assert not [t for t in manifest["tracks"] if t["kind"] == "pose"]
    assert "cannot express" in caplog.text


def test_empty_recording_refuses(tmp_path):
    model, _ = _compile()
    with pytest.raises(CaptureExportError, match="no samples"):
        write_capture(model, iter([]), tmp_path)
    assert not (tmp_path / "capture.json").exists(), "a refused export must leave no partial artifact"


def test_session_path_anchors_relative_output(monkeypatch, tmp_path):
    """A relative session path lands in the run's output dir, not the launch's working directory.

    The case this exists for: a campaign starts the world through a ROS launch file, so nothing passes
    an output directory down the way scenario-execution passes one to the adapter. Without the anchor
    the recording lands wherever the launch happened to leave the cwd, and the run view then 404s on an
    artifact that was written -- just not where the run collects results from.
    """
    from roqsim.runner import _session_path

    for var in ("RUN_OUTPUT_DIR", "OUTPUT_DIR", "SCENARIO_OUTPUT_DIR"):
        monkeypatch.delenv(var, raising=False)
    assert _session_path("run.npz") == pathlib.Path("run.npz")

    # The per-job directory, when that is the most specific thing on offer.
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / "job"))
    assert _session_path("run.npz") == tmp_path / "job" / "run.npz"

    # The run's own directory wins over it.
    monkeypatch.setenv("RUN_OUTPUT_DIR", str(tmp_path / "cfg" / "0"))
    assert _session_path("capture") == tmp_path / "cfg" / "0" / "capture"

    # SCENARIO_OUTPUT_DIR is the *campaign* root and must never anchor a per-run artifact: every run
    # of a sweep would write the same shared path, the last one winning. This is not hypothetical --
    # it is what a first cut did, and the capture landed beside campaign.db instead of in the run.
    monkeypatch.delenv("RUN_OUTPUT_DIR")
    monkeypatch.delenv("OUTPUT_DIR")
    monkeypatch.setenv("SCENARIO_OUTPUT_DIR", str(tmp_path / "campaign_root"))
    assert _session_path("capture") == pathlib.Path("capture")

    # An absolute path is never re-anchored.
    absolute = tmp_path / "elsewhere" / "run.npz"
    assert _session_path(str(absolute)) == absolute


def test_world_identity_distinguishes_no_overrides_from_unrecorded(tmp_path):
    """`{}` and absent must not look the same -- a consumer compiles geometry from this.

    A viewer that treats "not recorded" as "none applied" builds the *unoverridden* world for a run
    that varied it, and renders confidently wrong geometry. So the key is present only when the
    producer actually knows, and `{}` is a real answer meaning "none".
    """
    model, data = _compile()

    known = write_capture(model, _samples(model, data), tmp_path / "known",
                          world="w.yaml", overrides={"plugins": {"floorplan": {"size": 4.0}}},
                          packages={"roqsim": "0.1.0"})
    assert known["overrides"] == {"plugins": {"floorplan": {"size": 4.0}}}
    assert known["packages"] == {"roqsim": "0.1.0"}

    none_applied = write_capture(model, _samples(model, data), tmp_path / "none",
                                 world="w.yaml", overrides={})
    assert none_applied["overrides"] == {}, "an empty dict is 'none were applied', not 'unknown'"

    unrecorded = write_capture(model, _samples(model, data), tmp_path / "unknown", world="w.yaml")
    assert "overrides" not in unrecorded, "unknown must be absent, not {}"
    # `packages` is different in kind: a producer always knows its own versions, so it defaults to
    # them rather than being absent. Only `overrides` has an "I cannot tell" state.
    assert unrecorded["packages"]["roqsim"], "the producer always knows its own versions"
