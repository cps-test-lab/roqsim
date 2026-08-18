"""``roqsim state``: the selectors, the output shapes, and what a replayed sensor does and does not promise."""

from __future__ import annotations

import json

import mujoco
import numpy as np
import pytest

from roqsim import state as st
from roqsim.recording import open_recording

_WORLD_XML = """
<mujoco>
  <option timestep="0.002"/>
  <worldbody>
    <geom type="plane" size="6 6 .1"/>
    <body pos="2 0 .5"><geom type="box" size=".3 .3 .5"/></body>
    <body name="base" pos="0 0 .2">
      <geom type="cylinder" size=".2 .2"/>
      <site name="scan" pos="0 0 .3"/>
      <body name="arm" pos="0 0 .3">
        <joint name="j" type="hinge" axis="0 1 0"/>
        <geom type="capsule" size=".04" fromto="0 0 0 .4 0 0"/>
        <site name="tip" pos=".4 0 0"/>
      </body>
    </body>
  </worldbody>
  <actuator><position joint="j" kp="10"/></actuator>
  <sensor><jointpos joint="j" name="jpos"/></sensor>
</mujoco>
"""


@pytest.fixture
def recording(tmp_path, monkeypatch):
    """A short recording of a world with a body, a joint, an MJCF sensor and a lidar."""
    monkeypatch.setenv("MUJOCO_GL", __import__("os").environ.get("MUJOCO_GL", "egl"))
    import yaml

    scene = tmp_path / "s.xml"
    scene.write_text(_WORLD_XML)
    world = tmp_path / "w.yaml"
    world.write_text(
        yaml.safe_dump(
            {
                "sim": {"world": str(scene)},
                "plugins": [
                    {"lidar": {"name": "front", "site": "scan", "rays": 180, "range_stddev": 0.05}}
                ],
            }
        )
    )
    from roqsim.runner import run

    out = tmp_path / "run.npz"
    run(
        str(world),
        headless=True,
        pacing="asap",
        seconds=2.0,
        record=str(out),
        capture_fps=25,
        seed=7,
    )
    return out


# -- selectors -------------------------------------------------------------------------------------


def test_a_body_pose_has_position_and_orientation(recording, tmp_path):
    record = st.run_state(recording, bodies=["base"], at=1.0)
    keys = record["values"]
    for suffix in ("pos.x", "pos.y", "pos.z", "rot.roll", "rot.pitch", "rot.yaw"):
        assert f"base.{suffix}" in keys


def test_globs_match(recording):
    record = st.run_state(recording, bodies=["*"], at=1.0)
    assert any(k.startswith("base.") for k in record["values"])
    assert any(k.startswith("arm.") for k in record["values"])


def test_an_unmatched_selector_errors_and_suggests(recording):
    """Never an empty column: a silently missing series makes a downstream analysis quietly wrong."""
    with pytest.raises(st.StateError) as err:
        st.run_state(recording, bodies=["basex"], at=1.0)
    message = str(err.value)
    assert "matches nothing" in message
    assert "base" in message  # the near miss is named


def test_an_unmatched_glob_errors_too(recording):
    with pytest.raises(st.StateError, match="matches nothing"):
        st.run_state(recording, bodies=["nothing_*"], at=1.0)


def test_a_joint_reports_position_and_velocity(recording):
    record = st.run_state(recording, joints=["j"], at=1.0)
    assert "j.qpos" in record["values"] and "j.qvel" in record["values"]


def test_a_site_reports_its_world_position(recording):
    record = st.run_state(recording, sites=["tip"], at=1.0)
    assert {"tip.pos.x", "tip.pos.y", "tip.pos.z"} <= set(record["values"])


def test_an_mjcf_sensor_is_read_from_sensordata(recording):
    record = st.run_state(recording, mjcf_sensors=["jpos"], at=1.0)
    assert "jpos" in record["values"]


def test_selecting_nothing_is_an_error(recording):
    with pytest.raises(st.StateError, match="nothing selected"):
        st.run_state(recording, at=1.0)


def test_a_free_joint_is_sliced_by_address_not_by_id():
    """qpos[jid] would read a neighbour's value for a 7-wide free joint."""
    model = mujoco.MjModel.from_xml_string(
        "<mujoco><worldbody><body name='b'><freejoint name='f'/>"
        "<geom type='box' size='.1 .1 .1'/></body></worldbody></mujoco>"
    )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    columns = st.joint_columns(model, data, ["f"])
    assert len([k for k in columns if ".qpos" in k]) == 7
    assert len([k for k in columns if ".qvel" in k]) == 6


# -- the --at contract, inherited from the recording core ------------------------------------------


def test_at_reports_which_sample_it_used(recording):
    record = st.run_state(recording, bodies=["base"], at=1.0)
    assert record["requested_at"] == 1.0
    assert abs(record["at_error"]) <= 0.02 + 1e-9


def test_at_matches_what_the_renderer_reports(recording, tmp_path):
    """Both commands must agree, because both go through the same recording core."""
    from roqsim.render import render_target

    a = st.run_state(recording, bodies=["base"], at=1.23)
    b = render_target(None, tmp_path / "x.png", size="64x64", state=recording, at=1.23)
    assert (a["sim_time"], a["sample_index"]) == (b["sim_time"], b["sample_index"])


# -- output shapes ---------------------------------------------------------------------------------


def test_a_range_writes_csv_with_a_stated_header(recording, tmp_path):
    """insertion_task's rule: the observable, at a stated rate, with the frame in the header."""
    out = tmp_path / "b.csv"
    record = st.run_state(recording, bodies=["base"], start=0.5, stop=1.5, out=out)
    text = out.read_text()
    assert text.startswith("# recording:")
    assert "# frame: world" in text and "# rate_fps: 25" in text and "# seed: 7" in text
    assert record["rows"] > 0
    header_line = [ln for ln in text.splitlines() if not ln.startswith("#")][0]
    assert header_line.split(",")[:3] == ["sim_time", "wall_time", "sample_index"]
    # A wall column whose zero is not named in the header reads as a timestamp.
    assert "# wall_clock_origin: recorder start" in text


def test_the_csv_carries_both_clocks_and_they_disagree(recording, tmp_path):
    """Sim time and wall time are two measurements of one run, not two names for one column."""
    import csv as _csv

    out = tmp_path / "b.csv"
    st.run_state(recording, bodies=["base"], start=0.5, stop=1.5, out=out)
    body = [ln for ln in out.read_text().splitlines() if not ln.startswith("#")]
    rows = list(_csv.DictReader(body))
    sim = [float(r["sim_time"]) for r in rows]
    wall = [float(r["wall_time"]) for r in rows]
    assert sim == sorted(sim) and wall == sorted(wall)
    assert wall[0] >= 0.0, "elapsed seconds from the recorder's start, never a Unix timestamp"
    # `--pacing asap` on a toy world: a second of sim costs far less than a second of wall.
    assert (sim[-1] - sim[0]) > (wall[-1] - wall[0])


def test_csv_row_count_matches_the_sample_count_in_range(recording, tmp_path):
    rec = open_recording(recording)
    expected = sum(1 for _ in rec.range(0.5, 1.5))
    out = tmp_path / "b.csv"
    st.run_state(recording, bodies=["base"], start=0.5, stop=1.5, out=out)
    rows = [ln for ln in out.read_text().splitlines() if not ln.startswith("#")]
    assert len(rows) - 1 == expected  # minus the column header


def test_contacts_are_reported_per_contact(recording):
    record = st.run_state(recording, bodies=["base"], contacts=True, at=1.0)
    assert isinstance(record["contacts"], list)
    for row in record["contacts"]:
        assert {"geom1", "geom2", "pos.x", "dist", "force.normal"} <= set(row)


# -- sensors: kind comes from the endpoint, not from a table here ----------------------------------


def test_check_lists_the_worlds_sensors_and_their_kinds(recording):
    record = st.run_state(recording, check=True)
    assert record["sensors"], "a world with a lidar must offer at least one sensor"
    assert set(record["sensors"].values()) <= {st.KIND_SCALAR, st.KIND_ARRAY, st.KIND_IMAGE}


def test_an_undeclared_sensor_errors_and_lists_what_the_world_has(recording):
    with pytest.raises(st.StateError) as err:
        st.run_state(recording, sensors=["nosuch"], at=1.0)
    assert "no such sensor" in str(err.value) and "It has:" in str(err.value)


def test_an_array_sensor_to_csv_is_refused_naming_npz(recording, tmp_path):
    """A scan does not fit one row per sample; this is a shape rule, not a per-sensor rule."""
    names = st.run_state(recording, check=True)["sensors"]
    array_sensor = next((n for n, k in names.items() if k == st.KIND_ARRAY), None)
    assert array_sensor, "expected the lidar to be an array sensor"
    with pytest.raises(st.StateError, match=r"\.npz"):
        st.run_state(recording, sensors=[array_sensor], start=0.5, stop=1.0, out=tmp_path / "x.csv")


def test_an_array_sensor_to_npz_gives_one_row_per_sample(recording, tmp_path):
    names = st.run_state(recording, check=True)["sensors"]
    array_sensor = next(n for n, k in names.items() if k == st.KIND_ARRAY)
    out = tmp_path / "scan.npz"
    record = st.run_state(recording, sensors=[array_sensor], start=0.5, stop=1.0, out=out)
    archive = np.load(out, allow_pickle=False)
    rows, beams = record["arrays"][array_sensor]
    assert archive[array_sensor].shape == (rows, beams)
    assert len(archive["times"]) == rows
    # Both clocks, as separate named members: a reader must not have to guess which one it holds.
    assert len(archive["wall_times"]) == rows
    assert json.loads(str(archive["meta"]))["frame"] == "world"


def test_the_scalar_array_boundary_is_a_deliberate_width():
    """A short vector is CSV columns; a long one is an .npz array. The line is a width, not a name.

    A 32-beam scan really is usable as 32 columns, so the boundary is where columns stop being a row.
    """
    from roqsim.context import Endpoint

    just_under = Endpoint(name="s", direction="out", read=lambda: np.zeros(st._SCALAR_MAX))
    just_over = Endpoint(name="a", direction="out", read=lambda: np.zeros(st._SCALAR_MAX + 1))
    assert st.endpoint_kind(just_under) == st.KIND_SCALAR
    assert st.endpoint_kind(just_over) == st.KIND_ARRAY


def test_the_kind_comes_from_the_payload_shape_not_a_sensor_name():
    """A stub endpoint, so this cannot pass by coincidence on a real sensor.

    This is the whole design: `roqsim state` is another backend for the endpoint registry, so a new sensor
    type needs no change here at all.
    """
    from roqsim.context import Endpoint

    scalar = Endpoint(name="s", direction="out", read=lambda: np.array([1.0, 2.0]))
    array = Endpoint(name="a", direction="out", read=lambda: np.zeros(500))
    image = Endpoint(name="i", direction="out", read=lambda: np.zeros((8, 8, 3), np.uint8))
    assert st.endpoint_kind(scalar) == st.KIND_SCALAR
    assert st.endpoint_kind(array) == st.KIND_ARRAY
    assert st.endpoint_kind(image) == st.KIND_IMAGE


def test_an_endpoint_can_declare_its_kind_explicitly():
    """Via Endpoint.backend, the same inert-metadata channel the ROS 2 bridge already uses."""
    from roqsim.context import Endpoint

    endpoint = Endpoint(
        name="odd",
        direction="out",
        read=lambda: np.zeros(500),
        backend={"file": {"kind": st.KIND_SCALAR}},
    )
    assert st.endpoint_kind(endpoint) == st.KIND_SCALAR


def test_an_image_endpoint_is_refused_and_points_at_the_renderer():
    from roqsim.context import Endpoint

    endpoint = Endpoint(name="cam", direction="out", read=lambda: np.zeros((4, 4, 3), np.uint8))
    with pytest.raises(st.StateError) as err:
        st.check_sensor_kind("cam", endpoint, None)
    assert "roqsim render" in str(err.value) and "--camera" in str(err.value)


def test_only_outward_endpoints_are_offered():
    """An input port is something the world consumes, not something it can be asked for."""

    class _Ctx:
        class interface:
            @staticmethod
            def all():
                from roqsim.context import Endpoint

                return [
                    Endpoint(name="out1", direction="out", read=lambda: np.zeros(3)),
                    Endpoint(name="in1", direction="in", write=lambda v: None),
                ]

    assert set(st.replayable_sensors(_Ctx())) == {"out1"}


# -- what a replayed sensor promises ---------------------------------------------------------------


def test_replaying_the_same_recording_twice_is_identical(recording, tmp_path):
    """The guarantee that does hold: deterministic, and noise-correct for the restored state."""
    names = st.run_state(recording, check=True)["sensors"]
    sensor = next(n for n, k in names.items() if k == st.KIND_ARRAY)
    first, second = (tmp_path / "a.npz", tmp_path / "b.npz")
    for out in (first, second):
        st.run_state(recording, sensors=[sensor], start=0.5, stop=1.0, out=out)
    a, b = np.load(first)[sensor], np.load(second)[sensor]
    assert np.array_equal(np.nan_to_num(a), np.nan_to_num(b))


def test_a_replayed_sensor_carries_the_recorded_seed(recording):
    """Storing the seed and not applying it made every replayed scan differ while looking plausible."""
    rec = open_recording(recording)
    _model, ctx = rec.build()
    assert ctx.seed == 7
