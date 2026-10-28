"""The FANUC arms under the names of FANUC's ROS 2 driver, against the driver's own behaviour.

Each behaviour is the driver's, at the commit ``docs/interfaces.rst`` names, and the test says which
file of FANUC-CORPORATION/fanuc_driver it reproduces. The interface table in ``docs/interfaces.rst``
is read here and checked against what the simulated arm serves, so the two cannot drift. Run
without ROS: what the bridge does with each name is the bridge's own tests'.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import pytest
from roqsim_manipulation.plugins.arm_controller import ArmControllerPlugin
from roqsim_manipulation.plugins.fanuc_driver_interface import (
    STMO_INACTIVE,
    FanucDriverInterfacePlugin,
)

from roqsim.config import load_config_from_dict
from roqsim.controllers import ACTIVE, INACTIVE, STRICT, registry_for
from roqsim.engine import Engine

arms = pytest.mark.parametrize("name", ["m20id25", "crx25ia"])
DOCS = Path(__file__).resolve().parents[2] / "docs" / "interfaces.rst"


def _engine(name: str, extra: list | None = None, **arm) -> Engine:
    world = {
        "sim": {"timestep": 0.002},
        "components": [{"spawn_arm": {"model": name, **arm}, "name": "arm"}, *(extra or [])],
    }
    engine = Engine(load_config_from_dict(world, base_dir=Path(".")), preview=True)
    engine.setup()
    engine.reset()
    return engine


def _driver(engine) -> FanucDriverInterfacePlugin:
    return next(p for p in engine.plugins if isinstance(p, FanucDriverInterfacePlugin))


def _arm(engine) -> ArmControllerPlugin:
    return next(p for p in engine.plugins if isinstance(p, ArmControllerPlugin))


def _steps(engine, n: int) -> None:
    for _ in range(n):
        engine.step()


def _endpoints(engine) -> dict:
    return {e.name: e for e in engine.ctx.interface.all() if e.owner == "arm"}


def _call(engine, service: str, **request):
    """A service as the bridge calls it: the request mapping to the endpoint's write, on the
    physics thread, the reply mapping back."""
    ep = _endpoints(engine)[service.replace("/", ".")]
    return ep.write(request)


# -- the table -------------------------------------------------------------------------------------

#: The default type of an endpoint whose ros2 hints name none, by payload class name.
_DEFAULT_TYPES = {
    "JointState": "sensor_msgs/msg/JointState",
    "JointPositions": "trajectory_msgs/msg/JointTrajectory",
}


def _table() -> list[tuple[str, str, str, str]]:
    text = DOCS.read_text()
    start = text.index("``fanuc_driver`` against the simulated arm")
    rows = re.findall(
        r"   \* - (\w+)\n     - ``([^`]+)``\n     - ``([^`]+)``\n     - (served with a difference|served|not served)\n",
        text[start:],
    )
    assert len(rows) > 40, "the table was not found or not parsed"
    return [r for r in rows if r[0] != "kind"]


def _ros_names(engine) -> dict[str, str]:
    """``{name: type}`` of every topic, service and action the arm's endpoints put on ROS."""
    out = {}
    for ep in engine.ctx.interface.all():
        if ep.owner != "arm":
            continue
        hints = dict((ep.backend or {}).get("ros2") or {})
        if (ep.backend or {}).get("ros2", {}) is None:
            continue
        kind = hints.get("type") or hints.get("service") or hints.get("action")
        if kind is None and ep.payload_type is not None:
            kind = _DEFAULT_TYPES.get(ep.payload_type.cls.__name__)
        if kind is None:
            continue
        name = hints.get("topic") or hints.get("name") or ep.topic or ep.name
        out["/" + name.lstrip("/")] = kind.replace(".", "/")
    return out


@arms
def test_every_name_the_table_serves_is_served_with_its_type(name):
    engine = _engine(name)
    served = _ros_names(engine)
    registry = {c.name: c for c in registry_for(engine.ctx).all()}
    for kind, row_name, row_type, here in _table():
        if kind == "controller":
            if row_name == "controller_manager":
                continue  # the bridge's, for every robot with controllers
            if here == "not served":
                assert row_name not in registry, row_name
            else:
                assert registry[row_name].type == row_type, row_name
        elif kind in ("topic", "service", "action"):
            if here == "not served":
                assert not any(n.startswith(row_name.rstrip("*")) for n in served), row_name
            else:
                assert served.get(row_name) == row_type, (row_name, served.get(row_name))


@arms
def test_nothing_is_served_that_the_table_does_not_list(name):
    listed = {r[1] for r in _table() if r[0] in ("topic", "service", "action")}
    for served in _ros_names(_engine(name)):
        assert served in listed, f"{served} is served but not in the table"


@arms
def test_the_controllers_start_as_the_drivers_launch_spawns_them(name):
    engine = _engine(name)
    engine.step()
    registry = registry_for(engine.ctx)
    expected = {
        "joint_trajectory_controller": ACTIVE,
        "joint_state_broadcaster": ACTIVE,
        "fanuc_gpio_controller": ACTIVE,
        "fanuc_force_sensor_broadcaster": ACTIVE,
        "force_torque_sensor_broadcaster": ACTIVE,
        "forward_position_controller": INACTIVE,
    }
    for controller, state in expected.items():
        assert registry.get(controller).state == state, controller
    assert not registry.get("forward_position_controller").listed
    # The scaled controller reads the two status interfaces besides the joints.
    reads = registry.get("joint_trajectory_controller").reads
    assert {"Status/motion_possible", "Status/collaborative_speed_scaling"} <= set(reads)


# -- speed scaling (scaled_joint_trajectory_controller.cpp) ---------------------------------------


def _trajectory_duration(engine, scaling: int | None, settle: int = 600) -> float:
    """Sim seconds an in-process 1 s trajectory takes at *scaling*, set *settle* steps before."""
    arm = engine.ctx.blackboard.get("arm:arm")
    if scaling is not None:
        _endpoints(engine)["speed_scaling_factor"].write(scaling)
    _steps(engine, settle)
    start = np.array(arm.read_state()[1])
    arm.follow_trajectory(arm.joint_names, [0.0, 1.0], [start, start + 0.1])
    t0 = engine.ctx.data.time
    while arm.trajectory_active():
        engine.step()
    return engine.ctx.data.time - t0


@arms
def test_the_speed_scaling_stretches_a_trajectory_by_its_factor(name):
    plain = _trajectory_duration(_engine(name), None)
    half = _trajectory_duration(_engine(name), 50)
    quarter = _trajectory_duration(_engine(name), 25)
    assert plain == pytest.approx(1.0, abs=0.005)
    # 600 steps (1.2 s) of the 0.2 s lag leave 0.25 % of the step: within half a percent.
    assert half / plain == pytest.approx(2.0, rel=0.005)
    assert quarter / plain == pytest.approx(4.0, rel=0.005)


@arms
def test_the_scaling_follows_the_drivers_first_order_lag(name):
    engine = _engine(name)
    driver = _driver(engine)
    _endpoints(engine)["speed_scaling_factor"].write(50)
    _steps(engine, 100)  # 0.2 s, one time constant, at h = 2 ms
    expected = 50 + 50 * (1 - 0.002 / 0.2) ** 100
    assert driver.speed_scaling_factor() * 100 == pytest.approx(expected, rel=1e-9)


@arms
@pytest.mark.parametrize("value, held", [(-5, 0), (250, 100), (0, 0), (100, 100)])
def test_the_scaling_is_clamped_silently(name, value, held):
    engine = _engine(name)
    _endpoints(engine)["speed_scaling_factor"].write(value)
    _steps(engine, 2000)
    assert _driver(engine).speed_scaling_factor() * 100 == pytest.approx(held, abs=1e-3)


@arms
def test_at_zero_the_trajectory_stops_and_resumes(name):
    engine = _engine(name)
    arm = engine.ctx.blackboard.get("arm:arm")
    start = np.array(arm.read_state()[1])
    arm.follow_trajectory(arm.joint_names, [0.0, 1.0], [start, start + 0.2])
    _steps(engine, 200)
    _endpoints(engine)["speed_scaling_factor"].write(0)
    _steps(engine, 1500)
    paused = np.array(arm.read_state()[1])
    _steps(engine, 500)
    assert arm.trajectory_active()
    assert np.allclose(arm.read_state()[1], paused, atol=1e-4)
    _endpoints(engine)["speed_scaling_factor"].write(100)
    while arm.trajectory_active():
        engine.step()
    _steps(engine, 300)
    assert np.allclose(arm.read_state()[1], start + 0.2, atol=2e-3)


def test_the_crx_collaborative_scaling_multiplies_and_the_m20_has_none():
    engine = _engine("crx25ia")
    _driver(engine).set_collaborative_speed_scaling(0.5)
    _endpoints(engine)["speed_scaling_factor"].write(50)
    _steps(engine, 2000)
    assert _driver(engine).speed_scaling_factor() == pytest.approx(0.25, rel=1e-3)
    with pytest.raises(RuntimeError, match="industrial"):
        _driver(_engine("m20id25")).set_collaborative_speed_scaling(0.5)


@arms
@pytest.mark.parametrize("value", [0, 101, -5, 127])
def test_an_invalid_gen_override_is_refused_and_changes_nothing(name, value):
    engine = _engine(name)
    driver = _driver(engine)
    before = driver.speed_scaling_factor()
    assert _call(engine, "fanuc_gpio_controller/set_gen_override", value=value) == {"result": 1}
    assert driver._robot_status_ext()["gen_override"] == 100
    assert driver.speed_scaling_factor() == before


@arms
@pytest.mark.parametrize("value", [1, 50, 100])
def test_a_valid_gen_override_is_stored_and_retimes_nothing(name, value):
    engine = _engine(name)
    assert _call(engine, "fanuc_gpio_controller/set_gen_override", value=value) == {"result": 0}
    assert _driver(engine)._robot_status_ext()["gen_override"] == value
    assert _trajectory_duration(engine, None, settle=10) == pytest.approx(1.0, abs=0.005)


# -- I/O (fanuc_gpio_controller.cpp) --------------------------------------------------------------


def _io(engine, kind: str, index: int):
    return _call(engine, "fanuc_gpio_controller/get_bool_io", io_type={"type": kind}, index=index)


def test_the_crx_io_follows_the_drivers_gpio_map():
    engine = _engine("crx25ia")
    eps = _endpoints(engine)
    state = eps["fanuc_gpio_controller.io_state"].read()["values"]
    kinds = {(v["io_type"]["type"], v["index"]) for v in state}
    assert ("DI", 101) in kinds and ("RO", 2) in kinds and ("F", 32) in kinds
    assert len(state) == 12 + 12 + 2 + 2 + 32
    # io_cmd sets a configured output; an unconfigured entry is ignored.
    eps["fanuc_gpio_controller.io_cmd"].write(
        [
            {"io_type": {"type": "DO"}, "index": 101, "value": True},
            {"io_type": {"type": "DO"}, "index": 200, "value": True},
        ]
    )
    _steps(engine, 1)
    assert _io(engine, "DO", 101) == {"result": 0, "value": True}
    assert _io(engine, "DO", 200) == {"result": 0, "value": False}
    # An output io_cmd owns is rewritten from the command every period.
    assert _call(
        engine, "fanuc_gpio_controller/set_bool_io", io_type={"type": "DO"}, index=102, value=True
    ) == {"result": 0}
    _steps(engine, 1)
    assert _io(engine, "DO", 102)["value"] is False
    # One it does not own keeps what a service set.
    _call(
        engine, "fanuc_gpio_controller/set_bool_io", io_type={"type": "DO"}, index=120, value=True
    )
    _steps(engine, 1)
    assert _io(engine, "DO", 120)["value"] is True


@arms
@pytest.mark.parametrize("kind", ["DI", "RI", "UI", "UO", "GO"])
def test_set_bool_io_refuses_what_the_driver_refuses(name, kind):
    engine = _engine(name)
    reply = _call(
        engine, "fanuc_gpio_controller/set_bool_io", io_type={"type": kind}, index=1, value=True
    )
    assert reply == {"result": 1}


def test_the_m20_io_is_the_small_gpio_map():
    engine = _engine("m20id25")
    state = _endpoints(engine)["fanuc_gpio_controller.io_state"].read()["values"]
    assert [(v["io_type"]["type"], v["index"]) for v in state] == [("F", i) for i in range(1, 9)]


def test_an_input_is_driven_by_the_simulation():
    engine = _engine("crx25ia")
    _driver(engine).set_input("DI", 101, True)
    assert _io(engine, "DI", 101)["value"] is True


@arms
def test_registers(name):
    engine = _engine(name)
    assert _call(engine, "fanuc_gpio_controller/set_num_reg", index=1, value=2.5) == {"result": 0}
    assert _call(engine, "fanuc_gpio_controller/get_num_reg", index=1) == {
        "result": 0,
        "value": 2.5,
    }
    # R[3..5] are num_reg_cmd's: the command (0 until one arrives) rewrites them every period.
    _call(engine, "fanuc_gpio_controller/set_num_reg", index=4, value=9.0)
    _steps(engine, 1)
    assert _call(engine, "fanuc_gpio_controller/get_num_reg", index=4)["value"] == 0.0
    _endpoints(engine)["fanuc_gpio_controller.num_reg_cmd"].write([{"index": 4, "value": 7.0}])
    _steps(engine, 1)
    assert _call(engine, "fanuc_gpio_controller/get_num_reg", index=4)["value"] == 7.0
    reply = _call(
        engine,
        "fanuc_gpio_controller/set_pos_reg",
        index=2,
        representation="Joint",
        front=3,
        up=1,
        j1=10.0,
    )
    assert reply == {"result": 0}
    got = _call(engine, "fanuc_gpio_controller/get_pos_reg", index=2)
    assert got["representation"] == "Joint" and got["front"] == 1 and got["j1"] == 10.0


def test_the_analog_command_is_indexed_as_the_driver_indexes_it():
    """Two AO blocks: the driver's index array restarts at 0 for each block (no offset), so the
    second block's commands land on the first block's outputs, and the first's on output 0."""
    engine = _engine(
        "m20id25",
        extra=[],
    )
    driver = _driver(engine)
    driver.config["gpio_topic_config"] = {
        "analog_io_cmd": [
            {"type": "AO", "start": 1, "length": 2},
            {"type": "AO", "start": 10, "length": 2},
        ],
    }
    driver._reset_state()
    assert driver._analog_cmd_index == [2, 3, 0, 0]
    driver._on_analog_io_cmd([{"io_type": {"type": "AO"}, "index": 11, "value": 5.0}])
    _steps(engine, 1)
    # The driver writes every position through the array, in order: AO11's command (position 3)
    # goes through index[3] = 0 to AO1, last, so it is what AO1 holds; AO10's (position 2) went
    # to AO1 before it and is lost.
    assert driver._analog[("AO", 1)] == 5.0


# -- payload, frames, position ----------------------------------------------------------------------


@arms
def test_the_payload_reads_back_as_the_driver_scales_it(name):
    engine = _engine(name)
    _call(
        engine,
        "fanuc_gpio_controller/set_payload_value",
        payload_schedule_id=1,
        mass=10.0,
        cg_x=20.0,
        in_x=30000.0,
    )
    got = _call(engine, "fanuc_gpio_controller/get_payload_value", payload_schedule_id=1)
    assert (
        got["mass"] == 10.0
        and got["cg_x"] == pytest.approx(0.2)
        and got["in_x"] == pytest.approx(3.0)
    )


def test_a_crx_payload_change_ends_stream_motion_and_an_m20_one_does_not():
    crx, m20 = _engine("crx25ia"), _engine("m20id25")
    for engine in (crx, m20):
        _call(engine, "fanuc_gpio_controller/set_payload_value", payload_schedule_id=1, mass=1.0)
        _steps(engine, 1)
    assert not _driver(crx).motion_possible()
    assert _endpoints(crx)["fanuc_gpio_controller.connection_status"].read() == {
        "is_connected": False
    }
    assert _driver(m20).motion_possible()
    assert _call(
        m20, "fanuc_gpio_controller/set_payload_comp", payload_schedule_id=1, mass=1.0
    ) == {"result": 1}
    assert _call(
        crx, "fanuc_gpio_controller/set_payload_comp", payload_schedule_id=1, mass=1.0
    ) == {"result": 0}


@arms
def test_the_cartesian_position_is_the_flange_in_fanucs_world_frame(name):
    engine = _engine(name)
    d = engine.ctx.data
    got = _call(engine, "fanuc_gpio_controller/get_cartesian_position", group=0)
    flange, wbase = d.site("fanuc_flange").xpos, d.site("wbase").xpos
    assert got["result"] == 0 and got["group"] == 1
    assert [got["x"], got["y"], got["z"]] == pytest.approx(list((flange - wbase) * 1000), abs=1e-3)
    # At home the tool points straight down: W = 180 (or -180), P = 0.
    assert abs(abs(got["w"]) - 180.0) < 1e-3 and abs(got["p"]) < 1e-3
    # A UTool 100 mm out along the tool axis moves the TCP 100 mm down.
    _call(engine, "fanuc_gpio_controller/set_utool_data", utool_number=1, frame={"z": 100.0})
    _call(engine, "fanuc_gpio_controller/set_uframe_utool", uframe_number=0, utool_number=1)
    tool = _call(engine, "fanuc_gpio_controller/get_cartesian_position")
    assert tool["z"] == pytest.approx(got["z"] - 100.0, abs=1e-3)
    assert tool["utool"] == 1


# -- motion, alarms, force ------------------------------------------------------------------------


@arms
def test_stop_closes_the_arm_to_motion_and_start_opens_it(name):
    engine = _engine(name)
    arm = engine.ctx.blackboard.get("arm:arm")
    start = np.array(arm.read_state()[1])
    arm.follow_trajectory(arm.joint_names, [0.0, 1.0], [start, start + 0.2])
    _steps(engine, 100)
    assert _call(engine, "fanuc_gpio_controller/switch_control_state", status=0) == {"result": 0}
    _steps(engine, 1)
    assert arm.motion_blocked() == STMO_INACTIVE
    assert not arm.trajectory_active()
    status = _endpoints(engine)["fanuc_gpio_controller.robot_status"].read()
    assert status["motion_possible"] is False
    assert _call(engine, "fanuc_gpio_controller/switch_control_state", status=7) == {"result": -1}
    assert _call(engine, "fanuc_gpio_controller/switch_control_state", status=1) == {"result": 0}
    _steps(engine, 1)
    assert arm.motion_blocked() is None


#: A tool for the force tests: what an external flange sensor sees is the contact on a tool.
HAND_E = {"end_effector": {"model": "robotiq_hande"}}


def _push_world(name: str, tool: float = 0.0) -> list:
    """A fixed block just under the flange (or a tool *tool* m long) at home, so a move down
    presses into it."""
    z = (1.39 if name == "m20id25" else 1.14) - 0.12 - tool
    x = 0.965 if name == "m20id25" else 0.75
    y = 0.0 if name == "m20id25" else -0.185
    return [
        {
            "box": {
                "pose": {"position": {"x": x, "y": y, "z": z}},
                "size": [0.3, 0.3, 0.1],
                "motion": "static",
                "prefix": "block_",
            },
            "name": "block",
        }
    ]


def _press(engine, depth: float = 0.15) -> None:
    arm = engine.ctx.blackboard.get("arm:arm")
    q = np.array(arm.read_state()[1])
    q[1] += depth  # J2 forward lowers the flange onto the block
    arm.set_targets(arm.joint_names, list(q))
    _steps(engine, 1500)


def test_the_crx_stops_on_contact_and_reset_clears_the_alarm():
    engine = _engine("crx25ia", extra=_push_world("crx25ia"))
    driver = _driver(engine)
    assert _endpoints(engine)["fanuc_force_sensor_broadcaster.force_sensor"].read()["fs_type"] == 1
    _press(engine)
    assert not driver.motion_possible()
    status = _endpoints(engine)["fanuc_gpio_controller.robot_status"].read()
    assert status["in_error"] is True and status["contact_stop_mode"] == 2
    errors = _call(engine, "fanuc_gpio_controller/read_error", count=5)
    assert errors["count"] == 1 and errors["error_data"][0].startswith("SIM-001")
    assert _call(engine, "fanuc_gpio_controller/read_error", count=0) == {"result": 1}
    assert _call(engine, "fanuc_gpio_controller/reset") == {"result": 0}
    assert not driver._in_error


def test_the_m20_does_not_stop_on_contact_and_senses_nothing_unselected():
    engine = _engine("m20id25", extra=_push_world("m20id25", tool=0.17), **HAND_E)
    driver = _driver(engine)
    _press(engine)
    assert driver.motion_possible()
    sensed = _endpoints(engine)["fanuc_force_sensor_broadcaster.force_sensor"].read()
    assert sensed["fs_type"] == 0 and sensed["force_z"] == 0.0
    # An external sensor, selected, reads the contact on the tool.
    assert _call(
        engine, "fanuc_force_sensor_broadcaster/cfg_force_sensor", do_reset=0, fs_type=2
    ) == {"result": 0}
    _steps(engine, 2)
    sensed = _endpoints(engine)["fanuc_force_sensor_broadcaster.force_sensor"].read()
    assert (
        sensed["fs_type"] == 2
        and math.hypot(sensed["force_x"], sensed["force_y"], sensed["force_z"]) > 1.0
    )


def test_the_crx_force_estimate_agrees_with_the_contact_on_the_tool():
    """Pressed on the flange alone, the joints' view (embedded) reads what the contact on the tool
    is (an external sensor's view): two computations sharing nothing but the contacts."""
    engine = _engine("crx25ia", extra=_push_world("crx25ia", tool=0.17), **HAND_E)
    driver = _driver(engine)
    driver.contact_stop_force = 1e9  # measure, do not stop
    # The Hand-E's own weight is the controller's to know: declared, it is taken off. Declared in
    # the controller's store, as a cell configures it before streaming starts: on a CRX the
    # service ends Stream Motion.
    tool = sum(engine.ctx.model.body_mass[b] for b in driver._tool_bodies)
    driver._payload_value[1] = {"mass": float(tool)}
    _press(engine, depth=0.15)
    embedded = driver._force.copy()
    driver._fs_type = 2
    external = driver._sense(engine.ctx)
    assert np.linalg.norm(external[:3]) > 20.0, "the tool is not pressed"
    assert embedded[:3] == pytest.approx(external[:3], rel=0.02, abs=0.5)
    assert embedded[3:] == pytest.approx(external[3:], rel=0.02, abs=0.05)


@arms
def test_reset_restores_every_value(name):
    engine = _engine(name)
    driver = _driver(engine)
    _endpoints(engine)["speed_scaling_factor"].write(10)
    _call(engine, "fanuc_gpio_controller/set_num_reg", index=1, value=1.0)
    _call(engine, "fanuc_gpio_controller/set_gen_override", value=20)
    _call(engine, "fanuc_gpio_controller/switch_control_state", status=0)
    _steps(engine, 100)
    engine.reset()
    assert driver.motion_possible() and driver.speed_scaling_factor() == 1.0
    assert _call(engine, "fanuc_gpio_controller/get_num_reg", index=1)["value"] == 0.0
    assert driver._robot_status_ext()["gen_override"] == 100


@arms
def test_dynamic_joint_states_carries_the_hardwares_interfaces(name):
    engine = _engine(name)
    rows = dict(_endpoints(engine)["dynamic_joint_states"].read())
    assert set(rows["J1"]) == {"position", "velocity"}
    assert rows["Status"]["motion_possible"] == 1.0
    assert "fs_type" in rows["Force"] and "force.x" in rows["ft_sensor"]
    assert rows["ConnectionStatus"]["is_connected"] == 1.0


@arms
def test_a_forward_controller_hand_over_works_as_on_the_robot(name):
    engine = _engine(name)
    registry = registry_for(engine.ctx)
    registry.get("forward_position_controller").listed = True  # load_controller
    ok, _ = registry.switch(
        activate=["forward_position_controller"],
        deactivate=["joint_trajectory_controller"],
        strictness=STRICT,
    )
    assert ok
    target = [0.2, 0.1, 0.1, 0.0, -1.4, 0.2]
    _endpoints(engine)["forward_commands"].write({"data": target})
    _steps(engine, 1500)
    assert np.allclose(engine.ctx.blackboard.get("arm:arm").read_state()[1], target, atol=2e-3)
