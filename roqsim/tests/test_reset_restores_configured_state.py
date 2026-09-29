# Copyright (C) 2026 Frederik Pasch
# SPDX-License-Identifier: Apache-2.0
"""Every plugin's reset returns it to the state it was configured in.

For each plugin registered under ``roqsim.plugins``, in the smallest world it configures in:

A. set up, reset, record the per-trial state;
B. use it -- write every ``in`` endpoint of the world, step, call the case's setters (a
   controller's case switches every controller), step -- then reset and record again.

A trial that used a plugin must leave nothing behind, so A and B must be equal. The episode counter
is put back before the second reset, so a draw keyed on it is the same draw in both.

**What is compared.** The plugin's instance attributes, followed into the objects it owns (roqsim
types, dataclasses, containers, numpy arrays, generators' bit state), plus two things a plugin can
change that it does not own: the run's stop request on the context, and each controller's state in
the controller registry. A third-party object is compared by its type alone: its insides are that
library's bookkeeping. Not followed, because they are shared rather than the plugin's own: the
context and its registries, the entities, other plugins, MuJoCo objects (``model``, ``data``, a
scratch ``MjData``, a renderer), loggers, locks, threads and callables. Values are compared exactly:
both records are taken after the same reset, so a difference is a difference, not float noise.

A plugin that keeps some state across a reset on purpose is listed in :data:`EXEMPT` with the reason,
either whole or by attribute path. A plugin with no world it can configure in here is listed in
:data:`SKIPPED`. A case whose plugin has a known open defect names it (``defect=``) and is a strict
expected failure, so the fix that makes it pass has to remove it.

Every registered plugin must appear in exactly one of :data:`CASES`, :data:`EXEMPT` (whole) or
:data:`SKIPPED`, so a new plugin cannot go unchecked by being left out.
"""

from __future__ import annotations

import dataclasses
import functools
import logging
import queue
import threading
import types
from collections.abc import Callable
from dataclasses import dataclass, field
from importlib.metadata import entry_points
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from roqsim.config import load_config_from_dict
from roqsim.context import Blackboard, Entity, EntityRegistry, InterfaceRegistry, SimContext
from roqsim.controllers import ACTIVE, FORCE_AUTO, SERVICE_KEY, ControllerRegistry
from roqsim.endpoint import bind
from roqsim.engine import Engine
from roqsim.plugin import Plugin
from roqsim.types import JointState

#: Physics steps a trial runs for, split around the setter calls.
STEPS = 200

# -- worlds ------------------------------------------------------------------------------------------


def _world(*components: dict, **sim) -> dict:
    return {"sim": dict(sim), "components": list(components)}


def _robot(model: str, *components: dict, name: str = "bot", **spawn) -> dict:
    return {"spawn_robot": {"model": model, **spawn}, "name": name, "components": list(components)}


def _mobile(*components: dict) -> dict:
    """The smallest wheeled base, with a lidar from its manifest."""
    return _world(_robot("makerspet_mini", *components))


def _arm(*components: dict) -> dict:
    """A UR5e with its trajectory controller and a wrist force-torque sensor named ``ft``."""
    return _world(
        {
            "spawn_arm": {"model": "ur5e", "prefix": "ur5e_", "namespace": "ur5e"},
            "name": "ur5e",
            "components": [
                {"arm_controller": {}},
                {"force_torque": {"site": "fts_site", "frame": "world"}, "name": "ft"},
                *components,
            ],
        },
        timestep=0.001,
    )


def _sensor(model: str, *components: dict) -> dict:
    return _world(
        {
            "spawn_sensor": {
                "model": model,
                "prefix": f"{model}_",
                "pose": {"position": {"z": 0.5}},
            },
            "name": model,
            "components": list(components),
        },
        _box(),
    )


class _Mount(Plugin):
    """A post carrying a site and a camera by the names a sensor looks for, facing a box.

    For the sensors whose own housing is a vendor mesh that is generated rather than committed.
    """

    def build(self, spec, ctx) -> None:
        post = spec.worldbody.add_body(name="post", pos=[0.0, 0.0, 0.5])
        post.add_site(name=self.config["site"])
        post.add_camera(name=self.config["site"] + "_color", xyaxes=[0, -1, 0, 0, 0, 1])


def _mounted(site: str, *components: dict) -> dict:
    return _world({f"{__name__}:_Mount": {"site": site}}, *components, _box())


def _box() -> dict:
    return {
        "box": {"pose": {"position": {"x": 1.0, "y": 0.0, "z": 0.0}}, "size": [0.3, 0.3, 0.3]},
        "name": "crate",
    }


_BEAM = """<mujoco><worldbody><body name="holder" pos="0 0 .5">
<flexcomp name="beam" type="grid" count="6 2 2" spacing=".02 .02 .02" dim="3" mass=".05"
          radius=".001">
  <elasticity young="1e5" poisson="0.3" damping="0.01"/>
  <contact contype="0" conaffinity="0" selfcollide="none"/>
  <pin gridrange="0 0 0 0 1 1"/>
</flexcomp></body></worldbody></mujoco>"""


def _flex_world(tmp_path: Path) -> dict:
    path = tmp_path / "beam.xml"
    path.write_text(_BEAM, encoding="utf-8")
    return _world(
        {"spawn_model": {"model": str(path), "prefix": "p_", "motion": "static"}, "name": "beam"},
        {"flex_material": {"flex": "p_beam", "young": 3.0e5}},
        timestep=0.001,
    )


#: A three-wheel truck: a driven axle through base_link and a steered wheel 0.6 m behind it. No
#: tricycle model is bundled, so the case brings its own.
_TRICYCLE = """<mujoco><worldbody><body name="base_link" pos="0 0 .1">
<freejoint name="base_free"/>
<geom type="box" pos="-.3 0 .15" size=".4 .2 .05" mass="20"/>
<body name="left_link" pos="0 .25 0"><joint name="left" axis="0 1 0"/>
  <geom type="cylinder" size=".1 .03" quat=".7071 .7071 0 0" mass="1"/></body>
<body name="right_link" pos="0 -.25 0"><joint name="right" axis="0 1 0"/>
  <geom type="cylinder" size=".1 .03" quat=".7071 .7071 0 0" mass="1"/></body>
<body name="steer_link" pos="-.6 0 0"><joint name="steer" axis="0 0 1" range="-75 75"/>
  <geom type="box" size=".02 .02 .02" mass=".5"/>
  <body name="steer_wheel_link"><joint name="steer_wheel" axis="0 1 0"/>
    <geom type="cylinder" size=".1 .03" quat=".7071 .7071 0 0" mass="1"/></body></body>
</body></worldbody>
<actuator><position name="steer_motor" joint="steer" kp="500" kv="20"/>
<velocity name="left_motor" joint="left" kv="10"/>
<velocity name="right_motor" joint="right" kv="10"/></actuator></mujoco>"""


def _tricycle_world(tmp_path: Path) -> dict:
    path = tmp_path / "tricycle.xml"
    path.write_text(_TRICYCLE, encoding="utf-8")
    drive = {
        "steer_offset": -0.6,
        "wheel_radius": 0.1,
        "track": 0.5,
        "max_steer_angle": 1.2,
        "steer_actuator": "steer_motor",
        "steer_joint": "steer",
        "drive_actuators": ["left_motor", "right_motor"],
        "drive_joints": ["left", "right"],
        "passive_joints": ["steer_wheel"],
    }
    return _world(_robot(str(path), {"tricycle_drive": drive}))


def _trajectory_world(tmp_path: Path) -> dict:
    path = tmp_path / "square.csv"
    path.write_text("0,0\n50,0\n50,50\n0,50\n0,0\n", encoding="utf-8")
    return _world(
        {
            "prop_trajectory": {
                "path": str(path),
                "units": "mm",
                "speed": 0.03,
                "origin": [0.0, 0.0, 0.606],
                "plate": [0.07, 0.07, 0.006],
            },
            "name": "stage",
        }
    )


# -- using a plugin ----------------------------------------------------------------------------------


def _plugins(engine: Engine, name: str) -> list[Plugin]:
    cls = _entry_point_classes()[name]
    return [p for p in engine.plugins if type(p) is cls]


def _joint_command(engine: Engine, endpoint) -> tuple:
    """``(names, positions)`` a little off where the owner's joints are now."""
    states = [
        e
        for e in engine.ctx.interface.by_direction("out")
        if e.owner == endpoint.owner
        and e.name == "joint_states"
        and e.namespace == endpoint.namespace
    ]
    state = states[0].read()
    names, positions = (
        (state.names, state.positions) if isinstance(state, JointState) else state[:2]
    )
    return list(names), [float(p) + 0.1 for p in positions]


def _motor_command(engine: Engine, endpoint) -> list[float]:
    motors = engine.ctx.blackboard.get(f"motors:{endpoint.owner}")
    return [0.6] * motors.count


#: What to write into an ``in`` endpoint, by endpoint name: a value, or ``f(engine, endpoint)``.
PAYLOADS: dict[str, Any] = {
    "cmd_vel": (0.2, 0.0, 0.3),
    "ackermann_cmd": ({"steering_angle": 0.3, "speed": 0.2},),
    "follow_joint_trajectory": _joint_command,
    "joint_command": _joint_command,
    "joint_velocity": _joint_command,
    "gripper_cmd": 0.4,
    "target_frame": ([0.3, 0.2, 0.4], [1.0, 0.0, 0.0, 0.0]),
    "target_wrench": ([0.0, 0.0, -5.0], [0.0, 0.0, 0.0]),
    "navigate_to_pose": [(0.8, 0.3, 0.0)],
    "navigate_through_poses": [(0.5, 0.0, 0.0), (0.8, 0.4, 0.0)],
    "start_route": None,
    "cancel_route": None,
    "cmd_pos": ([0.0, 0.0, 0.5], [1.0, 0.0, 0.0, 0.0]),
    "motor_cmd": _motor_command,
    "speed": 0.2,
    "cmd": 1.0,
    "door": 1.0,
    "tare": None,
    "override": True,
    "attach": True,
}


def _write_every_in_endpoint(engine: Engine) -> None:
    for endpoint in engine.ctx.interface.by_direction("in"):
        if endpoint.name not in PAYLOADS:
            pytest.fail(
                f"no payload for the in endpoint {endpoint.name!r} (owner {endpoint.owner!r}): "
                "add one to PAYLOADS so a trial uses it"
            )
        payload = PAYLOADS[endpoint.name]
        if callable(payload):
            payload = payload(engine, endpoint)
        if endpoint.params is not None:
            payload = _named(endpoint, payload)
        engine.ctx.post(lambda _ctx, e=endpoint, p=payload: e.write(p))


def _named(endpoint, payload) -> dict:
    """*payload*, positional as in :data:`PAYLOADS`, as the named parameters a typed endpoint takes.

    Checked here: a typed write refuses a misfit into a future or a log line, which a trial would
    not notice, and the endpoint would go unused.
    """
    values = () if payload is None else payload if isinstance(payload, tuple) else (payload,)
    named = dict(zip((p.name for p in endpoint.params), values, strict=False))
    bind(endpoint.params, named, f"{endpoint.owner}/{endpoint.name}")
    return named


def _switch_every_controller(engine: Engine) -> None:
    """Deactivate what is active and activate what is not, as a scenario handing over would."""
    registry = engine.ctx.blackboard.get(SERVICE_KEY)
    if registry is None:
        return
    for namespace in {c.namespace for c in registry.all()}:
        mine = [c for c in registry.all(namespace) if c.claims]
        registry.switch(
            activate=[c.name for c in mine if c.state != ACTIVE],
            deactivate=[c.name for c in mine if c.state == ACTIVE],
            strictness=FORCE_AUTO,
            namespace=namespace,
            sim_time=engine.ctx.sim_time,
        )


def _override(engine: Engine) -> None:
    for plugin in _plugins(engine, "model_override"):
        plugin.set_active(not plugin.initial_active)


@dataclass
class Case:
    """The world a plugin is checked in, and what a trial does with it besides its endpoints."""

    world: Callable[[Path], dict]
    use: tuple[Callable[[Engine], None], ...] = ()
    #: Why this case cannot run here, asked at run time; ``None`` when it can.
    unavailable: Callable[[], str | None] = field(default=lambda: None)
    #: The open defect this case fails on, until the fix removes it.
    defect: str | None = None


def _needs_file(path_of: Callable[[], Path], what: str) -> Callable[[], str | None]:
    def check() -> str | None:
        return None if path_of().exists() else f"{what} is not present ({path_of()})"

    return check


def _spot_policy() -> Path:
    import roqsim_quadruped

    return Path(roqsim_quadruped.__file__).parent / "policy" / "spot_policy.pt"


def _static(*components: dict, **sim) -> Case:
    return Case(lambda _tmp: _world(*components, **sim))


CASES: dict[str, Case] = {
    # core
    "dummy": _static({"dummy": {}}),
    "ceiling": _static({"ceiling": {}}),
    "spawn_model": _static(
        {
            "spawn_model": {
                "model": "graspable_box",
                "prefix": "b_",
                "pose": {"position": {"x": 0.0, "y": 0.0, "z": 0.5}},
            }
        }
    ),
    "contact_monitor": Case(lambda _: _mobile({"contact_monitor": {}})),
    "contact_location": Case(lambda _: _mobile({"contact_location": {}})),
    "contact_impulse": Case(lambda _: _mobile({"contact_impulse": {}})),
    "clearance_monitor": Case(lambda _: _mobile({"clearance_monitor": {}})),
    "upright_monitor": Case(lambda _: _mobile({"upright_monitor": {}})),
    "energy_monitor": Case(lambda _: _mobile({"energy_monitor": {}})),
    "joint_state_publisher": Case(lambda _: _mobile({"joint_state_publisher": {}})),
    "pose_publisher": Case(lambda _: _world(_robot("turtlebot4"))),
    "bumper": Case(lambda _: _mobile({"bumper": {"zones": {"front": [-0.8, 0.8]}}})),
    "payload": Case(lambda _: _mobile({"payload": {"mass": 0.5}})),
    "attachment": Case(
        lambda _: _world(
            # The load first: the weld names both bodies at build, so the load must be built already.
            {
                "spawn_model": {
                    "model": "graspable_box",
                    "prefix": "b_",
                    "motion": "physics",
                    "pose": {"position": {"x": 0.4, "y": 0.0, "z": 0.1}},
                }
            },
            _robot("makerspet_mini", {"attachment": {"body": "graspable_box"}}),
        ),
    ),
    "model_override": Case(
        lambda _: _world(
            _box(),
            {
                "model_override": {
                    "overrides": [{"field": "geom_friction", "select": ["box"], "to": 0.1}]
                }
            },
        ),
        use=(_override,),
    ),
    "contact_pair_override": _static(
        _box(),
        {"contact_pair_override": {"a": {"geom": "box"}, "b": {"geom": "floor"}, "friction": 0.4}},
    ),
    "flex_material": Case(_flex_world),
    "heightfield": _static({"heightfield": {"size": [4.0, 4.0], "resolution": 32, "seed": 3}}),
    # mobile
    "diff_drive": Case(lambda _: _mobile()),
    "spawn_robot": Case(lambda _: _mobile()),
    "omni_drive": Case(lambda _: _world(_robot("lgdxrobot2"))),
    "floorplan": _static(
        {"floorplan": {"lines": [{"id": 0, "x0_m": 2.0, "y0_m": -2.0, "x1_m": 2.0, "y1_m": 2.0}]}}
    ),
    "ackermann_drive": Case(lambda _: _world(_robot("piracer"))),
    "tricycle_drive": Case(_tricycle_world),
    # navigation and people
    "navigator": Case(lambda _: _mobile({"navigator": {"speed": 0.3, "goals": [[1.0, 0.0]]}})),
    "walker": _static(
        {
            "walker": {"walker": "MaleVisitorWalk", "waypoints": [[-1.0, 0.0], [1.0, 0.0]]},
            "name": "ped",
        }
    ),
    # manipulation
    "spawn_arm": Case(lambda _: _arm()),
    "arm_controller": Case(lambda _: _arm(), use=(_switch_every_controller,)),
    "force_torque": Case(lambda _: _arm()),
    "cartesian_admittance": Case(
        lambda _: _arm({"cartesian_admittance": {"site": "tool_site", "ft": "ft"}}),
        use=(_switch_every_controller,),
    ),
    "force_limit": Case(
        # Low enough to trip within the trial, which is what a trial does with it.
        lambda _: _arm({"force_limit": {"ft": "ft", "max_force": 0.001}, "name": "safety"}),
    ),
    # sensors
    "lidar": Case(lambda _: _mobile()),
    "range_sensor": Case(lambda _: _world(_robot("turtlebot4"))),
    "imu": Case(lambda _: _mobile({"imu": {}})),
    "gnss": Case(lambda _: _mobile({"gnss": {"datum": {"lat": 47.4, "lon": 8.5, "alt": 400.0}}})),
    "spawn_sensor": Case(lambda _: _sensor("lds01")),
    "livox_mid360": Case(lambda _: _sensor("mid360")),
    "seyond_robin_w1g": Case(lambda _: _mounted("robin_w1g", {"seyond_robin_w1g": {}})),
    "oakd_camera": Case(lambda _: _sensor("oakd_pro")),
    "realsense_d415": Case(lambda _: _sensor("realsense_d415")),
    "realsense_d435": Case(lambda _: _sensor("realsense_d435")),
    "realsense_d455": Case(lambda _: _sensor("realsense_d455")),
    "zivid": Case(lambda _: _mounted("zivid", {"zivid": {}})),
    "fiducial_marker": _static(
        {
            "fiducial_marker": {
                "family": "apriltag_36h11",
                "id": 0,
                "size": 0.12,
                "pose": {"position": {"z": 0.5}},
            }
        }
    ),
    "object_detector": Case(
        lambda _: _world(
            _robot(
                "makerspet_mini",
                {
                    "object_detector": {
                        "frame": "base_link",
                        "objects": [{"body": "box", "class_id": "crate"}],
                    }
                },
            ),
            _box(),
        )
    ),
    "segmentation_camera": Case(
        lambda _: _mounted(
            "seg",
            {
                "segmentation_camera": {
                    "camera": "seg_color",
                    "classes": [{"class_id": 1, "name": "crate", "bodies": ["box"]}],
                }
            },
        )
    ),
    "sensor_coverage_probe": Case(
        lambda tmp: (
            _sensor("realsense_d435")
            | {
                "components": [
                    *_sensor("realsense_d435")["components"],
                    {
                        "sensor_coverage_probe": {
                            "sample": {
                                "volume": True,
                                "objects": False,
                                "resolution": 0.5,
                                "heights": [0.5],
                            },
                            "out": str(tmp / "coverage"),
                            "render": "none",
                        }
                    },
                ]
            }
        )
    ),
    # props
    "box": _static(_box()),
    "boxes": _static(
        {
            "boxes": {
                "instances": [
                    {"pose": {"position": {"x": 1.0, "y": 0.0, "z": 0.0}}, "size": [0.2, 0.2, 0.2]}
                ]
            }
        }
    ),
    "cylinder": _static(
        {
            "cylinder": {
                "pose": {"position": {"x": 1.0, "y": 0.0, "z": 0.0}},
                "radius": 0.1,
                "height": 0.3,
            }
        }
    ),
    "cylinders": _static(
        {
            "cylinders": {
                "instances": [
                    {
                        "pose": {"position": {"x": 1.0, "y": 0.0, "z": 0.0}},
                        "radius": 0.1,
                        "height": 0.3,
                    }
                ]
            }
        }
    ),
    "moving_box": _static(
        {
            "moving_box": {
                "pose": {"position": {"x": 0.0, "y": 0.0}},
                "size": [0.2, 0.2, 0.2],
                "speed": 0.5,
                "waypoints": [[1.0, 0.0]],
            }
        }
    ),
    "prop_trajectory": Case(_trajectory_world),
    "conveyor": _static({"conveyor": {}, "name": "conveyor"}),
    "door": _static({"door": {}, "name": "door"}),
    "window": _static({"window": {}, "name": "window"}),
    "shelf": _static({"shelf": {}, "name": "shelf"}),
    "workbench": _static({"workbench": {}, "name": "bench"}),
    "palm_tree": _static({"palm_tree": {}, "name": "palm"}),
    "duct": _static({"duct": {"prefix": "d_", "start": [1.0, 1.0], "end": [3.0, 1.0], "z": 3.2}}),
    "strip_light": _static(
        {"strip_light": {"prefix": "s_", "pose": {"position": {"x": 2.0, "y": 3.0, "z": 3.5}}}}
    ),
    "ceiling_panels": _static(
        {"ceiling_panels": {"prefix": "p_", "area": [0.0, 0.0, 2.0, 2.0], "z": 3.5}}
    ),
    # aerial
    "quadrotor_controller": Case(
        lambda _: _world(_robot("crazyflie_2", name="drone"), density=1.225, viscosity=1.8e-5)
    ),
    "multirotor_motors": Case(lambda _: _world(_robot("x500", name="drone"))),
    "wind_field": _static({"wind_field": {}}, density=1.225, viscosity=1.8e-5),
    # legged
    "g1_locomotion": Case(lambda _: _world(_robot("unitree_g1", name="robot"))),
    "oli_locomotion": Case(
        lambda _: _world(_robot("oli", {"oli_locomotion": {}}, name="robot"), timestep=0.001)
    ),
    "agibot_g2_controller": Case(lambda _: _world(_robot("agibot_g2", name="robot"))),
    "spot_locomotion": Case(
        lambda _: _world(_robot("spot", name="robot")),
        unavailable=_needs_file(_spot_policy, "the Spot policy (fetched, not shipped)"),
    ),
}

_RAY_SENSORS = ("lidar", "range_sensor", "livox_mid360", "seyond_robin_w1g")
_CAMERAS = ("realsense_d415", "segmentation_camera")
_DEPTH_CAMERAS = ("oakd_camera", "realsense_d435", "realsense_d455", "zivid")
_CAST_BUFFERS = "the cast's output buffers, rewritten by every cast before anything reads them"
_RENDERER = "the renderer, made at the first capture and kept for the run"
_DEPTH_MASK = (
    "the last depth frame's invalid pixels, read only with that frame, which the reset clears"
)
_CONTROLLERS = ("arm_controller", "cartesian_admittance")
_TRANSITIONS = (
    "the registry's transition log, which a bridge announces by position, so it runs for the "
    "whole process"
)
_MEASURED_ONCE = (
    "the footprint radius, measured from the model's geometry once and kept for the run"
)

#: Plugins, or attribute paths of a plugin, that keep state across a reset on purpose.
EXEMPT: dict[str, dict[str | None, str]] = {
    "px4_sitl": {
        None: "an autopilot in another process does not reset with the simulation, and the bridge "
        "keeps its link, arming and last controls across the episode boundary to match it",
    },
    **{name: {"_registered.transitions": _TRANSITIONS} for name in _CONTROLLERS},
    **{
        name: {"_obs": "the policy's input buffer, rebuilt in full before every policy call"}
        for name in ("g1_locomotion", "spot_locomotion")
    },
    "navigator": {
        "_radius": _MEASURED_ONCE,
        "_caution._radius": _MEASURED_ONCE,
    },
    **{name: {"_hits": _CAST_BUFFERS} for name in _RAY_SENSORS},
    **{name: {"_frames": _RENDERER} for name in _CAMERAS},
    **{name: {"_frames": _RENDERER, "_invalid": _DEPTH_MASK} for name in _DEPTH_CAMERAS},
}

#: Plugins with no world here to configure them in.
SKIPPED: dict[str, str] = {
    **{
        name: "a transport to a ROS 2 graph, which a unit test does not have; it declares itself "
        "transport_only, holding no simulation state"
        for name in ("ros2_bridge", "sim_interfaces")
    },
    "ipc_bridge": "a transport that binds a control socket and starts threads; it declares itself "
    "transport_only, holding no simulation state",
    "run_control": "the driver's pause/step/reset served as endpoints: its state is the driver's "
    "RunControl, and a reset is one of its commands (test_ipc_bridge drives it)",
    "entity_control": "entity placement and presence served as commands; it holds no state of its "
    "own (test_entity_control drives it)",
}

# -- what a plugin's state is ------------------------------------------------------------------------

#: Not the plugin's own: shared with the world, or not state at all.
_SHARED = (
    SimContext,
    Engine,
    Plugin,
    Blackboard,
    EntityRegistry,
    InterfaceRegistry,
    Entity,
    ControllerRegistry,
    logging.Logger,
    logging.LoggerAdapter,
    queue.Queue,
    type(threading.Lock()),
    type(threading.RLock()),
    threading.Thread,
    threading.Event,
    threading.Condition,
    types.ModuleType,
    types.FunctionType,
    types.MethodType,
    types.BuiltinFunctionType,
    types.BuiltinMethodType,
    functools.partial,
    type,
)


def _shared(value) -> bool:
    return isinstance(value, _SHARED) or type(value).__module__.split(".")[0] == "mujoco"


def _ours(value) -> bool:
    """Whether *value*'s insides are ours to compare: a roqsim type or a dataclass."""
    return type(value).__module__.startswith(("roqsim", "scenario_execution_roqsim")) or (
        dataclasses.is_dataclass(value)
    )


def _flatten(value, path: str, out: dict, seen: set[int]) -> None:
    """Every leaf of *value* the plugin owns, as ``out[path] = comparable``."""
    if value is None or isinstance(value, (bool, int, float, complex, str, bytes)):
        out[path] = value
        return
    if isinstance(value, np.generic):
        out[path] = value.item()
        return
    if isinstance(value, np.ndarray):
        out[path] = (value.dtype.str, value.shape, value.tobytes())
        return
    if _shared(value):
        return
    if id(value) in seen:
        return
    seen.add(id(value))
    if isinstance(value, np.random.Generator):
        _flatten(value.bit_generator.state, f"{path}.<rng>", out, seen)
    elif isinstance(value, dict):
        out[f"{path}.<keys>"] = sorted(repr(k) for k in value)
        for key, item in value.items():
            _flatten(item, f"{path}[{key!r}]", out, seen)
    elif isinstance(value, (list, tuple)) or type(value).__name__ == "deque":
        out[f"{path}.<len>"] = len(value)
        for i, item in enumerate(value):
            _flatten(item, f"{path}[{i}]", out, seen)
    elif isinstance(value, (set, frozenset)):
        out[path] = sorted(repr(v) for v in value)
    elif callable(value) and not hasattr(value, "__dict__"):
        return
    elif not _ours(value):
        out[path] = type(value).__qualname__
    else:
        attrs = dict(getattr(value, "__dict__", {}))
        for slot in getattr(type(value), "__slots__", ()):
            if hasattr(value, slot):
                attrs[slot] = getattr(value, slot)
        if not attrs:
            out[path] = type(value).__qualname__
        for key, item in attrs.items():
            _flatten(item, f"{path}.{key}", out, seen)


def _state(engine: Engine, name: str) -> dict:
    """The per-trial state a reset of *name*'s instances must restore."""
    out: dict = {}
    for i, plugin in enumerate(_plugins(engine, name)):
        root = f"{plugin.address}#{i}"
        for key, value in vars(plugin).items():
            _flatten(value, f"{root}.{key}", out, {id(plugin)})
    out["ctx.stop_requested"] = engine.ctx.stop_requested
    out["ctx.stop_reason"] = engine.ctx.stop_reason
    registry = engine.ctx.blackboard.get(SERVICE_KEY)
    for controller in registry.all() if registry is not None else ():
        out[f"controllers[{controller.namespace}/{controller.owner}/{controller.name}]"] = (
            controller.state
        )
    return out


def _exempted(name: str, path: str) -> bool:
    if "#" not in path:
        return False
    attr_path = path.split("#", 1)[1].split(".", 1)[1]
    return any(
        attr is not None and (attr_path == attr or attr_path.startswith((f"{attr}.", f"{attr}[")))
        for attr in EXEMPT.get(name, {})
    )


def _differences(name: str, before: dict, after: dict) -> list[str]:
    lines = []
    for path in sorted(set(before) | set(after)):
        if _exempted(name, path):
            continue
        a, b = before.get(path, "<absent>"), after.get(path, "<absent>")
        if not _equal(a, b):
            lines.append(f"  {path}: {_show(a)} -> {_show(b)}")
    return lines


def _equal(a, b) -> bool:
    if isinstance(a, float) and isinstance(b, float) and np.isnan(a) and np.isnan(b):
        return True
    return type(a) is type(b) and a == b


def _show(value) -> str:
    if isinstance(value, tuple) and len(value) == 3 and isinstance(value[2], bytes):
        array = np.frombuffer(value[2], dtype=np.dtype(value[0])).reshape(value[1])
        return np.array2string(array, threshold=8, precision=4)
    text = repr(value)
    return text if len(text) < 80 else text[:77] + "..."


# -- the test ----------------------------------------------------------------------------------------


@functools.cache
def _entry_point_classes() -> dict[str, type]:
    return {ep.name: ep.load() for ep in entry_points(group="roqsim.plugins")}


@functools.cache
def _registered() -> list[str]:
    return sorted({ep.name for ep in entry_points(group="roqsim.plugins")})


def test_every_registered_plugin_is_checked_exempted_or_skipped():
    whole = {name for name, attrs in EXEMPT.items() if None in attrs}
    listed = [set(CASES), whole, set(SKIPPED)]
    missing = [n for n in _registered() if not any(n in s for s in listed)]
    twice = [n for n in _registered() if sum(n in s for s in listed) > 1]
    assert not missing, f"no reset case, exemption or skip reason for: {missing}"
    assert not twice, f"listed more than once: {twice}"


def _params():
    for name in _registered():
        if name not in CASES:
            continue
        marks = []
        if defect := CASES[name].defect:
            marks.append(pytest.mark.xfail(reason=defect, raises=AssertionError, strict=True))
        yield pytest.param(name, id=name, marks=marks)


@pytest.mark.parametrize("name", list(_params()))
def test_a_used_plugin_resets_to_its_configured_state(name, tmp_path):
    case = CASES[name]
    if reason := case.unavailable():
        pytest.skip(reason)
    engine = Engine(load_config_from_dict(case.world(tmp_path), base_dir=tmp_path))
    engine.ctx.seed = 0
    engine.setup()
    try:
        if not _plugins(engine, name):
            pytest.fail(f"the world for {name!r} does not contain it")
        engine.reset()
        configured = _state(engine, name)

        _write_every_in_endpoint(engine)
        for _ in range(STEPS // 2):
            engine.step()
        for use in case.use:
            use(engine)
        for _ in range(STEPS // 2):
            engine.step()

        engine.ctx.episode -= 1
        engine.reset()
        used = _state(engine, name)
    finally:
        engine.shutdown()

    differences = _differences(name, configured, used)
    assert not differences, (
        f"{name}: a reset after a used trial does not restore the configured state:\n"
        + "\n".join(differences[:25])
    )
