"""The Franka Control Interface (FCI) messages, byte for byte, at protocol version 10.

The layouts are libfranka-common's (``include/research_interface/robot/{service_types,rbk_types,
error}.h``, ``#pragma pack(1)``, little-endian), at the commit ``tests/upstream/franka/SOURCES.toml``
pins; ``tests/test_franka_protocol.py`` parses those headers and checks every table here against them.

- TCP (port :data:`COMMAND_PORT`): a :data:`HEADER` (command, command id, total size) and a fixed
  payload per command, answered by the same header and a status byte (and, for ``Connect``, the
  version; for ``GetRobotModel``, the URDF).
- UDP: :data:`ROBOT_STATE` from the robot every millisecond, :data:`ROBOT_COMMAND` back while a
  motion runs, each carrying the ``message_id`` of the state it answers.
"""

from __future__ import annotations

import struct
from enum import IntEnum

#: ``service_types.h``.
VERSION = 10
COMMAND_PORT = 1337


class Command(IntEnum):
    CONNECT = 0
    MOVE = 1
    STOP_MOVE = 2
    SET_COLLISION_BEHAVIOR = 3
    SET_JOINT_IMPEDANCE = 4
    SET_CARTESIAN_IMPEDANCE = 5
    SET_GUIDING_MODE = 6
    SET_EE_TO_K = 7
    SET_NE_TO_EE = 8
    SET_LOAD = 9
    AUTOMATIC_ERROR_RECOVERY = 10
    GET_ROBOT_MODEL = 11


class ConnectStatus(IntEnum):
    SUCCESS = 0
    INCOMPATIBLE_LIBRARY_VERSION = 1


class MoveStatus(IntEnum):
    SUCCESS = 0
    MOTION_STARTED = 1
    PREEMPTED = 2
    PREEMPTED_DUE_TO_ACTIVATED_SAFETY_FUNCTIONS = 3
    COMMAND_REJECTED_DUE_TO_ACTIVATED_SAFETY_FUNCTIONS = 4
    COMMAND_NOT_POSSIBLE_REJECTED = 5
    START_AT_SINGULAR_POSE_REJECTED = 6
    INVALID_ARGUMENT_REJECTED = 7
    REFLEX_ABORTED = 8
    EMERGENCY_ABORTED = 9
    INPUT_ERROR_ABORTED = 10
    ABORTED = 11


class GetterSetterStatus(IntEnum):
    SUCCESS = 0
    COMMAND_NOT_POSSIBLE_REJECTED = 1
    INVALID_ARGUMENT_REJECTED = 2
    COMMAND_REJECTED_DUE_TO_ACTIVATED_SAFETY_FUNCTIONS = 3


class CommandStatus(IntEnum):
    """``CommandBase::Status``: ``StopMove``'s first three and ``GetRobotModel``'s."""

    SUCCESS = 0
    COMMAND_NOT_POSSIBLE_REJECTED = 1
    COMMAND_REJECTED_DUE_TO_ACTIVATED_SAFETY_FUNCTIONS = 2


class StopMoveStatus(IntEnum):
    SUCCESS = 0
    COMMAND_NOT_POSSIBLE_REJECTED = 1
    COMMAND_REJECTED_DUE_TO_ACTIVATED_SAFETY_FUNCTIONS = 2
    EMERGENCY_ABORTED = 3
    REFLEX_ABORTED = 4
    ABORTED = 5


class AutomaticErrorRecoveryStatus(IntEnum):
    SUCCESS = 0
    COMMAND_NOT_POSSIBLE_REJECTED = 1
    COMMAND_REJECTED_DUE_TO_ACTIVATED_SAFETY_FUNCTIONS = 2
    MANUAL_ERROR_RECOVERY_REQUIRED_REJECTED = 3
    REFLEX_ABORTED = 4
    EMERGENCY_ABORTED = 5
    ABORTED = 6


class MoveControllerMode(IntEnum):
    """``Move::ControllerMode`` (uint32 in the request)."""

    JOINT_IMPEDANCE = 0
    CARTESIAN_IMPEDANCE = 1
    EXTERNAL_CONTROLLER = 2


class MoveMotionGeneratorMode(IntEnum):
    """``Move::MotionGeneratorMode`` (uint32 in the request)."""

    JOINT_POSITION = 0
    JOINT_VELOCITY = 1
    CARTESIAN_POSITION = 2
    CARTESIAN_VELOCITY = 3
    NONE = 4


class MotionGeneratorMode(IntEnum):
    """``rbk_types.h`` ``MotionGeneratorMode`` (uint8 in the state)."""

    IDLE = 0
    JOINT_POSITION = 1
    JOINT_VELOCITY = 2
    CARTESIAN_POSITION = 3
    CARTESIAN_VELOCITY = 4
    NONE = 5


class ControllerMode(IntEnum):
    """``rbk_types.h`` ``ControllerMode`` (uint8 in the state)."""

    JOINT_IMPEDANCE = 0
    CARTESIAN_IMPEDANCE = 1
    EXTERNAL_CONTROLLER = 2
    OTHER = 3


class RobotMode(IntEnum):
    OTHER = 0
    IDLE = 1
    MOVE = 2
    GUIDING = 3
    REFLEX = 4
    USER_STOPPED = 5
    AUTOMATIC_ERROR_RECOVERY = 6


#: ``error.h`` ``enum class Error``, in order: the index of each flag in ``errors`` and
#: ``reflex_reason``, under libfranka's ``Errors`` field names (``include/franka/errors.h``).
ERRORS = (
    "joint_position_limits_violation",
    "cartesian_position_limits_violation",
    "self_collision_avoidance_violation",
    "joint_velocity_violation",
    "cartesian_velocity_violation",
    "force_control_safety_violation",
    "joint_reflex",
    "cartesian_reflex",
    "max_goal_pose_deviation_violation",
    "max_path_pose_deviation_violation",
    "cartesian_velocity_profile_safety_violation",
    "joint_position_motion_generator_start_pose_invalid",
    "joint_motion_generator_position_limits_violation",
    "joint_motion_generator_velocity_limits_violation",
    "joint_motion_generator_velocity_discontinuity",
    "joint_motion_generator_acceleration_discontinuity",
    "cartesian_position_motion_generator_start_pose_invalid",
    "cartesian_motion_generator_elbow_limit_violation",
    "cartesian_motion_generator_velocity_limits_violation",
    "cartesian_motion_generator_velocity_discontinuity",
    "cartesian_motion_generator_acceleration_discontinuity",
    "cartesian_motion_generator_elbow_sign_inconsistent",
    "cartesian_motion_generator_start_elbow_invalid",
    "force_controller_desired_force_tolerance_violation",
    "start_elbow_sign_inconsistent",
    "communication_constraints_violation",
    "power_limit_violation",
    "cartesian_motion_generator_joint_position_limits_violation",
    "cartesian_motion_generator_joint_velocity_limits_violation",
    "cartesian_motion_generator_joint_velocity_discontinuity",
    "cartesian_motion_generator_joint_acceleration_discontinuity",
    "cartesian_position_motion_generator_invalid_frame",
    "controller_torque_discontinuity",
    "joint_p2p_insufficient_torque_for_planning",
    "tau_j_range_violation",
    "instability_detected",
    "joint_move_in_wrong_direction",
    "cartesian_spline_motion_generator_violation",
    "joint_via_motion_generator_planning_joint_limit_violation",
    "base_acceleration_initialization_timeout",
    "base_acceleration_invalid_reading",
)
ERROR_INDEX = {name: i for i, name in enumerate(ERRORS)}

#: ``CommandHeader``: command, command_id, size (header included).
HEADER = struct.Struct("<III")

#: ``rbk_types.h`` ``RobotState``: (field, struct code, count), in order. Arrays are float32 since
#: protocol version 10; ``accelerometer_*`` is 6 x 3.
ROBOT_STATE_FIELDS: tuple[tuple[str, str, int], ...] = (
    ("message_id", "Q", 1),
    ("O_T_EE", "f", 16),
    ("O_T_EE_d", "f", 16),
    ("F_T_EE", "f", 16),
    ("EE_T_K", "f", 16),
    ("F_T_NE", "f", 16),
    ("NE_T_EE", "f", 16),
    ("m_ee", "f", 1),
    ("I_ee", "f", 9),
    ("F_x_Cee", "f", 3),
    ("m_load", "f", 1),
    ("I_load", "f", 9),
    ("F_x_Cload", "f", 3),
    ("elbow", "f", 2),
    ("elbow_d", "f", 2),
    ("tau_J", "f", 7),
    ("tau_J_d", "f", 7),
    ("dtau_J", "f", 7),
    ("q", "f", 7),
    ("q_d", "f", 7),
    ("dq", "f", 7),
    ("dq_d", "f", 7),
    ("ddq_d", "f", 7),
    ("joint_contact", "f", 7),
    ("cartesian_contact", "f", 6),
    ("joint_collision", "f", 7),
    ("cartesian_collision", "f", 6),
    ("tau_ext_hat_filtered", "f", 7),
    ("O_F_ext_hat_K", "f", 6),
    ("K_F_ext_hat_K", "f", 6),
    ("O_dP_EE_d", "f", 6),
    ("O_ddP_O", "f", 3),
    ("elbow_c", "f", 2),
    ("delbow_c", "f", 2),
    ("ddelbow_c", "f", 2),
    ("O_T_EE_c", "f", 16),
    ("O_dP_EE_c", "f", 6),
    ("O_ddP_EE_c", "f", 6),
    ("theta", "f", 7),
    ("dtheta", "f", 7),
    ("accelerometer_top", "f", 18),
    ("accelerometer_bottom", "f", 18),
    ("motion_generator_mode", "B", 1),
    ("controller_mode", "B", 1),
    ("errors", "?", 41),
    ("reflex_reason", "?", 41),
    ("robot_mode", "B", 1),
    ("control_command_success_rate", "f", 1),
)

#: ``rbk_types.h`` ``RobotCommand`` = message_id + ``MotionGeneratorCommand`` + ``ControllerCommand``.
ROBOT_COMMAND_FIELDS: tuple[tuple[str, str, int], ...] = (
    ("message_id", "Q", 1),
    ("q_c", "d", 7),
    ("dq_c", "d", 7),
    ("O_T_EE_c", "d", 16),
    ("O_dP_EE_c", "d", 6),
    ("elbow_c", "d", 2),
    ("valid_elbow", "?", 1),
    ("motion_generation_finished", "?", 1),
    ("tau_J_d", "d", 7),
    ("torque_command_finished", "?", 1),
)

#: ``service_types.h`` request payloads (after the header).
CONNECT_REQUEST = struct.Struct("<HH")  # version, udp_port
CONNECT_RESPONSE = struct.Struct("<BH")  # status, version
MOVE_REQUEST = struct.Struct(
    "<II3d3d?7d"
)  # controller, generator, path dev, goal dev, async, max vel
SET_COLLISION_BEHAVIOR_REQUEST = struct.Struct("<7d7d7d7d6d6d6d6d")
SET_JOINT_IMPEDANCE_REQUEST = struct.Struct("<7d")
SET_CARTESIAN_IMPEDANCE_REQUEST = struct.Struct("<6d")
SET_GUIDING_MODE_REQUEST = struct.Struct("<6??")
SET_EE_TO_K_REQUEST = struct.Struct("<16d")
SET_NE_TO_EE_REQUEST = struct.Struct("<16d")
SET_LOAD_REQUEST = struct.Struct("<d3d9d")
STATUS_RESPONSE = struct.Struct("<B")

#: The fixed request payload of each command (``None``: no payload).
REQUESTS: dict[Command, struct.Struct | None] = {
    Command.CONNECT: CONNECT_REQUEST,
    Command.MOVE: MOVE_REQUEST,
    Command.STOP_MOVE: None,
    Command.SET_COLLISION_BEHAVIOR: SET_COLLISION_BEHAVIOR_REQUEST,
    Command.SET_JOINT_IMPEDANCE: SET_JOINT_IMPEDANCE_REQUEST,
    Command.SET_CARTESIAN_IMPEDANCE: SET_CARTESIAN_IMPEDANCE_REQUEST,
    Command.SET_GUIDING_MODE: SET_GUIDING_MODE_REQUEST,
    Command.SET_EE_TO_K: SET_EE_TO_K_REQUEST,
    Command.SET_NE_TO_EE: SET_NE_TO_EE_REQUEST,
    Command.SET_LOAD: SET_LOAD_REQUEST,
    Command.AUTOMATIC_ERROR_RECOVERY: None,
    Command.GET_ROBOT_MODEL: None,
}


def _struct(fields) -> struct.Struct:
    return struct.Struct("<" + "".join(f"{n}{code}" for _, code, n in fields))


ROBOT_STATE = _struct(ROBOT_STATE_FIELDS)
ROBOT_COMMAND = _struct(ROBOT_COMMAND_FIELDS)


def pack_robot_state(state: dict) -> bytes:
    """``RobotState`` bytes from a dict of its fields (arrays as sequences, scalars as numbers)."""
    values: list = []
    for name, code, n in ROBOT_STATE_FIELDS:
        v = state[name]
        if n == 1:
            values.append(int(v) if code in "QB" else bool(v) if code == "?" else float(v))
        else:
            seq = list(v)
            if len(seq) != n:
                raise ValueError(f"RobotState.{name}: {len(seq)} values, the protocol has {n}")
            if code == "?":
                values.extend(bool(x) for x in seq)
            else:
                values.extend(float(x) for x in seq)
    return ROBOT_STATE.pack(*values)


def _unpack(fields, raw: tuple) -> dict:
    out, i = {}, 0
    for name, _, n in fields:
        out[name] = raw[i] if n == 1 else list(raw[i : i + n])
        i += n
    return out


def unpack_robot_state(data: bytes) -> dict:
    return _unpack(ROBOT_STATE_FIELDS, ROBOT_STATE.unpack(data))


def unpack_robot_command(data: bytes) -> dict:
    return _unpack(ROBOT_COMMAND_FIELDS, ROBOT_COMMAND.unpack(data))


def pack_robot_command(command: dict) -> bytes:
    values: list = []
    for name, code, n in ROBOT_COMMAND_FIELDS:
        v = command.get(name, 0 if n == 1 else [0] * n)
        if n == 1:
            values.append(int(v) if code == "Q" else bool(v) if code == "?" else float(v))
        else:
            values.extend(float(x) for x in v)
    return ROBOT_COMMAND.pack(*values)
