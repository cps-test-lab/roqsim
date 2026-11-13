"""The simulated FCI's wire format is libfranka-common's, at the pinned commit.

Every table in ``roqsim_manipulation.franka.protocol`` is checked against the headers it was taken
from (``upstream/franka/libfranka-common``): field order, types and counts of the UDP structs, the
command and status enums, the version and the port, the request payloads, and the order of the 41
error flags on the wire (``error.h``) under libfranka's names (``src/errors.cpp``).
"""

from __future__ import annotations

import re

import pytest
from _franka_upstream import (
    COMMON,
    LIBFRANKA,
    data_members,
    enum_values,
    strip_comments,
    struct_body,
    text,
)
from roqsim_manipulation.franka import protocol as P

RBK = strip_comments(text(COMMON / "rbk_types.h"))
SERVICE = strip_comments(text(COMMON / "service_types.h"))


def test_version_and_port():
    assert int(re.search(r"kVersion\s*=\s*(\d+)", SERVICE).group(1)) == P.VERSION
    assert int(re.search(r"kCommandPort\s*=\s*(\d+)", SERVICE).group(1)) == P.COMMAND_PORT


@pytest.mark.parametrize(
    "struct, ours",
    [
        ("RobotState", P.ROBOT_STATE_FIELDS),
        ("MotionGeneratorCommand", P.ROBOT_COMMAND_FIELDS[1:8]),
        ("ControllerCommand", P.ROBOT_COMMAND_FIELDS[8:]),
    ],
)
def test_udp_structs_match_the_header(struct, ours):
    assert data_members(struct_body(RBK, struct)) == list(ours)


def test_robot_command_is_message_id_motion_control():
    members = data_members(struct_body(RBK, "RobotCommand"))
    assert [m[0] for m in members] == ["message_id"]  # motion and control are nested structs
    assert re.search(
        r"uint64_t message_id;\s*MotionGeneratorCommand motion;\s*ControllerCommand control;",
        " ".join(struct_body(RBK, "RobotCommand").split()),
    )


@pytest.mark.parametrize(
    "enum, ours",
    [
        ("MotionGeneratorMode", P.MotionGeneratorMode),
        ("ControllerMode", P.ControllerMode),
        ("RobotMode", P.RobotMode),
    ],
)
def test_state_enums(enum, ours):
    values = enum_values(RBK, enum)
    assert len(values) == len(ours)
    for i, v in enumerate(values):
        assert ours(i).name.replace("_", "").lower() == v[1:].lower(), (enum, v)


def test_command_enum():
    values = enum_values(SERVICE, "Command")
    assert [v[1:].lower() for v in values] == [c.name.replace("_", "").lower() for c in P.Command]


def _status(struct: str) -> list[str]:
    body = struct_body(SERVICE, struct)
    return [v[1:].lower() for v in enum_values(body, "Status")]


@pytest.mark.parametrize(
    "struct, ours",
    [
        ("Connect", P.ConnectStatus),
        ("Move", P.MoveStatus),
        ("StopMove", P.StopMoveStatus),
        ("AutomaticErrorRecovery", P.AutomaticErrorRecoveryStatus),
    ],
)
def test_status_enums(struct, ours):
    assert _status(struct) == [s.name.replace("_", "").lower() for s in ours]


def test_getter_setter_status():
    body = struct_body(SERVICE, "GetterSetterCommandBase")
    assert [v[1:].lower() for v in enum_values(body, "Status")] == [
        s.name.replace("_", "").lower() for s in P.GetterSetterStatus
    ]


def test_move_modes():
    body = struct_body(SERVICE, "Move")
    assert [v[1:].lower() for v in enum_values(body, "ControllerMode")] == [
        s.name.replace("_", "").lower() for s in P.MoveControllerMode
    ]
    assert [v[1:].lower() for v in enum_values(body, "MotionGeneratorMode")] == [
        s.name.replace("_", "").lower() for s in P.MoveMotionGeneratorMode
    ]


def _request_size(struct: str) -> int:
    body = struct_body(struct_body(SERVICE, struct), "Request")
    size = 0
    sizes = {"Q": 8, "d": 8, "f": 4, "?": 1, "B": 1, "H": 2, "I": 4}
    for name, code, n in data_members(body):
        if name in ("controller_mode", "motion_generator_mode"):
            continue  # Move's are uint32 enums, counted below
        size += sizes[code] * n
    # members data_members cannot type: Version (uint16), the uint32 Move enums, Deviation (3 doubles)
    flat = " ".join(body.split())
    size += 2 * len(re.findall(r"\bVersion version;|uint16_t udp_port;", flat))
    size += 4 * len(
        re.findall(
            r"\bControllerMode controller_mode;|\bMotionGeneratorMode motion_generator_mode;", flat
        )
    )
    size += 24 * len(re.findall(r"\bDeviation \w+;", flat))
    return size


@pytest.mark.parametrize("command", [c for c in P.REQUESTS if P.REQUESTS[c] is not None])
def test_request_payload_sizes(command):
    name = "".join(w.capitalize() for w in command.name.lower().split("_"))
    name = {"SetEeToK": "SetEEToK", "SetNeToEe": "SetNEToEE"}.get(name, name)
    assert P.REQUESTS[command].size == _request_size(name), name


def test_errors_are_the_wire_order_under_libfrankas_names():
    enum = [v for v in enum_values(strip_comments(text(COMMON / "error.h")), "Error")]
    pairs = re.findall(
        r"([a-z_0-9]+)\(\s*errors_\[\s*static_cast<size_t>\(\s*Error::(k\w+)\s*\)\s*\]",
        text(LIBFRANKA / "src/errors.cpp"),
    )
    name_of = {k: n for n, k in pairs}
    assert len(enum) == len(P.ERRORS) == 41
    assert [name_of[k] for k in enum] == list(P.ERRORS)


def test_state_round_trips():
    state = {name: (7 if n == 1 else [0.5] * n) for name, code, n in P.ROBOT_STATE_FIELDS}
    state["errors"] = [False] * 41
    state["reflex_reason"] = [True] + [False] * 40
    back = P.unpack_robot_state(P.pack_robot_state(state))
    assert back["message_id"] == 7 and back["reflex_reason"][0] is True and back["q"] == [0.5] * 7
