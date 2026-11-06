# Copyright (C) 2026 Frederik Pasch
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""ROS messages as plain mappings, for the vendor interfaces whose messages nest.

A driver interface here publishes and serves message types this package does not import --
``fanuc_msgs`` among them -- whose fields are nested messages and sequences of them, which the
bridge's by-field-name mapping does not fill. So the plugin trades in plain mappings
(``{"values": [{"io_type": {"type": "DO"}, "index": 101, "value": True}]}``) and this module turns
them into the message and back, reading each field's type off the message class itself
(``get_fields_and_field_types``, which every rosidl Python message has). No ROS import: the bridge
hands a message in, and a nested message class is resolved by name only when one is needed.

Registered with the bridge through ``roqsim_manipulation.ros2_services`` (the
``roqsim_ros_bridge.extensions`` entry point):

- a converter (out) and a decoder (in) for each message type in :data:`MAPPED_MESSAGES`;
- one service handler for each service type in :data:`MAPPED_SERVICES`: the request goes to the
  endpoint's ``write`` as a mapping, and the mapping it returns fills the response. A write that
  raises, or that the physics thread does not run in time, answers ``result: 1`` where the response
  has one, the code FANUC's driver answers every failure with;
- a converter for ``control_msgs/DynamicJointState``, which ``arm_controller`` publishes as
  ``[(joint, {interface: value})]``.
"""

from __future__ import annotations

import importlib
import logging
from collections.abc import Mapping
from typing import Any

logger = logging.getLogger(__name__)

#: Message types a plugin publishes or subscribes as mappings.
MAPPED_MESSAGES = (
    "fanuc_msgs.msg.IOState",
    "fanuc_msgs.msg.IOCmd",
    "fanuc_msgs.msg.AnalogIOState",
    "fanuc_msgs.msg.AnalogIOCmd",
    "fanuc_msgs.msg.NumRegState",
    "fanuc_msgs.msg.NumRegCmd",
    "fanuc_msgs.msg.ConnectionStatus",
    "fanuc_msgs.msg.RobotStatus",
    "fanuc_msgs.msg.RobotStatusExt",
    "fanuc_msgs.msg.CollaborativeSpeedScaling",
    "fanuc_msgs.msg.ForceSensor",
)

#: Service types a plugin serves with a mapping in and a mapping out.
MAPPED_SERVICES = tuple(
    f"fanuc_msgs.srv.{name}"
    for name in (
        "CfgForceSensor",
        "GetAnalogIO",
        "GetBoolIO",
        "GetCartesianPosition",
        "GetGroupIO",
        "GetNumReg",
        "GetPayloadComp",
        "GetPayloadID",
        "GetPayloadValue",
        "GetPosReg",
        "GetTCPSpeed",
        "GetUFrameData",
        "GetUFrameUTool",
        "GetUToolData",
        "ReadError",
        "Reset",
        "SetAnalogIO",
        "SetBoolIO",
        "SetGenOverride",
        "SetGroupIO",
        "SetNumReg",
        "SetPayloadComp",
        "SetPayloadID",
        "SetPayloadValue",
        "SetPosReg",
        "SetUFrameData",
        "SetUFrameUTool",
        "SetUToolData",
        "SwitchControlState",
    )
)

_INTS = {
    "int8",
    "uint8",
    "int16",
    "uint16",
    "int32",
    "uint32",
    "int64",
    "uint64",
    "byte",
    "octet",
    "char",
}
_FLOATS = {"float", "double"}


def _message_class(spec: str):
    """``pkg/Type`` (as ``get_fields_and_field_types`` names a nested message) to its class."""
    pkg, _, name = spec.rpartition("/")
    return getattr(importlib.import_module(f"{pkg}.msg"), name)


def _element(spec: str) -> str:
    """The element type of a sequence or array field type, else the type itself."""
    if spec.startswith("sequence<"):
        return spec[len("sequence<") :].split(",")[0].rstrip(">")
    if spec.endswith("]") and "[" in spec:
        return spec[: spec.index("[")]
    return spec


def _is_sequence(spec: str) -> bool:
    return spec.startswith("sequence<") or (spec.endswith("]") and "[" in spec)


def _scalar(value: Any, spec: str) -> Any:
    base = spec.split("<")[0]
    if base in _INTS:
        return int(value)
    if base in _FLOATS:
        return float(value)
    if base == "boolean":
        return bool(value)
    if base.startswith("string") or base.startswith("wstring"):
        return str(value)
    return value


def fill(msg, value: Mapping[str, Any]):
    """Fill *msg* from *value*, field by field, nested messages and sequences of them included.

    A field *value* does not name keeps the message's default. Returns *msg*.
    """
    fields = type(msg).get_fields_and_field_types()
    for name, spec in fields.items():
        if name not in value:
            continue
        v = value[name]
        element = _element(spec)
        nested = "/" in element
        if _is_sequence(spec):
            if nested:
                cls = _message_class(element)
                setattr(msg, name, [fill(cls(), item) for item in v])
            else:
                setattr(msg, name, [_scalar(item, element) for item in v])
        elif nested:
            fill(getattr(msg, name), v)
        else:
            setattr(msg, name, _scalar(v, spec))
    return msg


def mapping(msg) -> dict[str, Any]:
    """*msg* as a plain mapping, nested messages and sequences of them included."""
    out: dict[str, Any] = {}
    for name, spec in type(msg).get_fields_and_field_types().items():
        v = getattr(msg, name)
        element = _element(spec)
        if "/" in element:
            out[name] = [mapping(item) for item in v] if _is_sequence(spec) else mapping(v)
        elif _is_sequence(spec):
            out[name] = list(v)
        else:
            out[name] = v
    return out


def mapped_service(request, response, ctx, on_payload, endpoint=None):  # noqa: ARG001
    """Serve a request as a mapping to the endpoint's write, and fill the response from its reply."""
    from roqsim_ros_bridge import physics

    try:
        reply = on_payload(mapping(request)).result(physics.DEFAULT_TIMEOUT_S)
    except Exception as exc:  # noqa: BLE001 - a failed command is the reply, as on the robot
        logger.warning("%s: %s: %s", getattr(endpoint, "name", "service"), type(exc).__name__, exc)
        reply = {"result": 1}
    if hasattr(response, "result") and "result" not in (reply or {}):
        reply = {**(reply or {}), "result": 0}
    return fill(response, reply or {})


def fill_mapped(msg, payload, stamp, hints) -> None:  # noqa: ARG001
    fill(msg, payload)
    if hasattr(msg, "header"):
        msg.header.stamp = stamp


def decode_mapped(msg) -> dict[str, Any]:
    return mapping(msg)


def fill_dynamic_joint_states(msg, payload, stamp, hints) -> None:  # noqa: ARG001
    """``[(name, {interface: value})]`` as a ``control_msgs/DynamicJointState``."""
    interface_value = _message_class("control_msgs/InterfaceValue")
    msg.header.stamp = stamp
    msg.joint_names = [name for name, _ in payload]
    msg.interface_values = [
        interface_value(interface_names=list(values), values=[float(x) for x in values.values()])
        for _, values in payload
    ]


def register() -> None:
    """Teach the bridge these types (called by the extension module the bridge imports)."""
    from roqsim_ros_bridge.registry import converter, decoder
    from roqsim_ros_bridge.services import service_handler

    for type_path in MAPPED_MESSAGES:
        converter(type_path)(fill_mapped)
        decoder(type_path)(decode_mapped)
    for type_path in MAPPED_SERVICES:
        service_handler(type_path)(mapped_service)
    converter("control_msgs.msg.DynamicJointState")(fill_dynamic_joint_states)
