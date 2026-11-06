"""Plain mappings to ROS messages and back, without ROS: stand-in message classes with rosidl's
``get_fields_and_field_types``, nested messages and sequences of them included."""

from __future__ import annotations

import sys
import types

import pytest

from roqsim_manipulation import ros2_mapping


class _Msg:
    FIELDS: dict[str, str] = {}

    def __init__(self, **kw):
        for name, spec in self.FIELDS.items():
            default = (
                []
                if ros2_mapping._is_sequence(spec)
                else (_resolve(spec)() if "/" in spec else ("" if spec.startswith("string") else 0))
            )
            setattr(self, name, kw.get(name, default))

    @classmethod
    def get_fields_and_field_types(cls):
        return dict(cls.FIELDS)


class IOType(_Msg):
    FIELDS = {"type": "string"}


class BoolIO(_Msg):
    FIELDS = {"io_type": "fake_msgs/IOType", "index": "uint16", "value": "boolean"}


class IOState(_Msg):
    FIELDS = {"values": "sequence<fake_msgs/BoolIO>"}


class ReadErrorResponse(_Msg):
    FIELDS = {"result": "int32", "count": "uint8", "error_data": "string[5]"}


def _resolve(spec):
    return {"fake_msgs/IOType": IOType, "fake_msgs/BoolIO": BoolIO}[spec]


@pytest.fixture(autouse=True)
def fake_package(monkeypatch):
    msg = types.ModuleType("fake_msgs.msg")
    msg.IOType, msg.BoolIO, msg.IOState = IOType, BoolIO, IOState
    monkeypatch.setitem(sys.modules, "fake_msgs", types.ModuleType("fake_msgs"))
    monkeypatch.setitem(sys.modules, "fake_msgs.msg", msg)


def test_a_mapping_fills_nested_messages_and_sequences_of_them():
    state = ros2_mapping.fill(
        IOState(), {"values": [{"io_type": {"type": "DO"}, "index": 101, "value": 1}]}
    )
    (entry,) = state.values
    assert isinstance(entry, BoolIO) and isinstance(entry.io_type, IOType)
    assert (entry.io_type.type, entry.index, entry.value) == ("DO", 101, True)


def test_a_message_reads_back_as_the_mapping():
    value = {"values": [{"io_type": {"type": "F"}, "index": 3, "value": False}]}
    assert ros2_mapping.mapping(ros2_mapping.fill(IOState(), value)) == value


def test_fields_left_out_keep_their_defaults_and_scalars_take_their_type():
    reply = ros2_mapping.fill(ReadErrorResponse(), {"count": 2.0, "error_data": ["a", "b"]})
    assert reply.result == 0 and reply.count == 2 and isinstance(reply.count, int)
    assert reply.error_data == ["a", "b"]
