# Copyright (C) 2026 Frederik Pasch
#
# SPDX-License-Identifier: Apache-2.0

"""A world is held to the config keys its plugins document.

A plugin with a schema refuses an unknown key itself. One without accepts any key, so a misspelled
or misplaced one leaves its setting at the default while the run looks configured. ``roqsim check``
and ``roqsim scenes describe`` say so as a warning naming the keys; ``enabled`` written inside a
config, where no plugin reads it, is refused, as is a ``topics:`` key the plugin never renames and an
override path through a value that is not a mapping.
"""

from __future__ import annotations

import textwrap

import pytest

from roqsim.check import check_world
from roqsim.config import resolve_plugin


def _world(tmp_path, body: str):
    path = tmp_path / "world.yaml"
    path.write_text(textwrap.dedent(body).lstrip(), encoding="utf-8")
    return str(path)


BOX = """
    sim: {{}}
    components:
      - box: {{pose: {{position: {{x: 1.0, y: 0.0}}}}, size: [0.4, 0.4, 0.4]{extra}}}
"""


def test_a_key_a_plugin_does_not_document_is_a_warning_naming_it(tmp_path):
    pytest.importorskip("roqsim_assets", reason="the box plugin lives in roqsim_assets")
    report = check_world(_world(tmp_path, BOX.format(extra=", mass: 3.0")))
    assert report["ok"] is True, "a plugin may read a key it does not document"
    warning = next(w for w in report["warnings"] if w["check"] == "config-key")
    assert warning["keys"] == ["mass"]
    assert "roqsim plugins describe" in warning["hint"]


def test_documented_keys_raise_no_warning(tmp_path):
    pytest.importorskip("roqsim_assets", reason="the box plugin lives in roqsim_assets")
    report = check_world(_world(tmp_path, BOX.format(extra=", friction: 0.8")))
    assert [w for w in report["warnings"] if w["check"] == "config-key"] == []


def test_enabled_inside_a_config_that_does_not_read_it_is_refused(tmp_path):
    pytest.importorskip("roqsim_assets", reason="the box plugin lives in roqsim_assets")
    report = check_world(_world(tmp_path, BOX.format(extra=", enabled: false")))
    assert report["ok"] is False
    message = report["problems"][0]["message"]
    assert "'enabled' is the reserved sibling key" in message and "Move it out one level" in message


def test_enabled_beside_the_plugin_ref_is_the_removal(tmp_path):
    pytest.importorskip("roqsim_assets", reason="the box plugin lives in roqsim_assets")
    report = check_world(
        _world(
            tmp_path,
            """
        sim: {}
        components:
          - box: {pose: {position: {x: 1.0, y: 0.0}}, size: [0.4, 0.4, 0.4]}
            enabled: false
    """,
        )
    )
    assert report["ok"] is True


def test_a_plugin_with_a_schema_is_left_to_its_schema():
    cls = resolve_plugin("spawn_robot")
    assert cls.CONFIG_SCHEMA
    assert cls.undocumented_keys({"modle": "turtlebot4"}) == []


def test_a_dotted_documented_key_is_its_top_level_key():
    pytest.importorskip("roqsim_mobile", reason="diff_drive lives in roqsim_mobile")
    cls = resolve_plugin("diff_drive")
    assert cls.undocumented_keys({"max_linear_vel": 0.3, "max_linar_vel": 0.3}) == ["max_linar_vel"]


CONTACT = """
    sim: {{}}
    components:
      - spawn_robot: {{model: turtlebot4}}
        name: robot
        components:
          - contact_monitor: {{topics: {{{key}: /bumped}}}}
"""


def test_a_topics_key_naming_the_topic_instead_of_the_endpoint_is_refused(tmp_path):
    pytest.importorskip("roqsim_mobile", reason="the turtlebot4 model lives in roqsim_mobile")
    report = check_world(_world(tmp_path, CONTACT.format(key="collision")))
    assert report["ok"] is False
    message = " ".join(p["message"] for p in report["problems"])
    assert "'collision'" in message and "this plugin's are: contact" in message


def test_a_topics_key_naming_the_endpoint_renames_it(tmp_path):
    pytest.importorskip("roqsim_mobile", reason="the turtlebot4 model lives in roqsim_mobile")
    report = check_world(_world(tmp_path, CONTACT.format(key="contact")))
    assert report["ok"] is True, report["problems"]
    assert "/bumped" in {e["topic"] for e in report["world"]["endpoints"]}


def test_an_override_through_a_value_that_is_not_a_mapping_is_refused(tmp_path):
    pytest.importorskip("roqsim_assets", reason="the box plugin lives in roqsim_assets")
    path = _world(tmp_path, BOX.format(extra=""))
    report = check_world(path, {"components": {"box": {"size": {"x": 1.0}}}})
    assert report["ok"] is False
    message = report["problems"][0]["message"]
    assert "walks into 'size'" in message and "list, not a mapping" in message
