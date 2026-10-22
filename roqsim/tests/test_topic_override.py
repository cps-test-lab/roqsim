"""Plugin.topic_override + validate_topics: the per-endpoint topic, absolute or namespace-relative."""

from __future__ import annotations

import pytest

from roqsim.plugin import Plugin, PluginError


def test_topic_override_returns_mapped_value():
    p = Plugin({"topics": {"image": "/camera/color/image_raw", "joint_states": "/joint_states"}})
    assert p.topic_override("image") == "/camera/color/image_raw"
    assert p.topic_override("joint_states") == "/joint_states"


def test_topic_override_absent_is_none():
    assert Plugin({}).topic_override("image") is None
    assert Plugin({"topics": {"depth": "/d"}}).topic_override("image") is None
    # A None/missing topics map is tolerated (default namespaced topic is used by the caller).
    assert Plugin({"topics": None}).topic_override("image") is None


def test_validate_topics_accepts_absolute_and_relative():
    assert Plugin.validate_topics({}) == []
    assert Plugin.validate_topics({"topics": {"image": "/camera/color/image_raw"}}) == []
    # Relative: a rename inside the endpoint's namespace (a vendor's `scan2` for a second scanner).
    assert (
        Plugin.validate_topics({"topics": {"scan": "scan2", "image": "camera/color/image_raw"}})
        == []
    )


def test_validate_topics_rejects_malformed_names():
    for bad in ("", "/", "scan/", "a//b", "my scan", 7):
        errs = Plugin.validate_topics({"topics": {"scan": bad}})
        assert errs and "topic name" in errs[0], bad

    errs = Plugin.validate_topics({"topics": ["not", "a", "map"]})
    assert errs and "mapping" in errs[0]


def test_a_name_first_asked_after_the_topics_check_is_refused():
    """The engine checks ``topics:`` after register_endpoints; a rename asked later would escape it."""
    p = Plugin({"topics": {"image": "/camera/color/image_raw"}})
    assert p.topic_override("image") == "/camera/color/image_raw"
    p.check_topic_keys()
    # A name asked before the check stays readable, so a repeated lookup keeps working.
    assert p.topic_override("image") == "/camera/color/image_raw"
    with pytest.raises(PluginError, match="configure or register_endpoints"):
        p.topic_override("depth")


def test_a_late_name_is_refused_without_a_topics_map_too():
    p = Plugin({})
    p.check_topic_keys()
    with pytest.raises(PluginError, match="topic_override\\('scan'\\)"):
        p.topic_override("scan")
