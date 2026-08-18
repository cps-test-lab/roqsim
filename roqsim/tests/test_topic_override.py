"""Plugin.topic_override + validate_topics: the per-endpoint absolute-topic hardwire."""

from __future__ import annotations

from roqsim.plugin import Plugin


def test_topic_override_returns_mapped_value():
    p = Plugin({"topics": {"image": "/camera/color/image_raw", "joint_states": "/joint_states"}})
    assert p.topic_override("image") == "/camera/color/image_raw"
    assert p.topic_override("joint_states") == "/joint_states"


def test_topic_override_absent_is_none():
    assert Plugin({}).topic_override("image") is None
    assert Plugin({"topics": {"depth": "/d"}}).topic_override("image") is None
    # A None/missing topics map is tolerated (default namespaced topic is used by the caller).
    assert Plugin({"topics": None}).topic_override("image") is None


def test_validate_topics_accepts_absolute_and_rejects_relative():
    assert Plugin.validate_topics({}) == []
    assert Plugin.validate_topics({"topics": {"image": "/camera/color/image_raw"}}) == []

    errs = Plugin.validate_topics({"topics": {"image": "camera/color/image_raw"}})
    assert errs and "absolute" in errs[0]

    errs = Plugin.validate_topics({"topics": ["not", "a", "map"]})
    assert errs and "mapping" in errs[0]
