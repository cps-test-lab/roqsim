"""Plugin.topic_override + validate_topics: the per-endpoint topic, absolute or namespace-relative."""

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
