"""The shared document checks: a stated version, and keys nothing reads."""

from __future__ import annotations

import pytest

from roqsim.document import check_version, nearest, refuse_unknown_keys


class _Refused(Exception):
    pass


def _version(doc):
    return check_version(doc, "version", reads=2, document="test document", where="doc")


def test_an_absent_stamp_is_version_1():
    assert _version({}) == 1


def test_a_version_up_to_the_reader_is_returned():
    assert _version({"version": 2}) == 2


def test_a_newer_version_is_refused_naming_both():
    with pytest.raises(ValueError, match=r"doc is test document version 3; .* reads up to 2"):
        _version({"version": 3})


@pytest.mark.parametrize("bad", [0, -1, "1", 1.5, True, None])
def test_a_version_that_is_not_a_positive_integer_is_refused(bad):
    with pytest.raises(ValueError, match="is not a test document version"):
        _version({"version": bad})


def test_the_caller_chooses_the_error():
    with pytest.raises(_Refused):
        check_version({"v": 9}, "v", reads=1, document="d", where="w", error=_Refused)


def test_nearest_is_tight():
    assert nearest("frame", {"frames", "fov"}) == "frames"
    assert nearest("banana", {"frames", "fov"}) is None


def test_known_keys_pass():
    refuse_unknown_keys({"a": 1, "b": 2}, {"a", "b", "c"}, "doc")


def test_an_unknown_key_is_refused_with_the_nearest_named():
    with pytest.raises(
        ValueError, match=r"doc: unknown key\(s\) 'widht_m' \(did you mean 'width_m'\?\)"
    ):
        refuse_unknown_keys({"widht_m": 1}, {"width_m", "t"}, "doc")


def test_a_far_key_is_refused_without_a_guess():
    with pytest.raises(ValueError, match=r"unknown key\(s\) 'banana'; it takes t, width_m\."):
        refuse_unknown_keys({"banana": 1}, {"width_m", "t"}, "doc")


def test_a_block_that_is_not_a_mapping_is_refused():
    with pytest.raises(_Refused, match="must be a mapping, not list"):
        refuse_unknown_keys([], {"a"}, "doc", error=_Refused)
