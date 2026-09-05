# Copyright (C) 2026 Frederik Pasch
#
# SPDX-License-Identifier: Apache-2.0

"""The key catalogue: what the handlers fly on, and what the window says they do, are one thing.

The point of `roqsim.keys` is that the F1 list cannot drift from the keys. These tests are where that
is actually held -- above all by pinning the maps `WalkKeys` polls to the values it used to carry
literally, so a catalogue edit cannot quietly change how the camera flies.
"""

from types import SimpleNamespace

import pytest

from roqsim import keys


def _source(*bindings):
    """Anything with a ``key_bindings`` attribute is a source -- that is the whole contract."""
    return SimpleNamespace(key_bindings=bindings)


# -- what the camera flies on ----------------------------------------------------------------------


def test_the_walk_maps_are_the_ones_the_camera_flew_on_before():
    """Derived now, but they must still be these: a catalogue edit cannot quietly change flight.

    Written out rather than computed from the catalogue -- a test that derived them the same way the
    code does would agree with any mistake.
    """
    assert keys.keycodes(keys.WALK) == {
        265: "w",  # Up: forward along the view
        264: "s",  # Down: back
        263: "a",  # Left: strafe left
        262: "d",  # Right: strafe right
        266: "e",  # Page Up: rise
        267: "q",  # Page Down: descend
        340: "shift",  # Left Shift: fly faster
        344: "shift",  # Right Shift
    }
    assert keys.keysyms(keys.WALK) == {
        "Up": "w",
        "Down": "s",
        "Left": "a",
        "Right": "d",
        "Prior": "e",
        "Next": "q",
        "Shift_L": "shift",
        "Shift_R": "shift",
    }


def test_one_line_can_be_two_keys_meaning_two_things():
    # The forward/back pair is one line in the list and two codes with different tokens; splitting it
    # would put four near-identical lines in a list whose point is being short.
    assert keys.CAMERA_FORWARD.tokens() == {265: "w", 264: "s"}
    assert keys.CAMERA_FORWARD.label == "Up/Down"


def test_one_line_can_be_two_keys_meaning_the_same_thing():
    assert keys.CAMERA_SPRINT.tokens() == {340: "shift", 344: "shift"}


def test_a_function_key_carries_no_token_or_keysym():
    # F8/F9/F10 are matched on the code by their own handler; nothing polls them.
    assert keys.SAVE_VIEW.tokens() == {} and keys.SAVE_VIEW.keysym_tokens() == {}


# -- merging, and refusing ------------------------------------------------------------------------


def test_merge_reads_key_bindings_off_anything_that_has_them():
    """The seam plugin-declared keys will arrive through: a bare object with the attribute."""
    assert keys.merge(_source(keys.SAVE_VIEW)) == (keys.SAVE_VIEW,)


def test_merge_skips_a_handler_this_run_did_not_install():
    # A headless-ish run wires no recorder; the caller should not have to branch on that.
    assert keys.merge(None, _source(keys.SHOW_HELP)) == (keys.SHOW_HELP,)


def test_a_source_with_no_keys_contributes_none():
    assert keys.merge(object(), _source()) == ()


def test_two_sources_claiming_one_keycode_are_refused_by_name():
    mine = keys.KeyBinding("acme.snapshot", "run", "F9", "snapshot", (keys.Key(keys.KEY_F9),))
    with pytest.raises(keys.KeyBindingError) as err:
        keys.merge(_source(keys.RECORD_TAKE), _source(mine))
    assert "298" in str(err.value)
    assert "run.record" in str(err.value) and "acme.snapshot" in str(err.value)


def test_two_sources_claiming_one_action_are_refused():
    twin = keys.KeyBinding("run.record", "run", "F12", "something else", (keys.Key(301),))
    with pytest.raises(keys.KeyBindingError):
        keys.merge(_source(keys.RECORD_TAKE), _source(twin))


def test_the_same_binding_from_two_sources_is_one_key_not_a_clash():
    # WalkKeys declares the camera keys and the help list is handed them too; that is one key.
    assert keys.merge(_source(keys.CAMERA_MODE), _source(keys.CAMERA_MODE)) == (keys.CAMERA_MODE,)


def test_the_order_is_the_catalogues_not_the_wirings():
    merged = keys.merge(_source(keys.SHOW_HELP), _source(keys.CAMERA_MODE, keys.CAMERA_SPRINT))
    assert [b.action for b in merged] == ["camera.mode", "camera.sprint", "run.help"]


def test_a_key_the_catalogue_does_not_name_sorts_last():
    mine = keys.KeyBinding("acme.snapshot", "run", "F12", "snapshot", (keys.Key(301),))
    merged = keys.merge(_source(mine), _source(keys.SHOW_HELP, keys.CAMERA_MODE))
    assert [b.action for b in merged] == ["camera.mode", "run.help", "acme.snapshot"]


# -- what the window shows -------------------------------------------------------------------------


def test_the_list_names_only_what_it_was_given():
    labels, _ = keys.help_columns(keys.merge(_source(*keys.CAMERA, keys.SHOW_HELP)))
    assert "F10" in labels and "F8" not in labels and "F9" not in labels


def test_a_group_with_nothing_left_in_it_is_not_a_heading_over_nothing():
    labels, _ = keys.help_columns(keys.merge(_source(keys.CAMERA_MODE)))
    assert "RUN" not in labels


def test_both_columns_have_the_same_number_of_lines():
    # mjr_overlay aligns the two strings line by line; a mismatch shifts every help text up a row.
    left, right = keys.help_columns(keys.CATALOGUE)
    assert len(left.split("\n")) == len(right.split("\n"))


def test_the_list_says_which_list_it_is():
    # Simulate's own help is on screen at the same time, in the opposite corner.
    assert keys.help_columns(keys.CATALOGUE)[0].startswith(keys.HELP_TITLE)


def test_every_line_is_ascii():
    # The overlay is drawn with MuJoCo's built-in bitmap font, where a multi-byte glyph is rubbish.
    assert all(column.isascii() for column in keys.help_columns(keys.CATALOGUE))


def test_every_line_is_short_enough_to_read_in_a_corner():
    for binding in keys.CATALOGUE:
        assert len(binding.label) + len(binding.help) <= 34, binding.action


def test_the_help_text_states_no_number():
    # A factor written here would be a second copy of viewer.WALK_SPRINT, free to drift from it.
    assert not any(c.isdigit() for b in keys.CATALOGUE for c in b.help)


# -- the keys we may take --------------------------------------------------------------------------


def test_only_the_help_key_is_one_simulate_owns():
    """F1 is shared on purpose -- it means help in both. Every other roqsim key is Simulate-free."""
    shared = [b for b in keys.CATALOGUE if set(b.codes) & keys.SIMULATE_FUNCTION_KEYS]
    assert shared == [keys.SHOW_HELP]


def test_no_two_bindings_in_the_catalogue_claim_one_key():
    codes = [code for binding in keys.CATALOGUE for code in binding.codes]
    assert len(codes) == len(set(codes))
