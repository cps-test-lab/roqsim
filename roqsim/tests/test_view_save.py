"""The F8 view save: the key handler, the derived block, and the surgical world-YAML edit."""

from __future__ import annotations

import textwrap

import pytest
import yaml

from roqsim.view_save import (
    KEY_F8,
    SaveViewKey,
    ViewSaveError,
    replace_sim_view,
    save_view,
    view_from_camera,
)


class FakeCam:
    def __init__(self, lookat=(1.0, 2.0, 3.0), distance=4.0, azimuth=130.0, elevation=-20.0):
        self.lookat = list(lookat)
        self.distance = distance
        self.azimuth = azimuth
        self.elevation = elevation


# -- the key handler -------------------------------------------------------------------------------


def test_f8_sets_a_pending_save():
    saver = SaveViewKey()
    assert saver.take_pending() is False
    saver.key_callback(KEY_F8)
    assert saver.take_pending() is True
    assert saver.take_pending() is False


def test_other_keys_are_ignored_but_still_chained():
    seen = []
    saver = SaveViewKey(chain=seen.append)
    for code in (ord(" "), ord("V"), 298):  # pause, MuJoCo's tendon flag, roqsim's F9
        saver.key_callback(code)
    assert saver.take_pending() is False
    assert seen == [ord(" "), ord("V"), 298]


def test_autorepeat_collapses_to_one_save():
    """A held F8 delivers several presses ~0.2 s apart; they must not queue several dialogs."""
    saver = SaveViewKey()
    for _ in range(5):
        saver.key_callback(KEY_F8)
    assert saver.take_pending() is True
    assert saver.take_pending() is False


# -- deriving the block ----------------------------------------------------------------------------


def test_free_camera_saves_its_whole_pose():
    view = view_from_camera(FakeCam())
    assert view == {
        "lookat": [1.0, 2.0, 3.0],
        "distance": 4.0,
        "azimuth": 130.0,
        "elevation": -20.0,
    }


def test_values_are_rounded_to_what_a_framing_decision_carries():
    cam = FakeCam(
        lookat=(1.23456, -0.00004, 3.0), distance=4.123456, azimuth=130.06, elevation=-20.04
    )
    view = view_from_camera(cam)
    assert view["lookat"] == [1.235, 0.0, 3.0]  # -0.00004 folds onto +0.0, not -0.0
    assert view["distance"] == 4.123
    assert (view["azimuth"], view["elevation"]) == (130.1, -20.0)


def test_tracking_view_keeps_track_and_drops_lookat():
    """MuJoCo drives lookat while tracking, so saving it would freeze the robot's position."""
    view = view_from_camera(FakeCam(), {"track": "robot"})
    assert view == {"track": "robot", "distance": 4.0, "azimuth": 130.0, "elevation": -20.0}


def test_follow_heading_saves_the_offset_not_the_live_azimuth():
    """cam.azimuth is yaw+offset, rewritten each frame; only the offset means anything in a world."""
    view = view_from_camera(
        FakeCam(azimuth=47.5), {"track": "robot", "follow_heading": True}, azimuth_offset=180.0
    )
    assert view["follow_heading"] is True
    assert view["azimuth"] == 180.0


# -- the surgical edit -----------------------------------------------------------------------------


VIEW = {"lookat": [4.0, 0.0, 1.0], "distance": 42.0, "azimuth": 90.0, "elevation": -45.0}


def _sim_view(text: str) -> dict:
    return yaml.safe_load(text)["sim"]["view"]


def test_replaces_a_block_style_view():
    before = textwrap.dedent("""\
        sim:
          name: Depot
          view:
            lookat: [0.0, 0.0, 0.0]
            distance: 1.0
            azimuth: 0.0
            elevation: 0.0

        components:
          - dummy: {}
        """)
    after = replace_sim_view(before, VIEW)
    assert _sim_view(after) == VIEW
    assert yaml.safe_load(after)["components"] == [{"dummy": {}}]


def test_replaces_a_flow_style_view_in_place():
    """Some worlds write the block on one line on purpose; a save must not expand it."""
    before = (
        "sim:\n  view: {lookat: [0, 0, 0.8], distance: 3.0, azimuth: 130.0, elevation: -15.0}\n"
    )
    after = replace_sim_view(before, VIEW)
    assert _sim_view(after) == VIEW
    assert after.count("\n") == before.count("\n")
    assert "view: {lookat: [4.0, 0.0, 1.0], distance: 42.0" in after


def test_comments_survive_the_write():
    """A world YAML is a commented document; a naive safe_dump round-trip would erase it."""
    before = textwrap.dedent("""\
        # How this scene was generated:
        #   roqsim scenes sdf-to-scene --world depot.sdf
        sim:
          name: Depot                  # model + viewer title
          world: depot/depot.xml       # MJCF file, relative to this YAML
          view:                        # initial free-camera framing (windowed runner only)
            lookat: [0.0, 0.0, 0.0]
            distance: 1.0

        components:
          # Ships OPEN (roofless) -- set `enabled: true` for the roofed warehouse.
          - ceiling: {above_z: 2.6, enabled: false}
        """)
    after = replace_sim_view(before, VIEW)
    assert _sim_view(after) == VIEW
    for comment in (
        "#   roqsim scenes sdf-to-scene --world depot.sdf",
        "# model + viewer title",
        "# MJCF file, relative to this YAML",
        "# initial free-camera framing (windowed runner only)",
        "# Ships OPEN (roofless)",
    ):
        assert comment in after, comment


def test_the_view_line_itself_is_left_alone():
    """Worlds align their trailing comments in a column; only the values should show up in a diff."""
    line = "  view:                        # initial free-camera framing (windowed runner only)"
    before = f"sim:\n  name: Depot\n{line}\n    lookat: [0.0, 0.0, 0.0]\n    distance: 1.0\n"
    after = replace_sim_view(before, VIEW)
    assert line in after.splitlines()
    assert _sim_view(after) == VIEW


def test_adds_a_view_to_a_sim_block_that_has_none():
    before = "sim:\n  name: Lab\n  pacing: realtime\n\nplugins:\n  - dummy: {}\n"
    after = replace_sim_view(before, VIEW)
    assert _sim_view(after) == VIEW
    assert yaml.safe_load(after)["sim"]["pacing"] == "realtime"


def test_adds_a_sim_block_to_a_world_that_has_none():
    before = "components:\n  - dummy: {}\n"
    after = replace_sim_view(before, VIEW)
    assert _sim_view(after) == VIEW
    assert yaml.safe_load(after)["components"] == [{"dummy": {}}]


def test_writes_into_the_child_of_an_extends_chain():
    """The leaf's sim block wins the merge, so the leaf is where a saved view has to land."""
    before = "extends: roqsim_scenes:depot\nsim:\n  timestep: 0.001\n"
    after = replace_sim_view(before, VIEW)
    assert yaml.safe_load(after) == {
        "extends": "roqsim_scenes:depot",
        "sim": {"timestep": 0.001, "view": VIEW},
    }


def test_a_comment_between_sim_and_plugins_is_not_swallowed():
    before = "sim:\n  name: Lab\n\n# The robot and its sensors:\nplugins:\n  - dummy: {}\n"
    after = replace_sim_view(before, VIEW)
    assert "# The robot and its sensors:" in after
    assert _sim_view(after) == VIEW


def test_floats_are_written_as_floats():
    """`distance: 42` would be an int; every hand-authored world writes these with a decimal point."""
    after = replace_sim_view(
        "sim:\n  name: Lab\n", {"distance": 42.0, "azimuth": 90.0, "elevation": -0.0}
    )
    assert "distance: 42.0" in after
    assert "elevation: 0.0" in after  # -0.0 is never a framing anyone chose


def test_a_flow_style_sim_block_keeps_its_other_keys():
    """`sim: {pacing: asap}` has no line range to splice into; the line is re-emitted instead."""
    after = replace_sim_view("sim: {name: Lab, pacing: asap}  # one-liner\n", VIEW)
    assert yaml.safe_load(after)["sim"] == {"name": "Lab", "pacing": "asap", "view": VIEW}
    assert "# one-liner" in after


def test_track_and_follow_heading_round_trip():
    view = {
        "track": "robot",
        "follow_heading": True,
        "distance": 3.4,
        "azimuth": 180.0,
        "elevation": -20.0,
    }
    after = replace_sim_view("sim:\n  name: Lab\n", view)
    assert _sim_view(after) == view


def test_the_written_block_satisfies_the_sim_view_schema():
    """A save must produce a world the loader accepts -- sim.view rejects unknown keys."""
    from roqsim.config import _VIEW_KEYS

    after = replace_sim_view("sim:\n  name: Lab\n", view_from_camera(FakeCam(), {"track": "robot"}))
    assert set(_sim_view(after)) <= _VIEW_KEYS


def test_a_non_mapping_world_is_refused():
    with pytest.raises(ViewSaveError):
        replace_sim_view("- a\n- b\n", VIEW)


def test_an_unterminated_flow_mapping_is_refused():
    with pytest.raises(ViewSaveError, match="never closed"):
        replace_sim_view("sim:\n  view: {lookat: [0, 0, 0]\n", VIEW)


# -- the write -------------------------------------------------------------------------------------


def test_save_view_rewrites_the_file(tmp_path):
    path = tmp_path / "world.yaml"
    path.write_text("sim:\n  name: Lab  # keep me\n\nplugins:\n  - dummy: {}\n")
    save_view(path, VIEW)
    assert _sim_view(path.read_text()) == VIEW
    assert "# keep me" in path.read_text()


def test_a_failed_save_leaves_the_world_untouched(tmp_path):
    path = tmp_path / "world.yaml"
    original = "- not a mapping\n"
    path.write_text(original)
    with pytest.raises(ViewSaveError):
        save_view(path, VIEW)
    assert path.read_text() == original
    assert list(tmp_path.iterdir()) == [path]  # no temp file left behind
