# Copyright (C) 2026 Frederik Pasch
#
# SPDX-License-Identifier: Apache-2.0

"""The one exit-status table: its codes, the errors that map onto them, and its one documentation."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from roqsim import exit_status
from roqsim.capture import RecordingError, RecordingNotFoundError
from roqsim.plugin import PluginError
from roqsim.render import RenderError, RenderGLError
from roqsim.rendering import GLBackendError
from roqsim.viewer import DisplayError

QUICKSTART = Path(__file__).resolve().parents[2] / "docs" / "quickstart.rst"


def test_the_table_is_six_distinct_codes():
    codes = [
        exit_status.OK,
        exit_status.CRASH,
        exit_status.BAD_INPUT,
        exit_status.NO_GL,
        exit_status.RECORDING,
        exit_status.FINDING,
    ]
    assert codes == list(range(6))
    assert sorted(exit_status.MEANINGS) == codes


def test_a_meaning_has_no_semicolon():
    """An epilog joins the meanings with '; ', which is how a reader (and a test) splits them."""
    assert not [m for m in exit_status.MEANINGS.values() if ";" in m]


@pytest.mark.parametrize(
    ("err", "code"),
    [
        (RecordingNotFoundError("x.mcap: no such recording"), exit_status.BAD_INPUT),
        (RecordingError("x.mcap is not a readable recording"), exit_status.RECORDING),
        (DisplayError("no display"), exit_status.NO_GL),
        (GLBackendError("glfw bound"), exit_status.NO_GL),
        (RenderGLError("set MUJOCO_GL"), exit_status.NO_GL),
        (RenderError("--size: expected WxH"), exit_status.BAD_INPUT),
        (PluginError("no such world"), exit_status.BAD_INPUT),
        (FileNotFoundError(2, "No such file or directory", "x.json"), exit_status.BAD_INPUT),
    ],
)
def test_an_error_class_states_its_status(err, code):
    assert exit_status.for_error(err) == code


def test_epilog_names_success_crash_and_what_it_was_given():
    text = exit_status.epilog(exit_status.RECORDING, note="Stdout is JSON.")
    assert text.startswith("exit status: 0 success; 1 an unexpected error")
    assert f"; 4 {exit_status.MEANINGS[4]}." in text and text.endswith("Stdout is JSON.")
    assert "; 2 " not in text


def test_epilog_refuses_a_code_outside_the_table():
    with pytest.raises(ValueError, match="not in roqsim.exit_status"):
        exit_status.epilog(7)


def test_the_documented_table_is_this_one():
    """Documented once, in the quickstart where the promise is made -- and it must say what runs."""
    section = QUICKSTART.read_text(encoding="utf-8").split(".. _exit-status:", 1)[1]
    table = section.split("\n=====  ", 3)[2]  # the body, between the header rule and the last one
    rows = re.findall(r"^``(\d+)``\s+(.+?)\s*$", table, re.M)
    assert {int(code): text for code, text in rows} == exit_status.MEANINGS
