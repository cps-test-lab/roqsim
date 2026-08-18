# Copyright (C) 2026 Frederik Pasch
# SPDX-License-Identifier: Apache-2.0
"""roqsim's CLI logging format: unchanged by default, stamped on request.

The default matters as much as the option. roqsim runs standalone -- `roqsim sim` is one command a
person watches in a terminal -- and an epoch timestamp on every line is worse there. The
stamped form exists for the other case, where roqsim is one producer in an aggregated log and a
line without a timestamp cannot be ordered against anything else.
"""

import logging
import subprocess
import sys

import pytest

from roqsim import logging_setup


def test_the_default_format_is_the_long_standing_one():
    """A change here is a change to every standalone user's terminal output."""
    assert logging_setup.log_format() == "%(levelname)s %(name)s: %(message)s"


def test_the_stamped_format_needs_no_clock_call_of_its_own():
    """`%(created)` is the record's own epoch time, taken when the event happened rather than
    when it was formatted -- so the stamp is the event's, not the handler's."""
    assert "%(created)" in logging_setup.STAMPED_FORMAT


@pytest.mark.parametrize("value,expected", [
    (None, logging_setup.PLAIN_FORMAT),
    ("plain", logging_setup.PLAIN_FORMAT),
    ("stamped", logging_setup.STAMPED_FORMAT),
])
def test_the_environment_selects_the_format(monkeypatch, value, expected):
    monkeypatch.delenv(logging_setup.FORMAT_ENV, raising=False)
    if value is not None:
        monkeypatch.setenv(logging_setup.FORMAT_ENV, value)
    assert logging_setup.log_format() == expected


def test_an_unknown_format_raises_rather_than_falling_back(monkeypatch):
    """A silent fallback would leave an aggregator that set the variable looking correctly
    configured while emitting lines nothing downstream can place. A typo is how that happens."""
    monkeypatch.setenv(logging_setup.FORMAT_ENV, "stampd")
    with pytest.raises(ValueError, match="stampd"):
        logging_setup.log_format()


@pytest.mark.parametrize("env,expected", [
    ({}, "INFO roqsim.engine: seed 42"),
    ({"ROQSIM_LOG_FORMAT": "stamped"}, "[INFO] ["),
])
def test_a_configured_logger_prints_that_format(env, expected, tmp_path):
    """End to end through a real process, because `basicConfig` is a no-op once the root logger
    has a handler -- an in-process test can pass while the CLI prints something else."""
    script = (
        "import logging\n"
        "from roqsim import logging_setup\n"
        "logging_setup.configure(verbose=False)\n"
        "logging.getLogger('roqsim.engine').info('seed 42')\n"
    )
    out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                         env={"PATH": "/usr/bin:/bin", "PYTHONPATH": ":".join(sys.path),
                             **env})
    assert out.returncode == 0, out.stderr
    assert expected in out.stderr


def test_every_cli_uses_the_helper_rather_than_its_own_basicConfig():
    """Six entry points repeated the same three-line call. The point of the helper is that the
    seventh does not have to rediscover the decision -- so a new `basicConfig` here is a
    regression even though it would work.

    `export_capture` is excluded deliberately: its format is bare `%(message)s` because that
    CLI's contract is one line of JSON on stdout.
    """
    from pathlib import Path
    src = Path(logging_setup.__file__).parent
    offenders = sorted(
        p.name for p in src.glob("*.py")
        if p.name not in ("logging_setup.py", "export_capture.py")
        and "logging.basicConfig" in p.read_text()
    )
    assert offenders == []
