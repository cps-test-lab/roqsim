# Copyright (C) 2026 Frederik Pasch
# SPDX-License-Identifier: Apache-2.0
"""An installation without requests (``pip install --no-deps``) still answers what needs no fetch."""

from __future__ import annotations

import subprocess
import sys

import pytest

BLOCK = "import sys; sys.modules['requests'] = None; "


@pytest.mark.parametrize("module", ["fuel_fetch", "sdf_to_scene"])
def test_it_answers_help_without_requests(module):
    code = f"{BLOCK}from roqsim_scenes.cli.{module} import main; main(['--help'])"
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    assert "usage" in done.stdout


def test_fetching_without_requests_names_what_is_missing():
    code = f"{BLOCK}from roqsim_scenes.cli.fuel_fetch import _requests; _requests()"
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert done.returncode != 0
    assert "fetching from Fuel needs `requests`" in done.stderr
