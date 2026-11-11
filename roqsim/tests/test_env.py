# Copyright (C) 2026 Frederik Pasch
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""A ``ROQSIM_*`` variable roqsim does not read is refused, and every one it reads is registered."""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

from roqsim import env, exit_status, runner
from roqsim.mcap_format import RecordingTargetError
from roqsim.scenario_adapter import MujocoSim

#: Built rather than spelled, so the scan below does not find a name no code reads.
UNREAD = env.PREFIX + "NOT_READ_BY_ANYTHING"

WORLD = "sim: {}\ncomponents:\n  - dummy: {}\n    name: d0\n"

#: The variables a campaign runner sets on a run, which must all keep being read.
CAMPAIGN_VARIABLES = (
    "ROQSIM_RECORD",
    "ROQSIM_LOG_FORMAT",
    "ROQSIM_CAPTURE_FPS",
    "ROQSIM_RECORD_TRACKS",
    "ROQSIM_RECORD_EXCLUDE",
    "ROQSIM_CONTROL",
    "ROQSIM_WORLD",
    "ROQSIM_WORLD_OVERRIDES",
)

REPO = Path(__file__).resolve().parents[2]

#: Where a ``ROQSIM_*`` name in this repository is a variable roqsim reads, documents or sets.
SCANNED = (
    "*/src/**/*.py",
    "*/tests/**/*.py",
    "docs/**/*.rst",
    "*/README.md",
    "README.md",
    "Makefile",
)

_NAME = re.compile(r"\bROQSIM_[A-Z0-9_]+\b")
#: A ``ROQSIM_*`` name read with ``os.environ.get``, ``os.getenv`` or ``os.environ[...]`` (not assigned to).
_DIRECT_READ = re.compile(
    r"""(?:environ\.get|getenv)\(\s*["']ROQSIM_|environ\[\s*["']ROQSIM_\w+["']\s*\](?!\s*=[^=])"""
)


@pytest.fixture
def world(tmp_path):
    path = tmp_path / "w.yaml"
    path.write_text(WORLD)
    return path


@pytest.fixture
def clean_env(monkeypatch):
    for name in list(os.environ):
        if name.startswith(env.PREFIX):
            monkeypatch.delenv(name)
    return monkeypatch


def test_roqsim_sim_refuses_an_unknown_roqsim_variable_by_name(world, clean_env, capsys):
    clean_env.setenv(UNREAD, "1")
    assert runner.main([str(world), "--headless", "--steps", "1", "--control", "none"]) == (
        exit_status.BAD_INPUT
    )
    err = capsys.readouterr().err
    assert err.startswith("roqsim sim: ROQSIM_* environment variables: unknown key(s) ")
    assert repr(UNREAD) in err
    for name in env.registry():
        assert name in err, f"the refusal lists what roqsim reads, and {name} is missing"


def test_the_scenario_adapter_refuses_an_unknown_roqsim_variable_by_name(world, clean_env):
    clean_env.setenv(UNREAD, "1")
    with pytest.raises(env.UnknownEnvironmentError, match=re.escape(repr(UNREAD))) as err:
        MujocoSim(world=str(world))
    assert exit_status.for_error(err.value) == exit_status.BAD_INPUT


@pytest.mark.parametrize("name", sorted(env.registry()))
def test_every_registered_variable_is_accepted(name):
    env.check({name: "1"})


@pytest.mark.parametrize("name", CAMPAIGN_VARIABLES)
def test_every_variable_a_campaign_runner_sets_is_registered(name):
    assert name in env.registry()


def test_the_campaign_variables_together_start_a_run(world, clean_env, tmp_path):
    overrides = tmp_path / "overrides.yaml"
    overrides.write_text("{}\n")
    values = {
        "ROQSIM_RECORD": str(tmp_path / "run.mcap"),
        "ROQSIM_LOG_FORMAT": "stamped",
        "ROQSIM_CAPTURE_FPS": "25",
        "ROQSIM_CONTROL": "none",
        "ROQSIM_WORLD": str(world),
        "ROQSIM_WORLD_OVERRIDES": str(overrides),
    }
    for name, value in values.items():
        clean_env.setenv(name, value)
    assert runner.main([str(world), "--headless", "--steps", "3", "--pacing", "asap"]) == 0
    assert (tmp_path / "run.mcap").exists()
    MujocoSim().shutdown()


def test_reading_an_unregistered_variable_is_a_bug_in_the_reader():
    with pytest.raises(KeyError, match="not a registered roqsim environment variable"):
        env.get(UNREAD)


def test_a_variable_set_outside_the_prefix_is_not_ours():
    env.check({"ROQSIMULATOR": "1", "MY_ROQSIM_THING": "1"})


def test_every_roqsim_name_in_the_repository_is_registered():
    """A name the code or the docs spell is one roqsim reads; one the registry lacks would be refused."""
    seen: dict[str, Path] = {}
    for pattern in SCANNED:
        for path in REPO.glob(pattern):
            for name in _NAME.findall(path.read_text(errors="replace")):
                seen.setdefault(name, path.relative_to(REPO))
    assert seen, "the scan found no ROQSIM_* name at all; the globs no longer match the tree"
    unregistered = {name: str(path) for name, path in seen.items() if name not in env.registry()}
    assert not unregistered, f"named but not registered in roqsim.env: {unregistered}"


def test_package_code_reads_roqsim_variables_through_the_registry():
    direct = [
        f"{path.relative_to(REPO)}:{n}"
        for path in REPO.glob("*/src/**/*.py")
        for n, line in enumerate(path.read_text().splitlines(), 1)
        if _DIRECT_READ.search(line)
    ]
    assert not direct, f"read with os.environ rather than roqsim.env.get: {direct}"


@pytest.mark.parametrize("target", ["run.npz", "run", "takes/"])
def test_roqsim_sim_refuses_a_record_target_that_is_not_an_mcap_file(
    world, tmp_path, capsys, target
):
    given = str(tmp_path / target) + ("/" if target.endswith("/") else "")
    args = [str(world), "--headless", "--steps", "1", "--control", "none", "--record", given]
    assert runner.main(args) == exit_status.BAD_INPUT
    assert f"roqsim sim: a recording is an .mcap file; got {given!r}" in capsys.readouterr().err
    assert sorted(p.name for p in tmp_path.iterdir()) == ["w.yaml"]


def test_a_roqsim_record_that_is_not_an_mcap_file_is_refused(world, clean_env, tmp_path, capsys):
    clean_env.setenv("ROQSIM_RECORD", str(tmp_path / "run.npz"))
    args = [str(world), "--headless", "--steps", "1", "--control", "none"]
    assert runner.main(args) == exit_status.BAD_INPUT
    assert "a recording is an .mcap file; got " in capsys.readouterr().err
    with pytest.raises(
        RecordingTargetError, match=r"a recording is an \.mcap file; got 'run\.npz'"
    ):
        clean_env.setenv("ROQSIM_RECORD", "run.npz")
        MujocoSim(world=str(world))
