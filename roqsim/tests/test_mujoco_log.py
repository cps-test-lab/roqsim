# Copyright (C) 2026 Frederik Pasch
# SPDX-License-Identifier: Apache-2.0
"""MuJoCo's warnings reach logging, and MuJoCo writes no MUJOCO_LOG.TXT into the current directory."""

import logging

import mujoco
import pytest

import roqsim  # noqa: F401  (its import installs the handler under test)
from roqsim import mujoco_log

_FREE_BODY = "<mujoco><worldbody><body><freejoint/><geom size='0.1'/></body></worldbody></mujoco>"


def _warn_bad_qpos() -> None:
    """A real MuJoCo warning: a NaN in qpos makes mj_step call mju_warning (mjWARN_BADQPOS)."""
    model = mujoco.MjModel.from_xml_string(_FREE_BODY)
    data = mujoco.MjData(model)
    data.qpos[0] = float("nan")
    mujoco.mj_step(model, data)
    assert data.warning[mujoco.mjtWarning.mjWARN_BADQPOS].number == 1


@pytest.fixture
def restore_handler():
    previous = mujoco.get_mju_user_warning()
    yield
    mujoco.set_mju_user_warning(previous)


def test_a_warning_is_logged_and_writes_no_file(tmp_path, monkeypatch, caplog):
    monkeypatch.chdir(tmp_path)
    with caplog.at_level(logging.WARNING, logger=mujoco_log.LOGGER.name):
        _warn_bad_qpos()
    records = [r for r in caplog.records if r.name == mujoco_log.LOGGER.name]
    assert len(records) == 1
    assert records[0].levelno == logging.WARNING
    assert "QPOS" in records[0].getMessage()
    assert list(tmp_path.iterdir()) == []


def test_route_warnings_is_idempotent(restore_handler):
    assert mujoco_log.route_warnings()
    assert mujoco_log.route_warnings()
    assert mujoco.get_mju_user_warning() is mujoco_log._log_warning


def test_a_handler_the_user_installed_is_kept(restore_handler):
    seen = []
    mujoco.set_mju_user_warning(seen.append)
    assert not mujoco_log.route_warnings()
    _warn_bad_qpos()
    assert len(seen) == 1


def test_without_a_handler_mujoco_writes_the_file(tmp_path, monkeypatch, restore_handler):
    """The behaviour the handler exists to prevent, and how to get it back."""
    monkeypatch.chdir(tmp_path)
    mujoco.set_mju_user_warning(None)
    _warn_bad_qpos()
    assert (tmp_path / "MUJOCO_LOG.TXT").exists()
