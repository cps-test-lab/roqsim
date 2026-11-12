# Copyright (C) 2026 Frederik Pasch
# SPDX-License-Identifier: Apache-2.0
"""A CUDA device MuJoCo Warp can run on, or a skip that says why there is none."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))


def _gpu_reason() -> str | None:
    """None when the backend opens here; else why it cannot, as the backend itself says it."""
    import mujoco  # noqa: PLC0415

    from roqsim_raycast_mjwarp.backend import MjwarpRaycaster  # noqa: PLC0415

    model = mujoco.MjModel.from_xml_string(
        "<mujoco><worldbody><geom size='1'/></worldbody></mujoco>"
    )
    backend = MjwarpRaycaster()
    try:
        backend.open(model)
    except RuntimeError as err:
        return str(err)
    finally:
        backend.close()
    return None


_REASON: list = []


@pytest.fixture(scope="session")
def gpu():
    """Skip, with the backend's own reason, where it cannot open (no Warp, no CUDA device...)."""
    if not _REASON:
        _REASON.append(_gpu_reason())
    if _REASON[0] is not None:
        pytest.skip(f"the mjwarp backend cannot run here: {_REASON[0]}")
