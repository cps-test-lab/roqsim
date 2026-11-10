# Copyright (C) 2026 Frederik Pasch
# SPDX-License-Identifier: Apache-2.0
"""Shared fixtures: a CUDA GPU or a skip that says so, and the splats the tests draw."""

from __future__ import annotations

import pytest


def _cuda() -> str | None:
    """Why the GPU tests cannot run here, or ``None`` when they can."""
    import torch  # a declared requirement: missing, it fails here, by name

    if not torch.cuda.is_available():
        return "needs a CUDA GPU: torch.cuda.is_available() is false on this machine"
    return None


@pytest.fixture
def gpu():
    """Skip, saying why, where no CUDA GPU is present; the splat renderer draws on one."""
    reason = _cuda()
    if reason:
        pytest.skip(reason)
